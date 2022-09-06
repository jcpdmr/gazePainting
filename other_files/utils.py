import argparse

import cv2 as cv2
import numpy as np
import torch
import torch.nn as nn
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin
from pathlib import Path
import subprocess
import re
from other_files.model import L2CS
import torchvision
import sys

GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
frame_width = 1280
frame_height = 720
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# These constants are used to access and elaborate the SQL database that is a matrix
PAINTING_NAME_COLUMN = 1
SECONDS_COLUMN = 2
GENDER_COLUMN = 3
GENDER_SCORE_COLUMN = 4
AGE_INTERVALS_COLUMN = 5
AGE_SCORE_COLUMN = 6


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


def draw_gaze(a, b, c, d, image_in, pitchyaw, thickness=2, color=(255, 255, 0), sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out


def select_device(device='', batch_size=None):
    device = '0'
    # device = "cpu"
    # s = f'YOLOv3 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    s = f'YOLOv3 🚀 {git_describe()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')


def spherical2cartesial(x):
    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])

    return output


def compute_angular_error(input, target):
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1, 3, 1)
    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180 * torch.mean(output_dot) / math.pi
    return output_dot


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                  np.array([frame.shape[1], frame.shape[0],
                            frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                                             10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation=inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str, required=True)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str, required=True)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int, required=True)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    parser.add_argument('--calib_painting_path', dest='cal_painting_path', help='Path of the folder of painting '
                                                                                'calibration [Example: C:/.../calibration/venere]',
                        type=str, required=True)

    parser.add_argument('--painting_name', dest='paint_name', help='Insert the name of the painting',
                        type=str, required=True)

    args = parser.parse_args()
    return args


# In the next functions result is a matrix where every row has datetime, painting_name, seconds, gender, gender_score,
# age, age_score. If you want to access a specific field of the table you need to do for example result[0][2], as
# you do in every matrix
def getPaintingNames(result):
    painting_names = []
    for row in range(len(result)):
        painting_name = result[row][PAINTING_NAME_COLUMN]
        if painting_name not in painting_names:
            painting_names.append(painting_name)
    return painting_names


def getInfoPerPainting(result):
    # This function creates arrays where each element has info [example: total_seconds, single_interactions] about
    # the painting_names element with the same index
    painting_names = getPaintingNames(result)
    # Pre allocate the seconds_array with all zeros, one array for male, one for female, one for the total
    total_seconds_array_male = [0] * len(painting_names)
    total_seconds_array_female = [0] * len(painting_names)
    total_seconds_array = [0] * len(painting_names)
    # Pre allocate the single_interactions_array with all zeros, one array for male, one for female, one for the total
    single_interactions_array_male = [0] * len(painting_names)
    single_interactions_array_female = [0] * len(painting_names)
    single_interactions_array_total = [0] * len(painting_names)
    # Pre allocate the age_interval_0_array,1,2,3... with all zeros for the array_total
    age_interval_0_array_total = [0] * len(painting_names)
    age_interval_1_array_total = [0] * len(painting_names)
    age_interval_2_array_total = [0] * len(painting_names)
    age_interval_3_array_total = [0] * len(painting_names)
    age_interval_4_array_total = [0] * len(painting_names)
    age_interval_5_array_total = [0] * len(painting_names)
    age_interval_6_array_total = [0] * len(painting_names)
    age_interval_7_array_total = [0] * len(painting_names)


    for row in range(len(result)):
        painting_name = result[row][PAINTING_NAME_COLUMN]
        # Find which painting of the list painting_names is in this row of the table
        painting_index = painting_names.index(painting_name)
        # Add the seconds to the total of this particular painting
        total_seconds_array[painting_index] += result[row][SECONDS_COLUMN]
        # Add one single interaction to the total interactions of this particular painting
        single_interactions_array_total[painting_index] += 1
        # Then find the gender of this row
        gender = result[row][GENDER_COLUMN]
        age_interval = result[row][AGE_INTERVALS_COLUMN]
        if gender == "Male":
            # You need to add the value of seconds in the male array at a precise index corresponding to a precise
            # painting
            total_seconds_array_male[painting_index] += result[row][SECONDS_COLUMN]
            # You need to add 1 to the number of interactions in the male array at a precise index corresponding to a
            # precise painting
            single_interactions_array_male[painting_index] += 1
        elif gender == "Female":
            # Same logic as gender == "Male" just different array
            total_seconds_array_female[painting_index] += result[row][SECONDS_COLUMN]
            single_interactions_array_female[painting_index] += 1
        else:
            print(
                f"Something went wrong while trying to classify the gender in row {row}, it's not 'Male' nor 'Female'")

        if age_interval == AGE_INTERVALS[0]:
            # Add one to the total interactions in this particular age_interval of this particular painting
            age_interval_0_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[1]:
            age_interval_1_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[2]:
            age_interval_2_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[3]:
            age_interval_3_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[4]:
            age_interval_4_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[5]:
            age_interval_5_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[6]:
            age_interval_6_array_total[painting_index] += 1
        elif age_interval == AGE_INTERVALS[7]:
            age_interval_7_array_total[painting_index] += 1
        else:
            print(f"Something went wrong while trying to classify the age intervals in row {row}")

    # AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

    return total_seconds_array_male, total_seconds_array_female, total_seconds_array, single_interactions_array_male,\
           single_interactions_array_female, single_interactions_array_total, age_interval_0_array_total, \
           age_interval_1_array_total, age_interval_2_array_total, age_interval_3_array_total, \
           age_interval_4_array_total, age_interval_5_array_total, age_interval_6_array_total, \
           age_interval_7_array_total
