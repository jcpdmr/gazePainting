import argparse
import os

import numpy as np
import cv2 as cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from utils import select_device, get_faces, image_resize
from PIL import Image

from face_detection import RetinaFace
from model import L2CS

yaw_calibration = []
pitch_calibration = []


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    parser.add_argument('--calib_painting_path', dest='cal_painting_path', help='Path of the folder of '
                        'painting calibration [Example: C:/.../calibration/venere]', type=str)

    args = parser.parse_args()
    return args


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


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot
    cal_painting_path = args.cal_painting_path

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x = 0

    with torch.no_grad():
        listdir = os.listdir(cal_painting_path)

        # Filtering other non .jpg files
        filtered_listdir = []
        for i in range(len(listdir)):
            if listdir[i].find(".jpg") != -1:
                filtered_listdir.append(listdir[i])
        if len(filtered_listdir) != 8:
            print("There are less than 8 photos for calibration!")
            exit()

        # Analyze every photo
        for i in range(len(filtered_listdir)):
            cap = cv2.VideoCapture(os.path.join(cal_painting_path, filtered_listdir[i]))
            success, frame = cap.read()

            faces = detector(frame)

            # Eliminate faces with a low score [false positive]
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min = int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img = transformations(im_pil)
                    img = Variable(img).cuda(gpu)
                    img = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

                    pitch_predicted_degree = pitch_predicted * 180.0 / np.pi
                    yaw_predicted_degree = yaw_predicted * 180.0 / np.pi

                    # Append the pitch and yaw of the current photo to pitch-yaw calibration list
                    pitch_calibration.append(pitch_predicted_degree)
                    yaw_calibration.append(yaw_predicted_degree)

        if (len(yaw_calibration) < 8) or (len(pitch_calibration) < 8):
            print("Less than 8 pitch-yaw calibration values, probably bad taken photo...")
            exit()

        # Choose the biggest/smallest values of pitch and yaw from calibration [float64 value]
        pitch_max = max(pitch_calibration)
        pitch_min = min(pitch_calibration)
        yaw_max = max(yaw_calibration)
        yaw_min = min(yaw_calibration)

        # Write the calibration settings in the calibration.txt file
        with open(os.path.join(cal_painting_path, "calibration.txt"), "w") as f:
            print(f"Writing the calibration settings in the {cal_painting_path}\\calibration.txt file : \n"
                  f"PITCH_MAX = {pitch_max:.2f}; PITCH_MIN = {pitch_min:.2f}; YAW_MAX = {yaw_max:.2f}; "
                  f"YAW_MIN = {yaw_min:.2f};")
            f.write(f"{pitch_max:.2f}" + "\n")
            f.write(f"{pitch_min:.2f}" + "\n")
            f.write(f"{yaw_max:.2f}" + "\n")
            f.write(f"{yaw_min:.2f}" + "\n")
            f.close()
