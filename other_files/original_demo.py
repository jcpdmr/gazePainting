import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision


from utils import select_device, draw_gaze, get_faces, image_resize, get_age_predictions, get_gender_predictions
from PIL import Image

from face_detection import RetinaFace
from model import L2CS

GENDER_MODEL = 'weights/deploy_gender.prototxt'
# The gender model pre-trained weights
GENDER_PROTO = 'weights/gender_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the gender classes
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# The model architecture
AGE_MODEL = 'weights/deploy_age.prototxt'
# The model pre-trained weights
AGE_PROTO = 'weights/age_net.caffemodel'
# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# Initialize frame size
frame_width = 1280
frame_height = 720
# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

counter = 0

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

    # cap = cv2.VideoCapture("prova.mp4")
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        print("Output started!")
        # Main Loop
        while True:
            success, frame = cap.read()
            start_fps = time.time()
            frame2 = frame.copy()

            faces = detector(frame)
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
                    # x_min = max(0,x_min-int(0.2*bbox_height))
                    # y_min = max(0,y_min-int(0.2*bbox_width))
                    # x_max = x_max+int(0.2*bbox_height)
                    # y_max = y_max+int(0.2*bbox_width)
                    # bbox_width = x_max - x_min
                    # bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img_copy = img
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

                    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
                    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0

                    pitch_predicted_degree = pitch_predicted * 180.0 / np.pi
                    yaw_predicted_degree = yaw_predicted * 180.0 / np.pi

                    # Mean of the last 5 value
                    # pitch_array[counter % 5] = pitch_predicted
                    # yaw_array[counter % 5] = yaw_predicted
                    #
                    # if (counter % 5 == 0):
                    #     pitch_predicted_degree = pitch_array.mean() * 180.0 / np.pi
                    #     yaw_predicted_degree = yaw_array.mean() * 180.0 / np.pi

                    if (counter % 8 == 0):
                        # Age - Gender prediction every 8 cycles

                        # predict age
                        age_preds = get_age_predictions(img_copy)

                        # predict gender
                        gender_preds = get_gender_predictions(img_copy)
                        i = gender_preds[0].argmax()
                        gender = GENDER_LIST[i]
                        gender_confidence_score = gender_preds[0][i]
                        i = age_preds[0].argmax()
                        age = AGE_INTERVALS[i]
                        age_confidence_score = age_preds[0][i]
                    counter += 1

                    # Drawing part

                    # Drawing gaze estimation
                    draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch_predicted, yaw_predicted),
                              color=(0, 0, 255))
                    pitch_label = f"pitch: {pitch_predicted_degree:.3f}"
                    yaw_label = f" yaw: {yaw_predicted_degree:.3f}"
                    cv2.putText(frame, pitch_label, (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.putText(frame, yaw_label, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0), 1,
                                cv2.LINE_AA)

                    # Drawing the grid
                    cv2.line(frame, (int((x_min + x_max) / 2), y_min), (int((x_min + x_max) / 2), y_max), (0, 255, 0),
                             thickness=1)
                    cv2.line(frame, (x_min, int((y_min + y_max) / 2)), (x_max, int((y_min + y_max) / 2)), (0, 255, 0),
                             thickness=1)

                    # Drawing Age-Gender estimation
                    agegender_label = f"{gender}-{gender_confidence_score * 100:.1f}%, {age}-{age_confidence_score * 100:.1f}%"
                    yPos = y_min - 15
                    while yPos < 15:
                        yPos -= 15
                    box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    # Label processed image
                    cv2.putText(frame, agegender_label, (x_min, yPos), cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)

            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)

            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
