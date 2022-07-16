from datetime import datetime
import os

import numpy as np
import cv2 as cv2
import time

# pip install --extra-index-url https://download.pytorch.org/whl/cu113/ "torch==1.11.0+cu113"
import torch
import torch.nn as nn
from sqlalchemy.orm import sessionmaker
from torch.autograd import Variable

from torchvision import transforms
import torch.backends.cudnn as cudnn

from other_files import tracker_utils

from other_files.utils import select_device, draw_gaze, get_age_predictions, get_gender_predictions, getArch, parse_args, \
    GENDER_LIST, AGE_INTERVALS


from PIL import Image

# pip install git+https://github.com/elliottzheng/face-detection.git@master
from face_detection import RetinaFace

import db

# Count the cycle of the program
counter = 0

# OPT: Enable/Disable Age-Gender Estimation True-False
age_gender_enabled = True

MAX_NUM_IDS_ARRAY_DIMENSION = 1000
# This array will be used to store the time that an id is looking in a precise zone
times_array = np.zeros(MAX_NUM_IDS_ARRAY_DIMENSION)
# This array will be used to store the gender of the ids
gender_array = [None] * MAX_NUM_IDS_ARRAY_DIMENSION
# This array will be used to store the age estimation of the ids
age_array = [None] * MAX_NUM_IDS_ARRAY_DIMENSION
# This array will be used to store the gender confidence scores of the ids
gender_score_array = np.zeros(MAX_NUM_IDS_ARRAY_DIMENSION)
# This array will be used to store the age confidence scores of the ids
age_score_array = np.zeros(MAX_NUM_IDS_ARRAY_DIMENSION)

# MAX_PITCH, MIN_PITCH, MAX_YAW, MIN_YAW = 0, 0, 0, 0
MAX_PITCH, MIN_PITCH, MAX_YAW, MIN_YAW = 30, -30, 30, -30

past_active_ids = []

Session = sessionmaker(bind=db.engine)
session = Session()

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot
    painting_name = args.paint_name
    # Search for calibration file
    cal_painting_path = args.cal_painting_path
    # Check if the calibration exists
    calibration_exists = os.path.exists(f"{cal_painting_path}/calibration.txt")
    if calibration_exists:
        # If calibration exists load pitch - yaw values
        with open(os.path.join(cal_painting_path, "calibration.txt"), "r+") as flist:
            lines = flist.readlines()
            MAX_PITCH = float(lines[0].strip())
            MIN_PITCH = float(lines[1].strip())
            MAX_YAW = float(lines[2].strip())
            MIN_YAW = float(lines[3].strip())
    else:
        print("Something has gone wrong while loading pitch - yaw calibration values, check the calibration folder...")
        exit()

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

    # Lowering the value gives different id to the same object if it's moving quickly
    tracker = tracker_utils.EuclideanDistTracker(50)

    # cap = cv2.VideoCapture("sample_short.mp4")
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        print("Output is starting...")

        # Main Loop
        while True:
            success, frame = cap.read()
            start_time = time.time()
            output_frame = np.zeros(1)

            faces = detector(frame)

            detected_faces = []
            active_ids = []

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

                    # Append the faces in the list
                    detected_faces.append([x_min, y_min, bbox_width, bbox_height])

            # Update the boxes adding a new value that is the id
            boxes_ids = tracker.update(detected_faces)

            for box_id in boxes_ids:
                x_min, y_min, bbox_width, bbox_height, id_num = box_id
                x_max = x_min + box_id[2]
                y_max = y_min + box_id[3]
                # Add this face id to the current monitoring faces that need to be put in database [time , gender]
                active_ids.append(id_num)

                # Crop image and make a original copy that will be used in Age-Gender estimation
                img = frame[y_min:y_max, x_min:x_max]
                img_copy = cv2.resize(img, (int(box_id[2] * 1.5), int(box_id[3] * 1.5)), interpolation=cv2.INTER_AREA)
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

                # Age - Gender prediction every 8 cycles if enabled
                if counter % 8 == 0 and age_gender_enabled:
                    # predict age
                    age_preds = get_age_predictions(
                        img_copy)

                    # predict gender
                    gender_preds = get_gender_predictions(
                        img_copy)
                    i = gender_preds[0].argmax()
                    gender = GENDER_LIST[i]
                    # Save them in their respective arrays
                    gender_array[id_num] = gender
                    gender_confidence_score = gender_preds[0][i]
                    gender_score_array[id_num] = gender_confidence_score
                    i = age_preds[0].argmax()
                    age = AGE_INTERVALS[i]
                    age_array[id_num] = age
                    age_confidence_score = age_preds[0][i]
                    age_score_array[id_num] = age_confidence_score

                # Checking if the current face is watching a precise zone and add the time
                if MIN_PITCH <= pitch_predicted_degree <= MAX_PITCH and MIN_YAW <= yaw_predicted_degree <= MAX_YAW:
                    times_array[id_num] = times_array[id_num] + (time.time() - start_time)

                # Drawing part
                output_frame = frame
                # Drawing gaze estimation
                draw_gaze(x_min, y_min, bbox_width, bbox_height, output_frame, (pitch_predicted, yaw_predicted),
                          color=(0, 0, 255))
                pitch_label = f"pitch: {pitch_predicted_degree:.3f}"
                yaw_label = f" yaw: {yaw_predicted_degree:.3f}"
                cv2.putText(output_frame, pitch_label, (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0),
                            1,
                            cv2.LINE_AA)
                cv2.putText(output_frame, yaw_label, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0),
                            1,
                            cv2.LINE_AA)

                # Drawing id
                id_label = f"   id: {id_num}"
                cv2.putText(output_frame, id_label, (x_min, y_max + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                            (0, 255, 0), 1, cv2.LINE_AA)

                # # Drawing the grid
                # cv2.line(output_frame, (int((x_min + x_max) / 2), y_min), (int((x_min + x_max) / 2), y_max), (0, 255, 0),
                #          thickness=1)
                # cv2.line(output_frame, (x_min, int((y_min + y_max) / 2)), (x_max, int((y_min + y_max) / 2)), (0, 255, 0),
                #          thickness=1)

                # Drawing Age-Gender estimation if enabled
                if age_gender_enabled:
                    age_gender_label = f"{gender}-{gender_confidence_score * 100:.1f}%, {age}-{age_confidence_score * 100:.1f}%"
                    yPos = y_min - 15
                    while yPos < 15:
                        yPos -= 15
                    box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                    cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), box_color, 1)
                    # Label processed image
                    cv2.putText(output_frame, age_gender_label, (x_min, yPos), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                                box_color,
                                2)
                else:
                    # Drawing bounding box
                    cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

                # Drawing watch time
                time_label = f"time: {times_array[id_num]:.1f}"
                cv2.putText(output_frame, time_label, (x_min, y_max + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                            (0, 255, 0), 1, cv2.LINE_AA)
            # Drawing other info
            if output_frame.any():
                myFPS = 1.0 / (time.time() - start_time)
                cv2.putText(output_frame, 'FPS: {:.1f}'.format(myFPS), (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.54,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA)

                cv2.putText(output_frame, 'AGE-GEN: {}'.format("Y" if age_gender_enabled else "N"), (545, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.54,
                            (0, 255, 0), 1,
                            cv2.LINE_AA)

                cv2.imshow("Demo", output_frame)
                # Increase the counter
                counter += 1
            else:
                cv2.imshow("Demo", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Let's find out the tracker ids that we've lost the tracking
            lost_tracking_ids = list(set(past_active_ids).difference(active_ids))
            for lost_id in lost_tracking_ids:
                # Check if the id has watched the zone for more than 4 seconds and save the infos
                if times_array[lost_id] > 4:
                    lost_label = f"The id {lost_id} has stopped watching after {times_array[lost_id]:.1f} seconds;  " \
                                 f"the gender is {gender_array[lost_id]} (probability: {gender_score_array[lost_id]:1f});  " \
                                 f"the age is {age_array[lost_id]} (probability: {age_score_array[lost_id]:1f})"
                    print(lost_label)
                    # Add data to our table in database [remember to commit the changes]
                    data = db.history(datetime.utcnow(), painting_name, round(times_array[lost_id], 1),
                                      gender_array[lost_id], int(gender_score_array[lost_id] * 100),
                                      str(age_array[lost_id]), int(age_score_array[lost_id] * 100))
                    session.add(data)
            # Update the past active ids with the ones founded in this cycle
            past_active_ids = active_ids
        # Before exiting the program we commit the changes made in the current session in the database
        print("Committing changes in the database")
        session.commit()
        cap.release()
        cv2.destroyAllWindows()
