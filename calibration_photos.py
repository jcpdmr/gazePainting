import cv2
import os

cam = cv2.VideoCapture(0)

img_counter = 1

name = str(input("Insert the name of the painting [use just letters and _ ]: "))

CALIBRATION_DIR = os.path.join(os.getcwd(), "calibration")
if not os.path.isdir(CALIBRATION_DIR):
    os.mkdir(CALIBRATION_DIR)

THIS_CALIBRATION_DIR = os.path.join(CALIBRATION_DIR, f"{name}")
if not os.path.isdir(THIS_CALIBRATION_DIR):
    os.mkdir(THIS_CALIBRATION_DIR)
else:
    # In case we are trying to save the photos in an already existing folder
    print(f"The name you've chosen already exist, check   calibration/{name}   folder to avoid overwriting")
    exit()

print("BE SURE TO MAKE A PHOTO WITH ONLY A FACE INSIDE")
print("img 1: Move to the right and look at the top left corner, then press SPACE")
print("img 2: Move to the right and look at the bottom left corner, then press SPACE")
print("img 3: Move to the right and look at the top right corner, then press SPACE")
print("img 4: Move to the right and look at the bottom right corner, then press SPACE")

print("img 5: Move to the left and look at the top left corner, then press SPACE")
print("img 6: Move to the left and look at the bottom left corner, then press SPACE")
print("img 7: Move to the left and look at the top right corner, then press SPACE")
print("img 8: Move to the left and look at the bottom right corner, then press SPACE")

cv2.namedWindow("Calibration")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Calibration", frame)

    k = cv2.waitKey(1)

    # First position
    if img_counter == 1:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1
    elif img_counter == 2:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg written!")
            img_counter += 1
    elif img_counter == 3:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1
    elif img_counter == 4:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1

    # Second position
    elif img_counter == 5:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1
    elif img_counter == 6:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1
    elif img_counter == 7:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1
    elif img_counter == 8:
        if k % 256 == 32:
            # SPACE pressed
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(os.path.join(THIS_CALIBRATION_DIR, img_name), frame)
            print(f"{name}_{img_counter}.jpg  written!")
            img_counter += 1
    elif img_counter == 9:
        print("All photos has been taken, exiting...")
        break

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()
