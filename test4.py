import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
nose_image = cv2.imread("itk_logo.png",1)
cap = cv2.VideoCapture(0)

_, frame = cap.read()
rows, cols, _ = frame.shape
#eyebrow_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    #eyebrow_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Nose coordinates
        top_eye = (landmarks.part(44).x, landmarks.part(44).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        left_eye = (landmarks.part(42).x, landmarks.part(42).y)

        head_width = int(hypot(left_eye[0] - right_eye[0],
                               left_eye[1] - right_eye[1]) * 4)
        head_height = int(head_width * 0.77)

        # New nose position
        # top_right=int(top_eyebrow[0] - )
        x1 = (int(top_eye[0] - head_width / 2))
        x2= (int(top_eye[1] - head_height / 2))
        y1 = (int(top_eye[0] + head_width / 2))
        y2= (int(top_eye[1] + head_height / 2))

        h, w = frame.shape[:2]

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h

            head_width = x2 - x1
            head_height = y2 - y1

            # Adding the new nose
        nose_pig = cv2.resize(nose_image, (head_width, head_height))
        mask = cv2.resize(orig_mask, (head_width, head_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (head_width, head_height), interpolation=cv2.INTER_AREA)

        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, eyebrow_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(eyebrow_mask)
        origheadHeight, origheadWidth = nose_image.shape[:2]

        roi = frame[y1:y2, x1:x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        roi_fg= cv2.bitwise_and(nose_pig, nose_pig, mask=mask)
        final_nose = cv2.add(roi_bg, roi_fg)

        frame[y1:y1, x1:x2] = final_nose

        orig_mask: object = nose_image[:, :, 3]

        orig_mask_inv = cv2.bitwise_not(orig_mask)

        imgEye = imgEye[:, :, 0:3]

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break