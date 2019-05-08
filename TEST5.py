# links testen mit rechts abgleichen... weinen
import numpy as np
import cv2
import dlib
from math import hypot

# Die Bibliothek laden
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Das Bild über Kopf laden
imgSprechblase = cv2.imread('itk_logo.png', -1)

# Maske für das Bild erstellen
orig_mask = imgSprechblase[:, :, 3]
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Das Bild in BGR konvertierten und das original Bild speichern
imgSprechblase = imgSprechblase[:, :, 0:3]
origHeight, origWidth = imgSprechblase.shape[:2]

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        landmarks = predictor(gray, face)

        top_eye = (landmarks.part(44).x, landmarks.part(44).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        left_eye = (landmarks.part(42).x, landmarks.part(42).y)

        head_width = int(hypot(left_eye[0] - right_eye[0],
                               left_eye[1] - right_eye[1]) * 4)
        head_height = int(head_width * 0.77)

        x1 = int(top_eye[0] - head_width / 2)
        x2 = int(top_eye[0] + head_height / 2)
        y1 = int(top_eye[1] - head_width / 2)
        y2 = int(top_eye[1] + head_height / 2)

        h, w = frame.shape[:2]

        # check for clipping
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h

        # re-calculate the size to avoid clipping
        head_width = x2 - x1
        head_height = y2 - y1

        roi_gray = gray[y1:y2 + head_height, x1:x2 + head_width]
        roi_color = frame[y1:y2 + head_height, x1:x2 + head_width]

        sprechblase = cv2.resize(imgSprechblase, (head_width, head_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (head_width, head_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (head_width, head_height), interpolation=cv2.INTER_AREA)

        roi = roi_color[y1:y2 + head_height, x1:x2 + head_width]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        roi_fg = cv2.bitwise_and(sprechblase, sprechblase, mask=mask)

        dst = cv2.add(roi_bg, roi_fg)
        roi_color[y1:y2 + head_height, x1:x2 + head_width] = dst
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # press any key to exit
    # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
