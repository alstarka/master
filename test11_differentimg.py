import cv2
import dlib
from math import hypot
import time

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

imgEye = cv2.imread('img/Bild4.png', -1)


def place(frame):
    global landmarks
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_eye = (landmarks.part(44).x, landmarks.part(44).y)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        sumeyerx=0
        sumeyery=0
        sumeyelx=0
        sumeyely=0
        sumfacex = 0
        sumfacey = 0

        for i in range(36, 41):
            sumeyelx += landmarks.part(i).x
            sumeyely += landmarks.part(i).y
        medianeyelx = sumeyelx / 6
        medianeyely = sumeyely / 6

        for i in range(42, 47):
            sumeyerx += landmarks.part(i).x
            sumeyery += landmarks.part(i).y
        medianeyerx = sumeyerx / 6
        medianeyery = sumeyery / 6

        distance=int(hypot(medianeyelx - medianeyerx,
                               medianeyely - medianeyery))

        for i in range(1, 67):
            sumfacex += landmarks.part(i).x
            sumfacey += landmarks.part(i).y
        medianfacex = sumfacex / 67
        medianfacey = sumfacey / 67



        head_width = int(hypot(left_eye[0] - right_eye[0],
                               left_eye[1] - right_eye[1]) / 2)

        # head_width = int(left_eye[0] - right_eye[0])/2

        head_height = int(head_width * 4)
        offsetx = distance
        offsety = -4*distance
        scale = distance*2
        x1 = int(offsetx + medianfacex)  # - (head_width / 2))
        x2 = int(offsetx + medianfacex + scale)  # right_eye[0] / 2) + (head_width / 2))
        y1 = int(offsety + medianfacey)  # - (center_eye[1] / 2) - (head_height / 2))
        y2 = int(offsety + medianfacey + scale)  # - (center_eye[1] / 2) + (head_height / 2))

        roi_width = (x2 - x1)
        roi_height = (y2 - y1)





        if x2 < 640 and x1 > 0 and y1 > 0:
            # calculate the masks for the overlay
            eyeOverlay = cv2.resize(imgEye, (roi_width, roi_height))
            #eyeOverlay_gray = cv2.cvtColor(eyeOverlay, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(orig_mask, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
            # ret, eye_mask = cv2.threshold(eyeOverlay_gray, 20, 255, cv2.THRESH_BINARY)

            # take ROI for the overlay from background, equal to size of the overlay image
            roi = frame[y1:y2, x1:x2]

            # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            # roi_fg contains the image pixels of the overlay only where the overlay should be
            roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            frame[y1:y2, x1:x2] = dst
    # ---------------------------------------------------------
    # Load and pre-process the eye-overlay
    # ---------------------------------------------------------


# Load the image to be used as our overlay

# Create the mask from the overlay image
orig_mask = imgEye[:, :, 3]

# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert the overlay image image to BGR
# and save the original image size
imgEye = imgEye[:, :, :3]
origEyeHeight, origEyeWidth = imgEye.shape[:2]

# Start capturing the WebCam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    place(frame)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame)

    ch = 0xFF & cv2.waitKey(1)

    if ch == ord('q'):
        break

cv2.destroyAllWindows()
