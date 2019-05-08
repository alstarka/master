import cv2
import dlib
from math import hypot
import time

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

def place(frame):
    global landmarks
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_eyebrow = (landmarks.part(24).x, landmarks.part(24).y)
        left_eyebrow = (landmarks.part(22).x, landmarks.part(22).y)
        right_eyebrow = (landmarks.part(26).x, landmarks.part(26).y)

        head_width = int(hypot(left_eyebrow[0] - right_eyebrow[0],
                               left_eyebrow[1] - right_eyebrow[1]) / 2)
        head_height = int(head_width * 2)

        x1 = int(left_eyebrow[0] - (head_width / 2))
        x2 = int(left_eyebrow[0] + (right_eyebrow[0] / 2) + (head_width *3))
        y1 = int(left_eyebrow[1] - (center_eyebrow[1] / 2) - (head_height * 2))
        y2 = int(left_eyebrow[1] - (center_eyebrow[1] / 4) + (head_height / 2))

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

        # calculate the masks for the overlay
        eyeOverlay = cv2.resize(img, (head_width, head_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (head_width, head_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (head_width, head_height), interpolation=cv2.INTER_AREA)

        # take ROI for the verlay from background, equal to size of the overlay image
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
def doRecognizePerson(faceNames, fid)
    time.sleep(2)
    faceNames[ fid ] = filters

filters= ['img/Bild1.png', 'img/Bild2.png', 'img/Bild3.png', 'img/Bild4.png']
filterIndex =0
img = cv2.imread(filters[filterIndex], -1)

# Create the mask from the overlay image
orig_mask = img[:, :, 3]
# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert the overlay image image to BGR
# and save the original image size
img = img[:, :, :3]
origEyeHeight, origEyeWidth = img.shape[:2]

# Start capturing the WebCam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    place(frame)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', frame)

    ch = 0xFF & cv2.waitKey(1)

    if ch == ord('q'):
        break

cv2.destroyAllWindows()