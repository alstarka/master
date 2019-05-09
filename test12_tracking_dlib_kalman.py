#!/usr/bin/python
'''
    Author: Guido Diepen <gdiepen@deloitte.nl>
'''

#Import the OpenCV and dlib libraries
import cv2
import dlib
import numpy as np
from imutils import face_utils

import threading
import time

#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
#Make sure that you copy this file from the opencv project to the root of this
#project folder
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600

textlist=[]*5
textlist=["hallo","was ist das","hab dich lieb","R+A","puppi"]

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Load the image to be used as our overlay
imgEye = cv2.imread('img/Bild4.png', -1)
# Create the mask from the overlay image
orig_mask = imgEye[:, :, 3]

# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert the overlay image image to BGR
# and save the original image size
imgEye = imgEye[:, :, :3]
origEyeHeight, origEyeWidth = imgEye.shape[:2]
origEyeHeight, origEyeWidth = imgEye[0].shape[:2]





# init kalman filter object

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.03

measurement = np.array((2,1), np.float32)
prediction = np.zeros((2,1), np.float32)


#We are not doing really face recognition
def doRecognizePerson(faceNames, fid):
    time.sleep(2)
    faceNames[ fid ] = "Person " + str(fid)




def detectAndTrackMultipleFaces():
    #Open the first webcame device
    capture = cv2.VideoCapture(0)

    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    cv2.moveWindow("base-image",0,100)
    cv2.moveWindow("result-image",400,100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}

    try:
        while True:
            #Retrieve the latest image from the webcam
            rc,fullSizeBaseImage = capture.read()

            #Resize the image to 320x240
            baseImage = cv2.resize( fullSizeBaseImage, ( 640, 480))

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                break



            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            resultImage = baseImage.copy()




            #STEPS:
            # * Update all trackers and remove the ones that are not
            #   relevant anymore
            # * Every 10 frames:
            #       + Use face detection on the current frame and look
            #         for faces.
            #       + For each found face, check if centerpoint is within
            #         existing tracked box. If so, nothing to do
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new face-id


            #Increase the framecounter
            frameCounter += 1



            #Update all the trackers and remove the ones for which the update
            #indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( baseImage )

                #If the tracking quality is good enough, we must delete
                #this tracker
                if trackingQuality < 7:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )




            #Every 10 frames, we will have to determine which faces
            #are present in the frame
            if (frameCounter % 10) == 0:



                #For the face detection, we need to make use of a gray
                #colored image so we will convert the baseImage to a
                #gray-based image
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                #Now use the haar cascade detector to find all faces
                #in the image
                faces = detector(gray,1)

                def rect_to_bb(rect):
                    # take a bounding predicted by dlib and convert it
                    # to the format (x, y, w, h) as we would normally do
                    # with OpenCV
                    x = rect.left()
                    y = rect.top()
                    w = rect.right() - x
                    h = rect.bottom() - y

                    # return a tuple of (x, y, w, h)
                    return (x, y, w, h)

                #Loop over all faces and check if the area for this
                #face is the largest so far
                #We need to convert it to int here because of the
                #requirement of the dlib tracker. If we omit the cast to
                #int here, you will get cast errors since the detector
                #returns numpy.int32 and the tracker requires an int
                for (fid) in faces:

                    (x, y, w, h) = face_utils.rect_to_bb(fid)

                    #calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h



                    #Variable holding information which faceid we
                    #matched with
                    matchedFid = None

                    #Now loop over all the trackers and check if the
                    #centerpoint of the face is within the box of a
                    #tracker
                    for fid in faceTrackers.keys():
                        tracked_position =  faceTrackers[fid].get_position()



                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())


                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        #check if the centerpoint of the face is within the
                        #rectangleof a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region
                        #detected as a face. If both of these conditions hold
                        #we have a match
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and
                             ( t_y <= y_bar   <= (t_y + t_h)) and
                             ( x   <= t_x_bar <= (x   + w  )) and
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid


                    #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:

                        print("Creating new tracker " + str(currentFaceID))

                        #Create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle( x-10,
                                                            y-20,
                                                            x+w+10,
                                                            y+h+20))

                        faceTrackers[ currentFaceID ] = tracker

                        #Start a new thread that is used to simulate
                        #face recognition. This is not yet implemented in this
                        #version :)
                        t = threading.Thread( target = doRecognizePerson ,
                                               args=(faceNames, currentFaceID))
                        t.start()

                        #Increase the currentFaceID counter

                        currentFaceID += 1

                        if currentFaceID>4:
                            currentFaceID=0


            #Now loop over all the trackers we have and draw the rectangle
            #around the detected faces. If we 'know' the name for this person
            #(i.e. the recognition thread is finished), we print the name
            #of the person, otherwise the message indicating we are detecting
            #the name of the person
            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                r_x = int(tracked_position.left())
                r_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                #coord = np.array([np.float32(r_x), np.float32(r_y),np.float32(r_w), np.float32(r_h)],np.float32)
                coord = np.array([np.float32(r_x), np.float32(r_y)], np.float32)
                kalman.correct(coord)
                prediction_coord = kalman.predict();
                t_x=int(prediction_coord[0])
                t_y=int(prediction_coord[1])
                #t_w=int(prediction_coord[2])
                #t_h=int(prediction_coord[3])
                """
                cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w , t_y + t_h),
                                        rectangleColor ,2)
                

                if fid in faceNames.keys():
                    cv2.putText(resultImage, textlist[fid] ,
                                (int(t_x + t_w/2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting..." ,
                                (int(t_x + t_w/2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                """
                offsety=-15
                offsetx=100
                x1=(t_x +offsetx)
                x2=t_x  #t_x + ((len(textlist[fid]))*15 +offsetx)*t_w/100
                y1=t_y + offsety
                y2=t_x  +offsety

                roi_width = x2-x1
                roi_height = y2-y1

                if x2 > 0 and x1 < 775 and y1 > 0 and y1 < 500:
                    # calculate the masks for the overlay
                    eyeOverlay = cv2.resize(imgEye, (roi_width, roi_height))
                    # eyeOverlay_gray = cv2.cvtColor(eyeOverlay, cv2.COLOR_BGR2GRAY)
                    mask = cv2.resize(orig_mask, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
                    mask_inv = cv2.resize(orig_mask_inv, (roi_width, roi_height), interpolation=cv2.INTER_AREA)
                    # ret, eye_mask = cv2.threshold(eyeOverlay_gray, 20, 255, cv2.THRESH_BINARY)

                    # take ROI for the overlay from background, equal to size of the overlay image
                    roi = resultImage[y1:y2, x1:x2]

                    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    # roi_fg contains the image pixels of the overlay only where the overlay should be
                    roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)
                    # join the roi_bg and roi_fg
                    dst = cv2.add(roi_bg, roi_fg)

                    # place the joined image, saved to dst back over the original image
                    resultImage[y1:y2, x1:x2] = dst



                if fid in faceNames.keys():
                    cv2.putText(resultImage, str(roi_width),#textlist[fid] ,
                                (int(t_x + t_w/2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting..." ,
                                (int(t_x + t_w/2), int(t_y)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)



            #Since we want to show something larger on the screen than the
            #original 320x240, we resize the image again
            #
            #Note that it would also be possible to keep the large version
            #of the baseimage and make the result image a copy of this large
            #base image and use the scaling factor to draw the rectangle
            #at the right coordinates.
            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

            #Finally, we want to show the images on the screen
            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)




    #To ensure we can also deal with the user pressing Ctrl-C in the console
    #we have to check for the KeyboardInterrupt exception and break out of
    #the main loop
    except KeyboardInterrupt as e:
        pass

    #Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    detectAndTrackMultipleFaces()
