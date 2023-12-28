# to demonstrate usage of the custom module

import cv2
import mediapipe as mp
import time

# importing module
import HandTrackingModule as HTM

# framerate
prevTime = 0
currTime = 0
cap = cv2.VideoCapture(0)
# creating obj for detecting
detector = HTM.HandDetector()  # we go with default params

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    # getting list of landmarks related to hand =0
    # hand =1 will give landmark list of other hand
    landmarkList = detector.findPosition(img)
    if len(landmarkList) != 0:
        print(landmarkList[4])

    # calculating fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
