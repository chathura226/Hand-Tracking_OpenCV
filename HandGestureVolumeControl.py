# changing volume using hand gesture
# this is configured for debian volume controllers

import cv2
import time
import numpy as np
import HandTrackingModule as HTM
import math
import subprocess  # to change volume


# volume change for debian
def changeVol(vol):
    # Define the command
    command = f'amixer sset "Master" {str(vol)}%'

    # Execute the command using subprocess
    subprocess.run(command, shell=True)


########params#############

camWidth = 640
camHeight = 480

####################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

# frame rate
prevTime, currTime = 0, 0

minVol = 0
maxVol = 100
vol=0
volBar=400 #for volume bar

# hand detector obj
detector = HTM.HandDetector(
    minDetectionConfidence=0.7)  # default detectionCon is 0.5 , we are increase here to really make sure

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    # thumb tip is index 4 and tip of index finger is index 8
    if len(landmarkList) != 0:
        # print(landmarkList[4],landmarkList[8])

        # x y coordinates seperately for thumb and index finger
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]
        # getting center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # drawing cirlce on them and line connecting them
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # taking length between the fingers
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # hand range after some tests was 40 - 280
        # volume range is from 0 - 100
        vol = np.interp(length, [40, 280], [minVol, maxVol])
        # adjusting vol to bar height
        volBar = np.interp(length, [40, 280], [400, 150])
        print(int(length), vol)
        changeVol(int(vol))

        if length < 37:  # after checking the length 37 was around the minimum when i touch two fingers
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    #volume bar to indicate current volume
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img, f'{str(int(vol))}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # calulating fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
