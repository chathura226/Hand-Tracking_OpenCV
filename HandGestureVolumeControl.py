# changing volume using hand gesture

import cv2
import time
import numpy as np
import HandTrackingModule as HTM

########params#############

camWidth=640
camHeight=480

####################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

#frame rate
prevTime,currTime=0,0

#hand detector obj
detector=HTM.HandDetector(minDetectionConfidence=0.7)  #default detectionCon is 0.5 , we are increase here to really make sure

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    landmarkList=detector.findPosition(img,draw=False)
    # thumb tip is index 4 and tip of index finger is index 8
    if len(landmarkList)!=0:
        print(landmarkList[4],landmarkList[8])

        # x y coordinates seperately for thumb and index finger
        x1,y1=landmarkList[4][1],landmarkList[4][2]
        x2, y2=landmarkList[8][1],landmarkList[8][2]

        #drawing cirlce on them and line connecting them
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)

    # calulating fps
    currTime=time.time()
    fps=1/(currTime-prevTime)
    prevTime=currTime
    cv2.putText(img,f'FPS: {str(int(fps))}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)


    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
