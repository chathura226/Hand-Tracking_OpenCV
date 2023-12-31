# virtual mouse
# using PyAutoGui for mouse control


import cv2
import numpy as np
import time
import pyautogui  # for mouse control
import HandTrackingModule as HTM

############params#########
frameWidth = 640
frameHeight = 480
screenWidth,screenHeight=pyautogui.size() #getting screen size
frameReduc=100 #frame reduction for utilizing corners of the frame
smoothening=6
######################

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# detector
detector = HTM.HandDetector(minDetectionConfidence=0.7, maxHands=1)

# for frame rate
prevTime = currTime = 0

# previous and current locations
prevX=prevY=0
currX=currY=0

while True:
    success, img = cap.read()
    if not success:
        break

    # fliping image for easier drawing
    img = cv2.flip(img, 1)  # flip horizontally

    # find hand landmarks
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img)

    if len(landmarkList) != 0:
        # get tip of the index and middle finger
        # index
        x1, y1 = landmarkList[8][1:]
        # middle
        x2, y2 = landmarkList[12][1:]

        # check which fingers are up
        upFingers = detector.fingersUp()


        cv2.rectangle(img, (frameReduc, frameReduc), (frameWidth - frameReduc, frameHeight - frameReduc), (255, 0, 255),
                      2)
        # only index finger - moving mode
        if upFingers[1] and not upFingers[2]:
            # convert coordinates to screen size
            mouseX=np.interp(x1,(frameReduc,frameWidth-frameReduc),(0,screenWidth))
            mouseY=np.interp(y1,(frameReduc,frameHeight-frameReduc),(0,screenHeight))
            # smoothen the values
            currX=prevX+(mouseX-prevX)/smoothening
            currY=prevY+(mouseY-prevY)/smoothening
            # move mouse
            pyautogui.moveTo(currX, currY)
            cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
            #updating
            prevX=currX
            prevY=currY

        # both index and middle fin up - click mode
        elif upFingers[1] and upFingers[2]:
            # find distance between fingers
            distance,img,lineInfo=detector.findDistance(8,12,img,draw=False)
            # click if distance is short
            if distance <40:
                cv2.circle(img, lineInfo[2], 15, (255, 0, 0), cv2.FILLED)
                pyautogui.click()

    # framerate
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # display
    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
