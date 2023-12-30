# project - virtual paint

import cv2
import time
import numpy as np
import os
import HandTrackingModule as HTM

##########params#############
frameWidth = 640
frameHeight = 480
############################

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# hand tracking obj
detector = HTM.HandDetector(minDetectionConfidence=0.8)

# importing headers
folderPath = "Vitual Paint Headers"
headerList = os.listdir(folderPath)
overlayImgList = []
# images are named with sec id and it will be imported such that index of list = sec id
for img in headerList:
    image = cv2.imread(f'{folderPath}/{img}')
    h, w, c = image.shape
    # changing height according to new width
    newHeight = (h / w) * frameWidth
    image = cv2.resize(image, (frameWidth, int(newHeight)))
    overlayImgList.append(image)
header = overlayImgList[0]  # default overlay header
# headerSection in x direction
# headers are designed such that part after (0.075*width) is divided into 4
logoDistance = frameWidth * 0.075
eachSecWidth = (frameWidth - logoDistance) / 4  # only 4 sections
# storing start and end of each section
secStartAndEnd = []
for i in range(0, 4):
    # appending (starting x cordi, ending x cordi, sec ID)
    ithSec = (logoDistance + (i * eachSecWidth), logoDistance + ((i + 1) * eachSecWidth), i)
    secStartAndEnd.append(ithSec)
# print(secStartAndEnd)


while True:
    # importing images
    sucess, img = cap.read()

    if not sucess:
        break
    # fliping image for easier drawing
    img = cv2.flip(img, 1)  # flip horizontally

    # finding land landmarks
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)

    if (len(landmarkList)) != 0:
        # print(landmarkList)

        # tip of index and middle fingers
        x1, y1 = landmarkList[8][1:]
        x2, y2 = landmarkList[12][1:]

        # checking which fingers are up
        upFingers = detector.fingersUp()
        # print(upFingers)

        # if selection mode - two fingers are up (checking for index and middle mfinger opennes)
        if upFingers[1] and upFingers[2]:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv2.FILLED)
            print("Selection mode")
            if y1 < header.shape[0]:
                # checking for each section
                for sec in secStartAndEnd:
                    if sec[0] < x1 < sec[1]:
                        header = overlayImgList[sec[2]]  # tuples 3rd element is sec id
                        break

        # if drawing mode - index finger is up
        if upFingers[1] and not upFingers[2]:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            print("Drawing mode")


    # overlay header
    h, w, c = header.shape
    img[0:h, 0:w] = header

    cv2.imshow('Virtual Paint', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
