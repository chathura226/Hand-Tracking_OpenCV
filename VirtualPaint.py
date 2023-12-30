# project - virtual paint

import cv2
import time
import numpy as np
import os
import HandTrackingModule as HTM
from UtilityFunctions import stackImages  # for checking results

##########params#############
frameWidth = 640
frameHeight = 480

brushThickenss = 15
eraserThickness = 70

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

# colors according to header section
colorList = [(196, 102, 255), (195, 136, 59), (87, 217, 126), (0, 0, 0)]
color = colorList[0]

# for drawing line, we need last position
prevX, prevY = 0, 0
# drawing canvas
drawCanvas = np.zeros((frameHeight, frameWidth, 3), np.uint8)

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
            # if selection mode, setting prev cordinaes to 0, so that dont draw unnecessary lines
            prevX, prevY = 0, 0

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), color, cv2.FILLED)
            print("Selection mode")
            if y1 < header.shape[0]:
                # checking for each section
                for sec in secStartAndEnd:
                    if sec[0] < x1 < sec[1]:
                        header = overlayImgList[sec[2]]  # tuples 3rd element is sec id
                        color = colorList[sec[2]]
                        break
        # if drawing mode - index finger is up
        elif upFingers[1] and not upFingers[2]:
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            print("Drawing mode")

            # if this is the first frame , ( to avoid line from origin to curr point)
            if prevX == 0 and prevY == 0:
                prevX, prevY = x1, y1

            # for erasing
            if color == (0, 0, 0):
                cv2.line(img, (prevX, prevY), (x1, y1), color, eraserThickness)
                cv2.line(drawCanvas, (prevX, prevY), (x1, y1), color, eraserThickness)
            else:
                cv2.line(img, (prevX, prevY), (x1, y1), color, brushThickenss)
                cv2.line(drawCanvas, (prevX, prevY), (x1, y1), color, brushThickenss)

            prevX, prevY = x1, y1

        else:  # not either modes
            # setting prev cordinaes to 0, so that dont draw unnecessary lines
            prevX, prevY = 0, 0

    # creting a gray image out of drawing canvas
    imgGray = cv2.cvtColor(drawCanvas, cv2.COLOR_BGR2GRAY)
    # getting inverse of grayscale
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # adding inv to img as a mask so it shows only the parts without paint from the frame
    img = cv2.bitwise_and(img, imgInv)
    # adding the color paint to the frame
    img = cv2.bitwise_or(img, drawCanvas)

    # overlay header
    h, w, c = header.shape
    img[0:h, 0:w] = header

    # img=cv2.addWeighted(img,0.5,drawCanvas,0.5,0)

    # to check how things work , uncomment below line
    # img=stackImages(0.5,([img,drawCanvas],[imgGray,imgInv]))
    cv2.imshow('Virtual Paint', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
