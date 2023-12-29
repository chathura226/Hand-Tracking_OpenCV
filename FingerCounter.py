# project - finger counter

import cv2
import time
import os

# importing hand track module
import HandTrackingModule as HTM

##############params##############
frameWidth = 640
frameHeight = 480
##############################

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# importing finger images for counting files
folderPath = "FingerCountingImages"
fingerImgList = os.listdir(folderPath)
overlayList = []
for imgPath in fingerImgList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    image = cv2.resize(image, (200, 200))  # resizing the images since they have high resolution
    overlayList.append(image)

print(len(overlayList))

# for frame rate
currTime = 0
prevTime = 0

# hand tracking obj
detector = HTM.HandDetector(minDetectionConfidence=0.7)
# landmark index in order (tip,dip) for each finger
tipAndPipLandmarkIDs = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        upFingers=[]
        # for thumb openness
        # for thumb we check the x coordinate of tip and dip of thumb
        if landmarkList[4][1]>landmarkList[3][1]:
            upFingers.append(1)
        else:
            upFingers.append(0)

        # checking y coredinate of landmark index for tip and pip of fingers to check whether open or close
        #except for thumb ( since closed thump doesnt show tip below pip)
        for tipID, pipID in tipAndPipLandmarkIDs[1:]:
            if landmarkList[tipID][2] < landmarkList[pipID][2]:
                upFingers.append(1) # append 1 if finger is up
            else:
                upFingers.append(0) # append 0 if finger is down


        # print(upFingers)
        totalFingers = upFingers.count(1)
        # print(totalFingers)

        # slicing for putting overlay image
        # have to resize since images have high resolution than 200
        img[0:200, 0:200] = overlayList[totalFingers]

        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_SIMPLEX,5,(255,0,0),20)
    # calculating and putting the framerate
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (480, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
