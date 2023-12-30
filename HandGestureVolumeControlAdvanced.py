# changing volume using hand gesture - advanced
# better version than HandGestureVolumeControl prj
# this is configured for debian volume controllers
#setting volume only when ring finger is down

import cv2
import time
import numpy as np
import HandTrackingModule as HTM
import math
import subprocess  # to change volume


# volume change for debian
def changeVol(vol):
    # Define the command
    command = f'amixer sset -q "Master" {str(vol)}%'

    # Execute the command using subprocess
    subprocess.run(command, shell=True)

def getCurrentVolume():
    # Command to get volume
    command = "amixer get 'Master' | grep 'Front Left:' | awk -F'[][]' '{print $2}' | sed 's/%//'"

    # Run the command and capture output
    output = subprocess.check_output(command, shell=True, text=True)

    # Print the output
    return int(output.strip())

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
volPercentage =getCurrentVolume() # adjusting vol
currentVol=getCurrentVolume()  #current volume
volBar = 400  # for volume bar
colorVol=(255, 0, 0) # to indicate when the vol is set


# hand detector obj
detector = HTM.HandDetector(
    minDetectionConfidence=0.7,
    maxHands=1)  # default detectionCon is 0.5 , we are increase here to really make sure

# boundingbox area (hand size)
area = 0


while True:
    success, img = cap.read()

    # find hand
    img = detector.findHands(img)
    landmarkList, boundingBox = detector.findPosition(img, draw=False, boundingBox=True)
    # thumb tip is index 4 and tip of index finger is index 8
    if len(landmarkList) != 0:

        #   filter based on size
        # print(boundingBox)
        cv2.rectangle(img, (boundingBox[0] - 20, boundingBox[1] - 20), (boundingBox[2] + 20, boundingBox[3] + 20),
                      (255, 0, 0), 2)
        boundingWidth, boundingHeight = boundingBox[2] - boundingBox[0], boundingBox[3] - boundingBox[1]
        area = (boundingWidth * boundingHeight) // 100
        # print(area)
        # only do other things only if the hand is in this size range
        if 250 < area < 1000:
            # find distance between index and thumb
            length, img, lineCoordinates = detector.findDistance(4, 8, img, draw=True)
            (cx, cy) = lineCoordinates[2]
            # print(length)

            # convert volume from length to volume percentage
            # since min and max volume is 0 and 100, the below will give percentage
            # hand range after some tests was 40 - 250
            volPercentage = np.interp(length, [40, 250], [minVol, maxVol])
            # adjusting vol to bar height
            volBar = np.interp(length, [40, 250], [400, 150])

            # reduce resolution to make it smoother
            smoothness=10
            volPercentage=smoothness*round(volPercentage/smoothness)


            # check which fingers up
            # if ring is down, set volume
            fingersUp = detector.fingersUp()
            if not fingersUp[3]:
                changeVol(int(volPercentage))
                currentVol=volPercentage
                cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    # drawings
    # volume bar to indicate current volume
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{str(int(volPercentage))}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f'Vol Set: {str(int(currentVol))}%', (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colorVol, 2)

    # framerate
    # calulating fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
