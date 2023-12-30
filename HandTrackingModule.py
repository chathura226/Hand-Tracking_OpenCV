# HandTracking custom module for getting list of positions of landmarks

# By this we can get a list of cordinates of landmarks of a hand with the
# relavant index number for landmark given by mediapipe

import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    # these constructor parameter are the parameter for mediapipe handobject initialization
    # they are set to default, but user can define if want
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        # landmark index in order (tip,dip) for each finger
        self.tipAndPipLandmarkIDs = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]

        self.landmarkList = []

        # creating obj from class hand
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,
                                        self.modelComplexity,
                                        self.minDetectionConfidence,
                                        self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    # function to find hands
    def findHands(self, img, draw=True):

        # converting BGR to RGB so that we can send it to hand object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # processing and extracting results from the above results given by hand obj
        if self.results.multi_hand_landmarks:
            # if multi_hand_landmarks exists => hand in the image,
            # but we dont know how many number of hands are there
            # so we loop through it
            for handLms in self.results.multi_hand_landmarks:
                # mpDraw.draw_landmarks(img,handLms)  #this will draw point in each landmarks for each hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # this will draw points connected with each land marks for each hand
        return img

    # to find positions of landmarks for a particular hand (only one hand)
    def findPosition(self, img, handNo=0, draw=True,boundingBox=False):

        # for bounding box
        xList=[]
        yList=[]
        bBox=[]

        self.landmarkList = []

        # if hand is detected
        if self.results.multi_hand_landmarks:
            # selecting the hand related to given parameter
            selectedHand = self.results.multi_hand_landmarks[handNo]

            # looping through landmarks of the selected hand
            for id, lm in enumerate(selectedHand.landmark):
                # print(id, lm)  # id -landmark index (0-21) and lm give x,y,z cordinate as a percentge
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)

                # print(id, cx, cy)
                self.landmarkList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # bounding box
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bBox = xMin, yMin, xMax, yMax

        if boundingBox:
            return self.landmarkList,bBox
        else:
            return self.landmarkList

    # return a list showing whether fingers are up or not ( 1- for open, 0- for close)
    def fingersUp(self):
        upFingers = []
        # for thumb openness
        # for thumb we check the x coordinate of tip and dip of thumb
        if self.landmarkList[4][1] < self.landmarkList[3][1]:
            upFingers.append(1)
        else:
            upFingers.append(0)

        # checking y coredinate of landmark index for tip and pip of fingers to check whether open or close
        # except for thumb ( since closed thump doesnt show tip below pip)
        for tipID, pipID in self.tipAndPipLandmarkIDs[1:]:
            if self.landmarkList[tipID][2] < self.landmarkList[pipID][2]:
                upFingers.append(1)  # append 1 if finger is up
            else:
                upFingers.append(0)  # append 0 if finger is down

        return upFingers


    # to find distance between fingers
    # returns length, img and coordinae tuples of each index and center point
    def findDistance(self,landmarkID1,landmarkID2,img,draw=True):
        # x y coordinates seperately for thumb and index finger
        x1, y1 = self.landmarkList[landmarkID1][1], self.landmarkList[landmarkID1][2]
        x2, y2 = self.landmarkList[landmarkID2][1], self.landmarkList[landmarkID2][2]

        # getting center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # drawing cirlce on them and line connecting them
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # taking length between the fingers
        length = math.hypot(x2 - x1, y2 - y1)
        return length,img,[(x1,y1),(x2,y2),(cx,cy)]


# dummy code to be run if not imported as a module
def main():
    # framerate
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    # creating obj for detecting
    detector = HandDetector()  # we go with default params

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


if __name__ == "__main__":
    main()
