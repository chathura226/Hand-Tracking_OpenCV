# HandTracking custom module for getting list of positions of landmarks

import cv2
import mediapipe as mp
import time


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
    def findPosition(self, img, handNo=0, draw=True):

        landmarkList = []

        # if hand is detected
        if self.results.multi_hand_landmarks:
            # selecting the hand related to given parameter
            selectedHand = self.results.multi_hand_landmarks[handNo]

            # looping through landmarks of the selected hand
            for id, lm in enumerate(selectedHand.landmark):
                # print(id, lm)  # id -landmark index (0-21) and lm give x,y,z cordinate as a percentge
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                landmarkList.append([id, cx, cy])

                if draw :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return landmarkList

#dummy code to be run if not imported as a module
def main():
    # framerate
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    # creating obj for detecting
    detector = HandDetector()  # we go with default params

    while True:
        success, img = cap.read()

        img=detector.findHands(img)
        landmarkList=detector.findPosition(img)
        if len(landmarkList)!=0:
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
