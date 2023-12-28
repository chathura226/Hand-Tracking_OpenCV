# minimum program for handtracking using google mediaPipe
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# creating obj from class hand
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # in here we are going with the default parameters.
mpDraw = mp.solutions.drawing_utils
# my note -  21 landmarks for a hand (landmarks -key points such as tips,base,knuckle etc)


# framerate
prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    # converting BGR to RGB so that we can send it to hand object
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # processing and extracting results from the above results given by hand obj
    if results.multi_hand_landmarks:
        # if multi_hand_landmarks exists => hand in the image,
        # but we dont know how many number of hands are there
        # so we loop through it
        for handLms in results.multi_hand_landmarks:
            # mpDraw.draw_landmarks(img,handLms)  #this will draw point in each land marks for each hand
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)  # this will draw points connected with each land marks for each hand

            # id - related to index number related to hand landmark (eg 4=>tip of thumb)
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)  # id -landmark index (0-21) and lm give x,y,z cordinate as a percentge
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id,cx,cy)
                if id==4 :
                    cv2.circle(img,(cx,cy),11,(255,0,255),cv2.FILLED)


    # calculating fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

    cv2.imshow('Result', img)
    cv2.waitKey(1)
