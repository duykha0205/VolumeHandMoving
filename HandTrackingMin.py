import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONQS)

    ctime = time.time()
    fps = 1/ (cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)))

    cv.imshow("image", img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()