# For now,This works for Right Hand only
import cv2
import numpy as np
import HandTracking_Module as htm
import os
import datetime

timestamp = datetime.datetime.now().strftime("%d-%m-%Y,%H-%M-%S")

print(timestamp)
brushThickness = 15
eraserThickness = 50

folderpath = "Headers"
overlayList = []
for imPath in os.listdir(folderpath):
    img = cv2.imread(f"{folderpath}/{imPath}")
    overlayList.append(img)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

header = overlayList[0]
h, w, c = header.shape
drawColor = (255, 0, 255)

detector = htm.HandDetection(detect_conf=0.80, max_hands=1)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
count = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findPosition(img=img, draw=False)

    if len(lmlist):
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[12][2]

        fingers = detector.fingersUp()
        # Selection Mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1-15), (x2, y2-15), drawColor, cv2.FILLED)

        # Drawing mode
        elif fingers[1] == True and fingers[2] == False:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0: xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # Explanation : https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
    # Converting all pixel values greater than 50 to 0(black color) and remaining to 255(white color)
    ret, imgInv = cv2.threshold(imgGray, 75, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # The AND operation of black pixels(value 0) with pixels of img will result in those pixels having value 0 too
    img = cv2.bitwise_and(img, imgInv)
    # the OR operation will convert those black pixels into the colored pixels
    img = cv2.bitwise_or(img, imgCanvas)
    imgSave = imgCanvas.copy()

    black_pixels = np.where(
        (imgCanvas[:, :, 0] == 0) &
        (imgCanvas[:, :, 1] == 0) &
        (imgCanvas[:, :, 2] == 0)
    )

    # set those pixels to white
    imgSave[black_pixels] = [255, 255, 255]

    img[0:h, 0:w] = header
    cv2.imshow("Image", img)
    # cv2.imshow("Image Canvas", imgCanvas)
    # cv2.imshow("White Background", imgSave)

    if cv2.waitKey(5) & 0xFF == ord('s'):
        cv2.imwrite(f"PaintFiles/File{timestamp}.jpg", imgSave)
        cv2.rectangle(imgSave, (0, 300), (1280, 600), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgSave, "File Saved", (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow("Result", imgSave)
        cv2.waitKey(200)
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): break