import cv2
import mediapipe as mp
import time

# Read the detailed explanation : https://google.github.io/mediapipe/solutions/hands.html#static_image_mode

class HandDetection():

    def __init__(self, mode=False, max_hands=2, detect_conf = 0.5, track_conf=0.5):  # all these are the default values
        self.mode = mode
        self.max_hands = max_hands
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands  # A formality type line
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils  # For drawing lines between the 21 landmarks
        self.tipIds = [4, 8, 12, 16, 20]

# This method is only for drawing the hands
    def findHands(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks) ==> This gives us {x,y,z} values when hand is displayed on the webcam
        # otherwise it gives None
        if self.results.multi_hand_landmarks:
            # Iterating over all hands
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Adding the "connections" argument helps to join the landmarks using lines otherwise only the points
                    # will get displayed
                    self.mpDraw.draw_landmarks(image=img, landmark_list=hand_lms, connections=self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0, draw=True):

        self.lmlist = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myhand.landmark):
                # print(f"ID :{id}\n Coordinates :\n{lm}")
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # {ID, X-coordinate, Y-coordinate}
                # print(id, cx, cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), 6)

        return self.lmlist

    def fingersUp(self):
        fingers = []
        # Generalised code for left and right hand needs to be added
        if len(self.lmlist) != 0:
            # For thumb
            if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0] - 1][1]:
                fingers.append(0)
            else:
                fingers.append(1)
            # For the four fingers
            for i in range(1, 5):

                if self.lmlist[self.tipIds[i]][2] < self.lmlist[self.tipIds[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

def main():
    c_time = 0
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetection()

    while True:
        success, img = cap.read()
        img = detector.findHands(img=img)
        lmlist = detector.findPosition(img=img)
        if len(lmlist) != 0:
            # printing info of landmark with id as 4
            print(lmlist[4])
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img=img, text=str(int(fps)), org=(20, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=3,
                    color=(255, 0, 255), thickness=3)
        cv2.imshow("image", img)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()