import cvzone
import cv2 # this is OpenCV btw
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageDraw, ImageFont
import math # for Euclidean distances
import random
from collections import deque

""" Computer Vision Snake Game """
# For those learning:
# Began from the tutorial: https://youtu.be/w26Ze6lP02Y?si=PUKYpcJLg4JMS1PZ
# But I developed and restructured some parts to be structured in a way that clicks for me more
# If you have any questions, feel free to contact me. I left a lot of comments to help.
# This is the best place to learn from imo: https://github.com/cvzone/cvzone

# Camera Initialisation
cap = cv2.VideoCapture(0) # opens default camera (index 0)
cap.set(3, 1280) # prop id 3 is for width
cap.set(4, 720) # 4 is height

# Build hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)  # standard 0.5 but to be more precise

def pixelText(img, text, pos, font_size, color):
    imgPil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imgPil)
    font = ImageFont.truetype("PressStart2P-Regular.ttf", font_size) # load font
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(imgPil), cv2.COLOR_RGB2BGR)

class SnakeGameClass:
    def __init__(self, pathFood: str):
        self.points = deque()  # list of all points of snake (x, y)
        self.lengths = deque()  # distances between each point
        self.currentLength = 0.0  # accumulated length of the snake
        self.maxLength = 150.0  # fixed value - total allowed length - tail is trimmed past this
        self.previousHead = None  # no head initially

        self.food = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)  # for clean overlay
        if self.food is None:
            raise FileNotFoundError(f"Food image not found: {pathFood}")
        self.heightFood, self.widthFood = self.food.shape[:2]  # to know placement so when snake hits, know it ate
        self.foodPos = (0, 0)  # initial position, will randomise after
        self.randomFoodPos()  # calls to immediately randomise the position

        # Gameplay state
        self.score = 0
        self.gameOver = False  # NEXT: if game over is true then show diff screen to end game

    def randomFoodPos(self):
        self.foodPos = random.randint(100, 1000), random.randint(100, 600)
        # random position from 100-1000 for height, 100-600 for width

    def reset(self):
        # reset snake and score when died
        self.points.clear()
        self.lengths.clear()
        self.currentLength = 0.0
        self.maxLength = 150.0
        self.previousHead = None
        self.score = 0
        self.gameOver = False

    def limitLength(self):
        # pop off any excess from the max length for the snake
        while self.lengths and self.currentLength > self.maxLength:
            self.currentLength -= self.lengths.popleft()
            self.points.popleft()

    def update(self, imgMain, currentHead):
        # update game for new frame
        if self.gameOver:  # if game-over, then diff screen but this is temporary
            overlay = imgMain.copy() # this for darkening the screen
            cv2.rectangle(overlay, (0, 0), (imgMain.shape[1], imgMain.shape[0]), (0, 0, 0), -1)
            alpha = 0.6  # transparency
            imgMain = cv2.addWeighted(overlay, 0.7, imgMain, 0.3, 0)

            imgMain = pixelText(imgMain, "YOU DIED", (300, 300), font_size=64, color=(255,0,0))
            imgMain = pixelText(imgMain, f"Score: {self.score}", (300, 400), font_size=48, color=(255,255,255))
            imgMain = pixelText(imgMain, "Press R to Restart the Game", (300, 550), font_size=25, color=(0,255,0))
            return imgMain
        else:
            cx, cy = currentHead
            if self.previousHead is None:
                self.previousHead = (cx, cy)
            px, py = self.previousHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py) # stable Euclidean distance
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = (cx, cy) # prepare for next appending by making current point previous

            self.limitLength() # if it needs to be shortened

            # Check for Eating
            rx, ry = self.foodPos  # check if index finger is in this region
            if rx - self.widthFood // 2 < cx < rx + self.widthFood // 2 and ry - self.heightFood // 2 < cy < ry + self.heightFood // 2:
                # foodx = foodwidth/2 < fingerx < foox + foodwidth/2 and
                # foody = foodheight/2 < fingery < fooy + foodheight/2
                self.randomFoodPos()
                self.maxLength += 50  # increase max length when eaten, generate new location for food
                self.score += 1  # increment score accordingly
                print(self.score)

            # Draw Snake
            for i in range(1, len(self.points)):
                cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 200, 0), 20, lineType=cv2.LINE_AA)
            if self.points:
                cv2.circle(imgMain, self.points[-1], 20, (0, 200, 0), cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(
                imgMain,
                self.food,
                (rx - self.widthFood // 2, ry - self.heightFood // 2)
            )
            # background img, front img, location (pushed to centre)

            # Live Score
            cv2.rectangle(imgMain, (20, 20), (255, 70), (255, 16, 240), -1)
            imgMain = pixelText(imgMain, f'Score:{self.score}', [30, 35], font_size= 25, color=(255,255,255))

            # Check for self-eat : create polygon from points other than last point and break if too low distance
            if len(self.points) > 4:
                points = np.array(list(self.points)[:-2], np.int32)  # convert to numpy array then use polygon feature
                points = points.reshape((-1, 1, 2))
                cv2.polylines(imgMain, [points], False, (0, 0, 0), 3)
                # red line is polygon line, if hit then game over
                minDistance = cv2.pointPolygonTest(points, (cx, cy), True)  # is head hitting any polygon points

                if -1 <= minDistance <= 1:  # increase for harder game
                    self.gameOver = True

        return imgMain


# game object - pass pizza pic
game = SnakeGameClass("pizza.png")  # location of image


while True:
    # cap.read gets frame, success will be false if camera fails
    success, img = cap.read()
    img = cv2.flip(img, 1)  # horizontal flip (0 would be vertical)
    hands, img = detector.findHands(img, flipType=False) # if theres a hand, it's a list of info

    if hands:
        # The github link gives a great break down on this (hand tracking section)
        handPoints = hands[0]['lmList'] # gets first hand seen and all the points
        pointIndex = handPoints[8][0:2]  # find pointer finger - 8 is index fingertip
        img = game.update(img, pointIndex)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.reset()

    if key == ord('q') or key == 27:  # 27 = esc btw
        break

cap.release()
cv2.destroyAllWindows()