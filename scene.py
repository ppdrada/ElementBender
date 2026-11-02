import math

import cv2
import numpy as np

from ascii_art import CELL_H, CELL_W, COLS, ROWS

BACK_BOX = (12, 10, 62, 38)  # back arrow button: left, top, right, bottom in pixels
BACK_SECONDS = 2  # how long the hand stays on the button to go back
MAX_CHARGE = 5  # after about 7 seconds of a closed hand the power almost fills the screen
SMOOTHING = 0.35  # how much of the new hand position is used each frame: lower is calmer but slower
STEADY_FRAMES = 3  # a gesture must stay this many frames before it counts, so a camera flicker doesn't throw
FLICK_SPEED = 1.6  # cells per frame: moving the hand faster than this is a throw


def on_back_button(x, y):
    left, top, right, bottom = BACK_BOX
    return left <= x <= right and top <= y <= bottom


class Scene:
    """What every bending scene shares: the hand or the mouse, the back arrow and the throw.
    Hand: a closed hand gathers power, five open fingers is the special power, and a quick swing throws.
    Mouse: left button gathers, right button is the special power, and a quick move throws."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.time = 0.0
        self.mouse = None
        self.mouse_left = False
        self.mouse_right = False
        self.hand = None  # (x, y) in cells
        self.history = []  # the last hand positions, to know the throw speed and direction
        self.fingers = None  # the gesture that has been steady
        self.new_fingers = None
        self.steady = 0
        self.back_fill = 0.0
        self.back = False  # True when the player wants to go back to the menu

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x / CELL_W, y / CELL_H)
        if event == cv2.EVENT_LBUTTONDOWN and on_back_button(x, y):
            self.back = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_left = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_left = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_right = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.mouse_right = False

    def read_input(self, seconds, fingers, hand):
        """Updates the hand position and returns (holding, open_hand)."""
        self.time += seconds
        if hand:
            target = (hand[0] * COLS, hand[1] * ROWS)
            if self.hand is None or self.fingers is None:
                self.hand = target
            else:  # move only part of the way, so the camera's small shakes disappear
                self.hand = (self.hand[0] + (target[0] - self.hand[0]) * SMOOTHING,
                             self.hand[1] + (target[1] - self.hand[1]) * SMOOTHING)
            self.steady = self.steady + 1 if fingers == self.new_fingers else 0
            self.new_fingers = fingers
            if self.fingers is None or self.steady >= STEADY_FRAMES - 1:
                self.fingers = fingers
            holding, open_hand = self.fingers == 0, self.fingers == 5
        else:
            self.fingers = self.new_fingers = None
            self.hand = self.mouse
            holding, open_hand = self.mouse_left, self.mouse_right
        self.history = (self.history + [self.hand])[-5:] if self.hand else []

        on_button = hand is not None and on_back_button(self.hand[0] * CELL_W, self.hand[1] * CELL_H)
        self.back_fill = min(1.0, self.back_fill + seconds / BACK_SECONDS) if on_button else 0.0
        if self.back_fill >= 1:
            self.back = True
        return holding and self.hand is not None, open_hand and self.hand is not None

    def flick(self):
        """If the hand is swinging fast, returns the throw speed (x, y) in cells per frame, otherwise None."""
        recent = self.history[-3:]
        if len(recent) < 3:
            return None
        vx, vy = (recent[-1][0] - recent[0][0]) / 2, (recent[-1][1] - recent[0][1]) / 2
        speed = math.hypot(vx, vy)
        if speed < FLICK_SPEED:
            return None
        boost = min(3.5, max(1.8, speed * 1.3)) / speed
        return vx * boost, vy * boost

    def shake(self, image, amount):
        if amount >= 1:
            image = np.roll(image, (int(self.rng.uniform(-amount, amount)), int(self.rng.uniform(-amount, amount))), axis=(0, 1))
        return image

    def draw_back(self, image):
        left, top, right, bottom = BACK_BOX
        color = (220, 220, 220) if self.back_fill > 0 else (140, 140, 140)
        middle = (top + bottom) // 2 - 2
        cv2.arrowedLine(image, (right - 10, middle), (left + 8, middle), color, 2, cv2.LINE_AA, tipLength=0.35)
        if self.back_fill > 0:
            cv2.line(image, (left + 8, bottom - 2), (left + 8 + int((right - left - 16) * self.back_fill), bottom - 2), color, 2)
        return image
