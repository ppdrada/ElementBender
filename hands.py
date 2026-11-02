import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

HAND_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"
# Each finger as (tip point, middle joint point); the thumb is left out, so 1 to 4 fingers are easy to show
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


class HandTracker:
    """Finds a hand in the webcam image: how many fingers are raised and where it is."""

    def __init__(self):
        self.detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
        ))

    def track(self, frame):
        """Returns (raised fingers, position). Fingers go from 0 to 4 without the thumb, and 5 means the whole
        hand is open, thumb too. The position is (x, y) from 0 to 1, mirrored like a mirror, taken from the
        middle of the palm; without a hand it returns (0, None)."""
        # OpenCV gives colors as BGR, MediaPipe expects RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(image, int(time.monotonic() * 1000))
        if not result.hand_landmarks:
            return 0, None
        hand = result.hand_landmarks[0]
        wrist, palm = hand[0], hand[9]  # palm: where the middle finger meets the hand
        # A finger is raised when its tip is farther from the wrist than its middle joint, in any direction
        fingers = sum(distance(hand[tip], wrist) > distance(hand[joint], wrist) * 1.1 for tip, joint in FINGERS)
        if fingers == 4 and distance(hand[4], palm) > distance(wrist, palm) * 0.55:
            fingers = 5  # the thumb tip is far from the palm, so the thumb is open too
        return fingers, (1 - palm.x, palm.y)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)
