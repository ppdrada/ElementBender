import ctypes
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)

WINDOW = "Element Bender"
HAND_MODEL = Path(__file__).parent / "models" / "hand_landmarker.task"


def hide_window_icon(title):
    user32 = ctypes.windll.user32
    window = user32.FindWindowW(None, title)
    user32.SetWindowLongW(window, -20, user32.GetWindowLongW(window, -20) | 1)
    user32.SetClassLongPtrW(window, -14, 0)


hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
))

camera = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)
if sys.platform == "win32":
    hide_window_icon(WINDOW)

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read the webcam.")
        break

    frame = cv2.flip(frame, 1)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = hands.detect_for_video(image, int(time.monotonic() * 1000))
    for hand in result.hand_landmarks:
        drawing_utils.draw_landmarks(frame, hand, HandLandmarksConnections.HAND_CONNECTIONS)

    cv2.imshow(WINDOW, frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()