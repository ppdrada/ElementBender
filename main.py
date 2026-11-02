import ctypes
import sys
import time

import cv2

from hands import HandTracker
from menu import Menu

WINDOW = "Element Bender"
# waitKey can't tell when a key is released, so a pressed key counts as held for a short time
KEY_HOLD_SECONDS = 0.6


def style_window(title):
    # Windows only: no icon, only the X button, black title bar with white text
    user32 = ctypes.windll.user32
    window = user32.FindWindowW(None, title)
    user32.SetWindowLongW(window, -20, user32.GetWindowLongW(window, -20) | 1)
    user32.SetClassLongPtrW(window, -14, 0)
    user32.SetWindowLongW(window, -16, user32.GetWindowLongW(window, -16) & ~0x30000)  # no minimize/maximize
    for attribute, color in ((35, 0x000000), (36, 0xFFFFFF)):  # title bar color, title text color
        value = ctypes.c_int(color)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(window, attribute, ctypes.byref(value), 4)
    user32.SetWindowPos(window, 0, 0, 0, 0, 0, 0x27)  # redraw the title bar


hands = HandTracker()
menu = Menu()

camera = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, menu.on_mouse)
if sys.platform == "win32":
    style_window(WINDOW)

key_option, key_time = None, 0.0
last = time.monotonic()

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read the webcam.")
        break

    now = time.monotonic()
    held = key_option if now - key_time < KEY_HOLD_SECONDS else None
    menu.update(now - last, hands.count_fingers(frame), held)
    last = now

    cv2.imshow(WINDOW, menu.draw())

    key = cv2.waitKey(1) & 0xFF
    if ord("1") <= key <= ord("4"):
        key_option, key_time = key - ord("1"), now
    if key == 27:  # Esc goes back to the menu
        menu.reset()
    if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
