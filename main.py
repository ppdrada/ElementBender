import ctypes
import sys
import time

import cv2

from air_scene import AirScene
from earth_scene import EarthScene
from fire_scene import FireScene
from hands import HandTracker
from menu import Menu
from water_scene import WaterScene

WINDOW = "Element Bender"
# waitKey can't tell when a key is released, so a pressed key counts as held for a short time
KEY_HOLD_SECONDS = 0.6
SCENES = {"fire": FireScene, "water": WaterScene, "earth": EarthScene, "air": AirScene}
ENTER_SECONDS = 1.5  # how long "entering ..." stays on screen


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
scene = None  # the bending scene of the chosen element, once it starts


def on_mouse(*event):
    (scene or menu).on_mouse(*event)


camera = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)
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
    fingers, hand = hands.track(frame)
    if scene:
        scene.update(now - last, fingers, hand)
        image = scene.draw()
    else:
        held = key_option if now - key_time < KEY_HOLD_SECONDS else None
        menu.update(now - last, fingers, held)
        image = menu.draw()
        if menu.chosen in SCENES and menu.entering > ENTER_SECONDS:
            scene = SCENES[menu.chosen]()
    last = now

    cv2.imshow(WINDOW, image)

    key = cv2.waitKey(1) & 0xFF
    if ord("1") <= key <= ord("4"):
        key_option, key_time = key - ord("1"), now
    if key == 27 or (scene and scene.back):  # Esc or the back arrow go back to the menu
        scene = None
        menu.reset()
    if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
