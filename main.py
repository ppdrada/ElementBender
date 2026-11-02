import ctypes

import cv2

WINDOW = "Element Bender"


def hide_window_icon(title):
    # The title bar belongs to Windows, not OpenCV, so we ask Windows directly
    user32 = ctypes.windll.user32
    window = user32.FindWindowW(None, title)
    user32.SetWindowLongW(window, -20, user32.GetWindowLongW(window, -20) | 1)  # dialog-style frame
    user32.SetClassLongPtrW(window, -14, 0)  # remove the icon


camera = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)
hide_window_icon(WINDOW)

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read the webcam.")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow(WINDOW, frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()