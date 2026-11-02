"""Shared tools to draw a screen made of colored characters."""
import cv2
import numpy as np

COLS, ROWS = 96, 40
CELL_W, CELL_H = 10, 12
WIDTH, HEIGHT = COLS * CELL_W, ROWS * CELL_H
# The first ten go from empty to full, the fire uses them as a ramp
CHARS = " .:-=+*#%@~/oO^_\\"

_ys, _xs = np.mgrid[0:ROWS, 0:COLS]
CELL_X = ((_xs + 0.5) * CELL_W).astype(np.float32)
CELL_Y = ((_ys + 0.5) * CELL_H).astype(np.float32)


def clamp(value):
    return max(0.0, min(1.0, value))


def ease(value):
    return value * value * (3 - 2 * value)


def lerp(a, b, k):
    return a + (b - a) * k


def make_glyphs():
    # One small black-and-white picture for each character
    glyphs = np.zeros((len(CHARS), CELL_H, CELL_W), np.uint8)
    for i, char in enumerate(CHARS):
        cv2.putText(glyphs[i], char, (1, CELL_H - 3), cv2.FONT_HERSHEY_PLAIN, 0.8, 255, 1, cv2.LINE_AA)
    # The thin font looks dim, so make every drawn pixel stronger
    return np.clip(glyphs.astype(np.int32) * 2, 0, 255).astype(np.uint8)


GLYPHS = make_glyphs()


def glow(center_y, radius, bgr, strength):
    """A soft light around the middle of the screen at height center_y, as one color per cell."""
    distance = np.hypot(CELL_X - WIDTH / 2, CELL_Y - center_y)
    return np.clip(1 - distance / radius, 0, 1)[..., None] * np.array(bgr, np.float32) * strength


def render(chars, colors, background=None):
    """chars: (ROWS, COLS) indexes into CHARS; colors and background: (ROWS, COLS, 3) BGR. Returns the image."""
    mask = GLYPHS[chars].transpose(0, 2, 1, 3).reshape(HEIGHT, WIDTH)
    colors = cv2.resize(np.clip(colors, 0, 255).astype(np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = cv2.multiply(colors, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), scale=1 / 255)
    if background is not None:
        background = np.clip(background, 0, 255).astype(np.uint8)
        image = cv2.max(image, cv2.resize(background, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
    return image
