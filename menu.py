import cv2

from air import Air
from ascii_art import HEIGHT, WIDTH, clamp
from earth import Earth
from fire import Fire
from water import Water

ELEMENTS = ["fire", "water", "earth", "air"]
EFFECTS = {"fire": Fire, "water": Water, "earth": Earth, "air": Air}
FILL_SECONDS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
# (dim, bright) label colors in BGR
LABEL_COLORS = {"fire": ((32, 74, 154), (71, 179, 255)), "water": ((154, 79, 31), (255, 191, 127)),
                "earth": ((42, 90, 74), (101, 204, 156)), "air": ((64, 96, 107), (138, 224, 255))}


def centered_x(text, scale):
    (width, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    return (WIDTH - width) // 2


class Menu:
    """Black screen with the four elements; hover, fingers or a key fill an element, a click enters it."""

    def __init__(self):
        self.effects = {name: effect() for name, effect in EFFECTS.items()}
        self.current = "fire"
        self.progress = 0.0
        self.seconds = 0.0
        self.lit = False
        self.hovered = None
        self.chosen = None
        self.option_boxes = self.layout_options()

    def layout_options(self):
        # (x, y, width, height) of each "1 fire", "2 water"... text, side by side in the middle
        labels = [f"{i + 1} {name}" for i, name in enumerate(ELEMENTS)]
        widths = [cv2.getTextSize(label, FONT, 0.6, 1)[0][0] for label in labels]
        gap = 36
        x = (WIDTH - sum(widths) - gap * (len(labels) - 1)) // 2
        boxes = []
        for width in widths:
            boxes.append((x, HEIGHT // 2 - 8, width, 20))
            x += width + gap
        return boxes

    def option_at(self, x, y):
        for i, (bx, by, bw, bh) in enumerate(self.option_boxes):
            if bx <= x <= bx + bw and by - 6 <= y <= by + bh + 6:
                return i
        return None

    def on_mouse(self, event, x, y, flags, param):
        self.hovered = self.option_at(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.hovered is not None:
            self.choose(ELEMENTS[self.hovered])

    def switch_to(self, element):
        if element != self.current:
            self.effects[element].reset()
            self.current = element

    def choose(self, element):
        if element in self.effects and self.chosen is None:
            self.switch_to(element)
            self.chosen = element
            self.progress = 1.0
            self.lit = True

    def reset(self):
        self.progress = 0.0
        self.lit = False
        self.chosen = None
        for effect in self.effects.values():
            effect.reset()

    def update(self, seconds, fingers, key_option):
        """seconds: time since the last frame; fingers: 0-4; key_option: 0-3 or None."""
        self.seconds = seconds
        if self.chosen:
            return
        pointed = [self.hovered, key_option, fingers - 1 if fingers else None]
        wanted = next((ELEMENTS[i] for i in pointed if i is not None and ELEMENTS[i] in self.effects), None)
        # A new element only starts after the current one has gone back to zero
        if wanted and (wanted == self.current or self.progress == 0):
            self.switch_to(wanted)
            self.lit = True
            self.progress = min(1.0, self.progress + seconds / FILL_SECONDS)
            if self.progress >= 1:
                self.choose(wanted)
        else:
            self.progress = max(0.0, self.progress - seconds * 1.2)
            if self.progress == 0:
                self.lit = False

    def draw(self):
        effect = self.effects[self.current]
        effect.step(self.progress, self.lit, self.seconds)
        image = effect.draw(self.lit)

        if self.chosen:
            text = f"entering {self.chosen}..."
            cv2.putText(image, text, (centered_x(text, 0.8), HEIGHT // 2), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            return image

        # The menu text fades out quickly when an element starts
        fade = clamp(1 - self.progress * 5)
        if fade == 0:
            return image
        title, hint = "Choose an element", "Hover or hold keys 1-4 (fingers) to fill, or click to enter now."
        cv2.putText(image, title, (centered_x(title, 0.9), HEIGHT // 2 - 80), FONT, 0.9, (221 * fade,) * 3, 1, cv2.LINE_AA)
        cv2.putText(image, hint, (centered_x(hint, 0.45), HEIGHT // 2 - 50), FONT, 0.45, (136 * fade,) * 3, 1, cv2.LINE_AA)
        for i, (x, y, _, h) in enumerate(self.option_boxes):
            name = ELEMENTS[i]
            dim, bright = LABEL_COLORS[name]
            color = bright if self.lit and self.current == name else dim
            cv2.putText(image, f"{i + 1} {name}", (x, y + h - 4), FONT, 0.6, tuple(c * fade for c in color), 1, cv2.LINE_AA)
        return image
