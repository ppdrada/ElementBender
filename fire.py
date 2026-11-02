import math

import cv2
import numpy as np

COLS, ROWS = 96, 40
CELL_W, CELL_H = 10, 12
WIDTH, HEIGHT = COLS * CELL_W, ROWS * CELL_H
RAMP = " .:-=+*#%@"
MAX_HEAT = 36


def clamp(value):
    return max(0.0, min(1.0, value))


def ease(value):
    return value * value * (3 - 2 * value)


def make_glyphs():
    # One small black-and-white picture for each character of the ramp
    glyphs = np.zeros((len(RAMP), CELL_H, CELL_W), np.uint8)
    for i, char in enumerate(RAMP):
        cv2.putText(glyphs[i], char, (1, CELL_H - 3), cv2.FONT_HERSHEY_PLAIN, 0.8, 255, 1, cv2.LINE_AA)
    # The thin font looks dim, so make every drawn pixel stronger
    return np.clip(glyphs.astype(np.int32) * 2, 0, 255).astype(np.uint8)


GLYPHS = make_glyphs()
# Distance of every cell from the bottom center (in pixels), used for the red glow
_ys, _xs = np.mgrid[0:ROWS, 0:COLS]
GLOW_DISTANCE = np.hypot((_xs + 0.5) * CELL_W - WIDTH / 2, (_ys + 0.5) * CELL_H - HEIGHT).astype(np.float32)


class Fire:
    """ASCII fire: a candle that becomes a bonfire, spreads to the sides and fills the screen."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.heat = np.zeros((ROWS, COLS), np.float32)
        self.column_heat = np.ones(COLS, np.float32)
        self.gust = 0.0
        self.gust_target = 0.0
        self.bursts = []  # [x, life, power]
        self.embers = []  # [x, y, speed_x, speed_y, life]
        self.time = 0.0
        self.strength = 0.0
        self.bonfire = 0.0

    def step(self, progress, lit):
        """Moves the fire one frame. progress goes from 0 (candle) to 1 (whole screen)."""
        rand = self.rng.random
        self.time += 1 / 24
        t = self.time
        center = COLS / 2

        # The three stages: bonfire in the middle, spread to the sides, whole screen
        bonfire = ease(clamp(progress / 0.55))
        spread = ease(clamp((progress - 0.55) / 0.27))
        everywhere = ease(clamp((progress - 0.82) / 0.18))
        strength = 0.2 * bonfire + 0.45 * spread + 0.35 * everywhere
        self.bonfire, self.strength = bonfire, strength

        if rand() < 0.04 + strength * 0.08:
            self.gust_target = (rand() - 0.5) * (0.2 + bonfire * 0.6 + spread * 3 + everywhere * 2)
        self.gust += (self.gust_target - self.gust) * 0.15

        half_width = 1.5 + 22 * bonfire + 60 * spread
        if bonfire > 0.4 and rand() < 0.06 + strength * 0.3:
            self.bursts.append([center + (rand() - 0.5) * 1.6 * half_width, 1.0, 12 + rand() * 24 * max(strength, 0.3)])
        self.column_heat += (0.55 + rand(COLS) * 0.9 - self.column_heat) * (0.25 + strength * 0.5)

        heat = 10 + 34 * bonfire + 4 * spread + 8 * everywhere
        decay_chance = 1 - 0.44 * bonfire - 0.06 * spread - 0.25 * everywhere
        wobble = 2 * bonfire + 4 * spread
        left = center - half_width - wobble * abs(math.sin(t * 3.1))
        right = center + half_width + wobble * abs(math.sin(t * 2.6 + 1))

        # Bottom row: where the fire is fed
        x = np.arange(COLS, dtype=np.float32)
        base = np.zeros(COLS, np.float32)
        if lit:
            edge = np.minimum(x - left, right - x)
            dome = 1.0 if spread > 0 else 0.55 + 0.45 * np.cos(np.minimum(1, np.abs(x - center) / half_width) * math.pi / 2)
            flicker = 0.85 + rand(COLS) * 0.3 if bonfire < 0.05 else self.column_heat
            base = np.where(edge > 0, heat * dome * np.clip(edge / 1.5, 0, 1) * flicker, 0)
            for burst_x, life, power in self.bursts:
                base += life * power * np.maximum(0, 1 - np.abs(x - burst_x) / (2 + strength * 9))
        for burst in self.bursts:
            burst[1] -= 0.25
        self.bursts = [b for b in self.bursts if b[1] > 0]
        self.heat[-1] = base

        if everywhere > 0:
            # Flames catch everywhere on the screen
            above = self.heat[:-1].reshape(-1)
            spots = self.rng.integers(0, above.size, int(everywhere * 90))
            above[spots] = np.maximum(above[spots], 14 + rand(spots.size) * 24)

        # Every cell takes the heat from a cell below it (a bit to the side), and cools down
        rows = np.arange(ROWS - 1, dtype=np.float32)
        sway = (self.gust * (1 - rows / ROWS * 0.3)
                + np.sin(rows * 0.45 + t * 4.5) * (0.35 + 1.2 * strength)
                + np.sin(rows * 0.15 - t * 1.7) * 0.8 * strength)
        shift = self.rng.integers(-1, 2, (ROWS - 1, COLS)) + np.rint(sway[:, None] * rand((ROWS - 1, COLS)) * 1.4)
        source_x = np.clip(np.arange(COLS) - shift, 0, COLS - 1).astype(int)
        below = np.take_along_axis(self.heat[1:], source_x, axis=1)
        cooling = np.where(rand((ROWS - 1, COLS)) < decay_chance, 0.8 + rand((ROWS - 1, COLS)) * 0.9, 0)
        new_heat = np.maximum(0, below - cooling)
        if everywhere > 0:
            new_heat = np.maximum(new_heat, self.heat[:-1] * 0.9)
        self.heat[:-1] = new_heat

        for _ in range(math.ceil(strength * 4)):
            if rand() < 0.7:
                self.embers.append([left + rand() * (right - left), ROWS - 4 - rand() * 26 * strength,
                                    (rand() - 0.5) * 1.5, -(0.8 + rand() * 1.4), 1.0])
        for ember in self.embers:
            ember[0] += ember[2] + self.gust * 0.7
            ember[1] += ember[3]
            ember[2] += (rand() - 0.5) * 0.4
            ember[4] -= 0.025
        self.embers = [e for e in self.embers if e[4] > 0 and e[1] >= 0 and 0 <= e[0] <= COLS]

    def draw(self, lit):
        """Returns the fire as a BGR image."""
        value = np.clip(self.heat / MAX_HEAT, 0, 1)
        index = np.where(value < 0.03, 0, np.clip(np.rint(value * 9), 1, 9)).astype(int)

        # Dark red at the edges, orange and yellow inside, white in the hottest part
        hot = value >= 0.35
        red = np.where(hot, 255, 110 + value * 400)
        green = np.where(hot, np.minimum(255, (value - 0.3) * 380), value * 40)
        blue = np.where(hot, np.clip((value - 0.9) * 1200, 0, 255), value * 30)
        colors = np.stack([blue, green, red], axis=-1).astype(np.uint8)

        # Put each character picture in its cell, painted with its color
        mask = GLYPHS[index].transpose(0, 2, 1, 3).reshape(HEIGHT, WIDTH)
        colors = cv2.resize(colors, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        image = cv2.multiply(colors, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), scale=1 / 255)

        if lit:
            pulse = 0.85 + 0.15 * math.sin(self.time * 9) + 0.1 * self.rng.random()
            radius = 80 + self.bonfire * 250 + self.strength * 600
            glow = np.clip(1 - GLOW_DISTANCE / radius, 0, 1) * (0.3 + self.bonfire * 0.15 + self.strength * 0.35) * pulse
            glow = np.stack([glow * 5, glow * 15, glow * 150], axis=-1).astype(np.uint8)
            image = cv2.max(image, cv2.resize(glow, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))

        for x, y, _, _, life in self.embers:
            color = (60, 150 + life * 100, 255)
            cv2.putText(image, "*" if life > 0.6 else ".", (int(x * CELL_W), int(y * CELL_H) + CELL_H),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, tuple(c * life for c in color), 1, cv2.LINE_AA)

        shake = (self.strength - 0.5) * 6 if self.strength > 0.5 else 0
        if shake:
            image = np.roll(image, (int(self.rng.uniform(-shake, shake)), int(self.rng.uniform(-shake, shake))), axis=(0, 1))
        return image
