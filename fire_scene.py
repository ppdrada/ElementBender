import math

import numpy as np

from ascii_art import CELL_H, CELL_W, CHARS, COLS, ROWS, glow, render
from fire import MAX_HEAT, fire_chars, fire_colors
from scene import MAX_CHARGE, Scene

X = np.arange(COLS, dtype=np.float32)[None, :]
Y = np.arange(ROWS, dtype=np.float32)[:, None]
BLAZE_SECONDS = 3  # how long five open fingers take to set the whole screen on fire


class FireScene(Scene):
    """Fire bending: a flame follows the hand; a closed hand charges a fireball that stays in the hand until
    a quick swing throws it; five open fingers set the whole screen on fire."""

    def __init__(self):
        super().__init__()
        self.heat = np.zeros((ROWS, COLS), np.float32)
        self.previous = None
        self.charge = 0.0
        self.blaze = 0.0  # 0: only the hand burns, 1: the whole screen burns
        self.balls = []  # [x, y, speed_x, speed_y, radius, power]
        self.sparks = []  # [x, y, speed_x, speed_y, life, gravity]
        self.gust = 0.0
        self.gust_target = 0.0
        self.flash = 0.0

    def update(self, seconds, fingers, hand):
        """fingers: raised fingers (0 = closed hand, 5 = open hand); hand: (x, y) from 0 to 1, or None."""
        rand = self.rng.random
        holding, open_hand = self.read_input(seconds, fingers, hand)
        if not self.hand:
            self.charge = 0.0  # the hand left: the fireball goes out
        swing = self.flick()
        if swing and self.charge >= 0.1:
            self.throw(swing)
        if open_hand:
            self.blaze = min(1.0, self.blaze + seconds / BLAZE_SECONDS)
        else:
            self.blaze = max(0.0, self.blaze - seconds / 1.5)

        if rand() < 0.05:
            self.gust_target = (rand() - 0.5) * 1.0
        self.gust += (self.gust_target - self.gust) * 0.1
        if holding:
            self.charge = min(MAX_CHARGE, self.charge + seconds * 0.7)

        if self.hand:
            self.flame_on_hand(rand)
        else:
            self.previous = None
        if self.blaze > 0:
            # Flames catch everywhere on the screen
            cells = self.heat.reshape(-1)
            spots = self.rng.integers(0, cells.size, int(self.blaze * self.blaze * 120))
            cells[spots] = np.maximum(cells[spots], 14 + rand(spots.size) * 24)
        self.move_balls(rand)
        self.spread_heat()
        self.flash *= 0.85
        self.move_sparks()

    def inject(self, x, y, radius, heat):
        """Adds a round spot of heat; flames rise from it."""
        distance = np.hypot((X - x) / 1.4, Y - y) / radius
        value = heat * (1 - distance * 0.45) * (0.8 + self.rng.random((ROWS, COLS)) * 0.4)
        np.maximum(self.heat, np.where(distance <= 1, value, 0), out=self.heat)

    def flame_on_hand(self, rand):
        x, y = self.hand
        radius, heat = 1.1 + self.charge * 3.4, 26 + self.charge * 20
        if self.previous:
            # A trail of fire along the way the hand moved
            px, py = self.previous
            steps = min(30, max(1, math.ceil(math.hypot(x - px, y - py) / 0.7)))  # at least 1: a still hand burns too
            for s in range(1, steps + 1):
                k = s / steps
                self.inject(px + (x - px) * k, py + (y - py) * k, radius * (0.6 + 0.4 * k), heat * (0.7 + 0.3 * k))
        else:
            self.inject(x, y, radius, heat)
        if self.charge > 0.05 and rand() < min(1, self.charge):
            # Sparks pulled into the growing fireball
            for _ in range(2):
                angle, reach = rand() * math.tau, radius * 2.2
                self.sparks.append([x + math.cos(angle) * reach * 1.4, y + math.sin(angle) * reach,
                                    -math.cos(angle) * 0.5, -math.sin(angle) * 0.35, 0.6, 0.0])
        if rand() < 0.25:
            self.sparks.append([x + (rand() - 0.5) * 2, y - 1, (rand() - 0.5) * 0.5 + self.gust * 0.4,
                                -(0.4 + rand() * 0.8), 1.0, 0.0])
        self.previous = self.hand

    def throw(self, velocity):
        x, y = self.hand
        self.balls.append([x, y, velocity[0], velocity[1], 1.2 + self.charge * 3, self.charge])
        self.charge = 0.0

    def move_balls(self, rand):
        alive = []
        for ball in self.balls:
            x, y, vx, vy, radius, power = ball
            vy += 0.02  # a little gravity
            nx, ny = x + vx, y + vy
            steps = math.ceil(math.hypot(vx, vy) / 0.6)
            for s in range(1, steps + 1):
                k = s / steps
                self.inject(x + (nx - x) * k, y + (ny - y) * k, radius * (0.55 + 0.45 * k), 36 + power * 10)
            if rand() < 0.6:
                self.sparks.append([nx, ny, -vx * 0.2 + (rand() - 0.5) * 0.5, -vy * 0.2 + (rand() - 0.5) * 0.5, 0.8, 0.02])
            if 1 <= nx <= COLS - 2 and 1 <= ny <= ROWS - 2:
                alive.append([nx, ny, vx, vy, radius, power])
            else:
                self.explode(min(COLS - 2, max(1, nx)), min(ROWS - 2, max(1, ny)), power)
        self.balls = alive

    def explode(self, x, y, power):
        rand = self.rng.random
        reach = 3 + power * 9
        for _ in range(14):
            angle, distance = rand() * math.tau, rand() * reach
            self.inject(x + math.cos(angle) * distance * 1.4, y + math.sin(angle) * distance, 1.5 + power * 2, 44)
        for _ in range(int(30 + power * 60)):
            angle, speed = rand() * math.tau, 0.5 + rand() * 2.2 * (0.5 + power)
            self.sparks.append([x, y, math.cos(angle) * speed * 1.4, math.sin(angle) * speed, 1.0, 0.04])
        self.flash = min(1.0, 0.4 + power * 0.6)

    def spread_heat(self):
        """Every cell takes heat from a cell below it (a bit to the side) and cools down, so flames rise."""
        rand = self.rng.random
        rows = np.arange(ROWS - 1, dtype=np.float32)
        sway = self.gust + np.sin(rows * 0.4 + self.time * 4) * 0.6
        shift = self.rng.integers(-1, 2, (ROWS - 1, COLS)) + np.rint(sway[:, None] * rand((ROWS - 1, COLS)))
        source = np.clip(np.arange(COLS) - shift, 0, COLS - 1).astype(int)
        below = np.take_along_axis(self.heat[1:], source, axis=1)
        cooling = np.where(rand((ROWS - 1, COLS)) < 0.85, 1 + rand((ROWS - 1, COLS)) * 1.1, 0)
        self.heat[:-1] = np.maximum(np.maximum(0, below - cooling), self.heat[:-1] * 0.5)
        self.heat[-1] *= 0.5

    def move_sparks(self):
        alive = []
        for spark in self.sparks:
            spark[0] += spark[2]
            spark[1] += spark[3]
            spark[3] += spark[5]
            spark[2] += self.gust * 0.03
            spark[4] -= 0.03
            if spark[4] > 0 and 0 <= spark[1] < ROWS:
                alive.append(spark)
        self.sparks = alive

    def draw(self):
        value = np.clip(self.heat / MAX_HEAT, 0, 1)
        chars, colors = fire_chars(value), fire_colors(value)
        for x, y, _, _, life, _ in self.sparks:
            col, row = int(round(x)), int(round(y))
            if 0 <= col < COLS and 0 <= row < ROWS:
                chars[row, col] = CHARS.index("*" if life > 0.6 else ".")
                colors[row, col] = np.array([60, 150 + life * 100, 255]) * min(1, life)
        if self.hand:
            col, row = int(round(self.hand[0])), int(round(self.hand[1]))
            if 0 <= col < COLS and 0 <= row < ROWS:
                chars[row, col] = CHARS.index("o")
                colors[row, col] = (128, 128, 128)

        background = np.zeros((ROWS, COLS, 3), np.float32) + np.array([30, 120, 255]) * self.flash * 0.35
        background += np.array([10, 30, 160]) * self.blaze * 0.45
        if self.hand:
            background += glow(self.hand[1] * CELL_H, 90 + self.charge * 200, (5, 30, 150), 0.3 + self.charge * 0.35,
                               self.hand[0] * CELL_W)
        image = render(chars, colors, background)
        image = self.shake(image, max((self.charge - 2) * 1.5, self.blaze * 3))  # big fire makes the screen tremble
        return self.draw_back(image)
