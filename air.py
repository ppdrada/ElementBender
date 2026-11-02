import math

import numpy as np

from ascii_art import CHARS, COLS, HEIGHT, ROWS, clamp, ease, glow, lerp, render

PARTICLES = 1200
MAX_TRAIL = 16
ASPECT = 1.6  # cells are taller than wide, so circles are stretched sideways
CENTER = COLS / 2
BASE = ROWS + 2  # the funnel starts just below the screen
X = np.arange(COLS, dtype=np.float64)
Y = np.arange(ROWS, dtype=np.float64)[:, None]
# Direction of a trail piece: horizontal, down-right, vertical, up-right
DIRECTION_CHARS = np.array([CHARS.index(c) for c in "-\\|/"])
DEBRIS_CHARS = "o@*#.'"


def index(char):
    return CHARS.index(char)


class Air:
    """ASCII air: a wavy breeze is caught by a hurricane funnel that grows until it consumes the screen."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.time = 0.0
        self.seconds = 0.0
        self.progress = 0.0
        self.x = np.zeros(PARTICLES)
        self.y = np.zeros(PARTICLES)
        self.caught = np.zeros(PARTICLES, bool)
        self.angle = np.zeros(PARTICLES)
        self.height = np.zeros(PARTICLES)  # 0 at the bottom of the funnel, 1 at the top
        self.size = np.ones(PARTICLES)
        self.warm = np.zeros(PARTICLES, bool)
        self.trail = np.zeros((MAX_TRAIL, PARTICLES, 5))  # x, y, speed x, speed y, front (>0) or back
        self.trail_length = np.zeros(PARTICLES, int)
        self.spawn(np.ones(PARTICLES, bool), anywhere=True)
        self.debris = []  # [angle, height, size, char, life]
        self.gusts = []  # [x, y, length, speed]

    def spawn(self, which, anywhere):
        n = int(which.sum())
        self.x[which] = self.rng.random(n) * COLS if anywhere else -2 - self.rng.random(n) * 6
        self.y[which] = self.rng.random(n) * ROWS
        self.caught[which] = False
        self.warm[which] = self.rng.random(n) < 0.15
        self.size[which] = 0.6 + self.rng.random(n) * 0.8
        self.trail_length[which] = 0

    def step(self, progress, lit, seconds):
        """Moves the air one frame. progress goes from 0 (calm) to 1 (the hurricane fills the screen)."""
        self.time += seconds
        self.seconds = seconds
        self.progress = progress
        if not lit and self.trail_length.any():
            self.reset()

    def breeze(self, x, y, strength):
        t = self.time
        vx = 0.9 + 0.4 * strength + np.sin(y * 0.4 + t) * 0.25
        vy = (np.sin(x * 0.13 + t * 1.4 + y * 0.21) * 0.7 + np.sin(x * 0.37 - t * 2.1 + y * 0.5) * 0.35
              + np.cos(x * 0.21 + t * 0.9) * np.sin(y * 0.33 - t * 0.7) * 0.5)
        return vx, vy

    def draw(self, lit):
        """Returns the air as a BGR image."""
        p, t, k, rand = self.progress, self.time, self.seconds * 24, self.rng.random
        breeze = ease(clamp(p / 0.3))
        swirl = ease(clamp((p - 0.2) / 0.45))
        consume = ease(clamp((p - 0.6) / 0.4))
        fury = 1 + consume * 3.5
        trail_size = 8 + round(consume * 8)
        funnel_height = lerp(4, ROWS * 0.95, swirl) + consume * ROWS * 0.6

        def radius(h):
            return (0.8 + h * (4 + swirl * 12)) * (1 + consume * 4.5)

        def axis(h):
            return CENTER + np.sin(h * 2.6 + t * 1.6 * fury) * (1.5 + swirl * 3) * (1 - consume * 0.5)

        chars = np.zeros((ROWS, COLS), int)
        colors = np.zeros((ROWS, COLS, 3), np.float32)
        if lit:
            if swirl > 0.05:
                self.draw_funnel(chars, colors, funnel_height, radius, axis, swirl, consume, fury)
            if consume > 0:
                if rand() < consume * 1.5 * k:
                    for _ in range(3):
                        self.gusts.append([-15, rand() * ROWS, 10 + rand() * 25, 3 + rand() * 3])
                if rand() < consume * 2 * k:
                    for _ in range(2):
                        self.debris.append([rand() * 6.28, rand() * 0.9, 0.5 + rand() * 0.9, DEBRIS_CHARS[int(rand() * 6)], 1.0])
            count = round(lerp(180, 520, clamp(breeze * 0.6 + swirl * 0.6)) + consume * 680)
            # Small steps keep the trails continuous even when the program runs at a low frame rate
            substeps = max(1, math.ceil(k / 0.4))
            for _ in range(substeps):
                self.move_wind(count, breeze, swirl, consume, fury, funnel_height, radius, axis, k / substeps)
            self.draw_wind(chars, colors, count, trail_size, consume)
            self.move_debris(chars, colors, funnel_height, radius, axis, fury, k)
            self.move_gusts(chars, colors, k)

        background = None
        if lit:
            light = 60 * consume
            background = glow(HEIGHT * 0.7, 120 + swirl * 300 + consume * 600,
                              (100 + light, 80 + light, 70 + light), 0.15 + swirl * 0.25 + consume * 0.35)
        image = render(chars, colors, background)
        shake = consume * consume * 10 if lit else 0
        if shake >= 1:
            image = np.roll(image, (int(self.rng.uniform(-shake, shake)), int(self.rng.uniform(-shake, shake))), axis=(0, 1))
        return image

    def draw_funnel(self, chars, colors, funnel_height, radius, axis, swirl, consume, fury):
        """The funnel's body: bands of wind that turn around it."""
        h = (BASE - Y) / funnel_height
        center, r = axis(h), radius(h) * ASPECT
        u = (X - center) / r
        inside = (h >= 0) & (h <= 1.05) & (np.abs(u) <= 1)
        band = np.sin(np.arcsin(np.clip(u, -1, 1)) * 3 - self.time * 7 * fury + Y * 0.55)
        show = inside & (band >= 0.55 - consume * 0.7)
        body = np.select([(consume > 0.3) & (band > 0.8), (consume > 0.15) & (band > 0.45), band > 0.85, band > 0.2],
                         [index("@"), index("%"), index(":"), index("~")], index("."))
        shade = (1 - np.abs(u) * 0.7) * (0.35 * swirl + 0.55 * consume)
        strength = np.minimum(1, shade * (0.6 + 0.4 * band))[..., None]
        color = np.array([245, 222 + 25 * consume, 210 + 40 * consume]) * strength
        chars[show] = body[show]
        colors[show] = color[show]

    def move_wind(self, count, breeze, swirl, consume, fury, funnel_height, radius, axis, k):
        active = np.arange(PARTICLES) < count
        x, y = self.x, self.y

        # Free wind follows the breeze; near the funnel it can get caught
        free = active & ~self.caught
        vx, vy = self.breeze(x, y, breeze)
        h = (BASE - y) / funnel_height
        hc = np.clip(h, 0, 1)
        center, r = axis(hc), radius(hc) * ASPECT
        near = (np.abs(x - center) < r + 3 + consume * 30) & (h >= 0) & (h <= 1)
        catch = free & near & (swirl > 0.05) & (self.rng.random(PARTICLES) < swirl * 0.5 + consume)
        self.caught |= catch
        self.height[catch] = hc[catch]
        self.angle[catch] = np.arcsin(np.clip((x - center) / np.maximum(r, 1e-6), -1, 1)[catch] * 0.999)
        self.trail_length[catch] = np.minimum(self.trail_length[catch], 2)
        speed = (0.5 + breeze * 0.5 + consume * 2) * k
        new_x, new_y = x + vx * speed, y + vy * speed

        # Caught wind turns around the funnel and climbs it
        turning = active & self.caught
        self.angle[turning] += (0.22 + 0.3 / (0.4 + self.height[turning])) * k * fury
        self.height[turning] += 0.006 * k * self.size[turning] * fury
        escaped = turning & (self.height > 1.05)
        turning &= ~escaped
        ring = radius(self.height) * self.size
        new_x = np.where(turning, axis(self.height) + np.sin(self.angle) * ring * ASPECT, new_x)
        new_y = np.where(turning, BASE - self.height * funnel_height, new_y)
        vx = np.where(turning, np.cos(self.angle) * ring * ASPECT, vx)
        vy = np.where(turning, -0.3, vy)
        front = np.where(turning, np.cos(self.angle), 1)

        # Remember where each piece of wind was, for its trail
        moving = active & ~escaped
        self.trail[1:, moving] = self.trail[:-1, moving]
        self.trail[0, moving] = np.stack([x, y, vx, vy, front], axis=1)[moving]
        self.trail_length[moving] = np.minimum(self.trail_length[moving] + 1, MAX_TRAIL)
        self.x = np.where(moving, new_x, x)
        self.y = np.where(moving, new_y, y)

        gone = active & ~self.caught & ((self.x > COLS + 2) | (self.x < -10) | (self.y < -3) | (self.y > ROWS + 3))
        self.spawn(gone, anywhere=False)
        self.spawn(escaped, anywhere=consume > 0.2)

    def draw_wind(self, chars, colors, count, trail_size, consume):
        active = np.arange(PARTICLES) < count
        for j in range(min(trail_size, MAX_TRAIL) - 1, -1, -1):  # oldest first, so the heads end on top
            tx, ty, tvx, tvy, front = self.trail[j].T
            col, row = np.rint(tx).astype(int), np.rint(ty).astype(int)
            show = active & (j < self.trail_length) & (col >= 0) & (col < COLS) & (row >= 0) & (row < ROWS)
            fade = (1 - j / trail_size) * np.where(front > 0, 1, 0.45 + consume * 0.3)
            if j == 0:
                char = np.full(PARTICLES, index("@" if consume > 0.4 else "~"))
            elif j < 4:
                angle = np.arctan2(tvy, tvx / ASPECT) % math.pi
                char = DIRECTION_CHARS[np.where((angle < math.pi / 8) | (angle > math.pi * 7 / 8), 0,
                                                np.where(angle < math.pi * 3 / 8, 1, np.where(angle < math.pi * 5 / 8, 2, 3)))]
            else:
                char = np.full(PARTICLES, index("."))
            color = np.where(self.warm[:, None], np.array([110, 214, 255]) * (fade * 0.95)[:, None],
                             np.array([255, 240, 230]) * np.minimum(1, fade * (0.9 + consume * 0.3))[:, None])
            chars[row[show], col[show]] = char[show]
            colors[row[show], col[show]] = color[show]

    def move_debris(self, chars, colors, funnel_height, radius, axis, fury, k):
        alive = []
        for d in self.debris:
            d[0] += 0.5 * k * fury
            d[1] += 0.012 * k * fury
            d[4] -= 0.01 * k
            if d[1] > 1.05 or d[4] <= 0:
                continue
            alive.append(d)
            col = int(round(float(axis(d[1])) + math.sin(d[0]) * radius(d[1]) * d[2] * ASPECT))
            row = int(round(BASE - d[1] * funnel_height))
            if 0 <= col < COLS and 0 <= row < ROWS:
                chars[row, col] = index(d[3])
                colors[row, col] = (154, 210, 232) if math.cos(d[0]) > 0 else (88, 122, 138)
        self.debris = alive

    def move_gusts(self, chars, colors, k):
        alive = []
        for g in self.gusts:
            g[0] += g[3] * k
            if g[0] - g[2] > COLS:
                continue
            alive.append(g)
            for j in range(int(g[2])):
                col = int(round(g[0] - j))
                row = int(round(g[1] + math.sin((g[0] - j) * 0.3) * 0.5))
                if 0 <= col < COLS and 0 <= row < ROWS:
                    chars[row, col] = index("=" if j < 2 else "-")
                    colors[row, col] = np.array([255, 255, 255]) * (1 - j / g[2]) * 0.8
        self.gusts = alive
