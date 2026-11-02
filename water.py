import math

import cv2
import numpy as np

from ascii_art import CELL_H, CELL_W, CHARS, COLS, HEIGHT, ROWS, WIDTH, clamp, ease, glow, lerp, render

# The shapes are drawn on a half-size canvas, then read cell by cell
SCALE = 0.5
BLOCK_H, BLOCK_W = int(CELL_H * SCALE), int(CELL_W * SCALE)
NAVY, BLUE, LIGHT, WHITE = 1, 2, 3, 4
BANDS = [WHITE, BLUE, WHITE, LIGHT, BLUE, WHITE, LIGHT, BLUE]
ROW_SPACING = 46
# Red, green and blue amount (0-255) of each label, used to mix the colors inside a cell
COLOR_TABLES = [np.zeros(256, np.uint8) for _ in range(3)]
for _label, _rgb in {NAVY: (0, 0, 255), BLUE: (0, 128, 255), LIGHT: (0, 255, 255), WHITE: (255, 255, 255)}.items():
    for _table, _value in zip(COLOR_TABLES, _rgb):
        _table[_label] = _value

# The great wave outline as Bezier curves, base at (0, 0) and curling to the left;
# it is drawn mirrored, so on screen it comes from the left and curls to the right
WAVE = [(380, 0), (200, -60, 20, -280, -150, -410), (-200, -440, -290, -440, -340, -390),
        (-380, -350, -370, -300, -320, -295), (-280, -292, -260, -320, -270, -340),
        (-220, -310, -200, -220, -240, -120), (-260, -60, -290, -20, -320, 0)]
CONTROL = np.array([WAVE[0]] + [seg[i:i + 2] for seg in WAVE[1:] for i in (0, 2, 4)], np.float32)
# Parts of the crest edge where the foam claws grow: (curve number, from, to)
CLAW_PARTS = [(1, 0.75, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1)]


def bezier(control, curve, u):
    p0, p1, p2, p3 = control[3 * curve - 3], control[3 * curve - 2], control[3 * curve - 1], control[3 * curve]
    u = np.asarray(u, np.float32)[:, None]
    v = 1 - u
    return v ** 3 * p0 + 3 * v * v * u * p1 + 3 * v * u * u * p2 + u ** 3 * p3


def profile(x, row, amplitude, length, speed, t):
    # A wave that rises slowly and falls fast, moving to the right
    f = ((x - t * speed + row * 137) / length) % 1
    return amplitude * np.where(f < 0.7, ease(f / 0.7), 1 - ease((f - 0.7) / 0.3))


def to_canvas(points):
    return np.round(np.asarray(points) * SCALE).astype(np.int32)


class Water:
    """ASCII water: two Great Waves ride a rising sea, crash, the sea covers the screen and the view sinks."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.time = 0.0
        self.seconds = 0.0
        self.progress = 0.0
        self.lit = False
        self.crashed = [False, False]
        self.spray = []  # [x, y, speed_x, speed_y, life] in cells
        self.bubbles = []  # [x, y, size, life] in cells
        self.canvas = np.zeros((int(HEIGHT * SCALE), int(WIDTH * SCALE)), np.uint8)

    def row_y(self, x, row, top):
        t = self.time
        amplitude = (18 + 8 * (row % 2)) * (0.8 + 0.3 * math.sin(t * 0.9 + row))
        return (top + row * ROW_SPACING - profile(x, row, amplitude, 180 + 40 * (row % 3), 90 + 20 * row, t)
                + np.sin(x * 0.013 + t * 1.7 + row) * 10 + np.sin(x * 0.031 - t * 2.3 + row * 2) * 6)

    def bend(self, points, crash, phase):
        """Makes the wave sway like water, and melt forward when it crashes."""
        t = self.time
        x, y = points[:, 0], points[:, 1]
        height = np.clip(-y / 440, 0, 1)
        tip = np.clip((-x - 150) / 200, 0, 1)
        new_x = x + np.sin(y * 0.012 + t * 2.6 + phase) * 28 * height + math.sin(t * 3.7 + phase) * 18 * tip
        new_y = y + np.sin(x * 0.015 + t * 2.1 + phase) * 16 * height + math.cos(t * 3.1 + phase) * 14 * tip
        if crash > 0:
            new_x = new_x - (crash * height ** 2 * 300 + crash * tip * 120)
            new_y = (new_y * (1 - 0.92 * crash) + crash * (1 - crash) * height * tip * 120
                     + np.sin(x * 0.03 + t * 6) * 14 * crash * height * (1 - crash * 0.6))
        return np.stack([new_x, np.minimum(0, new_y)], axis=1)

    def draw_wave(self, x0, y0, size, crash, phase):
        """Draws one Great Wave and returns the points of its claws (in cells)."""
        if size < 0.02:
            return []
        t = self.time
        control = self.bend(CONTROL, crash, phase)
        to_world = lambda p: np.stack([x0 - size * p[:, 0], y0 + size * p[:, 1]], axis=1)
        outline = np.concatenate([bezier(control, curve, np.linspace(0, 1, 12)) for curve in range(1, 7)])

        # Body with moving bands, kept inside the wave
        body = np.zeros_like(self.canvas)
        cv2.fillPoly(body, [to_canvas(to_world(outline))], 1)
        bands = np.zeros_like(self.canvas)
        anchor = self.bend(np.array([[110, -20]], np.float32), crash, phase)[0]
        flow = (t * 0.5 + phase) % 1
        for j, band in enumerate(BANDS):
            shrink = 1 - (j + flow) * 0.1
            if shrink > 0.05:
                width = (20 if band == WHITE else 15) * SCALE
                cv2.polylines(bands, [to_canvas(to_world(anchor + (outline - anchor) * shrink))], True, band, max(1, round(width)))
        inside = body > 0
        self.canvas[inside] = np.where(bands[inside] > 0, bands[inside], NAVY)

        # Foam edge along the crest
        crest = np.concatenate([bezier(control, curve, np.linspace(0, 1, 11)) for curve in (2, 3, 4)])
        cv2.polylines(self.canvas, [to_canvas(to_world(crest))], False, WHITE, round((26 + crash * 20) * SCALE))

        # Foam claws pointing into the curl
        center = self.bend(np.array([[-290, -350]], np.float32), crash, phase)[0]
        if crash > 0.95:
            return []
        claws = []
        for curve, start, end in CLAW_PARTS:
            u = np.arange(start, end, 0.08, dtype=np.float32)
            points = bezier(control, curve, u)
            direction = center - points
            direction /= np.maximum(np.hypot(direction[:, 0], direction[:, 1]), 1)[:, None]
            wiggle = np.sin(t * 4 + u * 10 + curve + phase * 9)[:, None]
            reach = (30 + 14 * np.sin(t * 6 + curve * 7 + u * 20 + phase * 5)) * (0.6 if curve == 4 else 1) * (1 - crash * 0.8)
            sideways = np.stack([-direction[:, 1], direction[:, 0]], 1)
            tips = points + direction * reach[:, None] + sideways * 10 * wiggle
            bases, ends = to_world(points), to_world(tips)
            for a, b in zip(to_canvas(bases).tolist(), to_canvas(ends).tolist()):
                cv2.line(self.canvas, a, b, WHITE, round(14 * SCALE))
            claws += (bases / [CELL_W, CELL_H]).tolist()
        return claws

    def draw_sea(self, top):
        xs = np.arange(0, WIDTH + 8, 8, dtype=np.float32)
        t = self.time
        for row in range(math.ceil((HEIGHT - top) / ROW_SPACING) + 2):
            ys = self.row_y(xs, row, top)
            cv2.fillPoly(self.canvas, [to_canvas(np.concatenate([np.stack([xs, ys], 1), [[WIDTH, HEIGHT + 10], [0, HEIGHT + 10]]]))], NAVY)
            for label, offset, width in ((BLUE, 22, 14), (LIGHT, 10, 12), (WHITE, 0, 10)):
                line = np.stack([xs, ys + offset + np.sin(xs * 0.05 + t * 3 + offset) * 3], 1)
                cv2.polylines(self.canvas, [to_canvas(line)], False, label, round(width * SCALE))
            # Crests that curl forward, like in a storm
            shape = profile(xs, row, 1, 180 + 40 * (row % 3), 90 + 20 * row, t)
            for i in np.nonzero((shape[:-1] > 0.93) & (shape[1:] < shape[:-1]))[0]:
                radius = 12 + 5 * math.sin(t * 5 + xs[i] * 0.1)
                center = to_canvas([xs[i] + radius * 0.6, ys[i] + radius * 0.9]).tolist()
                cv2.ellipse(self.canvas, center, (round(radius * SCALE),) * 2, 0, 189, 387, WHITE, round(11 * SCALE))
                cv2.circle(self.canvas, to_canvas([xs[i] + radius * 1.4, ys[i] + radius * 1.4]).tolist(), 2, WHITE, -1)

    def foam_carpet(self, x, size, crash, top, row):
        if crash <= 0.3:
            return
        strength = clamp((crash - 0.3) / 0.4) * min(1, size * 2)
        t = self.time
        for fx in np.arange(x - size * 100, x + size * (300 + crash * 260), 14):
            radius = (9 + 7 * math.sin(t * 7 + fx * 0.05)) * strength
            if radius > 1:
                center = to_canvas([fx + math.sin(t * 5 + fx) * 4, self.row_y(fx, row, top) + 2]).tolist()
                cv2.circle(self.canvas, center, round(radius * SCALE), WHITE, -1)

    def burst(self, x, y, size, count):
        rand = self.rng.random
        for _ in range(count):
            self.spray.append([(x + size * 320) / CELL_W + rand() * size * 30, y / CELL_H - rand() * 16 * size,
                               (rand() - 0.15) * 2.2, -(0.4 + rand() * 2 * size), 1.0])

    def step(self, progress, lit, seconds):
        """Moves the water one frame. progress goes from 0 (calm sea) to 1 (deep under water)."""
        self.time += seconds
        self.seconds = seconds
        self.progress, self.lit = progress, lit
        if not lit:
            self.crashed = [False, False]

    def stages(self):
        p = self.progress
        return {
            "rise_small": ease(clamp(p / 0.28)), "crash_small": ease(clamp((p - 0.24) / 0.3)),
            "rise_big": ease(clamp((p - 0.06) / 0.44)), "crash_big": ease(clamp((p - 0.48) / 0.28)),
            "cover": ease(clamp((p - 0.56) / 0.26)), "under": ease(clamp((p - 0.76) / 0.24)),
        }

    def draw(self, lit):
        """Returns the water as a BGR image."""
        s = self.stages()
        t, rand, k = self.time, self.rng.random, self.seconds * 24  # k: how many 1/24 s steps passed
        top = lerp(lerp(HEIGHT - 14, HEIGHT - 170, ease(clamp(self.progress / 0.6))), -90, s["cover"])
        self.canvas[:] = 0
        claws = []
        if lit:
            big_x = lerp(40, 250, s["rise_big"]) + s["crash_big"] * 160
            big_size = 0.06 + 0.75 * s["rise_big"]
            claws += self.draw_wave(big_x, self.row_y(big_x, 0, top) + 14, big_size, s["crash_big"], 0)
            self.draw_sea(top)
            self.foam_carpet(big_x, big_size, s["crash_big"], top, 0)

            small_x = lerp(160, 420, s["rise_small"]) + s["crash_small"] * 110
            small_size = 0.05 + 0.36 * s["rise_small"]
            claws += self.draw_wave(small_x, self.row_y(small_x, 1, top) + 10, small_size, s["crash_small"], 1.7)
            self.foam_carpet(small_x, small_size, s["crash_small"], top, 1)

            if s["crash_small"] > 0.45 and not self.crashed[0]:
                self.crashed[0] = True
                self.burst(small_x, self.row_y(small_x, 1, top), small_size, 40)
            if s["crash_big"] > 0.45 and not self.crashed[1]:
                self.crashed[1] = True
                self.burst(big_x, self.row_y(big_x, 0, top), big_size, 100)

        labels = self.classify(self.canvas)
        under = s["under"]
        surface_y = lerp(HEIGHT * 0.55, -40, under)
        if under > 0:
            labels = self.sink(labels, under, surface_y)

        chars, colors = self.paint(labels, under, surface_y / CELL_H)

        if lit and s["crash_big"] < 0.6:
            for x, y in claws:
                if rand() < 0.04 * k:
                    self.spray.append([x, y, rand() * 0.8, -rand() * 0.5, 1.0])
        if under > 0.1 and rand() < under * k:
            for _ in range(5):
                self.bubbles.append([rand() * COLS, ROWS - 1 + rand() * 2, rand(), 1.0])
        self.move_particles(k, surface_y / CELL_H, chars, colors)

        background = None
        if lit:
            background = glow(0 if under > 0.5 else HEIGHT, 140 + s["rise_big"] * 350 + s["cover"] * 500,
                              (140, 50, 10), 0.3 + s["rise_big"] * 0.2 + s["cover"] * 0.2)
        return render(chars, colors, background)

    @staticmethod
    def classify(canvas):
        """Turns the drawing into one label per cell, by mixing the colors inside the cell."""
        # Each label as a color: navy (0, 0, 1), blue (0, 0.5, 1), light (0, 1, 1), white (1, 1, 1)
        red, green, blue = (cv2.resize(cv2.LUT(canvas, table), (COLS, ROWS), interpolation=cv2.INTER_AREA) / 255
                            for table in COLOR_TABLES)
        labels = np.zeros((ROWS, COLS), np.uint8)
        labels[blue > 0.39] = NAVY
        labels[(blue > 0.39) & (green > 0.35)] = BLUE
        labels[(blue > 0.39) & (green > 0.75)] = LIGHT
        labels[(blue > 0.39) & (red > 0.59)] = WHITE
        return labels

    def sink(self, labels, under, surface_y):
        """Under water: foam and crests fade into deep blue, and the surface is seen from below."""
        fade = under * 0.95
        sunk = labels.copy()
        sunk[(labels == WHITE) & (fade >= 0.41)] = BLUE if fade < 0.65 else NAVY
        sunk[(labels == LIGHT) & (fade >= 0.25)] = BLUE if fade < 0.65 else NAVY
        sunk[(labels == BLUE) & (fade >= 0.3)] = NAVY
        if fade > 0.4:
            sunk[labels == 0] = NAVY

        t = self.time
        layer = np.zeros_like(self.canvas)
        xs = np.arange(0, WIDTH + 10, 10, dtype=np.float32)
        surface = surface_y + np.sin(xs * 0.02 + t * 3) * 10 + np.sin(xs * 0.051 - t * 2) * 6
        cv2.polylines(layer, [to_canvas(np.stack([xs, surface], 1))], False, WHITE if under < 0.82 else BLUE, round(12 * SCALE))
        shine = surface_y + 22 + np.sin(xs * 0.024 - t * 2.6) * 8
        cv2.polylines(layer, [to_canvas(np.stack([xs, shine], 1))], False, BLUE, round(10 * SCALE))
        if under > 0.9:
            for j in range(5):
                current = surface_y + 90 + j * 80 + np.sin(xs * 0.01 + t * 0.8 + j * 1.3) * 30 + np.sin(xs * 0.027 - t * 0.5 + j) * 12
                cv2.polylines(layer, [to_canvas(np.stack([xs, current], 1))], False, BLUE, round(10 * SCALE))
        drawn = self.classify(layer)
        sunk[drawn > 0] = drawn[drawn > 0]
        return sunk

    def paint(self, labels, under, surface_row):
        """Chooses a character and a color for every cell."""
        t = self.time
        rows, cols = np.mgrid[0:ROWS, 0:COLS]
        depth = rows / ROWS
        dark = 1 - 0.55 * under
        chars = np.zeros((ROWS, COLS), int)
        colors = np.zeros((ROWS, COLS, 3), np.float32)

        def put(where, char, bgr):
            chars[where] = CHARS.index(char) if isinstance(char, str) else char[where]
            colors[where] = bgr if np.ndim(bgr) == 1 else bgr[where]

        wave_pattern = np.sin(cols * 0.35 + rows * 0.9 - t * 2.2) > 0.2
        navy_color = np.stack([(210 - 90 * depth) * dark, (90 - 50 * depth) * dark, (30 - 20 * depth) * dark], -1)
        put(labels == NAVY, np.where(wave_pattern, CHARS.index("-"), CHARS.index("~")), navy_color)
        put(labels == BLUE, "~" if under > 0.5 else "=", np.array([224, 120, 58]) * dark)
        put(labels == LIGHT, "~", np.array([255, 196, 143]))

        if under > 0.15:
            ray = ((labels == NAVY) | (labels == BLUE)) & (rows > surface_row + 1) & (
                np.sin((cols * 0.55 - rows * 0.35) * 0.45 + t * 0.6 + math.sin(t * 0.3) * 2) > 0.9)
            strength = (0.7 * np.clip(1 - (rows - surface_row) / ROWS, 0, 1) * under + 0.1)[..., None]
            put(ray, "/", np.array([255, 200, 150]) * strength)

        foam = np.array([CHARS.index(c) for c in "@%o*#"])[(cols * 3 + rows + int(t * 8)) % 5]
        put(labels == WHITE, foam, np.array([255, 248, 244]))
        return chars, colors

    def move_particles(self, k, surface_row, chars, colors):
        alive = []
        for drop in self.spray:
            drop[0] += drop[2] * k
            drop[1] += drop[3] * k
            drop[3] += 0.07 * k
            drop[4] -= 0.022 * k
            if drop[4] > 0 and 0 <= drop[1] < ROWS and 0 <= drop[0] < COLS:
                alive.append(drop)
                chars[int(drop[1]), int(drop[0])] = CHARS.index("*" if drop[4] > 0.6 else ".")
                colors[int(drop[1]), int(drop[0])] = (255, 255, 255)
        self.spray = alive

        alive = []
        for i, bubble in enumerate(self.bubbles):
            bubble[1] -= (0.4 + bubble[2] * 0.6) * k
            bubble[0] += math.sin(self.time * 4 + i) * 0.25 * k
            bubble[3] -= 0.012 * k
            if bubble[3] > 0 and bubble[1] >= max(0, surface_row) and 0 <= bubble[0] < COLS:
                alive.append(bubble)
                if bubble[1] >= ROWS:  # born just below the screen
                    continue
                char = "O" if bubble[2] > 0.7 else ("o" if bubble[2] > 0.3 else ".")
                chars[int(bubble[1]), int(bubble[0])] = CHARS.index(char)
                colors[int(bubble[1]), int(bubble[0])] = np.array([255, 230, 200]) * min(1, bubble[3] * 1.5)
        self.bubbles = alive
