import math

import numpy as np

from ascii_art import CHARS, COLS, HEIGHT, ROWS, clamp, ease, glow, lerp, render

CENTER = COLS / 2
# (offset from the center, half width, height, seed); the first one is the big mountain
MOUNTAINS = [(0, 30, ROWS * 0.82, 0), (-24, 20, ROWS * 0.5, 1), (25, 22, ROWS * 0.56, 2),
             (-42, 16, ROWS * 0.36, 3), (42, 17, ROWS * 0.4, 4)]
# Floating rocks: (offset from the center, height above the ground, radius)
ROCKS = [(-22, 13, 2.2), (21, 11, 1.8), (-35, 17, 1.6), (35, 16, 2.0),
         (-9, 30, 1.4), (10, 32, 1.3), (-45, 12, 1.2), (45, 10, 1.4)]
STRATA = np.array([[40, 66, 96], [52, 86, 120], [64, 74, 84], [62, 104, 140], [36, 52, 70]], np.float32)  # BGR

X = np.arange(COLS, dtype=np.float64)
Y = np.arange(ROWS, dtype=np.float64)[:, None]


def hash_(n):
    s = np.sin(np.asarray(n, np.float64) * 127.1) * 43758.5453
    return s - np.floor(s)


def noise(x):
    i = np.floor(x)
    f = x - i
    return lerp(hash_(i), hash_(i + 1), f * f * (3 - 2 * f))


def back_out(v):
    # Goes a little past 1 and comes back, like something pushed up that settles
    v -= 1
    return v * v * (2.2 * v + 1.2) + 1


def index(char):
    return CHARS.index(char)


class Earth:
    """ASCII earth: the ground shakes and cracks, mountains rise, rocks float and earth layers bury the screen."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.time = 0.0
        self.seconds = 0.0
        self.progress = 0.0
        self.last_rise = 0.0
        self.dust = []  # [x, y, speed_x, speed_y, life]
        self.falling = []  # [x, y, speed_x, speed_y, life]
        self.shake = 0.0

    def step(self, progress, lit, seconds):
        """Moves the earth one frame. progress goes from 0 (calm ground) to 1 (buried)."""
        self.time += seconds
        self.seconds = seconds
        self.progress = progress

    def draw(self, lit):
        """Returns the earth as a BGR image."""
        p, t, rand, k = self.progress, self.time, self.rng.random, self.seconds * 24
        tremor = ease(clamp(p / 0.25))
        rise = clamp((p - 0.1) / 0.45)
        spread = ease(clamp((p - 0.48) / 0.32))
        cover = ease(clamp((p - 0.78) / 0.22))
        rise_eased = back_out(rise) if rise > 0 else 0

        chars = np.zeros((ROWS, COLS), int)
        colors = np.zeros((ROWS, COLS, 3), np.float32)
        ground = 2 + noise(X * 0.3) * 1.5 + noise(X * 0.07 + 3) * 1.2
        mountains = np.zeros(COLS)
        top = ground

        if lit:
            mountains, owner, sunk = self.mountains(rise_eased, spread)
            top = np.maximum(ground, mountains)
            self.draw_ground(chars, colors, ground)
            self.draw_mountains(chars, colors, ground, mountains, owner, sunk, top)
            self.draw_cracks(chars, colors, ground, mountains, tremor * 12 + spread * 40)
            if spread > 0:
                self.draw_rocks(chars, colors, spread, k)
            if cover > 0:
                self.draw_cover(chars, colors, cover)

            speed = max(0.0, rise_eased - self.last_rise) / max(self.seconds, 0.001)
            if rise_eased > 0 and rand() < min(1, 0.2 + speed * 0.8) * k:
                for side in (-1, 1):
                    for _ in range(2):
                        self.dust.append([CENTER + side * (MOUNTAINS[0][1] * 0.6 + rand() * 8), ROWS - 3,
                                          side * (0.2 + rand() * 0.5), -(0.05 + rand() * 0.3), 1.0])
            if rand() < (0.3 + tremor * 0.6) * k * (1 - cover):
                self.dust.append([CENTER + (rand() - 0.5) * (10 + spread * 80), ROWS - 3,
                                  (rand() - 0.5) * 0.4, -(0.1 + rand() * 0.3), 1.0])
            if 0 < rise < 1 and rand() < 0.6 * k:
                x = CENTER + (rand() - 0.5) * 20
                self.falling.append([x, ROWS - top[int(x)] - 1, (x - CENTER) * 0.03, -0.2, 1.0])
        self.last_rise = rise_eased
        self.move_particles(chars, colors, top, k)

        background = None
        if lit:
            background = glow(HEIGHT, 120 + rise_eased * 300 + spread * 400, (25, 60, 90), 0.3 + rise_eased * 0.2 + spread * 0.2)
        image = render(chars, colors, background)

        shake = (tremor * 1.5 + spread * 3) * (1 - cover * 0.7) * 2 if lit else 0
        if shake >= 1:
            image = np.roll(image, (int(self.rng.uniform(-shake, shake)), int(self.rng.uniform(-shake, shake))), axis=(0, 1))
        return image

    def mountains(self, rise_eased, spread):
        """Height of the tallest mountain in every column, which mountain it is, and how deep it still is."""
        heights = np.zeros(COLS)
        owner = np.full(COLS, -1)
        sunk = np.zeros(COLS)
        for i, (offset, half, height, seed) in enumerate(MOUNTAINS):
            lift = rise_eased if i == 0 else (back_out(clamp(spread * 1.3 - (i - 1) * 0.12)) if spread > 0 else 0)
            if lift <= 0:
                continue
            distance = np.abs(X - (CENTER + offset)) / half
            ridge = noise(X * 0.35 + seed * 13) * 2.2 + noise(X * 0.9 + seed * 39) * 1.2 + noise(X * 2.1 + seed * 91) * 0.5
            shape = height * np.clip(1 - distance, 0, None) ** 1.15 + ridge * (1 - distance) * 2.5 - 1
            depth = (1 - lift) * (height + 4)
            h = np.where(distance < 1, shape - depth, -1)
            higher = h > heights
            heights[higher], owner[higher], sunk[higher] = h[higher], i, depth
        return heights, owner, sunk

    def draw_ground(self, chars, colors, ground):
        surface = ROWS - ground
        inside = Y >= np.floor(surface)
        depth = Y - surface
        n = noise(X * 0.7 + Y * 1.3)
        body = np.select([n > 0.72, n > 0.5, n > 0.3], [index("@"), index("#"), index("%")], index("="))
        strata = STRATA[(np.floor((Y + noise(X * 0.2) * 3) / 3) % len(STRATA)).astype(int)]
        shade = np.maximum(0.45, 1 - depth * 0.03)[..., None]
        left, right = np.roll(ground, 1), np.roll(ground, -1)
        edge_char = np.where(left < ground - 0.6, index("/"), np.where(right < ground - 0.6, index("\\"), index("_")))
        edge = inside & (depth < 1)
        fill = inside & ~edge
        chars[fill] = body[fill]
        colors[fill] = (strata * shade)[fill]
        chars[edge] = np.broadcast_to(edge_char, chars.shape)[edge]
        colors[edge] = (74, 195, 139)  # moss

    def draw_mountains(self, chars, colors, ground, mountains, owner, sunk, top):
        surface = ROWS - mountains
        inside = (owner >= 0) & (mountains > ground - 0.5) & (Y >= np.floor(surface)) & (Y < ROWS - ground + 0.5)
        depth = Y - surface
        slope = np.roll(top, -1) - np.roll(top, 1)
        light = np.where(slope > 0.3, 1, np.where(slope < -0.3, -1, 0))
        big_height = MOUNTAINS[0][2]
        high = (owner == 0) & (ROWS - Y > big_height * 0.75)

        # Edges: peak, slopes and flat parts, with snow on the big one
        left, right = np.roll(top, 1), np.roll(top, -1)
        edge_char = np.where((left < mountains - 0.5) & (right < mountains - 0.5), index("^"),
                             np.where(left < mountains - 0.4, index("/"),
                                      np.where(right < mountains - 0.4, index("\\"), index("_"))))
        edge = inside & (depth < 1)
        chars[edge] = np.broadcast_to(edge_char, chars.shape)[edge]
        edge_color = np.where(high[..., None], [242, 242, 242],
                              np.where((light >= 0)[:, None], [168, 201, 216], [112, 138, 154])[None])
        colors[edge] = edge_color[edge]

        # Rock body with diagonal veins; the texture moves up with the mountain
        fill = inside & ~edge
        texture_y = np.floor(Y + sunk)
        n = hash_(X * 3.7 + texture_y * 5.3 + owner * 11)
        direction = np.where(light >= 0, 1, -1)
        vein = np.abs(np.sin((X * 0.5 * direction + (Y + sunk) * 0.9) * 0.7)) < 0.12
        snow = (owner == 0) & (ROWS - Y > big_height * 0.78) & (depth < 3)
        body = np.select([snow, vein], [np.where(n > 0.5, index("*"), index(".")),
                                         np.where(direction > 0, index("/"), index("\\"))],
                         np.where(n > 0.66, index("#"), np.where(n > 0.33, index("%"), index("="))))
        base = np.where((light > 0)[:, None], [132, 160, 176], np.where((light < 0)[:, None], [76, 92, 104], [104, 126, 140]))
        rock = base[None] * np.maximum(0.55, 1 - depth * 0.012)[..., None] * np.where(vein, 0.7, 1)[..., None]
        rock = np.where(snow[..., None], [242, 238, 232], rock)
        chars[fill] = body[fill]
        colors[fill] = rock[fill]

    def draw_cracks(self, chars, colors, ground, mountains, length):
        for direction in (-1, 1):
            for branch in range(2):
                x, y = CENTER + direction * (3 + branch * 2), ROWS - 1 - branch * 2
                for s in range(int(length)):
                    wobble = float(hash_(s * 7 + branch * 5 + direction * 11))
                    x += direction * (0.8 + float(hash_(s * 3 + branch * 17 + direction)) * 0.6)
                    column = int(round(min(COLS - 1, max(0, x))))
                    y = min(ROWS - 1, max(ROWS - ground[column] + 0.5, y + (wobble - 0.55) * 1.1))
                    if mountains[column] > ground[column] or not 0 <= round(x) < COLS:
                        continue
                    char = "-" if abs(wobble - 0.55) < 0.15 else ("\\" if (wobble > 0.55) == (direction > 0) else "/")
                    chars[int(round(y)), int(round(x))] = index(char)
                    colors[int(round(y)), int(round(x))] = (102, 204, 255) if s > length - 3 else (10, 18, 26)

    def draw_rocks(self, chars, colors, spread, k):
        t = self.time
        for i, (offset, altitude, radius) in enumerate(ROCKS):
            amount = ease(clamp(spread * 1.3 - i * 0.08))
            if amount <= 0:
                continue
            rx = CENTER + offset + math.sin(t * 1.3 + i) * 0.6
            ry = ROWS - amount * altitude - 2 + math.sin(t * 2 + i * 1.7) * 0.8
            r = radius * amount
            for y in range(math.floor(ry - r), math.floor(ry + r) + 1):
                for x in range(math.floor(rx - r * 1.3), math.floor(rx + r * 1.3) + 1):
                    dx, dy = (x - rx) / 1.3, y - ry
                    if dx * dx + dy * dy <= r * r and 0 <= x < COLS and 0 <= y < ROWS:
                        edge = dx * dx + dy * dy > (r - 0.8) ** 2
                        chars[y, x] = index("O" if edge else ("@" if hash_(x * 3 + y) > 0.5 else "#"))
                        colors[y, x] = (114, 146, 168) if edge else (72, 98, 122)
            if self.rng.random() < 0.15 * k:
                self.falling.append([rx + (self.rng.random() - 0.5) * r * 2, ry + r, 0.0, 0.0, 1.0])

    def draw_cover(self, chars, colors, cover):
        t = self.time
        cover_top = lerp(ROWS + 1, -2, cover) + np.sin(X * 0.15 + t * 1.5) * 1.2 + noise(X * 0.25) * 2
        inside = Y >= np.floor(cover_top)
        layer = np.floor((Y - cover_top + noise(X * 0.1) * 4) / 2.5)
        n = noise(X * 0.8 + Y * 1.7 + layer)
        body = np.select([Y - cover_top < 1, n > 0.66, n > 0.4, n > 0.2],
                         [index("~"), index("#"), index("%"), index("=")], index("-"))
        color = STRATA[(layer % len(STRATA)).astype(int)]
        chars[inside] = body[inside]
        colors[inside] = color[inside]

    def move_particles(self, chars, colors, top, k):
        alive = []
        for d in self.dust:
            d[0] += d[2] * k
            d[1] += d[3] * k
            d[2] *= 0.97
            d[4] -= 0.02 * k
            x, y = int(round(d[0])), int(round(d[1]))
            if d[4] > 0:
                alive.append(d)
                if 0 <= x < COLS and 0 <= y < ROWS:
                    chars[y, x] = index("@" if d[4] > 0.6 else (":" if d[4] > 0.3 else "."))
                    colors[y, x] = np.array([130, 170, 190]) * d[4]
        self.dust = alive

        alive = []
        for f in self.falling:
            f[3] += 0.06 * k
            f[0] += f[2] * k
            f[1] += f[3] * k
            f[4] -= 0.01 * k
            x, y = int(round(f[0])), int(round(f[1]))
            if f[4] > 0 and 0 <= x < COLS and y < ROWS - top[x] + 0.5:
                alive.append(f)
                if 0 <= y < ROWS:
                    chars[y, x] = index("o")
                    colors[y, x] = (104, 136, 160)
        self.falling = alive
