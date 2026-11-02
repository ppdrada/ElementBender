import math

import numpy as np

from ascii_art import CELL_H, CELL_W, CHARS, COLS, ROWS, glow, render
from earth import hash_
from scene import Scene

ASPECT = 1.4  # cells are taller than wide, so round things are stretched sideways
GROUND = ROWS - 5  # the ground's surface
ROCK = 3  # the rock's radius, always the same
WALL_MAX = ROWS * 0.5  # the tallest a wall can grow
WALL_HALF = 7 * ASPECT * 0.5  # half of the wall's width, in cells
WALL_STAY = 3  # seconds a released wall stands before it sinks
PULL = 3  # rows the hand must rise above where the fist closed to pull the wall out of the ground
X = np.arange(COLS, dtype=np.float64)
Y = np.arange(ROWS, dtype=np.float64)[:, None]
# Colors are BGR
WALL_SHADES = np.array([[110, 150, 176], [78, 112, 140], [52, 76, 96]])  # lit side, middle, shadow side
DUST = np.array([140, 170, 190])


def index(char):
    return CHARS.index(char)


class EarthScene(Scene):
    """Earth bending: a rock is ripped out of the ground and floats with the hand; a quick swing throws it.
    A closed hand turns it into a wall that rises from the ground: move the hand sideways to drag it, raise it
    clearly to pull the wall out, then swing to throw it. Five open fingers turn the rock into lava."""

    def __init__(self):
        super().__init__()
        self.state = "empty"  # empty -> ripping (out of the ground) -> held (a rock in the hand) / wall
        self.wait = 0.0
        self.rip_k = 0.0
        self.rip_x = COLS / 2
        self.rock = None  # [x, y, spin, seed]
        self.lava = 0.0
        self.wall = None  # the wall the hand is holding
        self.merging = None  # [x, y]: the rock sinking into a new wall
        self.walls = []  # {x, bottom, h, stay, seed, anchor, falling, speed}
        self.holes = []  # [x, width, depth]
        self.flying = []  # [x, y, speed_x, speed_y, spin, seed, lava]
        self.slabs = []  # [x, bottom, height, speed_x, speed_y, seed]: thrown walls
        self.rubble = []  # [x, y, speed_x, speed_y, char, life, lava, bounces]
        self.dust = []  # [x, y, speed_x, speed_y, life]
        self.drips = []  # [x, y, speed_y]: lava dripping from the rock
        self.shake_amount = 0.0

    def ground(self, x):
        y = GROUND + np.sin(x * 0.3) * 0.4 + hash_(np.floor(x)) * 0.5
        for hx, width, depth in self.holes:
            d = (x - hx) / (width * ASPECT)
            y = y + np.maximum(0, 1 - d * d) * depth
        return y

    def update(self, seconds, fingers, hand):
        """fingers: raised fingers (0 = closed hand, 5 = open hand); hand: (x, y) from 0 to 1, or None."""
        rand = self.rng.random
        fist, open_hand = self.read_input(seconds, fingers, hand)
        swing = self.flick()
        heating = open_hand and self.state == "held" and not fist
        self.lava += ((1 if heating else 0) - self.lava) * (0.08 if heating else 0.04)

        if fist and self.wall is None and self.state in ("held", "empty"):
            self.raise_wall()
        if self.wall:
            self.hold_wall(fist, swing, seconds)
        self.move_walls(seconds)

        if not self.hand and self.state != "empty":
            if self.state == "held":
                self.crumble(self.rock[0], self.rock[1], ROCK, self.lava > 0.5, False, False)
            self.state, self.lava, self.rock = "empty", 0.0, None
        if self.hand and self.state == "empty" and not fist:
            self.wait -= seconds
            if self.wait <= 0:
                self.state, self.rip_k, self.rip_x = "ripping", 0.0, self.hand[0]
                self.holes.append([self.rip_x, 2.5, 0.0])
        if self.state == "ripping":
            self.rip_out(seconds, rand)
        if self.state == "held":
            self.hold_rock(swing, rand)

        self.move_flying(rand)
        self.move_slabs(rand)
        if self.merging:
            target_y = GROUND - self.wall["h"] / 2 if self.wall else GROUND
            target_x = self.wall["x"] if self.wall else self.merging[0]
            self.merging[0] += (target_x - self.merging[0]) * 0.3
            self.merging[1] += (target_y - self.merging[1]) * 0.3
            if math.hypot(target_x - self.merging[0], target_y - self.merging[1]) < 1:
                self.merging = None
        for hole in self.holes[:]:
            if not (self.state == "ripping" and hole is self.holes[-1]):
                hole[2] -= 0.01  # holes slowly fill back
                if hole[2] <= 0:
                    self.holes.remove(hole)
        self.move_particles()
        self.shake_amount *= 0.85

    # ---------- the rock ----------

    def rip_out(self, seconds, rand):
        self.rip_k = min(1.0, self.rip_k + seconds * 1.6)
        self.holes[-1][2] = min(2.5, self.rip_k * 3)
        if rand() < 0.5:
            self.dust.append([self.rip_x + (rand() - 0.5) * 6, GROUND, (rand() - 0.5) * 0.4, -(0.1 + rand() * 0.3), 1.0])
        if self.rip_k >= 0.35:
            k = (self.rip_k - 0.35) / 0.65
            k = k * k * (3 - 2 * k)
            seed = self.rock[3] if self.rock else rand() * 10
            self.rock = [self.rip_x + (self.hand[0] - self.rip_x) * k, GROUND + (self.hand[1] - GROUND) * k, 0.0, seed]
            if self.rip_k >= 1:
                self.state = "held"

    def hold_rock(self, swing, rand):
        rock = self.rock
        rock[0] += (self.hand[0] - rock[0]) * 0.25  # heavy: it follows the hand slowly
        rock[1] += (self.hand[1] - rock[1] + math.sin(self.time * 2) * 0.3) * 0.25
        rock[2] += 0.02
        if self.lava > 0.5 and rand() < 0.4:
            self.drips.append([rock[0] + (rand() - 0.5) * ROCK * ASPECT, rock[1] + ROCK * 0.8, 0.1])
        if swing:
            self.flying.append([rock[0], rock[1], swing[0], swing[1], rock[2], rock[3], self.lava > 0.5])
            self.state, self.wait, self.lava, self.rock = "empty", 0.6, 0.0, None

    # ---------- the wall ----------

    def raise_wall(self):
        self.wall = {"x": self.hand[0], "bottom": GROUND + 0.5, "h": 0.0, "stay": WALL_STAY,
                     "seed": self.rng.random() * 9, "anchor": self.hand[1], "falling": False, "speed": 0.0}
        self.walls.append(self.wall)
        if len(self.walls) > 3:
            self.walls[0]["stay"] = 0  # only three walls at a time: the oldest sinks
        self.merging = [self.rock[0], self.rock[1]] if self.rock else None
        self.state, self.rock, self.lava = "wall", None, 0.0

    def hold_wall(self, fist, swing, seconds):
        wall, rand = self.wall, self.rng.random
        if not fist:
            wall["falling"] = wall["bottom"] < GROUND  # a lifted wall drops when the hand opens
            self.wall, self.state, self.wait = None, "empty", 0.6
            return
        before = wall["h"]
        wall["h"] = min(WALL_MAX, wall["h"] + seconds * 24)
        wall["x"] += (self.hand[0] - wall["x"]) * 0.3
        pull = max(0.0, wall["anchor"] - self.hand[1] - PULL)  # only a clear upward pull lifts it
        wall["bottom"] += (GROUND + 0.5 - pull * 1.3 - wall["bottom"]) * 0.3
        grounded = wall["bottom"] >= GROUND
        if grounded and (wall["h"] > before or abs(self.hand[0] - wall["x"]) > 0.3):
            for side in (-1, 1):
                if rand() < 0.6:
                    self.dust.append([wall["x"] + side * (WALL_HALF + 1), GROUND, side * (0.2 + rand() * 0.4),
                                      -(0.05 + rand() * 0.25), 1.0])
        if not grounded and rand() < 0.3:
            self.rubble.append([wall["x"] + (rand() - 0.5) * WALL_HALF * 2, wall["bottom"], 0.0, 0.0, ".", 0.6, False, 2])
        if swing and not grounded:
            self.slabs.append([wall["x"], wall["bottom"], wall["h"], swing[0], swing[1], wall["seed"]])
            self.walls.remove(wall)
            self.wall, self.state, self.wait = None, "empty", 0.6

    def move_walls(self, seconds):
        rand = self.rng.random
        for wall in self.walls[:]:
            if wall is self.wall:
                continue
            if wall["falling"]:
                wall["speed"] += 0.08
                wall["bottom"] += wall["speed"]
                if wall["bottom"] >= GROUND + 0.5:
                    wall["bottom"], wall["falling"], self.shake_amount = GROUND + 0.5, False, 6
                    for _ in range(30):
                        self.dust.append([wall["x"] + (rand() - 0.5) * 16, GROUND, (rand() - 0.5) * 1.2,
                                          -(0.05 + rand() * 0.3), 1.0])
                continue
            wall["stay"] -= seconds
            if wall["stay"] <= 0:
                wall["h"] -= seconds * 14  # it sinks back into the ground
                if rand() < 0.5:
                    self.dust.append([wall["x"] + (rand() - 0.5) * 14, GROUND, (rand() - 0.5) * 0.5, -0.2, 0.8])
                if wall["h"] <= 0:
                    self.walls.remove(wall)

    def wall_at(self, x, y):
        for wall in self.walls:
            if abs(x - wall["x"]) <= WALL_HALF + 1 and wall["bottom"] - wall["h"] <= y <= wall["bottom"]:
                return wall
        return None

    # ---------- things that fly and break ----------

    def move_flying(self, rand):
        alive = []
        for rock in self.flying:
            rock[3] += 0.045  # heavy: it falls faster than water
            rock[0] += rock[2]
            rock[1] += rock[3]
            rock[4] += 0.15
            x, y, vx, vy, _, _, lava = rock
            if rand() < 0.5:
                self.dust.append([x - vx * 2, y - vy * 2, 0.0, -0.05, 0.6])
            if lava and rand() < 0.6:
                self.drips.append([x, y + ROCK * 0.5, 0.0])
            on_ground = y + ROCK * 0.6 >= self.ground(x)
            wall = self.wall_at(x + math.copysign(ROCK * ASPECT * 0.8, vx), y)
            if on_ground or wall or x < 1 or x > COLS - 2 or y < 1:
                self.crumble(min(COLS - 2, max(1, x)), self.ground(x) - 1 if on_ground else max(1, y), ROCK, lava,
                             on_ground, False)
                if wall:
                    wall["h"] = max(0.0, wall["h"] - 3)  # the rock breaks a piece of the wall
            else:
                alive.append(rock)
        self.flying = alive

    def move_slabs(self, rand):
        alive = []
        for slab in self.slabs:
            slab[4] += 0.05
            slab[0] += slab[3]
            slab[1] += slab[4]
            x, bottom, height, vx, _, _ = slab
            if rand() < 0.7:
                self.dust.append([x - vx * 2, bottom - height / 2, 0.0, -0.05, 0.7])
            on_ground = bottom >= self.ground(x)
            if on_ground or x - WALL_HALF < 0 or x + WALL_HALF > COLS - 1 or bottom - height < 0:
                self.crumble(min(COLS - 3, max(2, x)), min(GROUND - 1, bottom - height / 2), height / 2, False,
                             on_ground, True)
            else:
                alive.append(slab)
        self.slabs = alive

    def crumble(self, x, y, size, lava, on_ground, big):
        """Breaks into rubble that bounces on the ground, with dust; a big crash for a thrown wall."""
        rand = self.rng.random
        for _ in range(120 if big else 45):
            angle = -math.pi * (0.05 + rand() * 0.9) if on_ground else rand() * math.tau
            speed = 0.4 + rand() * (2.2 if big else 1.6)
            self.rubble.append([x + (rand() - 0.5) * size * (2.5 if big else 1), y + (rand() - 0.5) * size * 0.6,
                                math.cos(angle) * speed * ASPECT, math.sin(angle) * speed - 0.3,
                                "#%o@."[int(rand() * 5)], 1.0, lava, 0])
        for _ in range(70 if big else 30):
            self.dust.append([x + (rand() - 0.5) * size * 2.5, y + (rand() - 0.5) * size, (rand() - 0.5) * 0.8,
                              -(0.05 + rand() * 0.4), 1.0])
        if on_ground:
            self.holes.append([x, 6 if big else 3, 3 if big else 2])
        self.shake_amount = 9 if big else 5

    def move_particles(self):
        rand = self.rng.random
        alive = []
        for piece in self.rubble:
            piece[3] += 0.06
            piece[0] += piece[2]
            piece[1] += piece[3]
            piece[5] -= 0.006
            ground = self.ground(piece[0])
            if piece[1] >= ground - 0.5 and piece[3] > 0:
                if piece[7] < 2 and piece[3] > 0.3:  # bounce
                    piece[3], piece[2], piece[7], piece[1] = -piece[3] * 0.35, piece[2] * 0.6, piece[7] + 1, ground - 0.6
                else:
                    piece[2], piece[3], piece[1] = 0.0, 0.0, ground - 0.6
                    piece[5] -= 0.02
            if piece[5] > 0 and 0 <= piece[0] <= COLS:
                alive.append(piece)
        self.rubble = alive

        alive = []
        for drip in self.drips:
            drip[2] += 0.05
            drip[1] += drip[2]
            if drip[1] >= self.ground(drip[0]):
                if rand() < 0.3:  # a little lava stays on the ground for a moment
                    self.rubble.append([drip[0], drip[1] - 0.6, 0.0, 0.0, "~", 0.6, True, 3])
            else:
                alive.append(drip)
        self.drips = alive

        alive = []
        for d in self.dust:
            d[0] += d[2]
            d[1] += d[3]
            d[2] *= 0.96
            d[4] -= 0.015
            if d[4] > 0:
                alive.append(d)
        self.dust = alive

    # ---------- drawing ----------

    def draw_rock(self, chars, colors, cx, cy, radius, spin, heat, seed):
        top, bottom = max(0, math.floor(cy - radius - 1)), min(ROWS - 1, math.floor(cy + radius + 1))
        left, right = max(0, math.floor(cx - radius * ASPECT - 1)), min(COLS - 1, math.floor(cx + radius * ASPECT + 1))
        if top > bottom or left > right:
            return
        ys = np.arange(top, bottom + 1)[:, None].astype(np.float64)
        xs = np.arange(left, right + 1)[None, :].astype(np.float64)
        dx, dy = (xs - cx) / ASPECT, ys - cy
        r, angle = np.hypot(dx, dy), np.arctan2(dy, dx)
        edge = radius * (0.82 + 0.18 * np.sin(angle * 5 + seed) + 0.1 * np.sin(angle * 11 + seed * 3))  # bumpy outline
        inside = r <= edge
        lit = (-dx - dy) / (radius * 1.4) * 0.5 + 0.5  # light from the top left
        n = hash_(np.floor(xs - cx + spin * 2) * 7 + np.floor(ys - cy) * 13 + seed)
        rim = r > edge - 1
        if heat > 0.5:
            crack = np.abs(np.sin((dx * 1.3 + dy) * 1.7 + spin)) < 0.25
            shine = 0.5 + 0.5 * np.sin(self.time * 6 + xs * 0.7 + ys)
            char = np.where(crack, index("~"), np.where(rim, index("@"), np.where(n > 0.5, index("#"), index("%"))))
            cracks = np.stack([80 + shine * 80, 200 + shine * 55, np.full_like(shine, 255)], -1)
            stone = np.stack([np.full_like(lit, 20), 60 + 60 * lit * heat, np.full_like(lit, 180 + 70 * heat)], -1)
            color = np.where(crack[..., None], cracks, stone)
        else:
            light = lit * 0.6 + 0.4
            char = np.where(rim, np.where(n > 0.5, index("@"), index("O")),
                            np.where(n > 0.66, index("#"), np.where(n > 0.33, index("%"), index("="))))
            color = np.stack([85 * light, 120 * light + heat * 30, 150 * light + heat * 120], -1)
        area_chars, area_colors = chars[top:bottom + 1, left:right + 1], colors[top:bottom + 1, left:right + 1]
        area_chars[inside] = char[inside]
        area_colors[inside] = color[inside]

    def draw_wall(self, chars, colors, cx, bottom, height, seed):
        x0, x1 = round(cx - WALL_HALF), round(cx + WALL_HALF)
        top = bottom - height
        left, right = max(0, x0), min(COLS - 1, x1)
        first, last = max(0, math.ceil(top)), min(ROWS - 1, math.floor(bottom))
        if left > right or first > last:
            return
        xs = np.arange(left, right + 1)[None, :].astype(np.float64)
        ys = np.arange(first, last + 1)[:, None].astype(np.float64)
        jag = np.where((xs == x0) | (xs == x1), 1, (hash_(xs - x0 + seed * 7) > 0.55).astype(int))  # broken top
        inside = ys >= np.ceil(top + jag)
        width = x1 - x0
        side = np.where(xs - x0 < width * 0.3, 0, np.where(x1 - xs < width * 0.3, 2, 1))
        base = WALL_SHADES[side[0]][None].repeat(len(ys), 0)
        layer = np.floor(ys - bottom + 500)  # the layers move with the wall
        is_top = ys - top < 1 + jag
        loose = bool(bottom < GROUND - 0.5) & (bottom - ys < 1)  # pebbles under a lifted wall
        strata = layer % 3 == 0
        n = hash_((xs - x0) * 3.1 + layer * 7.7 + seed)
        char = np.select([is_top, loose, strata, n > 0.66, n > 0.33],
                         [np.where(jag == 1, index("^"), index("_")),
                          np.where(hash_(xs + seed) > 0.5, index("v"), index("'")),
                          index("="), index("#"), index("%")], index("+"))
        color = np.select([is_top[..., None], loose[..., None], strata[..., None]],
                          [np.array([140, 180, 200]), np.array([80, 112, 138]), base * 0.7], base)
        area_chars, area_colors = chars[first:last + 1, left:right + 1], colors[first:last + 1, left:right + 1]
        area_chars[inside] = np.broadcast_to(char, inside.shape)[inside]
        area_colors[inside] = np.broadcast_to(color, inside.shape + (3,))[inside]

    def draw(self):
        t = self.time
        chars = np.zeros((ROWS, COLS), int)
        colors = np.zeros((ROWS, COLS, 3), np.float32)

        # The ground: grass on top, layers of earth below
        surface = self.ground(X)
        depth = Y - surface
        inside = Y >= np.ceil(surface)
        n = hash_(X * 3 + Y * 7)
        body = np.where(n > 0.66, index("#"), np.where(n > 0.33, index("%"), index("=")))
        grass = np.where(hash_(X) > 0.4, index("_"), index(","))
        chars[inside] = np.where(depth < 1, grass, body)[inside]
        earth = np.clip(np.stack([52 - depth * 5, 86 - depth * 8, 120 - depth * 10], -1), 0, 255)
        colors[inside] = np.where((depth < 1)[..., None], [66, 179, 124], earth)[inside]

        if self.state == "ripping" and self.rip_k < 0.35:
            for i in range(-6, 7):  # the ground cracks under the hand
                if hash_(i * 3 + math.floor(t * 10)) > 0.5:
                    x = self.rip_x + i
                    self.put(chars, colors, x, float(self.ground(x)) + abs(i) % 2, "/" if i < 0 else "\\", (18, 29, 42))
        for wall in self.walls:
            self.draw_wall(chars, colors, wall["x"], wall["bottom"], wall["h"], wall["seed"])
        for x, bottom, height, _, _, seed in self.slabs:
            self.draw_wall(chars, colors, x, bottom, height, seed)
        if self.merging:
            self.draw_rock(chars, colors, self.merging[0], self.merging[1], ROCK * 0.8, 0, 0, 1)
        if self.rock and self.state in ("held", "ripping"):
            self.draw_rock(chars, colors, self.rock[0], self.rock[1], ROCK, self.rock[2], self.lava, self.rock[3])
        for x, y, _, _, spin, seed, lava in self.flying:
            self.draw_rock(chars, colors, x, y, ROCK, spin, 1.0 if lava else 0.0, seed)
        for x, y, _, _, char, life, lava, _ in self.rubble:
            color = (40, 100 + life * 120, 255) if lava else (80 * life + 20, 115 * life + 30, 150 * life + 40)
            self.put(chars, colors, x, y, char, color)
        for x, y, _ in self.drips:
            self.put(chars, colors, x, y, "'", (64, 176, 255))
        for x, y, _, _, life in self.dust:
            self.put(chars, colors, x, y, ":" if life > 0.6 else ".", DUST * life * 0.8)

        background = None
        if self.hand:
            if self.lava > 0.3:
                color, strength = (10, 70, 200), 0.25 + self.lava * 0.35
            else:
                color, strength = (35, 65, 90), 0.3
            background = glow(self.hand[1] * CELL_H, 100 + self.lava * 100, color, strength, self.hand[0] * CELL_W)
        image = render(chars, colors, background)
        rising = 1.2 if self.wall and self.wall["h"] < WALL_MAX else 0
        image = self.shake(image, self.shake_amount + rising)
        return self.draw_back(image)

    @staticmethod
    def put(chars, colors, x, y, char, color):
        col, row = int(round(x)), int(round(y))
        if 0 <= col < COLS and 0 <= row < ROWS:
            chars[row, col] = index(char)
            colors[row, col] = color
