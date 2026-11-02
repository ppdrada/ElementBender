import math

import numpy as np

from ascii_art import CELL_H, CELL_W, CHARS, COLS, ROWS, glow, render
from scene import MAX_CHARGE, Scene

ASPECT = 1.4  # cells are taller than wide, so round things are stretched sideways
SEA = ROWS - 5  # the sea's surface
LINKS = 24  # points of the water whip
X = np.arange(COLS, dtype=np.float64)
Y = np.arange(ROWS, dtype=np.float64)[:, None]
# Colors are BGR, as (water, ice)
EDGE = (np.array([255, 244, 232]), np.array([255, 255, 255]))
LIGHT = (np.array([255, 191, 127]), np.array([255, 235, 190]))
DEEP = (np.array([214, 111, 47]), np.array([255, 210, 150]))
ICE_EDGE, ICE_DARK, ICE_LIGHT = np.array([255, 235, 200]), np.array([230, 130, 60]), np.array([255, 225, 170])
WHITE = np.array([255, 255, 255])


def index(char):
    return CHARS.index(char)


def mix(a, b, k):
    return a + (b - a) * k


def orb_radius(charge):
    return 1.5 + charge * 3.2


class WaterScene(Scene):
    """Water bending: water rises from the sea and breaks free into a whip that follows the hand.
    A closed hand gathers it into an orb that stays in the hand; five open fingers freeze the water into ice;
    a quick swing throws the orb, or the frozen whip like a spear. Ice shatters where it hits."""

    def __init__(self):
        super().__init__()
        self.state = "empty"  # empty -> drawing (rising from the sea) -> free (a whip in the hand)
        self.wait = 0.0
        self.draw_k = 0.0  # how far the water has risen from the sea, 0 to 1
        self.base_x = COLS / 2
        self.whip = 0.0
        self.ice = 0.0
        self.frozen = None  # the whip's shape, kept while it is ice
        self.frozen_time = 0.0
        self.holding = False
        self.charge = 0.0
        self.drained = 0.0
        self.flash = 0.0
        self.chain = [[COLS / 2, ROWS / 2] for _ in range(LINKS)]
        self.drops = []  # [x, y, speed_x, speed_y, life]
        self.orbs = []  # [x, y, speed_x, speed_y, radius, power, spin, frozen]
        self.spears = []  # [points, speed_x, speed_y]: thrown frozen whips
        self.shards = []  # [x, y, speed_x, speed_y, life]: pieces of shattered ice
        self.cracks = []  # [x, y, angle, length, life]: short lines where ice shattered
        self.frost = 0.0  # pale flash of shattering ice
        self.ripples = []  # [x, y, radius, max_radius, life, delay, on_sea]
        self.feed = []  # [x, y]: drops flying up from the sea into the orb

    def surface(self, x):
        t = self.time
        y = SEA + self.drained * 0.8 + np.sin(x * 0.18 + t * 1.6) * 0.5 + np.sin(x * 0.07 - t * 0.9) * 0.4
        if self.state == "drawing":
            y = y - 2 * np.exp(-((x - self.base_x) / (4 * ASPECT)) ** 2) * (1 - self.draw_k * 0.5)
        for rx, _, r, _, life, delay, on_sea in self.ripples:
            if on_sea and delay <= 0:
                d = np.abs(x - rx) / ASPECT - r
                y = y - np.exp(-d * d) * life
        return y

    def column_point(self, k):
        """A point of the column rising from the sea to the hand, k from 0 (sea) to 1 (hand)."""
        bx, by = self.base_x, float(self.surface(self.base_x)) + 1
        hx, hy = self.hand
        cx, cy = bx + (hx - bx) * 0.25, by + (hy - by) * 0.55
        u = 1 - k
        return u * u * bx + 2 * u * k * cx + k * k * hx, u * u * by + 2 * u * k * cy + k * k * hy

    def update(self, seconds, fingers, hand):
        """fingers: raised fingers (0 = closed hand, 5 = open hand); hand: (x, y) from 0 to 1, or None."""
        rand = self.rng.random
        holding_input, open_hand = self.read_input(seconds, fingers, hand)
        freezing = open_hand and self.state == "free" and not holding_input
        self.ice += ((1 if freezing else 0) - self.ice) * (0.15 if freezing else 0.06)
        self.holding = holding_input and self.state == "free" and self.ice < 0.3
        if self.holding:
            self.charge = min(MAX_CHARGE, self.charge + seconds * 0.7)
        self.drained += ((self.charge if self.holding else 0) - self.drained) * (0.2 if self.holding else 0.03)

        swing = self.flick()
        if swing and self.state == "free":
            if self.charge >= 0.1:
                self.throw_orb(swing)
            elif self.ice > 0.5:
                self.throw_spear(swing)

        if not self.hand and self.state != "empty":
            if self.state == "free":
                self.drop_chain()
            self.state, self.ice, self.charge = "empty", 0.0, 0.0
        if self.hand and self.state == "empty":
            self.wait -= seconds
            if self.wait <= 0:
                self.state, self.draw_k, self.base_x = "drawing", 0.0, self.hand[0]
        if self.state == "drawing":
            self.rise_from_sea(seconds)
        if self.state == "free":
            self.move_whip(rand)

        sea = self.surface(X)
        self.move_feed(sea, rand)
        self.move_orbs(sea, rand)
        self.move_spears(sea)
        self.move_shards(sea, rand)
        self.move_ripples()
        self.move_drops(sea, rand)
        self.flash *= 0.85
        self.frost *= 0.85

    def rise_from_sea(self, seconds):
        self.draw_k = min(1.0, self.draw_k + seconds * 1.8)
        self.base_x += (self.hand[0] - self.base_x) * 0.1
        if self.draw_k >= 1:
            # The water breaks free: the bottom falls back, the top becomes the whip
            rand = self.rng.random
            for _ in range(30):
                x, y = self.column_point(rand() * 0.55)
                self.drops.append([x, y, (rand() - 0.5) * 0.4, rand() * 0.3, 1.0])
            for i, link in enumerate(self.chain):
                link[:] = self.column_point(1 - i / LINKS * 0.45)
            self.state, self.whip = "free", 1.0

    def move_whip(self, rand):
        hx, hy = self.hand
        if self.ice > 0.5:
            # Ice: the whip keeps its shape and moves with the hand like one solid piece
            if self.frozen is None:
                self.frozen = [(x - hx, y - hy) for x, y in self.chain]
                self.frozen_time = self.time
            for link, (ox, oy) in zip(self.chain, self.frozen):
                link[:] = hx + ox, hy + oy
            return
        self.frozen = None
        target = max(0.0, 1 - self.charge * 2) if self.charge > 0 else 1.0  # the orb takes the whip's water
        self.whip += (target - self.whip) * 0.2
        head = self.chain[0]
        head[0] += (hx - head[0]) * 0.6
        head[1] += (hy - head[1]) * 0.6
        segment = 0.9 * self.whip + 0.05
        for previous, link in zip(self.chain, self.chain[1:]):
            dx, dy = link[0] - previous[0], link[1] - previous[1]
            length = math.hypot(dx / ASPECT, dy) or 1
            step = min(length, segment) / length
            link[:] = previous[0] + dx * step, previous[1] + dy * step + 0.05 * self.whip
        speed = math.hypot(self.chain[0][0] - self.chain[2][0], self.chain[0][1] - self.chain[2][1])
        if self.ice < 0.3 and rand() < min(0.8, speed * 0.3) * self.whip:
            x, y = self.chain[int(rand() * LINKS * 0.7)]
            self.drops.append([x, y, (rand() - 0.5) * 0.6, rand() * 0.3, 1.0])
        if self.holding and rand() < 0.5 + min(0.5, self.charge * 0.15):
            x = hx + (rand() - 0.5) * orb_radius(self.charge) * 3
            self.feed.append([x, float(self.surface(x))])

    def drop_chain(self):
        rand = self.rng.random
        for x, y in self.chain:
            if rand() < 0.6:
                self.drops.append([x, y, (rand() - 0.5) * 0.5, rand() * 0.2, 1.0])
        self.whip = 0.0

    def throw_orb(self, velocity):
        frozen = self.ice > 0.5
        self.orbs.append([self.hand[0], self.hand[1], velocity[0], velocity[1], orb_radius(self.charge), self.charge,
                          0.0, frozen])
        self.empty_hand()

    def throw_spear(self, velocity):
        self.spears.append([[list(link) for link in self.chain], velocity[0], velocity[1]])
        self.empty_hand()

    def empty_hand(self):
        # After a throw the hand is empty; new water rises from the sea in half a second
        self.state, self.wait, self.whip, self.charge, self.ice, self.frozen = "empty", 0.5, 0.0, 0.0, 0.0, None

    def shatter(self, x, y, power):
        rand = self.rng.random
        for _ in range(int(30 + power * 50)):
            angle, speed = rand() * math.tau, 0.5 + rand() * 2.0 * (0.6 + power * 0.4)
            self.shards.append([x, y, math.cos(angle) * speed * ASPECT, math.sin(angle) * speed - 0.4, 1.0])
        for _ in range(8):
            self.cracks.append([x, y, rand() * math.tau, 2 + rand() * (3 + power * 4), 1.0])
        self.frost = min(1.0, 0.5 + power * 0.4)

    def move_spears(self, sea):
        alive = []
        for points, vx, vy in self.spears:
            vy += 0.02
            for point in points:
                point[0] += vx
                point[1] += vy
            hit = next(((x, y) for x, y in points if x < 1 or x > COLS - 2 or y < 1
                        or y >= sea[min(COLS - 1, max(0, int(round(x))))]), None)
            if hit:
                self.shatter(min(COLS - 2, max(1, hit[0])), min(ROWS - 2, max(1, hit[1])), 1.0)
            else:
                alive.append([points, vx, vy])
        self.spears = alive

    def move_shards(self, sea, rand):
        alive = []
        for shard in self.shards:
            shard[3] += 0.05
            shard[0] += shard[2]
            shard[1] += shard[3]
            shard[4] -= 0.015
            column = min(COLS - 1, max(0, int(round(shard[0]))))
            if shard[1] >= sea[column] and shard[3] > 0:
                if rand() < 0.1:
                    self.ripples.append([shard[0], shard[1], 0.0, 2.0, 1.0, 0, True])
            elif shard[4] > 0:
                alive.append(shard)
        self.shards = alive
        self.cracks = [crack[:4] + [crack[4] - 0.12] for crack in self.cracks if crack[4] > 0.12]

    def splash(self, x, y, power, into_sea):
        rand = self.rng.random
        for _ in range(int(40 + power * 70)):
            angle = -math.pi * (0.1 + rand() * 0.8) if into_sea else rand() * math.tau
            speed = 0.4 + rand() * 1.8 * (0.6 + power * 0.5)
            self.drops.append([x, y - (1 if into_sea else 0), math.cos(angle) * speed * ASPECT,
                               math.sin(angle) * speed - 0.5, 1.0])
        for k in range(3):
            self.ripples.append([x, y, 0.0, 6 + power * 14 + k * 4, 1.0, k * 4, into_sea])
        self.flash = min(1.0, 0.4 + power * 0.5)

    def move_feed(self, sea, rand):
        alive = []
        if self.hand and self.state == "free" and self.holding:  # the drops only fly while the hand gathers
            hx, hy = self.hand
            for drop in self.feed:
                dx, dy = hx - drop[0], hy - drop[1]
                distance = math.hypot(dx / ASPECT, dy) or 1
                if distance >= orb_radius(self.charge):
                    drop[0] += dx / distance * 0.9 * ASPECT * 0.6
                    drop[1] += dy / distance * 0.9
                    alive.append(drop)
        self.feed = alive

    def move_orbs(self, sea, rand):
        alive = []
        for orb in self.orbs:
            orb[3] += 0.02
            orb[0] += orb[2]
            orb[1] += orb[3]
            orb[6] += 0.3
            x, y, vx, vy, radius, power, _, frozen = orb
            if not frozen and rand() < 0.7:
                self.drops.append([x - vx * radius * 0.5, y - vy * radius * 0.5, -vx * 0.1, 0.0, 0.8])
            column = min(COLS - 1, max(0, int(round(x))))
            into_sea = y + radius * 0.5 >= sea[column]
            if into_sea or x < 1 or x > COLS - 2 or y < 1:
                x, y = min(COLS - 2, max(1, x)), float(sea[column]) if into_sea else max(1, y)
                if frozen:
                    self.shatter(x, min(ROWS - 2, y), power)
                else:
                    self.splash(x, y, power, into_sea)
            else:
                alive.append(orb)
        self.orbs = alive

    def move_ripples(self):
        alive = []
        for ripple in self.ripples:
            if ripple[5] > 0:
                ripple[5] -= 1
            else:
                ripple[2] += 0.9
                ripple[4] = 1 - ripple[2] / ripple[3]
            if ripple[4] > 0:
                alive.append(ripple)
        self.ripples = alive

    def move_drops(self, sea, rand):
        alive = []
        for drop in self.drops:
            drop[3] += 0.06
            drop[0] += drop[2]
            drop[1] += drop[3]
            drop[4] -= 0.008
            column = min(COLS - 1, max(0, int(round(drop[0]))))
            if drop[1] >= sea[column] and drop[3] > 0:
                if rand() < 0.15:
                    self.ripples.append([drop[0], drop[1], 0.0, 3.0, 1.0, 0, True])
            elif drop[4] > 0:
                alive.append(drop)
        self.drops = alive

    # ---------- drawing ----------

    def water_cell(self, chars, colors, x, y, edge, flow, frozen):
        col, row = int(round(x)), int(round(y))
        if not (0 <= col < COLS and 0 <= row < ROWS):
            return
        if frozen > 0.5:
            crystal = math.sin(x * 1.7 + y * 2.3) > 0.4
            char = "*" if edge else ("#" if crystal else ("+" if flow > 0 else "/"))
            color = mix(ICE_EDGE, WHITE, frozen) if edge else mix(ICE_DARK, ICE_LIGHT, frozen * (1 if crystal else 0.8))
        else:
            char = ("@" if flow > 0.3 else "o") if edge else ("~" if flow > 0.2 else "=")
            palette = EDGE if edge else (LIGHT if flow > 0.2 else DEEP)
            color = mix(palette[0], palette[1], frozen)
        chars[row, col] = index(char)
        colors[row, col] = color

    def draw_ribbon(self, chars, colors, points, thickness, clock, frozen):
        """The whip (or a thrown frozen whip): the same width all along, a bit thinner at the ends."""
        for i, (a, b) in enumerate(zip(points, points[1:])):
            ends = min(1.0, (i + 1) / 3, (len(points) - 2 - i) / 3)
            width = (0.7 + 1.0 * ends) * thickness
            dx, dy = (b[0] - a[0]) / ASPECT, b[1] - a[1]
            length = math.hypot(dx, dy) or 1
            side_x, side_y = -dy / length * ASPECT, dx / length  # across the whip, so it is thick in any direction
            steps = math.ceil(length * 2) + 1
            for s in range(steps):
                k = s / steps
                x, y = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
                for o in np.arange(-width, width + 0.01, 0.6):
                    self.water_cell(chars, colors, x + side_x * o, y + side_y * o, abs(o) > width - 0.8,
                                    math.sin(i * 0.9 - clock * 9 + o), frozen)

    def draw_orb(self, chars, colors, cx, cy, radius, spin, shine, frozen=False):
        top, bottom = max(0, math.floor(cy - radius)), min(ROWS - 1, math.floor(cy + radius))
        left, right = max(0, math.floor(cx - radius * ASPECT)), min(COLS - 1, math.floor(cx + radius * ASPECT))
        if top > bottom or left > right:
            return
        ys = np.arange(top, bottom + 1)[:, None]
        xs = np.arange(left, right + 1)[None, :]
        dx, dy = (xs - cx) / ASPECT, ys - cy
        r = np.hypot(dx, dy)
        inside = r <= radius
        band = np.sin(np.arctan2(dy, dx) * 3 + r * 0.8 - spin)
        rim = r > radius - 1.2
        light = shine & (dx < -radius * 0.3) & (dy < -radius * 0.3) & (r > radius * 0.5) & (r < radius * 0.75)
        if frozen:
            crystal = np.sin(xs * 1.7 + ys * 2.3) > 0.4
            char = np.where(rim, index("*"), np.where(crystal, index("#"), np.where(band > 0, index("+"), index("/"))))
            color = np.where(rim[..., None], WHITE, np.where(crystal[..., None], ICE_LIGHT, (ICE_DARK + ICE_LIGHT) / 2))
        else:
            char = np.where(rim, np.where(band > 0.3, index("@"), index("o")), np.where(band > 0.2, index("~"), index("=")))
            color = np.where(rim[..., None], EDGE[0], np.where((band > 0.2)[..., None], LIGHT[0], DEEP[0]))
        char = np.where(light, index("*"), char)
        color = np.where(light[..., None], WHITE, color)
        area_chars, area_colors = chars[top:bottom + 1, left:right + 1], colors[top:bottom + 1, left:right + 1]
        area_chars[inside] = char[inside]
        area_colors[inside] = color[inside]

    def draw(self):
        t = self.time
        chars = np.zeros((ROWS, COLS), int)
        colors = np.zeros((ROWS, COLS, 3), np.float32)

        # The sea
        sea = self.surface(X)
        depth = Y - sea
        inside = Y >= np.ceil(sea)
        top_char = np.where(np.sin(X * 0.6 + t * 3) > 0.6, index("@"), index("~"))
        body_char = np.where(np.sin(X * 0.35 + Y * 0.9 - t * 2.2) > 0.2, index("="), index("-"))
        chars[inside] = np.where(depth < 1, top_char, body_char)[inside]
        deep = np.stack([220 - depth * 10, 110 - depth * 8, 40 - depth * 3], -1)
        colors[inside] = np.where((depth < 1)[..., None], [255, 230, 207], np.clip(deep, 0, 255))[inside]

        if self.state == "drawing":
            steps = 40
            for i in range(int(steps * self.draw_k) + 1):
                k = i / steps
                x, y = self.column_point(k)
                width = 3.2 * (1 - k) + 1.1 * k
                for o in np.arange(-width, width + 0.01, 0.6):
                    self.water_cell(chars, colors, x + o * ASPECT * 0.7, y, abs(o) > width - 0.7,
                                    math.sin(k * 14 - t * 10 + o), 0)

        if self.state == "free":
            clock = self.frozen_time if self.ice > 0.5 else t  # frozen water stops flowing
            if self.whip > 0.05:
                self.draw_ribbon(chars, colors, self.chain, self.whip, clock, self.ice)
            if self.ice > 0.5 and self.rng.random() < 0.6:
                x, y = self.chain[int(self.rng.random() * LINKS)]
                self.put(chars, colors, x + (self.rng.random() - 0.5) * 3, y + (self.rng.random() - 0.5) * 2, "*", WHITE)
            if self.charge > 0:
                self.draw_orb(chars, colors, self.hand[0], self.hand[1], orb_radius(self.charge), t * 5, True,
                              self.ice > 0.5)

        for x, y in self.feed:
            self.put(chars, colors, x, y, "o", (255, 208, 159))
        for x, y, _, _, radius, _, spin, frozen in self.orbs:
            self.draw_orb(chars, colors, x, y, radius, spin, False, frozen)
        for points, _, _ in self.spears:
            self.draw_ribbon(chars, colors, points, 1.0, 0, 1.0)
        for x, y, angle, length, life in self.cracks:
            for d in np.arange(0, length * (1.3 - life * 0.3), 0.5):
                self.put(chars, colors, x + math.cos(angle) * d * ASPECT, y + math.sin(angle) * d,
                         "-" if abs(math.cos(angle)) > 0.8 else ("|" if abs(math.sin(angle)) > 0.8 else
                         ("\\" if math.cos(angle) * math.sin(angle) > 0 else "/")), WHITE * life)
        for x, y, vx, vy, life in self.shards:
            spin = "*/+\\"[int(t * 12 + x * 3) % 4]  # the pieces turn while they fly
            self.put(chars, colors, x, y, spin, mix(ICE_LIGHT, WHITE, life))
        for x, y, r, _, life, delay, on_sea in self.ripples:
            if not on_sea and delay <= 0:
                for angle in np.arange(0, math.tau, 0.6 / max(1, r)):
                    self.put(chars, colors, x + math.cos(angle) * r * ASPECT, y + math.sin(angle) * r, "~",
                             np.array([255, 225, 190]) * life)
        for x, y, _, vy, life in self.drops:
            self.put(chars, colors, x, y, "'" if vy > 0.6 else ("o" if life > 0.5 else "."),
                     np.array([255, 215, 170]) * min(1, life + 0.2))

        background = np.zeros((ROWS, COLS, 3), np.float32) + np.array([255, 140, 60]) * self.flash * 0.3
        background += np.array([255, 245, 225]) * self.frost * 0.12
        if self.hand:
            if self.ice > 0.3:
                color, strength = (255, 190, 120), 0.2 + self.ice * 0.25
            else:
                color, strength = (140, 50, 10), 0.3 + min(1, self.charge) * 0.3
            background += glow(self.hand[1] * CELL_H, 90 + self.charge * 180 + self.ice * 120, color, strength,
                               self.hand[0] * CELL_W)
        image = render(chars, colors, background)
        image = self.shake(image, (self.charge - 2) * 1.5)
        return self.draw_back(image)

    @staticmethod
    def put(chars, colors, x, y, char, color):
        col, row = int(round(x)), int(round(y))
        if 0 <= col < COLS and 0 <= row < ROWS:
            chars[row, col] = index(char)
            colors[row, col] = color
