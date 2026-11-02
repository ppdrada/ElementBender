import math

import numpy as np

from ascii_art import CELL_H, CELL_W, CHARS, COLS, ROWS, glow, render
from scene import MAX_CHARGE, Scene

ASPECT = 1.6  # cells are taller than wide, so round things are stretched sideways
LINKS = 26  # points of the wind whip
TRAIL = 7  # how many past positions a wisp of wind remembers
# Colors are BGR
WIND = np.array([255, 232, 220])
WHITE_WIND = np.array([255, 248, 240])
PALE_WIND = np.array([235, 205, 185])
EDGE_WIND = np.array([240, 215, 200])
GOLD = np.array([140, 225, 255])


def index(char):
    return CHARS.index(char)


def direction_char(vx, vy):
    """A line character that points the way the wind moves."""
    angle = math.atan2(vy, vx / ASPECT) % math.pi
    if angle < math.pi / 8 or angle > math.pi * 7 / 8:
        return "-"
    if angle < math.pi * 3 / 8:
        return "\\"
    return "|" if angle < math.pi * 5 / 8 else "/"


def ball_radius(charge):
    return 2 + charge * 3


class AirScene(Scene):
    """Air bending, like water but made of wind: wisps gather into a light wind whip that follows the hand.
    A closed hand winds it into an air ball that stays in the hand; a quick swing throws it into a shockwave;
    five open fingers blow a first-person gust toward the hand."""

    def __init__(self):
        super().__init__()
        self.state = "empty"  # empty -> forming (wisps gather) -> free (a whip in the hand)
        self.wait = 0.0
        self.form = 0.0
        self.whip = 0.0
        self.charge = 0.0
        self.gust = 0.0
        self.flash = 0.0
        self.chain = [[COLS / 2, ROWS / 2] for _ in range(LINKS)]
        self.wisps = []  # [x, y, speed_x, speed_y, curl, life, trail, pulled]
        self.balls = []  # [x, y, speed_x, speed_y, radius, power, spin]
        self.waves = []  # [x, y, radius, max_radius, power]: shockwaves
        self.streaks = []  # [start_x, start_y, progress, speed]: the first-person gust

    def add_wisp(self, x, y, vx=0.0, vy=0.0, curl=0.0, life=1.0, pulled=False):
        self.wisps.append([x, y, vx, vy, curl, life, [], pulled])

    def update(self, seconds, fingers, hand):
        """fingers: raised fingers (0 = closed hand, 5 = open hand); hand: (x, y) from 0 to 1, or None."""
        rand = self.rng.random
        holding, open_hand = self.read_input(seconds, fingers, hand)
        fist = holding and self.state == "free"
        five = open_hand and not holding
        if fist:
            self.charge = min(MAX_CHARGE, self.charge + seconds * 0.7)
        self.gust += ((1 if five else 0) - self.gust) * (0.1 if five else 0.06)

        if not self.hand and self.state != "empty":
            for x, y in self.chain:  # the hand left: the whip scatters
                if rand() < 0.5:
                    self.add_wisp(x, y, (rand() - 0.5) * 0.8, (rand() - 0.5) * 0.6, (rand() - 0.5) * 0.3)
            self.state, self.charge, self.whip = "empty", 0.0, 0.0
        if self.hand and self.state == "empty":
            self.wait -= seconds
            if self.wait <= 0:
                self.state, self.form = "forming", 0.0
                for link in self.chain:
                    link[:] = self.hand
        if self.state == "forming":
            self.form = min(1.0, self.form + seconds * 2.2)
            for _ in range(4):  # wisps come from all around and gather in the hand
                angle, distance = rand() * math.tau, 10 + rand() * 12
                self.add_wisp(self.hand[0] + math.cos(angle) * distance * ASPECT,
                              self.hand[1] + math.sin(angle) * distance, pulled=True)
            self.whip = self.form
            if self.form >= 1:
                self.state = "free"

        swing = self.flick()
        if swing and self.charge >= 0.1 and self.state == "free":
            self.balls.append([self.hand[0], self.hand[1], swing[0], swing[1], ball_radius(self.charge), self.charge, 0.0])
            self.state, self.wait, self.whip, self.charge = "empty", 0.5, 0.0, 0.0
        if self.state in ("free", "forming"):
            self.move_whip(fist, rand)

        self.move_wisps(fist)
        self.move_balls(rand)
        self.waves = [[x, y, r + 1.6, top, power] for x, y, r, top, power in self.waves if r + 1.6 <= top]
        self.move_gust(rand)
        self.flash *= 0.85

    def move_whip(self, fist, rand):
        hx, hy = self.hand
        if self.state == "free":
            target = max(0.0, 1 - self.charge * 2) if self.charge > 0 else 1.0  # the ball takes the whip's wind
            self.whip += (target - self.whip) * 0.2
        head = self.chain[0]
        head[0] += (hx - head[0]) * 0.6
        head[1] += (hy - head[1]) * 0.6
        segment = 0.9 * self.whip + 0.05
        t = self.time
        for i, (previous, link) in enumerate(zip(self.chain, self.chain[1:]), start=1):
            dx, dy = link[0] - previous[0], link[1] - previous[1]
            length = math.hypot(dx / ASPECT, dy) or 1
            step = min(length, segment) / length
            # Air floats: a soft sway and a little lift instead of gravity
            link[:] = (previous[0] + dx * step + math.sin(t * 2 + i * 0.5) * 0.05,
                       previous[1] + dy * step - 0.02 * self.whip + math.cos(t * 1.7 + i * 0.4) * 0.04)
        speed = math.hypot(self.chain[0][0] - self.chain[2][0], self.chain[0][1] - self.chain[2][1])
        if rand() < min(1.0, speed * 0.35) * self.whip:  # a fast whip sheds wisps
            i = int(rand() * LINKS)
            x, y = self.chain[i]
            px, py = self.chain[max(0, i - 1)]
            self.add_wisp(x, y, (x - px) * 0.3 + (rand() - 0.5) * 0.4, (y - py) * 0.3 + (rand() - 0.5) * 0.3,
                          (rand() - 0.5) * 0.3, 0.8)
        if fist and rand() < 0.8:
            for _ in range(2):  # wind pulled into the growing ball
                angle, distance = rand() * math.tau, ball_radius(self.charge) + 6 + rand() * 8
                self.add_wisp(hx + math.cos(angle) * distance * ASPECT, hy + math.sin(angle) * distance, pulled=True)

    def move_wisps(self, fist):
        alive = []
        for wisp in self.wisps:
            x, y, vx, vy, curl, life, trail, pulled = wisp
            if pulled and self.hand and (self.state == "forming" or fist):
                dx, dy = (self.hand[0] - x) / ASPECT, self.hand[1] - y
                distance = math.hypot(dx, dy) + 0.01
                if distance < (ball_radius(self.charge) if fist else 2):
                    continue  # it reached the hand
                vx = dx / distance * 1.1 * ASPECT + (-dy / distance) * 0.6 * ASPECT  # spiral in
                vy = dy / distance * 1.1 + (dx / distance) * 0.6
            else:
                pulled = False
                cos, sin = math.cos(curl), math.sin(curl)  # wisps curl as they slow down
                vx, vy = (vx * cos - vy * sin) * 0.96, (vx * sin + vy * cos) * 0.96
                life -= 0.02
            for wx, wy, r, _, power in self.waves:
                dx, dy = (x - wx) / ASPECT, y - wy
                distance = math.hypot(dx, dy) + 0.01
                if abs(distance - r) < 2.5:  # the shockwave pushes the wind away
                    vx += dx / distance * power * 0.6 * ASPECT
                    vy += dy / distance * power * 0.6
            trail.insert(0, (x, y, vx, vy))
            del trail[TRAIL:]
            x, y = x + vx, y + vy
            if life > 0 and -5 <= x <= COLS + 5 and -5 <= y <= ROWS + 5:
                alive.append([x, y, vx, vy, curl, life, trail, pulled])
        self.wisps = alive

    def move_balls(self, rand):
        alive = []
        for ball in self.balls:
            ball[0] += ball[2]
            ball[1] += ball[3]
            ball[6] += 0.4
            x, y, vx, vy, _, power, _ = ball
            if rand() < 0.8:
                self.add_wisp(x, y, -vx * 0.3 + (rand() - 0.5) * 0.6, -vy * 0.3 + (rand() - 0.5) * 0.6,
                              (rand() - 0.5) * 0.2, 0.7)
            if 1 <= x <= COLS - 2 and 1 <= y <= ROWS - 2:
                alive.append(ball)
                continue
            x, y = min(COLS - 2, max(1, x)), min(ROWS - 2, max(1, y))
            self.waves.append([x, y, 0.0, 20 + power * 14, 1 + power * 0.6])
            for _ in range(int(40 + power * 30)):
                angle, speed = rand() * math.tau, 0.6 + rand() * 1.6
                self.add_wisp(x, y, math.cos(angle) * speed * ASPECT, math.sin(angle) * speed, (rand() - 0.5) * 0.3)
            self.flash = min(1.0, 0.4 + power * 0.4)
        self.balls = alive

    def move_gust(self, rand):
        if self.gust > 0.05 and self.hand:
            for _ in range(round(self.gust * 10)):  # streaks start at the screen's edges, close to you
                side = int(rand() * 4)
                if side == 0:
                    start = (rand() * COLS, -1)
                elif side == 1:
                    start = (rand() * COLS, ROWS)
                elif side == 2:
                    start = (-1, rand() * ROWS)
                else:
                    start = (COLS, rand() * ROWS)
                self.streaks.append([start[0], start[1], 0.0, 0.03 + rand() * 0.03])
        alive = []
        if self.hand:
            for streak in self.streaks:
                streak[2] += streak[3] * (1 - streak[2] * 0.6)  # slower as it goes far away
                if streak[2] < 0.97:
                    alive.append(streak)
        self.streaks = alive

    # ---------- drawing ----------

    def draw_whip(self, chars, colors):
        t = self.time
        for i, (a, b) in enumerate(zip(self.chain, self.chain[1:])):
            ends = min(1.0, (i + 1) / 3, (LINKS - 2 - i) / 3)
            width = (0.7 + 1.0 * ends) * self.whip
            dx, dy = (b[0] - a[0]) / ASPECT, b[1] - a[1]
            length = math.hypot(dx, dy) or 1
            side_x, side_y = -dy / length * ASPECT, dx / length
            steps = math.ceil(length * 2) + 1
            for s in range(steps):
                k = s / steps
                x, y = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
                for o in np.arange(-width, width + 0.01, 0.6):
                    flow = math.sin(i * 0.9 - t * 11 + o * 1.7)
                    if flow < -0.25:
                        continue  # gaps: air is not solid
                    edge = abs(o) > width - 0.8
                    if edge:
                        char, color = ("~" if flow > 0.6 else "."), EDGE_WIND * 0.6
                    elif flow > 0.55:
                        char, color = "~", WHITE_WIND * 0.95
                    else:
                        char, color = ("-" if flow > 0.1 else "."), PALE_WIND * 0.75
                    self.put(chars, colors, x + side_x * o, y + side_y * o, char, color)
            if i % 5 == 2 and self.whip > 0.3:  # small whirls along the whip
                radius = 1.2 * self.whip
                for k in range(6):
                    angle = t * 7 + i + k * 1.05
                    self.put(chars, colors, a[0] + math.cos(angle) * radius * ASPECT, a[1] + math.sin(angle) * radius,
                             "o" if k == 0 else ".", GOLD * 0.9 if k == 0 else WIND * 0.7)

    def draw_ball(self, chars, colors, cx, cy, radius, spin):
        """Rings of wind turning around the ball at different tilts, like a spinning sphere."""
        for k in range(6):
            tilt, offset = k * 0.5, spin * (1 + k * 0.15) + k * 1.05
            for a in np.arange(0, math.tau, 0.12):
                fade = a / math.tau
                if fade < 0.35:
                    continue
                angle = a + offset
                x, y = math.cos(angle) * radius, math.sin(angle) * radius * math.cos(tilt) * 0.85
                front = math.sin(angle) * math.sin(tilt) > 0
                char = "@" if fade > 0.9 else ("~" if front else "-")
                color = np.array([255, 245, 235]) * fade if front else np.array([210, 175, 150]) * fade * 0.7
                self.put(chars, colors, cx + x * ASPECT, cy + y, char, color)
        self.put(chars, colors, cx, cy, "o", GOLD * 0.8)

    def draw(self):
        chars = np.zeros((ROWS, COLS), int)
        colors = np.zeros((ROWS, COLS, 3), np.float32)

        for x, y, r, top, _ in self.waves:
            life = 1 - r / top
            for angle in np.arange(0, math.tau, 0.5 / max(2, r)):
                self.put(chars, colors, x + math.cos(angle) * r * ASPECT, y + math.sin(angle) * r,
                         "|" if abs(math.cos(angle)) > 0.7 else "~", np.array([255, 240, 230]) * life)
        if self.hand:
            hx, hy = self.hand
            for sx, sy, progress, _ in self.streaks:
                eased = 1 - (1 - progress) ** 2.2
                x, y = sx + (hx - sx) * eased, sy + (hy - sy) * eased
                dx, dy = hx - sx, hy - sy
                distance = math.hypot(dx / ASPECT, dy) or 1
                ux, uy = dx / distance, dy / distance
                near = 1 - progress  # long and bright near you, short and faint far away
                length = near * 9 + 1
                char = direction_char(ux, uy)
                for j in np.arange(0, length, 0.5):
                    head = j < 0.5
                    self.put(chars, colors, x - ux * j, y - uy * j, ("=" if near > 0.6 else "-") if head else char,
                             np.array([255, 240, 230]) * (1 - j / length) * (0.3 + near * 0.7))
        for x, y, _, _, _, life, trail, _ in self.wisps:
            for j in range(len(trail) - 1, -1, -1):
                tx, ty, tvx, tvy = trail[j]
                fade = (1 - j / TRAIL) * min(1, life * 1.3)
                char = "~" if j == 0 else (direction_char(tvx, tvy) if j < 3 else ".")
                self.put(chars, colors, tx, ty, char, WIND * fade * 0.8)
        if self.state in ("free", "forming") and self.whip > 0.05:
            self.draw_whip(chars, colors)
        if self.charge > 0 and self.hand:
            self.draw_ball(chars, colors, self.hand[0], self.hand[1], ball_radius(self.charge), self.time * 6)
        for x, y, _, _, radius, _, spin in self.balls:
            self.draw_ball(chars, colors, x, y, radius, spin)

        background = np.zeros((ROWS, COLS, 3), np.float32) + np.array([255, 235, 220]) * self.flash * 0.15
        if self.hand:
            light = 80 * self.gust
            background += glow(self.hand[1] * CELL_H, 90 + self.charge * 160 + self.gust * 160,
                               (125 + light, 100 + light, 90 + light),
                               0.25 + min(1, self.charge) * 0.2 + self.gust * 0.2, self.hand[0] * CELL_W)
        image = render(chars, colors, background)
        image = self.shake(image, max(0, (self.charge - 2) * 1.2) + self.gust * 2)
        return self.draw_back(image)

    @staticmethod
    def put(chars, colors, x, y, char, color):
        col, row = int(round(x)), int(round(y))
        if 0 <= col < COLS and 0 <= row < ROWS:
            chars[row, col] = index(char)
            colors[row, col] = color
