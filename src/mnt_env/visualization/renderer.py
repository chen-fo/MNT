"""Pygame spectator window. Does not call env.step or feed state to actors."""
import math

from mnt_env.env import DELTAS, STATUS_NAMES
from .geometry import shot_segments

BACKGROUND = (17, 24, 39)
TEXT = (227, 233, 242)
MUTED = (151, 166, 187)
ALLY = (75, 171, 255)
ENEMY = (255, 115, 96)


class Renderer:
    def __init__(self, size):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('Install the optional viewer: python -m pip install -e ".[visualization]"') from exc
        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.cell = max(12, min(48, 624 // size))
        self.grid = self.cell * size
        self.left, self.top = 28, 104
        self.panel = self.left + self.grid + 30
        self.width, self.height = self.panel + 440, max(810, self.top + self.grid + 72)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MNT | Local match viewer")
        # Bundled font avoids Windows system-font registry errors.
        self.font = pygame.font.Font(None, 23)
        self.small = pygame.font.Font(None, 19)
        self.title = pygame.font.Font(None, 32)

    def text(self, text, x, y, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(str(text), True, color), (x, y))

    def center(self, position):
        return (self.left + (float(position[0]) + .5) * self.cell,
                self.top + (float(position[1]) + .5) * self.cell)

    def draw(self, snapshot, *, observations, infos, rewards, returns, actions,
             events, episode, selected=0, paused=False, speed=5, finished=False):
        pg, canvas = self.pg, self.screen
        canvas.fill(BACKGROUND)
        size = snapshot["config"]["size"]
        names = snapshot["agent_order"]
        selected %= len(names)
        self.text("MNT / MATCH VIEWER", 28, 22, font=self.title)
        mode = "FINISHED - N for next match" if finished else "PAUSED" if paused else "PLAYING"
        self.text(f"Episode {episode}   Step {snapshot['steps']}/{snapshot['config']['max_steps']}"
                  f"   {mode}   {speed:g} steps/s", 28, 60, MUTED)
        for i in range(size + 1):
            x, y = self.left + i * self.cell, self.top + i * self.cell
            pg.draw.line(canvas, (43, 55, 73), (x, self.top), (x, self.top + self.grid))
            pg.draw.line(canvas, (43, 55, 73), (self.left, y), (self.left + self.grid, y))
        pg.draw.rect(canvas, MUTED, (self.left, self.top, self.grid, self.grid), 2)
        target = self.center(snapshot["target"])
        radius = max(4, int(self.cell * .32))
        pg.draw.rect(canvas, (54, 183, 128), (target[0]-radius, target[1]-radius, radius*2, radius*2), border_radius=4)
        self.text("G", target[0]-6, target[1]-8, font=self.small)
        for mine in snapshot["mines"]:
            x, y = self.center(mine)
            pg.draw.circle(canvas, (216, 163, 71), (x, y), max(3, radius-2), 2)
            pg.draw.line(canvas, (216, 163, 71), (x-4, y-4), (x+4, y+4), 2)
            pg.draw.line(canvas, (216, 163, 71), (x-4, y+4), (x+4, y-4), 2)
        for start, end, hit in shot_segments(snapshot):
            pg.draw.line(canvas, (255, 221, 92), self.center(start), self.center(end), 3)
            pg.draw.circle(canvas, ENEMY if hit else (255, 221, 92), self.center(end), 5, 2)
        groups = {}
        for i, position in enumerate(snapshot["positions"]):
            groups.setdefault(tuple(position), []).append(i)
        for members in groups.values():
            for slot, i in enumerate(members):
                x, y = self.center(snapshot["positions"][i])
                if len(members) > 1:
                    angle = slot * 2 * math.pi / len(members)
                    x += math.cos(angle) * self.cell * .23
                    y += math.sin(angle) * self.cell * .23
                live = snapshot["statuses"][i] == 0
                color = ALLY if names[i].startswith("ally") else ENEMY
                r = max(4, int(self.cell * (.20 if len(members) > 1 else .30)))
                pg.draw.circle(canvas, color if live else (91, 101, 116), (x, y), r, 0 if live else 2)
                dx, dy = DELTAS[snapshot["headings"][i]]
                pg.draw.line(canvas, TEXT, (x, y), (x+dx*(r+5), y+dy*(r+5)), 2)
                if i == selected:
                    pg.draw.circle(canvas, (244, 234, 158), (x, y), r+4, 1)
                label = ("A" if names[i].startswith("ally") else "E") + names[i].split("_")[-1]
                self.text(label, x-8, y-7, BACKGROUND if live else TEXT, self.small)
                if not live:
                    pg.draw.line(canvas, MUTED, (x-r, y-r), (x+r, y+r), 2)
        foot = self.top + self.grid + 12
        self.text("SPECTATOR: full map. Actors still use local sonar only.", self.left, foot, MUTED, self.small)
        self.text("Blue: ally   Red: enemy   Gold X: mine   Green G: goal", self.left, foot+22, MUTED, self.small)
        x = self.panel
        self.text("INDIVIDUAL ENVIRONMENT RETURNS", x, 104)
        self.text("Unit / status                     last reward       total", x, 132, MUTED, self.small)
        # Paginate the roster around the selected unit for larger custom arenas.
        first = (selected // 6) * 6
        for row, i in enumerate(range(first, min(first+6, len(names)))):
            a, y = names[i], 160 + row*37
            color = ALLY if a.startswith("ally") else ENEMY
            status = STATUS_NAMES[snapshot["statuses"][i]]
            self.text(f"{'>' if i == selected else ' '} {a}: {status}", x, y, color)
            self.text(f"{rewards.get(a, 0):+7.2f}", x+251, y)
            self.text(f"{returns.get(a, 0):+7.2f}", x+337, y)
        a = names[selected]
        obs = observations.get(a)
        self.text(f"SELECTED: {a} / ACTUAL OBSERVATION", x, 400)
        if obs is not None and snapshot["statuses"][selected] == 0:
            self.text(f"Heading {int(obs[0])}    Goal delta ({int(obs[1])}, {int(obs[2])})", x, 432)
            self.text("Sonar: relative -90 / -45 / 0 / +45 / +90 degrees", x, 461, MUTED, self.small)
            distances = [f"{1/float(v):.0f}" if v > 0 else "--" for v in obs[3:8]]
            self.text("Nearest distance:   " + "   /   ".join(distances), x, 486)
            self.text("--: no detection; object type is UNKNOWN", x, 513, MUTED, self.small)
        else:
            self.text("Inactive: no further decisions", x, 435, MUTED)
        action = actions.get(a)
        labels = ("move -90", "move -45", "move forward", "move +45", "move +90",
                  "fire -45", "fire forward", "fire +45")
        self.text("Last action: " + (labels[action] if action is not None else "--"), x, 545)
        self.text("Events: " + ", ".join(infos.get(a, {}).get("events", [])), x, 573, MUTED, self.small)
        self.text("RECENT EVENTS", x, 610)
        for i, event in enumerate(events):
            self.text(event, x, 637+i*21, MUTED, self.small)
        if finished:
            self.text(f"End: {snapshot['end_reason']}", x, 755, (255, 221, 92))
        self.text("Space pause | Right step | +/- speed | Tab select | N next | Esc exit",
                  28, self.height-26, TEXT, self.small)
        pg.display.flip()

    def close(self):
        self.pg.display.quit()
        self.pg.font.quit()
