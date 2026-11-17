import asyncio
import os
import re
import subprocess
import sys

import pygame

from .claw_lib import ClawCtl
from .game_flow import run_game
from .strength_test import run_grip_strength_test

GAME_MONITOR = "other"   # "other" = the screen that is NOT the main one, "primary", or a number (0, 1)


def _monitors():
    """Monitors from `xrandr --listmonitors`: list of (index, is_primary, width, height, x, y)."""
    try:
        text = subprocess.run(["xrandr", "--listmonitors"], capture_output=True,
                              text=True, timeout=5).stdout
    except Exception as e:
        print(f"[SCREEN] xrandr failed: {e}")
        return []
    mons = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+):\s+\+(\*?)\S*\s+(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)", line)
        if m:
            mons.append((int(m.group(1)), m.group(2) == "*", int(m.group(3)), int(m.group(4)),
                         int(m.group(5)), int(m.group(6))))
    return mons


def make_screen(monitor=GAME_MONITOR):
    """Borderless window that fills one monitor (default: the second screen)."""
    mons = _monitors()
    chosen = None
    if mons:
        if monitor == "other":
            chosen = next((m for m in mons if not m[1]), mons[0])
        elif monitor == "primary":
            chosen = next((m for m in mons if m[1]), mons[0])
        else:
            chosen = next((m for m in mons if m[0] == int(monitor)), mons[0])
    if chosen is None:
        print("[SCREEN] no monitor info, opening a normal 800x600 window")
        pygame.display.init()
        return pygame.display.set_mode((800, 600))
    _, _, w, h, x, y = chosen
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"     # must be set before the window opens
    pygame.display.init()
    print(f"[SCREEN] monitor {chosen[0]}: {w}x{h} at +{x}+{y}")
    return pygame.display.set_mode((w, h), pygame.NOFRAME)


def main(args=None):
    claw_ctl = ClawCtl(args)
    pygame.init()
    screen = make_screen()

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running grip strength test mode...")
        asyncio.run(run_grip_strength_test(screen, claw_ctl, ir_detector=None))
    else:
        print("Running normal game mode...")
        asyncio.run(run_game(screen, claw_ctl))


if __name__ == '__main__':
    main()