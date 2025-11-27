import asyncio
import sys
import pygame
from .claw_lib import ClawCtl
from .game_flow import run_game
from .strength_test import run_grip_strength_test

def main(args=None):
    claw_ctl = ClawCtl(args)
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    # Decide mode from command line
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running grip strength test mode...")
        asyncio.run(run_grip_strength_test(screen, claw_ctl))
    else:
        print("Running normal game mode...")
        asyncio.run(run_game(screen, claw_ctl))

if __name__ == '__main__':
    main()
