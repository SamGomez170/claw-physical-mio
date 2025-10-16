import asyncio
import sys
import pygame
from .claw_lib import ClawCtl
import time
import textwrap 
from std_msgs.msg import UInt8
from .game_flow import run_game
#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
def main(args=None):
    claw_ctl = ClawCtl(args)
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    asyncio.run(run_game(screen, claw_ctl))
    
    
if __name__ == '__main__':
    main()

    