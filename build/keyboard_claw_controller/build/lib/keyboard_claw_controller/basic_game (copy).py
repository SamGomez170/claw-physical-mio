from .claw_lib import ClawCtl
import time


def main(args=None):
    claw_ctl = ClawCtl(args)

    claw_ctl.disable_joystick()
    time.sleep(.3)
    claw_ctl.move_home()

    while True:
        claw_ctl.enable_joystick()
        
        claw_ctl.wait_fire_button()

        claw_ctl.disable_joystick()

        speed = 255
        grip = 160
        claw_ctl.grab_sequence(speed, grip)
        '''claw_ctl.open_claw()
        claw_ctl.claw_down(speed)
        time.sleep(1)
        claw_ctl.close_claw(grip)
        time.sleep(1)
        claw_ctl.claw_up(speed)
        time.sleep(1)'''

        claw_ctl.move_home()
        time.sleep(2)

        claw_ctl.open_claw()
        time.sleep(2)


    
    
if __name__ == '__main__':
    main()

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

from .claw_lib import ClawCtl
import time
def main(args=None):
    claw_ctl = ClawCtl(args)
    claw_ctl.disable_joystick()
    time.sleep(.3)
    claw_ctl.move_home()
    # Just down and up - nothing else
    print("Going DOWN...")
    claw_ctl.claw_down(255)
    time.sleep(3)  # Wait 3 seconds at bottom
if __name__ == '__main__':
    main()

