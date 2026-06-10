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
        grip = 255
        #claw_ctl.grab_sequence(speed, grip)
        claw_ctl.open_claw()
        claw_ctl.claw_down(speed)
        time.sleep(1)
        claw_ctl.close_claw(grip)
        time.sleep(1)
        claw_ctl.claw_up(speed)
        time.sleep(1)

        claw_ctl.move_home()
        time.sleep(2)

        claw_ctl.open_claw()
        time.sleep(2)


    
    
if __name__ == '__main__':
    main()