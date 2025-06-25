import asyncio
import sys
import pygame
from .claw_lib import ClawCtl
import time
import textwrap 
from std_msgs.msg import UInt8

async def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

async def display_and_wait(screen, claw_ctl, render_items):
    """
    render_items: list of dicts
      { "text": str,         # paragraph text
        "color": (r,g,b)      # optional override color for the whole line
      }
    """
    pygame.event.clear()
    pygame.font.init()
    font = pygame.font.Font(None, 28)
    screen.fill((255, 255, 255))

    max_width = screen.get_width() - 80  # margins
    y = 40

    wrapped_pass = []
    for item in render_items:
        if not item["text"]:
            wrapped_pass.append([{"text": "", "color": None}])
            continue

        lines = await wrap_text(item["text"], font, max_width)
        wrapped_pass.append([{"text": line, "color": item.get("color")} for line in lines])

    # Draw all lines
    for block in wrapped_pass:
        for seg in block:
            text_surf = font.render(seg["text"], True, seg["color"] or (0,0,0))
            text_rect = text_surf.get_rect(topleft=(40, y))
            screen.blit(text_surf, text_rect)
            y += 35
        y += 20  # spacing between paragraphs

    pygame.display.flip()

    # Wait on button
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, claw_ctl.wait_fire_button)

    # blank out for the next screen
    screen.fill((255,255,255))
    pygame.display.flip()

# intro
INTRO = [
    {"text": "Welcome and thank you for the interest in our experiment! In this experiment you will play a simulation of the Claw Machine arcade. Your goal is to pick up the balls and drop them in the box."},
    {"text": "You can control the position of the claw with the LEFT and RIGHT arrows on your keyboard. When you're happy with the position of the claw over the ball, you can press the fire button for it to go down and pick the ball."},
    {"text": "Please use your dominant hand to control the claw."},
    {"text": "Once you press the fire button, the whole pick-and-drop sequence goes automatically."},
    {"text": "There will be 4 different types of ball, and sometimes it will be easier to pick-and-drop them, and sometimes harder: some balls are lighter on average, some are heavier. Also some balls have little variability in their weight, and their weight is quite predictable, while others have more variability, and their weight is less predictable."},
    {"text": "You will be able to tell different ball types by their color."},
    {"text": "The claw remains the same across the entire experiment."},
    {"text": "Before the main part of the experiment, you will have a chance to do a number of training trials"},
    {"text": "Thank you for your time and good luck!"},
    {"text": ""},  # blank line
]

# training instructions
PRE_TRAINING = [
    {"text": "Welcome to the training phase of the experiment."},
    {"text": "In this part, you will get familiar with the claw machine and how different types of balls behave."},
    {"text": "You will see balls of different colors, each representing a different type of ball:"},
    {"text": ""}, 
    {"text": "BLUE balls are lighter on average and have very predictable weights", "color": (30,144,255)},
    {"text": "GREEN balls are lighter on average but have more variable weights", "color": (50,205,50)},
    {"text": "YELLOW balls are heavier on average and have more variable weights", "color": (255,215,0)},
    {"text": "RED balls are heavier on average and have very predictable weights", "color": (220,20,60)},
    {"text": ""},
    {"text": "Your task is to practice picking up and dropping these balls into the box."},
    {"text": "Please pay attention to how the different types of balls move and behave."},
    {"text": ""},
    {"text": "Press the fire button to begin the training trials."}
]

def main(args=None):
    claw_ctl = ClawCtl(args)
    
    # flag to move the claw disabled for the intructions phase
    claw_ctl.ctl.axis_enabled = False

    # init pygame display
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    
    # instructions
    asyncio.run(display_and_wait(screen, claw_ctl, INTRO))
    asyncio.run(display_and_wait(screen, claw_ctl, PRE_TRAINING))

    claw_ctl.ctl.axis_enabled = True
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

    