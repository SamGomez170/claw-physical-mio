import asyncio
import sys
import pygame
import threading
from .claw_lib import ClawCtl
import time
import textwrap 
from std_msgs.msg import UInt8


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
    {"text": "Press the fire button to begin the training trials."},
]

# post training
POST_TRAINING = [
    {"text": "You completed the training phase and then proceeded to the main experiment."},
    {"text": "You were asked to read the instructions carefully your understanding of them was tested at the end, and you could only continue the experiment if you answered at least 4 out of 5 questions correctly."},
    {"text": ""},
    {"text": "On each trial, you had to choose a ball to pick and drop between two options."},
    {"text": "If you managed to pick and drop this ball into the box, you received the corresponding reward."},
    {"text": "If not – you did not lose any points."},
    {"text": ""},
    {"text": "Different ball types brought different reward points."},
    {"text": ""},
    {"text": "Later, reward points were converted into money with £1 given for every 200 points."}
]

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

async def display_trial_start(
    screen,
    claw_ctl,
    trial_number,
    automatic_mode=False,
    is_training=False,
    total_training_trials=2,
    delay=2000
):
    # clear out any old events, prep font
    pygame.event.clear()
    pygame.font.init()
    screen.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 36)

    # choose message
    if is_training:
        msg = f"Training Trial {trial_number}/{total_training_trials}. Fire to start"
    else:
        if automatic_mode:
            msg = f"Trial {trial_number}. Starting automatically…"
        else:
            msg = f"Action Selection Trial {trial_number}/{total_training_trials}. Fire to start"

    # render & blit
    surf = font.render(msg, True, (0, 0, 0))
    rect = surf.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
    screen.blit(surf, rect)

    draw_legend(screen)
    pygame.display.flip()

    # wait for fire button
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, claw_ctl.wait_fire_button)


    # clear screen
    screen.fill((255,255,255))
    pygame.display.flip()

async def display_and_wait(screen, claw_ctl=None, render_items=None,
                           title=None, color=(0,0,0), legend=False,
                           automatic_mode=False, delay=0):

    pygame.event.clear()
    pygame.font.init()
    screen.fill((255,255,255))

    # Draw title or text blocks
    if title is not None:
        font = pygame.font.SysFont(None, 36)
        surf = font.render(title, True, color)
        rect = surf.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
        screen.blit(surf, rect)
    elif render_items:
        font = pygame.font.Font(None, 28)
        max_w = screen.get_width() - 80
        y = 40
        for item in render_items:
            lines = await wrap_text(item.get('text',''), font, max_w)
            for line in lines:
                surf = font.render(line, True, item.get('color') or (0,0,0))
                rect = surf.get_rect(topleft=(40, y))
                screen.blit(surf, rect)
                y += 35
            y += 20

    pygame.display.flip()

    # wait for input
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, claw_ctl.wait_fire_button)

    # clear screen
    screen.fill((255,255,255))
    pygame.display.flip()

def draw_legend(screen):
    font = pygame.font.Font(None, 24)
    y, x = 20, 10
    colors = {
        "Lighter, More Variable": (50, 205, 50),
        "Lighter, Less Variable": (30, 144, 255),
        "Heavier, More Variable": (255, 215, 0),
        "Heavier, Less Variable": (220, 20, 60)
    }
    for name, col in colors.items():
        pygame.draw.circle(screen, col, (x+10, y+10), 10)
        txt = font.render(name, True, (0,0,0))
        screen.blit(txt, (x+25, y))
        y += 25


async def run_trial(claw_ctl, screen,
                    speed=255, grip=160,
                    home_delay=2, open_delay=2,
                    current_ball=None, total_balls=None):

    """
    Enable joystick, wait for fire, perform grab sequence, move home, open claw.
    """
    draw_legend(screen)

    pygame.display.flip()
    claw_ctl.enable_joystick()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, claw_ctl.wait_fire_button)

    # disable joystick when dont want the user to move it during automatic sequence
    #claw_ctl.disable_joystick()
    
    # automatic sequence grab -> up -> go home -> open claw 
    claw_ctl.grab_sequence(speed, grip)
    claw_ctl.move_home()
    await asyncio.sleep(home_delay)
    claw_ctl.open_claw()
    await asyncio.sleep(open_delay)

async def run_game(screen, claw_ctl, total_trials=3, training_trials=2):
    # disable movement during instructions
    claw_ctl.ctl.axis_enabled = False

    # show intro/instructions
    await display_and_wait(screen, claw_ctl, INTRO)
    await display_and_wait(screen, claw_ctl, PRE_TRAINING)

    # run each trial
    for trial in range(1, training_trials + 1):
        claw_ctl.ctl.axis_enabled = False
        await display_trial_start(
            screen, claw_ctl,
            trial_number=trial,
            is_training=True,
            total_training_trials=training_trials,
            automatic_mode=False,
            delay=2000
        )
        claw_ctl.ctl.axis_enabled = True
        await run_trial(claw_ctl, screen)

    # training complete
    await display_and_wait(
        screen,
        claw_ctl,
        title="Training completed. Press fire to begin the experiment."
    )
    await display_and_wait(screen, claw_ctl, POST_TRAINING)

    # actual experiment trials
    for trial in range(1, total_trials + 1):
        claw_ctl.ctl.axis_enabled = False
        await display_trial_start(
            screen, claw_ctl,
            trial_number=trial,
            is_training=False,
            total_training_trials=total_trials,
            automatic_mode=False,
            delay=2000
        )
        claw_ctl.ctl.axis_enabled = True
        await run_trial(claw_ctl, screen)
