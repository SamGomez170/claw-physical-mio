import asyncio
import sys
import pygame
import threading
from .claw_lib import ClawCtl
import time
import textwrap 
from std_msgs.msg import UInt8
import queue
import rclpy


# intro
INTRO = [
    {"text": "Welcome and thank you for the interest in our experiment! In this experiment you will play the Claw Machine arcade. Your goal is to pick up the balls and drop them in the box."},
    {"text": "You can control the position of the claw with the joystick. When you're happy with the position of the claw over the ball, you can press the red button for it to go down and pick the ball."},
    {"text": "Please use your dominant hand to control the claw."},
    {"text": "Once you press the fire button, the whole pick-and-drop sequence goes automatically."},
    {"text": "There will be 4 different types of grip fopr the claw, and sometimes it will be easier to pick-and-drop them, and sometimes harder: some grips are weaker on average, some are stronger. Also some have litte variability in their strenght, so they are more predictable while others have more variablity, so they are less predictable."},
    {"text": "You will be able to tell different grip types by the color of the light in the claw."},
    {"text": "Before the main part of the experiment, you will have a chance to do a number of training trials"},
    {"text": "Thank you for your time and good luck!"},
    {"text": "Press red button to begin the training trials."},
]

# training instructions
PRE_TRAINING = [
    {"text": "Welcome to the training phase of the experiment."},
    {"text": "In this part, you will get familiar with the claw machine and how different types of grips behave."},
    {"text": "You will see lights of different colors, each representing a different type of grip:"},
    {"text": ""}, 
    {"text": "BLUE claw is weaker on average and have very predictable grip strength", "icon": True, "color": (30,144,255)},
    {"text": "GREEN claw is weaker on average but have more variable grip strength", "icon": True, "color": (50,205,50)},
    {"text": "YELLOW claw is stronger on average and have more variable grip strength", "icon": True, "color": (255,215,0)},
    {"text": "RED claw is stronger on average and have very predictable grip strength", "icon": True, "color": (220,20,60)}, 
    {"text": ""},
    {"text": "Your task is to practice picking up and dropping balls into the box."},
    {"text": "Please pay attention to how the different types of claws move and behave."},
    {"text": ""},
    {"text": "Press the red button to begin the training trials."},
]

# post training
POST_TRAINING = [
    {"text": "Now that you've completed the training phase, you will proceed to the main experiment."},
    {"text": "You will have to choose to pick a claw between two options. If you then manage to pick and drop the ball with the chosen claw into the box, you will get the corresponding reward. If not you won't lose points."},
    {"text": ""},
    {"text": "After your choice and before your action, you will be asked two questions:"},
    {"text": "1) about your confidence in choosing the ball that is the best for your overall reward score, and"},
    {"text": "2) about your confidence in being able to pick and drop the chosen ball into the box."},
    {"text": ""},
    {"text": "Please think about your confidence rating carefully, and then report it on a scale from 1 (very unsure) to 6 (very sure), using the corresponding keys on the keyboard."},
    {"text": ""},
    {"text": "Then you'll have a chance to pick and drop the ball you've chosen."},
    {"text": "Press Red Button to continue."},
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
        msg = f"Training Trial {trial_number}/{total_training_trials}. Red Button to continue"
    else:
        if automatic_mode:
            msg = f"Trial {trial_number}. Starting automatically…"
        else:
            msg = f"Action Selection Trial {trial_number}/{total_training_trials}. Red Button to continue"

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

def draw_small_claw(screen, color, topleft):
    x0, y0 = topleft
    cord_len = 15
    bar_w, bar_h = 20, 4
    pr_w, pr_h = 3, 12

    # cord
    cx = x0 + bar_w // 2
    pygame.draw.line(screen, 'black', (cx, y0), (cx, y0 + cord_len), 2)

    # horizontal bar
    bar_y = y0 + cord_len
    pygame.draw.rect(screen, color, (x0, bar_y, bar_w, bar_h))

    # prongs
    prong_y = bar_y + bar_h
    # left prong
    pygame.draw.rect(screen, color, (x0, prong_y, pr_w, pr_h))
    # right prong
    pygame.draw.rect(screen, color,
                     (x0 + bar_w - pr_w, prong_y, pr_w, pr_h))

    return bar_w

async def display_and_wait(screen, claw_ctl=None, render_items=None,
                           title=None, color=(0,0,0), show_claw=False):

    pygame.event.clear()
    pygame.font.init()
    screen.fill((255,255,255))

    # Draw title or text blocks
    if title is not None:
        font = pygame.font.SysFont(None, 36)
        surf = font.render(title, True, color)
        rect = surf.get_rect(center=(screen.get_width() / 2,
                                      screen.get_height() / 2))
        screen.blit(surf, rect)

    # Otherwise draw centered text blocks with extra spacing
    elif render_items:
        font = pygame.font.Font(None, 26)
        y_offset = 80
        max_width = screen.get_width() - 40

        for item in render_items:
            # inside your for item in render_items:
            paragraph   = item.get('text', '')
            text_color  = item.get('color', color)
            draw_icon   = item.get('icon', False)

            # wrap the text to a single line if it’s short enough
            wrapped = await wrap_text(paragraph, font, max_width)
            for line in wrapped:
                # if this line needs a claw icon, split out the color‐word
                if draw_icon:
                    # split on the color word itself
                    cname = line.split()[0]       # "RED"
                    crgb  = text_color            # (220,20,60)

                    before = ""                   # nothing before “RED” in this example
                    after  = line[len(cname):]    # the rest of the sentence

                    x = (screen.get_width() - font.size(line)[0]) // 2
                    y = y_offset

                    # 1) draw ‘before’ text (if any)
                    if before:
                        s0 = font.render(before, True, (0,0,0))
                        screen.blit(s0, (x, y))
                        x += s0.get_width()

                    # 2) draw the mini‑claw icon
                    icon_y = y + (font.get_height() - (15 + 4 + 12)) // 2
                    claw_w = draw_small_claw(screen, crgb, (x, icon_y))
                    x += claw_w + 8    # gap after icon

                    # 3) draw the color word in its color
                    s1 = font.render(cname, True, crgb)
                    screen.blit(s1, (x, y))
                    x += s1.get_width()

                    # 4) draw the rest of the sentence
                    if after.strip():
                        s2 = font.render(after, True, (0,0,0))
                        screen.blit(s2, (x, y))
                else:
                    # normal rendering for lines without the icon
                    text_surface = font.render(line, True, text_color)
                    text_rect    = text_surface.get_rect(
                                    center=(screen.get_width()/2, y_offset)
                                )
                    screen.blit(text_surface, text_rect)

                y_offset += font.get_linesize()

            # extra paragraph spacing
            y_offset += font.get_linesize() * 1.2

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
        "Weaker, Less Variable": (30, 144, 255),  # Blue
        "Weaker, More Variable": (50, 205, 50),     # Green   
        "Stronger, More Variable": (255, 215, 0),    # Yellow
        "Stronger, Less Variable": (220, 20, 60)   # Red
    }
    for name, col in colors.items():
        pygame.draw.circle(screen, col, (x+10, y+10), 10)
        txt = font.render(name, True, (0,0,0))
        screen.blit(txt, (x+25, y))
        y += 25


def draw_training_claw_counter(screen, current_claw, total_claws):
    WIDTH = 800

    font = pygame.font.Font(None, 24)
    
    # Draw first instruction line about arrow keys
    instruction1_text = "Move <- and -> the joystick to control the claw horizontally"
    instruction1_surface = font.render(instruction1_text, True, (0, 0, 0))
    instruction1_rect = instruction1_surface.get_rect(topright=(WIDTH - 10, 10))
    screen.blit(instruction1_surface, instruction1_rect)
    
    # Draw second instruction line about the 'd' key
    instruction2_text = "Press red button for the claw to go down and pick the object"
    instruction2_surface = font.render(instruction2_text, True, (0, 0, 0))
    instruction2_rect = instruction2_surface.get_rect(topright=(WIDTH - 10, 40))
    screen.blit(instruction2_surface, instruction2_rect)
    
    # Draw the claw counter below both instructions
    counter_text = f"Claws: {current_claw}/{total_claws}"
    text_surface = font.render(counter_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(topright=(WIDTH - 10, 70))
    screen.blit(text_surface, text_rect)

async def display_claw_selection(screen, first_type, second_type, claw_ctl, rewards):
    # Enable UI nav (captures left/right; blocks axis movement)
    claw_ctl.ctl.ui_enabled = True
    claw_ctl.ctl.axis_enabled = False

    # Clear any stale events
    claw_ctl.ctl.red_button_event.clear()
    while not claw_ctl.ctl.ui_nav_queue.empty():
        claw_ctl.ctl.ui_nav_queue.get_nowait()

    current_selection = 0  # 0 = left, 1 = right
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Precompute distribution text
    dist = {
        "wide_high": f"High force\nHigh variability\nReward: {rewards['wide_high']}",
        "narrow_high": f"High force\nLow variability\nReward: {rewards['narrow_high']}",
        "wide_low":  f"Low force\nHigh variability\nReward: {rewards['wide_low']}",
        "narrow_low":f"Low force\nLow variability\nReward: {rewards['narrow_low']}"
    }

    def draw_claw(surface, cx, cy, ctype):
        # (same as your draw_claw)
        claw_colors = {
            "narrow_low": (30, 144, 255),
            "wide_low":   (50, 205, 50),
            "wide_high":  (255, 215, 0),
            "narrow_high":(220, 20, 60)
        }
        color = claw_colors.get(ctype, (100, 100, 100))
        base_w, arm_l, arm_w, cord_l = 60, 50, 10, 100

        # cord
        pygame.draw.line(surface, (0,0,0),
                         (cx, cy - cord_l),
                         (cx, cy - arm_w//2), 2)
        # base
        base_rect = pygame.Rect(cx - base_w//2,
                                cy - arm_w//2,
                                base_w, arm_w)
        pygame.draw.rect(surface, color, base_rect)
        # arms
        lx, rx = cx - base_w//2, cx + base_w//2
        sy = cy
        ey = cy + arm_l
        pygame.draw.line(surface, color, (lx, sy), (lx, ey), arm_w)
        pygame.draw.line(surface, color, (rx, sy), (rx, ey), arm_w)
        # endpoints
        for x in (lx, rx):
            pygame.draw.circle(surface, color, (x, ey), 4)

    # Main loop
    while True:
        screen.fill((255,255,255))
        w, h = screen.get_size()

        # box sizes & positions
        normal, selected = (200,200), (240,240)
        lx, rx, by = 100, 500, 200

        # build rects
        if current_selection == 0:
            left = pygame.Rect(lx-20, by-20, *selected)
            right = pygame.Rect(rx, by, *normal)
        else:
            left = pygame.Rect(lx, by, *normal)
            right = pygame.Rect(rx-20, by-20, *selected)

        # draw outlines
        highlight = (50,255,255)
        pygame.draw.rect(screen,
                         highlight if current_selection==0 else (0,0,0),
                         left, 4 if current_selection==0 else 2)
        pygame.draw.rect(screen,
                         highlight if current_selection==1 else (0,0,0),
                         right,4 if current_selection==1 else 2)

        # draw claws
        draw_claw(screen, left.centerx, left.centery+30, first_type)
        draw_claw(screen, right.centerx, right.centery+30, second_type)

        # distribution info
        lines1 = dist[first_type].split('\n')
        lines2 = dist[second_type].split('\n')
        for i, line in enumerate(lines1):
            txt = small_font.render(line, True, (0,0,0))
            screen.blit(txt, 
                        (left.centerx - txt.get_width()//2,
                         left.bottom + 15 + i*20))
        for i, line in enumerate(lines2):
            txt = small_font.render(line, True, (0,0,0))
            screen.blit(txt,
                        (right.centerx - txt.get_width()//2,
                         right.bottom + 15 + i*20))

        # instructions
        instr = font.render("Left/Right to choose, Red button to confirm", True, (0,0,0))
        screen.blit(instr, (w//2 - instr.get_width()//2, 100))

        pygame.display.flip()

        # handle nav
        try:
            nav = claw_ctl.ctl.ui_nav_queue.get_nowait()
            if nav == 'Key.left':
                current_selection = 0
            elif nav == 'Key.right':
                current_selection = 1
        except queue.Empty:
            pass

        # spin ROS for joystick/red‑button
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)

        # confirm?
        if claw_ctl.ctl.red_button_event.is_set():
            claw_ctl.ctl.red_button_event.clear()
            break

        await asyncio.sleep(0.02)
        clock.tick(60)

    # teardown UI mode
    claw_ctl.ctl.ui_enabled = False
    claw_ctl.ctl.axis_enabled = True

    return first_type if current_selection == 0 else second_type


async def run_trial(claw_ctl, screen,
                    speed=255, grip=160,
                    home_delay=2, open_delay=2,
                    trial_number=None, total_trials=None):

    """
    Enable joystick, wait for fire, perform grab sequence, move home, open claw.
    """
    draw_legend(screen)
    if trial_number is not None and total_trials is not None:
        draw_training_claw_counter(screen, trial_number, total_trials)


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
    rewards = {
        "wide_high": 10,
        "narrow_high": 8,
        "wide_low": 5,
        "narrow_low": 3
    }

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
        await run_trial(claw_ctl, screen, trial_number=trial, total_trials=training_trials)

    # training complete
    claw_ctl.ctl.axis_enabled = False
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
        choice = await display_claw_selection(
            screen,
            first_type="wide_high",
            second_type="narrow_low",
            claw_ctl=claw_ctl,
            rewards=rewards
        )

        with claw_ctl.ctl.ui_nav_queue.mutex:
            claw_ctl.ctl.ui_nav_queue.queue.clear()

        screen.fill((255, 255, 255))
        pygame.display.flip()

        claw_ctl.ctl.axis_enabled = True
        await run_trial(claw_ctl, screen)
