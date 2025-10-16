import asyncio
import sys
import pygame
from .claw_lib import ClawCtl
from std_msgs.msg import UInt8
import queue
import rclpy
import random
import json
import os
import secrets
import numpy as np
import datetime
from .data_logger import DataLogger, save_choice_data, save_event_data, save_trial_json, make_session_dir, save_session_metadata
import time 
from .read_rfid import RFIDReader

rfid = RFIDReader(port=None, baud=115200, timeout=0.1, verbose=False)
rfid.start()
    

SELECTION_COLOR = (0, 128, 0)   # e.g. a green for “selection” 
OUTCOME_COLOR   = (128, 0, 0)   # e.g. a red for “outcome”
rewards = {
    "wide_low": 14,      # Base reward
    "narrow_low": 18,  # Slight penalty for playing it safe
    "wide_high": 10,     # High reward for risky choice  
    "narrow_high": 11    # Good reward for skill-based choice
}
WIDTH = 800
HEIGHT = 600  # You may need to adjust this based on your screen height
grip_distribution_types = {
    "wide_high": {
        "force": {"mu": 184.5, "sigma": 0.3},
        "rate": 255
    }, #yellow
    "narrow_high": {
        "force": {"mu": 184.5, "sigma": 0.12},
        "rate": 255
    },#red
    "wide_low": {
        "force": {"mu": 183.4, "sigma": 0.3},
        "rate": 255
    }, #green
    "narrow_low": {
        "force": {"mu": 183.4, "sigma": 0.12},
        "rate": 255
    }#blue
}



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

# end
END = [
    {"text": "Thank you! This is the end of the experiment."},
    {"text": "Your total reward: {total_reward:.1f} points"},
    {"text": "In monetary terms, your reward is: £{monetary_reward:.2f}"},
    {"text": "Your data are being saved. You will now proceed to the next part."},
    {"text": "Press SPACE or ENTER to continue."},
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
        draw_legend(screen)
    else:
        if automatic_mode:
            msg = f"Trial {trial_number}. Starting automatically…"
        else:
            msg = f"Action Selection Trial {trial_number}/{total_training_trials}. Red Button to continue"

    # render & blit
    surf = font.render(msg, True, (0, 0, 0))
    rect = surf.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
    screen.blit(surf, rect)

    pygame.display.flip()

    # wait for fire button
    #loop = asyncio.get_running_loop()
    #await loop.run_in_executor(None, claw_ctl.wait_fire_button)
    # wait until the fire button is pressed, but keep UI responsive
    while not claw_ctl.ctl.red_button_event.is_set():
        # let rclpy process incoming joystick/red-button messages
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)

        # ensure pygame processes window messages (prevents "Not Responding")
        pygame.event.pump()

        # yield to the asyncio loop so other tasks can run
        await asyncio.sleep(0.01)
    # once set, clear it or use it as needed
    claw_ctl.ctl.red_button_event.clear()

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
                           title=None, color=(0,0,0), total_reward=0):

    pygame.event.clear()
    pygame.font.init()
    screen.fill((255,255,255))
    monetary_reward = total_reward / 200 

    # Draw title or text blocks
    if title is not None:
        font = pygame.font.SysFont(None, 36)
        surf = font.render(title, True, color)
        rect = surf.get_rect(center=(screen.get_width() / 2,
                                      screen.get_height() / 2))
        screen.blit(surf, rect)

    # draw centered text blocks with extra spacing
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
    # wait until the fire button is pressed, but keep UI responsive
    while not claw_ctl.ctl.red_button_event.is_set():
        # let rclpy process incoming joystick/red-button messages
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)

        # ensure pygame processes window messages (prevents "Not Responding")
        pygame.event.pump()

        # yield to the asyncio loop so other tasks can run
        await asyncio.sleep(0.01)
    # once set, clear it or use it as needed
    claw_ctl.ctl.red_button_event.clear()

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

def draw_reward_display(screen, grip_distribution_type, total_reward, training_mode=False):
    if training_mode:  # Don't display rewards during training
        return
        
    font = pygame.font.Font(None, 24)
    reward_x = WIDTH - 300  # Position in upper right corner
    reward_y = 10  # Near the top

    # Create a larger font for the claw text
    large_font_size = 36
    large_font = pygame.font.Font(None, large_font_size)

    
    # Get potential reward for current trial
    potential_reward = rewards[grip_distribution_type] if grip_distribution_type in rewards else 0
    
    # Draw potential and total rewards
    potential_text = f"Potential Reward in This Trial: {potential_reward:.1f}"
    total_text = f"Total Reward: {total_reward:.1f}"
    
    potential_surface = font.render(potential_text, True, (0, 0, 0))
    total_surface = font.render(total_text, True, (0, 0, 0))

    # Draw the new text at the top center behind everything else
    claw_text = "You are playing now with this claw:"
    claw_surface = large_font.render(claw_text, True, (200, 200, 200))  # Light gray for background effect
    claw_rect = claw_surface.get_rect()
    claw_rect.centerx = WIDTH // 2
    claw_rect.top = 150
    screen.blit(claw_surface, claw_rect)
    
    screen.blit(potential_surface, (reward_x, reward_y))
    screen.blit(total_surface, (reward_x, reward_y + 25))


def draw_training_claw_counter(screen, current_claw, total_claws):
    # Create a smaller font for instructions and counter
    font = pygame.font.Font(None, 24)

    # Create a larger font for the claw text
    large_font_size = 36
    large_font = pygame.font.Font(None, large_font_size)

    # Draw the new text at the top center behind everything else
    claw_text = "You are playing now with this claw:"
    claw_surface = large_font.render(claw_text, True, (200, 200, 200))  # Light gray for background effect
    claw_rect = claw_surface.get_rect()
    claw_rect.centerx = WIDTH // 2
    claw_rect.top = 150
    screen.blit(claw_surface, claw_rect)

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
            "narrow_low": (30, 144, 255), #blue
            "wide_low":   (50, 205, 50),  #green
            "wide_high":  (255, 215, 0),  #yellow
            "narrow_high":(220, 20, 60)   #red
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

async def display_rating(screen, claw_ctl,
                         question_lines, scale_texts,
                         question_type, trial):

    # — INITIAL SETUP —
    pygame.font.init()
    claw_ctl.ctl.ui_enabled   = True
    claw_ctl.ctl.axis_enabled = False

    # clear any pending input
    claw_ctl.ctl.red_button_event.clear()
    while not claw_ctl.ctl.ui_nav_queue.empty():
        claw_ctl.ctl.ui_nav_queue.get_nowait()

    # headers & fonts
    header_text  = ("BALL SELECTION CONFIDENCE"
                    if question_type=="selection"
                    else "ACTION OUTCOME CONFIDENCE")
    header_color = (0,128,255) if question_type=="selection" else (255,100,0)
    trial_text   = f"Trial {trial}."
    header_font  = pygame.font.SysFont(None, 40)
    font         = pygame.font.SysFont(None, 32)
    label_font   = pygame.font.SysFont(None, 28)
    clock        = pygame.time.Clock()
    current      = 0
    locked_in    = False
    locked_value = None


    def draw():
        nonlocal locked_in, locked_value
        screen.fill((255,255,255))
        w,h = screen.get_size()
        y = 50

        # Header
        surf = header_font.render(header_text, True, header_color)
        screen.blit(surf, surf.get_rect(center=(w//2,y)))
        y += header_font.get_linesize() + 10

        # Trial #
        surf = font.render(trial_text, True, (0,0,0))
        screen.blit(surf, surf.get_rect(center=(w//2, y)))
        y += font.get_linesize() + 10

        # Question lines
        for line in question_lines:
            surf = font.render(line, True, (0,0,0))
            screen.blit(surf, surf.get_rect(center=(w//2,y)))
            y += font.get_linesize() + 5
        y += 20

        # Options
        for idx, label in enumerate(scale_texts):
            col = (0,0,255) if idx==current else (0,0,0)
            surf = label_font.render(label, True, col)
            screen.blit(surf, surf.get_rect(center=(w//2,y)))
            y += label_font.get_linesize() + 15

        # Two‐step instructions:
        if not locked_in:
            prompt = "Navigate with Up/Down, then press Red Button"
        else:
            prompt = "Press Red again to continue".format(locked_value)
        instr = label_font.render(prompt, True, (0,0,0))
        screen.blit(instr, instr.get_rect(center=(w//2, h-40)))

        pygame.display.flip()

    # draw initial frame
    draw()

    # — MAIN LOOP —
    while True:
        updated = False  # Track if UI needs redraw

        # handle nav
        try:
            nav = claw_ctl.ctl.ui_nav_queue.get_nowait()
            if nav == 'Key.up':
                current = max(0, current - 1)
                draw()
            elif nav == 'Key.down':
                current = min(len(scale_texts) - 1, current + 1)
                draw()
        except queue.Empty:
            pass


        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)

        if updated:
            draw()  # Redraw screen when selection changes

        # one-step red button confirm
        if claw_ctl.ctl.red_button_event.is_set():
            claw_ctl.ctl.red_button_event.clear()
            locked_value = current + 1   # 1-based scale
            break

        # always redraw to update the prompt/selection highlight
        draw()
        await asyncio.sleep(0.01)
        clock.tick(60)

     # — TEARDOWN & RETURN —
    claw_ctl.ctl.ui_enabled   = False
    claw_ctl.ctl.axis_enabled = True

    return locked_value

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

def generate_participant_id(prefix='P', length=8):
    """Return a random hex id like 'P8b3a2f1c' (length is number of chars after prefix)."""
    # token_hex(4) -> 8 hex chars; adjust if you want longer
    nbytes = max(1, (length + 1) // 2)
    return prefix + secrets.token_hex(nbytes)[:length]


async def run_trial(claw_ctl, screen, grip_type,
                    speed, grip,
                    home_delay=2, open_delay=2,
                    trial_number=None, total_trials=None,
                    training_mode=False, total_reward=0, rfid_reader=None):
    """
    Run one trial while logging telemetry. Returns a trial_record dict.
    """
    # prepare logger
    logger = DataLogger(claw_ctl, sample_interval=0.01)  # 100 Hz sampling
    logger.log_event('trial_init', {'grip_type': grip_type, 'speed': speed, 'grip_sampled': grip, 'training_mode': training_mode})
    logger.start()

    def _movement_handler(movement, raw_msg=None):
        """
        movement: string like 'Key.up' or 'Key.stop'
        raw_msg: the original ROS message (optional)
        """
        payload = {
            'movement': movement,
            'timestamp': time.time()
        }
        # if raw_msg has header or seq you might include it:
        try:
            if raw_msg is not None and hasattr(raw_msg, 'header'):
                payload['msg_header'] = {
                    'stamp': str(getattr(raw_msg, 'header').stamp) if getattr(raw_msg, 'header', None) else None,
                    'frame_id': getattr(raw_msg, 'header').frame_id if getattr(raw_msg, 'header', None) else None
                }
        except Exception:
            pass

        # write to the same logger used for trial events
        try:
            logger.log_event('joystick_movement', payload)
        except Exception as e:
            # be defensive: don't let logging errors break the trial
            print(f"[WARN] movement logging failed: {e}")

    # attach to the hardware control object
    if hasattr(claw_ctl, 'ctl'):
        claw_ctl.ctl.on_movement = _movement_handler
    else:
        # fallback if ClawCtl exposes the node directly
        setattr(claw_ctl, 'on_movement', _movement_handler)

    # show UI overlays depending on mode
    if training_mode:
        if trial_number is not None and total_trials is not None:
            draw_legend(screen)
            draw_training_claw_counter(screen, trial_number, total_trials)
    else:
        draw_reward_display(screen, grip_type, total_reward, training_mode)

    cx, cy = screen.get_width() // 2, screen.get_height() // 2
    draw_claw(screen, cx, cy, grip_type)
    pygame.display.flip()

    # Wrap the key claw_ctl actions to add precise event logging
    # Save originals so we can restore later
    orig_grab = getattr(claw_ctl, 'grab_sequence', None)
    orig_move_home = getattr(claw_ctl, 'move_home', None)
    orig_open_claw = getattr(claw_ctl, 'open_claw', None)

    def _wrap(method_name, orig):
        if not orig:
            return None
        def wrapped(*a, **kw):
            logger.log_event(f'{method_name}_start', {'args': a, 'kwargs': kw})
            try:
                res = orig(*a, **kw)
            except Exception as e:
                logger.log_event(f'{method_name}_exception', {'exc': str(e)})
                raise
            logger.log_event(f'{method_name}_end')
            return res
        return wrapped

    if orig_grab:
        setattr(claw_ctl, 'grab_sequence', _wrap('grab_sequence', orig_grab))
    if orig_move_home:
        setattr(claw_ctl, 'move_home', _wrap('move_home', orig_move_home))
    if orig_open_claw:
        setattr(claw_ctl, 'open_claw', _wrap('open_claw', orig_open_claw))

    # enable joystick and wait for user fire (samples already collecting)
    claw_ctl.enable_joystick()

    # wait until the fire button is pressed, but keep UI responsive
    while not claw_ctl.ctl.red_button_event.is_set():
        # let rclpy process incoming joystick/red-button messages
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)

        # ensure pygame processes window messages (prevents "Not Responding")
        pygame.event.pump()

        # yield to the asyncio loop so other tasks can run
        await asyncio.sleep(0.01)
    # once set, clear it or use it as needed
    claw_ctl.ctl.red_button_event.clear()

    logger.log_event('fire_pressed')

    # At this point the automatic sequence will run:
    # grab -> move_to_box -> open_claw -> move_home (random)
    try:
        # Note: the wrapped functions will log start/end events
        claw_ctl.grab_sequence(speed, grip)

        # move to the fixed drop-box and wait for arrival (if you added move_to_box)
        if hasattr(claw_ctl, 'move_to_box'):
            claw_ctl.move_to_box(x=0.0, y=150.0)

        claw_ctl.open_claw()
        await asyncio.sleep(open_delay)

        # THEN move to a random home position (move_home still uses the random logic)
        claw_ctl.move_home()

        # optional extra wait after move to random new position
        #await asyncio.sleep(home_delay)

    except Exception as e:
        logger.log_event('run_sequence_exception', {'exc': str(e)})
 

    # restore original methods
    if orig_grab:
        setattr(claw_ctl, 'grab_sequence', orig_grab)
    if orig_move_home:
        setattr(claw_ctl, 'move_home', orig_move_home)
    if orig_open_claw:
        setattr(claw_ctl, 'open_claw', orig_open_claw)

    # --- attempt to read RFID tag and record it in the trial_end event ---
    info_value = None
    try:
        if rfid_reader:
            # prefer waiting for a new tag that arrives during the open window
            # set require_new=True if you want to ensure it was presented after the open
            tag = rfid_reader.read_tag(timeout=3.0, require_new=False)
            # or, if you want only tags that arrived after a certain moment, call with require_new=True
            # tag = rfid_reader.read_tag(timeout=3.0, require_new=True)
            if tag:
                info_value = f"SUCCESSFUL BALL, tag: {tag}"

        # fallback: try to read last_tag (very recent)
        if not info_value and rfid_reader:
            info_value = rfid_reader.get_last_tag(max_age=0.5)
    except Exception as e:
        print(f"[WARN] RFID check failed: {e}")

    if not info_value:
        info_value = "NO BALL SUCCESS"
    logger.log_event('trial_end', info_value)

    # stop the logger after logging the event
    logger.stop()

    # detach movement handler (clean up)
    try:
        if hasattr(claw_ctl, 'ctl') and hasattr(claw_ctl.ctl, 'on_movement'):
            claw_ctl.ctl.on_movement = None
        elif hasattr(claw_ctl, 'on_movement'):
            claw_ctl.on_movement = None
    except Exception:
        pass

    # try to read the node's current home target (this should be the random point chosen by move_home)
    original_position = None
    try:
        ctl_node = getattr(claw_ctl, 'ctl', None)
        if ctl_node is not None:
            # if there's a lock, use it
            home_lock = getattr(ctl_node, 'home_lock', None)
            if home_lock is not None:
                with home_lock:
                    original_position = (float(getattr(ctl_node, 'home_x', None)),
                                         float(getattr(ctl_node, 'home_y', None)))
            else:
                original_position = (float(getattr(ctl_node, 'home_x', None)),
                                     float(getattr(ctl_node, 'home_y', None)))
    except Exception:
        original_position = None

    # build trial record
    trial_record = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'trial_number': trial_number,
        'total_trials': total_trials,
        'training_mode': bool(training_mode),
        'grip_type': grip_type,
        'grip_params': grip_distribution_types.get(grip_type, {}),
        'grip_sampled_value': grip,
        #'speed_value': speed,
        'total_reward_before': total_reward,
        'logger': logger.as_dict(),
        'original_claw_position': original_position,
        'rfid': info_value
    }

    # return the data; caller (run_game) will handle saving or aggregate collection
    
    return trial_record

async def run_game(screen, claw_ctl, total_trials=3, training_trials=2):
    # show intro/instructions
    await display_and_wait(screen, claw_ctl, INTRO)
    await display_and_wait(screen, claw_ctl, PRE_TRAINING)

        # === RFID: start session-wide background reader ===
    rfid = RFIDReader(port=None, baud=115200, timeout=0.1, verbose=True)
    rfid.start()


    all_records = []
    training_types = list(grip_distribution_types.keys())
    random.shuffle(training_types)

    participant_id = generate_participant_id()   # e.g. "P3f8a9d2b"
    session_dir = make_session_dir(participant_id, base_dir='claw_data')
    #save_session_metadata(session_dir, participant_id, extra={'experiment': 'claw_v1', 'notes': ''})

    # TRAINING TRIALS
    
    for i, grip_type in enumerate(training_types, start=1):
        claw_ctl.ctl.axis_enabled = False
        await display_trial_start(
            screen, claw_ctl,
            trial_number=i,
            is_training=True,
            total_training_trials=len(training_types),
            automatic_mode=False,
            delay=2000
        )
        claw_ctl.ctl.axis_enabled = True

        grip_params = grip_distribution_types[grip_type]
        grip_force = random.gauss(
            grip_params["force"]["mu"],
            grip_params["force"]["sigma"]
        )

        grip_value = grip_force
        speed_value = grip_distribution_types[grip_type]["rate"]

        trial_record = await run_trial(
            claw_ctl,
            screen,
            grip_type = grip_type,
            speed=speed_value,
            grip=grip_value,
            trial_number=i,
            training_mode=True,
            total_trials=len(training_types),
            rfid_reader=rfid
        )

        # SAVING DATA
        ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        _, ev_path, _ = save_event_data(trial_record,
                                       participant_id=participant_id,
                                       timestamp=ts,
                                       save_locally=True,
                                       out_dir=session_dir,
                                       filename_prefix=f"train_trial_{i}",
                                       include_samples=False)   # set True if you want samples too
        if ev_path:
            print(f"[DATA] Saved training events JSON to {ev_path}")
        all_records.append(trial_record)
    
    # TRAINING COMPLETE
    claw_ctl.ctl.axis_enabled = False
    await display_and_wait(
        screen,
        claw_ctl,
        title="Training completed. Press fire to begin the experiment."
    )
    await display_and_wait(screen, claw_ctl, POST_TRAINING)

    # EXPERIMENT TRIALS
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

        available = list(grip_distribution_types.keys())
        first_type = random.choice(available)
        available.remove(first_type)
        second_type = random.choice(available)

        choice = await display_claw_selection(
            screen,
            first_type=first_type,
            second_type=second_type,
            claw_ctl=claw_ctl,
            rewards=rewards
        )

        selection_confidence = await display_rating(
            screen,
            claw_ctl,
            question_lines=[
                "How confident are you that you selected the ball",
                "that would give you the highest reward?"
            ],
            scale_texts=[
                "1 - very unsure",
                "2 - unsure",
                "3 - somewhat unsure",
                "4 - somewhat sure",
                "5 - sure",
                "6 - very sure",
            ],
            question_type="selection",
            trial=trial
        )

        outcome_confidence = await display_rating(
            screen,
            claw_ctl,
            question_lines=[
                "How sure are you that you made the best decision",
                "for your reward score?"
            ],
            scale_texts=[
                "1 - very unsure",
                "2 - unsure",
                "3 - somewhat unsure",
                "4 - somewhat sure",
                "5 - sure",
                "6 - very sure",
            ],
            question_type="outcome",
            trial=trial
        )

        # clear nav queue
        with claw_ctl.ctl.ui_nav_queue.mutex:
            claw_ctl.ctl.ui_nav_queue.queue.clear()
        screen.fill((255, 255, 255))
        pygame.display.flip()

        grip_params = grip_distribution_types[choice]
        grip_force = random.gauss(
            grip_params["force"]["mu"],
            grip_params["force"]["sigma"]
        )
        grip_value = grip_force
        speed_value = grip_distribution_types[choice]["rate"]

        claw_ctl.ctl.axis_enabled = True
        current_total_reward = 0

        trial_record = await run_trial(
            claw_ctl,
            screen,
            grip_type=choice,
            speed=speed_value,
            grip=grip_value,
            trial_number=trial,
            total_trials=total_trials,
            training_mode=False,
            total_reward=current_total_reward, rfid_reader=rfid
        )

        # augment record with participant responses & choices
        trial_record['choice'] = choice
        trial_record['selection_confidence'] = selection_confidence
        trial_record['outcome_confidence'] = outcome_confidence
        trial_record['chosen_reward'] = rewards.get(choice)
        trial_record['options_displayed'] = (first_type, second_type)

        # Save experiment trial into the participant/session folder
        #p = save_trial_json(trial_record, out_dir=session_dir)
        #print(f"[DATA] Saved experiment trial to {p}")
        #all_records.append(trial_record)

        # Save minimal choice summary (instead of logger/events)
        choice_path = save_choice_data(trial_record,
                                    participant_id=participant_id,
                                    out_dir=session_dir,
                                    filename_prefix=f"choice_trial_{trial}")
        print(f"[DATA] Saved choice summary to {choice_path}")
        all_records.append(trial_record)

        claw_ctl.ctl.axis_enabled = False

    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')

    def _is_success(rfid_val):
        """Return True if RFID suggests success; False otherwise."""
        if not rfid_val:
            return False
        s = str(rfid_val).upper()
        if 'NO' in s and 'NO BALL' in s:
            return False
        if 'NO BALL SUCCESS' in s or 'NO SUCCESS' in s:
            return False
        # anything mentioning SUCCESS or a tag we treat as success
        if 'SUCCESS' in s or 'TAG' in s or 'SUCCESSFUL' in s:
            return True
        # fallback: if it's not the explicit NO string assume success if non-empty
        return True

    # partition
    training_trials = [t for t in all_records if t.get('training_mode')]
    experiment_trials = [t for t in all_records if not t.get('training_mode')]

    # build minimal summaries
    training_summary = []
    for t in training_trials:
        training_summary.append({
            'participant_id': participant_id,
            'trial_number': t.get('trial_number'),
            'grip_force': t.get('grip_sampled_value'),
            'rfid': t.get('rfid'),
            'success': _is_success(t.get('rfid')),
        })

    experiment_summary = []
    for t in experiment_trials:
        experiment_summary.append({
            'participant_id': participant_id,
            'trial_number': t.get('trial_number'),
            'grip_force': t.get('grip_sampled_value'),
            'rfid': t.get('rfid'),
            'success': _is_success(t.get('rfid')),
            'selection_confidence': t.get('selection_confidence'),
            'outcome_confidence': t.get('outcome_confidence'),
        })

    # ensure session dir exists
    os.makedirs(session_dir, exist_ok=True)

    train_path = os.path.join(session_dir, f'combined_all_training_{participant_id}_{ts}.json')
    exp_path   = os.path.join(session_dir, f'combined_all_trials_{participant_id}_{ts}.json')

    try:
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, default=str)
        print(f"[DATA] Saved combined training summary to {train_path}")
    except Exception as e:
        print(f"[WARN] failed saving combined training summary: {e}")

    try:
        with open(exp_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, default=str)
        print(f"[DATA] Saved combined experiment summary to {exp_path}")
    except Exception as e:
        print(f"[WARN] failed saving combined experiment summary: {e}")
    claw_ctl.ctl.axis_enabled = False
        # at end of run_game, before exiting:
    try:
        rfid.stop()
    except Exception:
        pass


    await display_and_wait(
        screen,
        claw_ctl,
        title="Experiment completed",
        total_reward=10
    )
