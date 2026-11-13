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
import serial
from .read_IR import IRDetector
import signal
from statistics import NormalDist


#total_trials=
#NUM_TRAINING_ROUNDS = 6 
SELECTION_COLOR = (0, 128, 0)
OUTCOME_COLOR   = (128, 0, 0)
rewards = {
    "wide_low": 14,
    "narrow_low": 18,
    "wide_high": 11,
    "narrow_high": 10
}
WIDTH = 800     # layout was designed for 800 x 600; everything is scaled to the real screen
HEIGHT = 600
UI_SCALE = 1.0  # set from the real screen size at the start of run_game


def set_ui_scale(screen):
    """Scale fonts and distances to the screen (e.g. 1280x1024 -> 1.6)."""
    global UI_SCALE
    w, h = screen.get_size()
    UI_SCALE = max(1.0, min(w / WIDTH, h / HEIGHT))
    print(f"[SCREEN] {w}x{h}, text scale {UI_SCALE:.2f}")


def F(size):
    """Font of `size` (as designed for 800x600), scaled to the screen."""
    return pygame.font.Font(None, max(8, int(round(size * UI_SCALE))))


def P(v):
    """Distance/size in pixels (as designed for 800x600), scaled to the screen."""
    return int(round(v * UI_SCALE))
# Arena limits for the random start position (same units as move_to_box)
ARENA_X = (300.0, 750.0)
ARENA_Y = (100.0, 480.0)
NUM_TRAINING_ROUNDS = 4  # each round shows every claw once -> 12 trials per claw (48 in total)
SKIP_TRAINING = False   # True = go straight from the intro to the experiment
STRATIFIED_TRAINING = True  # True = training forces cover each claw's distribution evenly
                            # (stratified sampling); False = plain random draws (random.gauss)

grip_distribution_types = {
    "wide_high": {
        "force": {"mu": 197, "sigma": 2},
        "rate": 255
    },
    "narrow_high": {
        "force": {"mu": 197, "sigma": 0.5},
        "rate": 255
    },
    "wide_low": {
        "force": {"mu": 191, "sigma": 2},
        "rate": 255
    },
    "narrow_low": {
        "force": {"mu": 191, "sigma": 0.5},
        "rate": 255
    }
}


# intro
INTRO = [
    {"text": "Welcome and thank you for the interest in our experiment! In this experiment you will play the Claw Machine arcade. Your goal is to pick up the balls and drop them in the box."},
    {"text": "You can control the position of the claw with the joystick. When you're happy with the position of the claw over the ball, you can press the red button for it to go down and pick the ball."},
    {"text": "Please use your dominant hand to control the claw."},
    {"text": "Once you press the fire button, the whole pick-and-drop sequence goes automatically."},
    {"text": "There will be 4 different types of claw, and sometimes it will be easier to pick-and-drop the ball with them, and sometimes harder: some grips are weaker on average, some are stronger. Also some claws have little variability in their strenght, so they are more predictable while others have more variability, so they are less predictable."},
    {"text": "You will be able to tell different grip types by the color of the claw in the screen."},
    {"text": "Before the main part of the experiment, you will have a chance to do a number of training trials"},
    {"text": "Thank you for your time and good luck!"},
    {"text": "Press red button to begin the training trials."},
]

# training instructions
PRE_TRAINING = [
    {"text": "Welcome to the training phase of the experiment."},
    {"text": "In this part, you will get familiar with the claw machine and how different types of grips behave."},
    {"text": "You will see claws of different colors, each representing a different type of grip:"},
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

# post training: main-experiment instructions, one string per screen
POST_TRAINING_SCREENS = [
    (
        "Now that you've completed the training phase, you will proceed to the main experiment.\n\n"
        "Please read these instructions carefully.\n\n"
        "On each trial, you will have to choose a claw between two options. If you then manage to pick and drop the ball into the box, you will get the corresponding reward. If not - you won't lose any points.\n\n"
        "Different claw types bring different reward points.\n\n"
        "Later, reward points will be converted into money - with £1 every 200 points."
    ),
    (
        "After your choice and before your action, you will be asked TWO separate questions:\n\n"
        "QUESTION #1: \"How sure are you that you made the best decision for your reward score?\"\n\n"
        "QUESTION #2: \"How sure are you that you will be able to pick and drop the ball successfully in the upcoming trial?\"\n\n"
        "Please note the difference between these two questions:\n"
        "The first one refers to your confidence in choosing the claw that is the best for your OVERALL reward score, considering both the potential reward points AND your chance of successfully picking and dropping it.\n\n"
        "The second one is about your confidence in being able to physically pick and drop the ball with the chosen claw into the box in this particular trial (regardless of the potential reward).\n\n"
        "These questions are NOT the same questions with different wording. Please pay attention to them."
    ),
    (
        "While it's not impossible that confidence ratings for both questions in some trials might coincide, we urge you to consider them separately, and as carefully as possible.\n\n"
        "Please take your time and think about your confidence ratings in each case VERY CAREFULLY.\n\n"
        "You will need to report your confidence on a scale from 1 (very unsure) to 6 (very sure), by moving the joystick up/down and confirming with the red button. Please choose lower confidence ratings if you feel less confident than in previous trials and higher confidence ratings when you feel more confident than in previous trials, for this particular confidence question."
    ),
    (
        "NOTE: Always choosing the same confidence ratings in all cases is uninformative for us. We will be monitoring your confidence ratings throughout the experiment. If we detect that you're clearly selecting confidence ratings without thoughtful consideration, this may affect your eligibility for continued participation.\n\n"
        "After you have submitted your confidence reports, you'll have a chance to pick and drop the ball with the claw you've chosen. Confidence ratings are not used to determine your bonus points, only the success in dropping the ball into the box on each trial determines bonus points."
    ),
]


class LEDController:
    """
    Serial link to the LED Arduino. If the USB device drops out
    (Errno 5 / Input/output error), it closes the dead handle and
    reopens the port on the next command instead of failing forever.
    """
    def __init__(self, port="/dev/led_controller", baud=9600, timeout=0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self._last_reconnect_attempt = 0.0
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud,
                                     timeout=self.timeout,
                                     write_timeout=1)
            time.sleep(2)  # opening the port resets the Arduino; wait for setup()
            self.ser.reset_input_buffer()
            print(f"[LED] Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            self.ser = None
            print(f"[LED] Could not connect to Arduino on {self.port}: {e}")
            return False

    def _drop(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _send(self, payload: bytes, expect: str):
        for attempt in range(3):
            if self.ser is None or not self.ser.is_open:
                now = time.time()
                if now - self._last_reconnect_attempt < 3.0:
                    print(f"[LED] Skipped {payload!r}: reconnect throttled")
                    return False
                self._last_reconnect_attempt = now
                if not self._connect():
                    return False
            try:
                self.ser.reset_input_buffer()      # drop stale replies / READY
                self.ser.write(payload)
                self.ser.flush()
                reply = self.ser.readline().decode(errors="ignore").strip()
            except (serial.SerialException, OSError) as e:
                print(f"[LED] Serial error ({e}); reconnecting...")
                self._drop()
                continue
            if reply == expect:
                return True
            print(f"[LED] Expected {expect!r}, got {reply!r} for {payload!r} (attempt {attempt+1})")
        return False

    def set_grip(self, grip_type: str):
        self._send(f"SET_GRIP:{grip_type}\n".encode(), f"OK:{grip_type}")

    def off(self):
        self._send(b"OFF\n", "OK:off")

    def close(self):
        if self.ser:
            try:
                self.ser.write(b"OFF\n")
                self.ser.flush()
            except Exception:
                pass
            self._drop()


class RedButtonGate:
    """
    Accepts a red-button press only if the button has been quiet for
    `quiet` seconds (since the screen appeared or since the last press).
    Every press, accepted or not, restarts the quiet timer, so double
    presses and held buttons can't skip a screen.
    """
    def __init__(self, claw_ctl, quiet=0.6):
        self.ev = claw_ctl.ctl.red_button_event
        self.quiet = quiet
        self.ev.clear()
        self.last = time.time()

    def pressed(self):
        if not self.ev.is_set():
            return False
        self.ev.clear()
        now = time.time()
        ok = (now - self.last) >= self.quiet
        self.last = now
        return ok


async def wait_for_red(claw_ctl, quiet=0.6):
    gate = RedButtonGate(claw_ctl, quiet)
    while True:
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)
        pygame.event.pump()
        if gate.pressed():
            return
        await asyncio.sleep(0.01)


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
    automatic_mode=True,
    is_training=False,
    total_training_trials=2,
    total_trials=1,
    delay=2000
):
    pygame.event.clear()
    pygame.font.init()
    screen.fill((255, 255, 255))
    font = F(36)

    if is_training:
        msg = "Training trials. Press Red Button to continue"
        draw_legend(screen)
    else:
        if automatic_mode:
            msg = f"Trial {trial_number}/{total_trials}. Press Red Button to continue"
        else:
            msg = f"Action Selection Trial {trial_number}/{total_training_trials}. Red Button to continue"

    surf = font.render(msg, True, (0, 0, 0))
    rect = surf.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
    screen.blit(surf, rect)

    pygame.display.flip()

    await wait_for_red(claw_ctl)

    screen.fill((255,255,255))
    pygame.display.flip()


def draw_small_claw(screen, color, topleft):
    x0, y0 = topleft
    cord_len = P(15)
    bar_w, bar_h = P(20), P(4)
    pr_w, pr_h = P(3), P(12)

    cx = x0 + bar_w // 2
    pygame.draw.line(screen, 'black', (cx, y0), (cx, y0 + cord_len), 2)

    bar_y = y0 + cord_len
    pygame.draw.rect(screen, color, (x0, bar_y, bar_w, bar_h))

    prong_y = bar_y + bar_h
    pygame.draw.rect(screen, color, (x0, prong_y, pr_w, pr_h))
    pygame.draw.rect(screen, color, (x0 + bar_w - pr_w, prong_y, pr_w, pr_h))

    return bar_w


async def display_and_wait(screen, claw_ctl=None, render_items=None,
                           title=None, color=(0,0,0), total_reward=0):

    pygame.event.clear()
    pygame.font.init()
    screen.fill((255,255,255))
    monetary_reward = total_reward / 200

    if title is not None:
        font = F(36)
        surf = font.render(title, True, color)
        rect = surf.get_rect(center=(screen.get_width() / 2,
                                      screen.get_height() / 2))
        screen.blit(surf, rect)

    elif render_items:
        font = F(26)
        max_width = screen.get_width() - P(80)
        # height of the whole block, so it can be centred vertically
        total_h = 0
        for item in render_items:
            total_h += len(await wrap_text(item.get('text', ''), font, max_width)) * font.get_linesize()
            total_h += font.get_linesize() * 1.2
        y_offset = max(P(30), (screen.get_height() - total_h) // 2)

        for item in render_items:
            paragraph  = item.get('text', '')
            text_color = item.get('color', color)
            draw_icon  = item.get('icon', False)

            wrapped = await wrap_text(paragraph, font, max_width)
            for line in wrapped:
                if draw_icon:
                    cname = line.split()[0]
                    crgb  = text_color

                    before = ""
                    after  = line[len(cname):]

                    x = (screen.get_width() - font.size(line)[0]) // 2
                    y = y_offset

                    if before:
                        s0 = font.render(before, True, (0,0,0))
                        screen.blit(s0, (x, y))
                        x += s0.get_width()

                    icon_y = y + (font.get_height() - P(15 + 4 + 12)) // 2
                    claw_w = draw_small_claw(screen, crgb, (x, icon_y))
                    x += claw_w + P(8)

                    s1 = font.render(cname, True, crgb)
                    screen.blit(s1, (x, y))
                    x += s1.get_width()

                    if after.strip():
                        s2 = font.render(after, True, (0,0,0))
                        screen.blit(s2, (x, y))
                else:
                    text_surface = font.render(line, True, text_color)
                    text_rect    = text_surface.get_rect(
                                    midtop=(screen.get_width()/2, y_offset)
                                )
                    screen.blit(text_surface, text_rect)

                y_offset += font.get_linesize()

            y_offset += font.get_linesize() * 1.2

    pygame.display.flip()
    await wait_for_red(claw_ctl)

    screen.fill((255,255,255))
    pygame.display.flip()


def _paragraph_color(p):
    # same colors as the two rating screens
    if p.startswith("QUESTION #1"):
        return (0, 128, 255)
    if p.startswith("QUESTION #2"):
        return (255, 100, 0)
    return (0, 0, 0)


async def show_instruction_screen_with_text(screen, claw_ctl, text,
                                            footer="Press the red button to continue."):
    pygame.font.init()
    w, h = screen.get_size()
    margin = P(50)
    max_width = w - 2 * margin
    paragraphs = text.split("\n") + ["", footer]

    # largest font (same style as the other screens) where everything fits
    for size in range(P(26), P(15) - 1, -1):
        font = pygame.font.Font(None, size)
        line_h = int(font.get_linesize() * 1.3)
        gap_h = int(font.get_linesize() * 0.9)
        rows = []
        for p in paragraphs:
            if not p.strip():
                rows.append((None, None))
                continue
            col = _paragraph_color(p)
            for line in await wrap_text(p, font, max_width):
                rows.append((line, col))
        total = sum(line_h if line is not None else gap_h for line, _ in rows)
        if total <= h - 2 * margin:
            break

    screen.fill((255, 255, 255))
    y = (h - total) // 2          # center the whole block vertically
    for line, col in rows:
        if line is None:
            y += gap_h
            continue
        surf = font.render(line, True, col)
        screen.blit(surf, surf.get_rect(midtop=(w // 2, y)))
        y += line_h

    pygame.display.flip()
    await wait_for_red(claw_ctl)

    screen.fill((255, 255, 255))
    pygame.display.flip()


def draw_legend(screen, training_results=None, per_claw_total=None):
    font = F(24)
    y, x = P(20), P(10)
    rows = [
        ("Weaker, Less Variable",   "narrow_low",  (30, 144, 255)),
        ("Weaker, More Variable",   "wide_low",    (50, 205, 50)),
        ("Stronger, More Variable", "wide_high",   (255, 215, 0)),
        ("Stronger, Less Variable", "narrow_high", (220, 20, 60)),
    ]
    for name, grip_type, col in rows:
        label = name
        # during training: successes so far / trials per claw (e.g. 3/20)
        if training_results is not None and per_claw_total is not None:
            succ = training_results.get(grip_type, {}).get("successes", 0)
            label = f"{name}   {succ}/{per_claw_total}"
        pygame.draw.circle(screen, col, (x + P(10), y + P(10)), P(10))
        txt = font.render(label, True, (0,0,0))
        screen.blit(txt, (x + P(25), y))
        y += P(25)


def draw_grip_debug(screen, grip_sampled=None, force_mu=None, force_sigma=None):
    font = F(26)
    debug_x = P(10)
    debug_y = P(10)
    debug_color = (30, 30, 200)
    line_h = P(24)

    if force_mu is not None:
        surf = font.render(f"\u03bc (mu): {force_mu:.2f}", True, debug_color)
        screen.blit(surf, (debug_x, debug_y))
        debug_y += line_h
    if force_sigma is not None:
        surf = font.render(f"\u03c3 (sigma): {force_sigma:.2f}", True, debug_color)
        screen.blit(surf, (debug_x, debug_y))
        debug_y += line_h
    if grip_sampled is not None:
        surf = font.render(f"Sampled force: {grip_sampled:.2f}", True, debug_color)
        screen.blit(surf, (debug_x, debug_y))


def draw_reward_display(screen, grip_distribution_type, total_reward, training_mode=False,
                        grip_sampled=None, force_mu=None, force_sigma=None):
    font = F(24)
    W = screen.get_width()
    reward_y = P(10)

    large_font_size = 36
    large_font = F(large_font_size)

    claw_text = "You are playing now with this claw:"
    claw_surface = large_font.render(claw_text, True, (200, 200, 200))
    claw_rect = claw_surface.get_rect()
    claw_rect.centerx = W // 2
    claw_rect.top = P(150)
    screen.blit(claw_surface, claw_rect)

    if not training_mode:
        potential_reward = rewards[grip_distribution_type] if grip_distribution_type in rewards else 0
        potential_text = f"Potential Reward in This Trial: {potential_reward:.1f}"
        total_text = f"Total Reward: {total_reward:.1f}"
        potential_surface = font.render(potential_text, True, (0, 0, 0))
        total_surface = font.render(total_text, True, (0, 0, 0))
        screen.blit(potential_surface, potential_surface.get_rect(topright=(W - P(10), reward_y)))
        screen.blit(total_surface, total_surface.get_rect(topright=(W - P(10), reward_y + P(25))))


def draw_training_claw_counter(screen, current_claw, total_claws,
                                grip_sampled=None, force_mu=None, force_sigma=None):
    font = F(24)
    large_font_size = 36
    large_font = F(large_font_size)

    W = screen.get_width()
    claw_text = "You are playing now with this claw:"
    claw_surface = large_font.render(claw_text, True, (200, 200, 200))
    claw_rect = claw_surface.get_rect()
    claw_rect.centerx = W // 2
    claw_rect.top = P(150)
    screen.blit(claw_surface, claw_rect)

    instruction1_text = "Move <- and -> the joystick to control the claw horizontally"
    instruction1_surface = font.render(instruction1_text, True, (0, 0, 0))
    instruction1_rect = instruction1_surface.get_rect(topright=(W - P(10), P(10)))
    screen.blit(instruction1_surface, instruction1_rect)

    instruction2_text = "Press red button for the claw to go down and pick the object"
    instruction2_surface = font.render(instruction2_text, True, (0, 0, 0))
    instruction2_rect = instruction2_surface.get_rect(topright=(W - P(10), P(40)))
    screen.blit(instruction2_surface, instruction2_rect)

    counter_text = f"Claw {current_claw}/{total_claws}"
    text_surface = font.render(counter_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(topright=(W - P(10), P(70)))
    screen.blit(text_surface, text_rect)


def draw_tag_popup(screen, text, padding=20, alpha=200, font_size=28):
    if screen is None:
        return
    padding = P(padding)
    font = F(font_size)

    lines = text.splitlines()
    rendered = [font.render(line, True, (255, 255, 255)) for line in lines]
    max_w = max(s.get_width() for s in rendered)
    total_h = sum(s.get_height() for s in rendered) + (len(rendered)-1) * 6

    popup_w = max_w + padding*2
    popup_h = total_h + padding*2
    screen_w, screen_h = screen.get_width(), screen.get_height()
    popup_x = (screen_w - popup_w) // 2
    popup_y = (screen_h - popup_h) // 2

    surf = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
    rect_color = (0, 0, 0, alpha)
    pygame.draw.rect(surf, rect_color, surf.get_rect(), border_radius=12)
    pygame.draw.rect(surf, (255,255,255,50), surf.get_rect(), width=2, border_radius=12)
    y_offset = padding
    for r in rendered:
        tx = (popup_w - r.get_width()) // 2
        surf.blit(r, (tx, y_offset))
        y_offset += r.get_height() + 6

    screen.blit(surf, (popup_x, popup_y))


async def show_popup(screen, text, duration=2.0, fps=30, claw_ctl=None):
    if screen is None:
        await asyncio.sleep(duration)
        return

    start = time.time()
    period = 1.0 / float(max(1, fps))
    try:
        while True:
            now = time.time()
            if now - start >= duration:
                break
            try:
                if claw_ctl is not None and hasattr(claw_ctl, 'ctl'):
                    rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)
            except Exception:
                pass
            draw_tag_popup(screen, text)
            pygame.display.flip()
            pygame.event.pump()
            await asyncio.sleep(period)
    except Exception as e:
        print(f"[WARN] show_popup failed: {e}")


async def display_claw_selection(screen, first_type, second_type, claw_ctl, rewards):
    claw_ctl.ctl.ui_enabled = True
    claw_ctl.ctl.axis_enabled = False

    gate = RedButtonGate(claw_ctl)
    while not claw_ctl.ctl.ui_nav_queue.empty():
        claw_ctl.ctl.ui_nav_queue.get_nowait()

    current_selection = None
    clock = pygame.time.Clock()
    font = F(36)
    small_font = F(24)

    dist = {
        "wide_high": f"High force\nHigh variability\nReward: {rewards['wide_high']}",
        "narrow_high": f"High force\nLow variability\nReward: {rewards['narrow_high']}",
        "wide_low":  f"Low force\nHigh variability\nReward: {rewards['wide_low']}",
        "narrow_low":f"Low force\nLow variability\nReward: {rewards['narrow_low']}"
    }

    def draw_claw(surface, cx, cy, ctype):
        claw_colors = {
            "narrow_low": (30, 144, 255),
            "wide_low":   (50, 205, 50),
            "wide_high":  (255, 215, 0),
            "narrow_high":(220, 20, 60)
        }
        color = claw_colors.get(ctype, (100, 100, 100))
        base_w, arm_l, arm_w, cord_l = P(60), P(50), P(10), P(100)

        pygame.draw.line(surface, (0,0,0),
                         (cx, cy - cord_l),
                         (cx, cy - arm_w//2), max(2, P(2)))
        base_rect = pygame.Rect(cx - base_w//2,
                                cy - arm_w//2,
                                base_w, arm_w)
        pygame.draw.rect(surface, color, base_rect)
        lx, rx = cx - base_w//2, cx + base_w//2
        sy = cy
        ey = cy + arm_l
        pygame.draw.line(surface, color, (lx, sy), (lx, ey), arm_w)
        pygame.draw.line(surface, color, (rx, sy), (rx, ey), arm_w)
        for x in (lx, rx):
            pygame.draw.circle(surface, color, (x, ey), P(4))

    while True:
        screen.fill((255,255,255))
        w, h = screen.get_size()

        normal, selected = P(200), P(240)
        cy_box = int(h * 0.48)

        def box(cx, big):
            size = selected if big else normal
            r = pygame.Rect(0, 0, size, size)
            r.center = (cx, cy_box)
            return r

        left  = box(int(w * 0.30), current_selection == 0)
        right = box(int(w * 0.70), current_selection == 1)

        highlight = (50,255,255)
        pygame.draw.rect(screen,
                        highlight if current_selection == 0 else (0, 0, 0),
                        left,  P(4) if current_selection == 0 else P(2))
        pygame.draw.rect(screen,
                        highlight if current_selection == 1 else (0, 0, 0),
                        right, P(4) if current_selection == 1 else P(2))

        draw_claw(screen, left.centerx, left.centery + P(10), first_type)
        draw_claw(screen, right.centerx, right.centery + P(10), second_type)

        lines1 = dist[first_type].split('\n')
        lines2 = dist[second_type].split('\n')
        for i, line in enumerate(lines1):
            txt = small_font.render(line, True, (0,0,0))
            screen.blit(txt,
                        (left.centerx - txt.get_width()//2,
                         left.bottom + P(15) + i * small_font.get_linesize()))
        for i, line in enumerate(lines2):
            txt = small_font.render(line, True, (0,0,0))
            screen.blit(txt,
                        (right.centerx - txt.get_width()//2,
                         right.bottom + P(15) + i * small_font.get_linesize()))

        instr = font.render("Choose a claw (Left/Right to choose, Red button to confirm)", True, (0,0,0))
        screen.blit(instr, instr.get_rect(center=(w // 2, int(h * 0.15))))

        pygame.display.flip()

        try:
            nav = claw_ctl.ctl.ui_nav_queue.get_nowait()
            if nav == 'Key.left':
                current_selection = 0
            elif nav == 'Key.right':
                current_selection = 1
        except queue.Empty:
            pass

        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)

        if gate.pressed() and current_selection is not None:
            break

        await asyncio.sleep(0.02)
        clock.tick(60)

    claw_ctl.ctl.ui_enabled = False
    claw_ctl.ctl.axis_enabled = True

    return first_type if current_selection == 0 else second_type


async def display_rating(screen, claw_ctl,
                         question_lines, scale_texts,
                         question_type, trial):
    pygame.font.init()
    claw_ctl.ctl.ui_enabled   = True
    claw_ctl.ctl.axis_enabled = False
    gate = RedButtonGate(claw_ctl)
    while not claw_ctl.ctl.ui_nav_queue.empty():
        claw_ctl.ctl.ui_nav_queue.get_nowait()

    header_text  = ("BALL SELECTION CONFIDENCE"
                    if question_type == "selection"
                    else "ACTION OUTCOME CONFIDENCE")
    header_color = (0, 128, 255) if question_type == "selection" else (255, 100, 0)
    trial_text   = f"Trial {trial}."
    header_font  = F(40)
    font         = F(32)
    label_font   = F(28)
    clock        = pygame.time.Clock()
    current      = random.randint(0, len(scale_texts) - 1)
    locked_in    = False
    locked_value = None

    def draw():
        nonlocal locked_in, locked_value
        screen.fill((255, 255, 255))
        w, h = screen.get_size()
        block = (header_font.get_linesize() + P(10) + font.get_linesize() + P(10)
                 + len(question_lines) * (font.get_linesize() + P(5)) + P(20)
                 + len(scale_texts) * (label_font.get_linesize() + P(15)))
        y = max(P(50), (h - P(60) - block) // 2)

        surf = header_font.render(header_text, True, header_color)
        screen.blit(surf, surf.get_rect(center=(w // 2, y)))
        y += header_font.get_linesize() + P(10)

        surf = font.render(trial_text, True, (0, 0, 0))
        screen.blit(surf, surf.get_rect(center=(w // 2, y)))
        y += font.get_linesize() + P(10)

        for line in question_lines:
            surf = font.render(line, True, (0, 0, 0))
            screen.blit(surf, surf.get_rect(center=(w // 2, y)))
            y += font.get_linesize() + P(5)
        y += P(20)

        for idx, label in enumerate(scale_texts):
            col = (0, 0, 255) if idx == current else (0, 0, 0)
            surf = label_font.render(label, True, col)
            screen.blit(surf, surf.get_rect(center=(w // 2, y)))
            y += label_font.get_linesize() + P(15)

        prompt = "Navigate with Up/Down, then press Red Button" if not locked_in else "Press Red again to continue"
        instr = label_font.render(prompt, True, (0, 0, 0))
        screen.blit(instr, instr.get_rect(center=(w // 2, h - P(40))))
        pygame.display.flip()

    draw()

    while True:
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

        if gate.pressed():
            locked_value = current + 1
            break

        draw()
        await asyncio.sleep(0.01)
        clock.tick(60)

    claw_ctl.ctl.ui_enabled   = False
    claw_ctl.ctl.axis_enabled = True
    return locked_value


def draw_claw(surface, cx, cy, ctype):
    claw_colors = {
        "narrow_low": (30, 144, 255),
        "wide_low":   (50, 205, 50),
        "wide_high":  (255, 215, 0),
        "narrow_high":(220, 20, 60)
    }
    color = claw_colors.get(ctype, (100, 100, 100))
    base_w, arm_l, arm_w, cord_l = P(60), P(50), P(10), P(100)

    pygame.draw.line(surface, (0,0,0),
                     (cx, cy - cord_l),
                     (cx, cy - arm_w//2), max(2, P(2)))
    base_rect = pygame.Rect(cx - base_w//2,
                            cy - arm_w//2,
                            base_w, arm_w)
    pygame.draw.rect(surface, color, base_rect)
    lx, rx = cx - base_w//2, cx + base_w//2
    sy = cy
    ey = cy + arm_l
    pygame.draw.line(surface, color, (lx, sy), (lx, ey), arm_w)
    pygame.draw.line(surface, color, (rx, sy), (rx, ey), arm_w)
    for x in (lx, rx):
        pygame.draw.circle(surface, color, (x, ey), P(4))


def read_claw_position(claw_ctl):
    """Latest measured X-Carve position (x, y), or None if not known yet."""
    node = getattr(claw_ctl, 'ctl', None)
    x = getattr(node, 'current_x', None) if node is not None else None
    y = getattr(node, 'current_y', None) if node is not None else None
    if x is None or y is None:
        return None
    try:
        return (float(x), float(y))
    except (TypeError, ValueError):
        return None


def _xy(pos):
    if pos is None:
        return None, None
    return pos[0], pos[1]


def build_training_summary(records, participant_id):
    rows = []
    for t in records:
        if not t.get('training_mode'):
            continue
        _gp = t.get('grip_params', {}).get('force', {})
        sx, sy = _xy(t.get('start_position'))
        gx, gy = _xy(t.get('grab_position'))
        rows.append({
            'participant_id': participant_id,
            'trial_number': t.get('trial_number'),
            'grip_type': t.get('grip_type'),
            'force_mu': _gp.get('mu'),
            'force_sigma': _gp.get('sigma'),
            'grip_force': t.get('grip_sampled_value'),
            'start_x': sx,
            'start_y': sy,
            'grab_x': gx,
            'grab_y': gy,
            'ir_detection': t.get('ir_detection'),
            'ir_success': t.get('ir_success'),
        })
    return rows


def build_experiment_summary(records, participant_id):
    rows = []
    for t in records:
        if t.get('training_mode'):
            continue
        _gp = t.get('grip_params', {}).get('force', {})
        sx, sy = _xy(t.get('start_position'))
        gx, gy = _xy(t.get('grab_position'))
        rows.append({
            'participant_id': participant_id,
            'trial_number': t.get('trial_number'),
            'options_displayed': t.get('options_displayed'),
            'grip_type': t.get('grip_type'),
            'chosen_reward': t.get('chosen_reward'),
            'force_mu': _gp.get('mu'),
            'force_sigma': _gp.get('sigma'),
            'grip_force': t.get('grip_sampled_value'),
            'selection_confidence': t.get('selection_confidence'),
            'outcome_confidence': t.get('outcome_confidence'),
            'start_x': sx,
            'start_y': sy,
            'grab_x': gx,
            'grab_y': gy,
            'ir_detection': t.get('ir_detection'),
            'ir_success': t.get('ir_success'),
        })
    return rows


def write_json_atomic(path, data):
    """Write to a temp file first, then rename, so a crash mid-write
    never leaves a half-written (corrupt) summary file."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def save_summaries(all_records, participant_id, train_path, exp_path):
    """Rewrite both combined summaries; called after every trial."""
    try:
        write_json_atomic(train_path, build_training_summary(all_records, participant_id))
    except Exception as e:
        print(f"[WARN] failed saving combined training summary: {e}")
    try:
        write_json_atomic(exp_path, build_experiment_summary(all_records, participant_id))
    except Exception as e:
        print(f"[WARN] failed saving combined experiment summary: {e}")


def generate_stratified_forces(mu, sigma, n_samples, q_min=0.001, q_max=0.999):
    """n_samples forces from N(mu, sigma) that cover the whole distribution evenly.

    The probability range [q_min, q_max] is cut into n_samples equal slices
    (strata). One random point is drawn inside each slice and turned into a
    force with the inverse normal CDF, so every part of the bell curve
    (low tail, middle, high tail) gets exactly its share of samples.
    The list is shuffled so the order is still random.
    """
    if n_samples <= 0:
        return []
    if n_samples == 1:
        return [float(mu)]
    dist = NormalDist(mu, sigma)
    width = (q_max - q_min) / n_samples
    forces = [dist.inv_cdf(q_min + (k + random.random()) * width) for k in range(n_samples)]
    random.shuffle(forces)
    return forces


def generate_participant_id(prefix='P', length=8):
    nbytes = max(1, (length + 1) // 2)
    return prefix + secrets.token_hex(nbytes)[:length]


async def run_trial(claw_ctl, screen, grip_type,
                    speed, grip,
                    home_delay=2, open_delay=2,
                    trial_number=None, total_trials=None,
                    training_mode=False, total_reward=0,
                    led=None, IR=None, training_results=None):
    """
    Run one trial while logging telemetry. Returns a trial_record dict.
    """
    # Clear IR detection if available
    if IR is not None:
        IR.clear_detection()

    logger = DataLogger(claw_ctl, sample_interval=0.01)
    logger.log_event('trial_init', {'grip_type': grip_type, 'speed': speed, 'grip_sampled': grip, 'training_mode': training_mode})
    logger.start()

    def _movement_handler(movement, raw_msg=None):
        payload = {
            'movement': movement,
            'timestamp': time.time()
        }
        try:
            if raw_msg is not None and hasattr(raw_msg, 'header'):
                payload['msg_header'] = {
                    'stamp': str(getattr(raw_msg, 'header').stamp) if getattr(raw_msg, 'header', None) else None,
                    'frame_id': getattr(raw_msg, 'header').frame_id if getattr(raw_msg, 'header', None) else None
                }
        except Exception:
            pass

        try:
            logger.log_event('joystick_movement', payload)
        except Exception as e:
            print(f"[WARN] movement logging failed: {e}")

    if hasattr(claw_ctl, 'ctl'):
        claw_ctl.ctl.on_movement = _movement_handler
    else:
        setattr(claw_ctl, 'on_movement', _movement_handler)
    
    # Trajectory: log every measured X-Carve position during the trial
    def _position_handler(msg):
        logger.log_event('position', {
            'x': float(msg.x),
            'y': float(msg.y),
            'z': float(getattr(msg, 'z', 0.0)),
        })

    start_position = None
    if hasattr(claw_ctl, 'ctl'):
        claw_ctl.ctl.on_position = _position_handler
        # where the claw is at the start of the trial
        start_position = read_claw_position(claw_ctl)
        if start_position is not None:
            logger.log_event('start_position', {'x': start_position[0],
                                                'y': start_position[1]})

    grip_params = grip_distribution_types.get(grip_type, {})
    force_mu    = grip_params.get("force", {}).get("mu")
    force_sigma = grip_params.get("force", {}).get("sigma")

    if training_mode:
        if trial_number is not None and total_trials is not None:
            draw_legend(screen, training_results,
                        total_trials // len(grip_distribution_types))
            draw_training_claw_counter(
                screen, trial_number, total_trials,
                grip_sampled=grip, force_mu=force_mu, force_sigma=force_sigma
            )
    else:
        draw_reward_display(
            screen, grip_type, total_reward, training_mode,
            grip_sampled=grip, force_mu=force_mu, force_sigma=force_sigma
        )

    cx, cy = screen.get_width() // 2, screen.get_height() // 2
    draw_claw(screen, cx, cy, grip_type)
    pygame.display.flip()

    # Set LED color for this grip type
    # (in experiment trials it is already lit from the moment the claw was chosen;
    #  sending it again here is harmless)
    if led is not None:
        led.set_grip(grip_type)

    orig_grab      = getattr(claw_ctl, 'grab_sequence', None)
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

    claw_ctl.enable_joystick()
    await wait_for_red(claw_ctl, quiet=1.0)

    # Where the claw goes down: its measured position at the moment of firing
    grab_position = None
    ctl_node = getattr(claw_ctl, 'ctl', None)
    grab_position = read_claw_position(claw_ctl)
    if grab_position is None:
        # fallback: last logged position before firing
        for ev in reversed(getattr(logger, 'events', [])):
            if ev['name'] == 'position':
                grab_position = (ev['info']['x'], ev['info']['y'])
                break
    logger.log_event('fire_pressed', {'grab_x': grab_position[0] if grab_position else None,
                                      'grab_y': grab_position[1] if grab_position else None})

    try:
        claw_ctl.open_claw()
        claw_ctl.claw_down(speed)
        claw_ctl.close_claw(grip)
        claw_ctl.disable_joystick()
        time.sleep(1)
        claw_ctl.claw_up(speed)
        if hasattr(claw_ctl, 'move_to_box'):
            claw_ctl.move_to_box(x=0.0, y=150.0)

        claw_ctl.open_claw()

        # Turn LED off after drop
        if led is not None:
            led.off()

        await asyncio.sleep(open_delay)
        sequence_ok = True
    except Exception as e:
        logger.log_event('run_sequence_exception', {'exc': str(e)})
        # Make sure the LED doesn't stay lit if the sequence failed
        if led is not None:
            led.off()
        sequence_ok = False

    if orig_grab:
        setattr(claw_ctl, 'grab_sequence', orig_grab)
    if orig_move_home:
        setattr(claw_ctl, 'move_home', orig_move_home)
    if orig_open_claw:
        setattr(claw_ctl, 'open_claw', orig_open_claw)

    # IR detection
    ir_value = None
    ir_success = False
    if IR is not None:
        try:
            ir_detection = IR.get_last_detection(max_age=None)
            if ir_detection:
                ir_value = ir_detection
                ir_success = True
                try:
                    await show_popup(screen, f"IR Ball Detected!\n{ir_detection}", duration=2.0, fps=30, claw_ctl=claw_ctl)
                except Exception as e:
                    print(f"[WARN] failed to show IR popup: {e}")
            else:
                ir_value = "No IR detection"
        except Exception as e:
            print(f"[WARN] IR check failed: {e}")
            ir_value = "IR check failed"
    else:
        ir_value = "IR not available"

    # After the IR check: move to a random position for the next trial
    if sequence_ok:
        rand_x = random.uniform(*ARENA_X)
        rand_y = random.uniform(*ARENA_Y)
        logger.log_event('random_reposition', {'x': rand_x, 'y': rand_y})
        try:
            claw_ctl.move_to_box(x=rand_x, y=rand_y)
        except Exception as e:
            logger.log_event('random_reposition_exception', {'exc': str(e)})
    claw_ctl.enable_joystick()

    logger.log_event('trial_end', {'ir_detection': ir_value})
    logger.stop()

    try:
        if hasattr(claw_ctl, 'ctl') and hasattr(claw_ctl.ctl, 'on_movement'):
            claw_ctl.ctl.on_movement = None
            claw_ctl.ctl.on_position = None
        elif hasattr(claw_ctl, 'on_movement'):
            claw_ctl.on_movement = None
    except Exception:
        pass

    original_position = None
    try:
        ctl_node = getattr(claw_ctl, 'ctl', None)
        if ctl_node is not None:
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

    trial_record = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'trial_number': trial_number,
        'total_trials': total_trials,
        'training_mode': bool(training_mode),
        'grip_type': grip_type,
        'grip_params': grip_distribution_types.get(grip_type, {}),
        'grip_sampled_value': grip,
        'total_reward_before': total_reward,
        'logger': logger.as_dict(),
        'original_claw_position': original_position,
        'start_position': start_position,
        'grab_position': grab_position,
        'ir_detection': ir_value,
        'ir_success': ir_success
    }

    return trial_record


async def run_game(screen, claw_ctl, total_trials=50, training_trials=1):
    set_ui_scale(screen)

    # --- Device startup check ---
    print("[STARTUP] Checking devices...")
    for name, path in [("IR detector", "/dev/ir_detector"),
                       ("LED controller", "/dev/led_controller"),
                       ("XCarve", "/dev/xcarve")]:
        if not os.path.exists(path):
            print(f"[WARN] {name} not found at {path}")
        else:
            print(f"[OK] {name} found at {path}")

    # --- Init IR (safe, won't crash game if missing) ---
    IR = None
    try:
        IR = IRDetector(port="/dev/ir_detector", baud=115200, verbose=True)
        IR.start()
        print("[OK] IR detector started")
    except Exception as e:
        print(f"[WARN] IR detector not available: {e}")

    # --- Init LED (safe, won't crash game if missing) ---
    led = None
    try:
        led = LEDController(port="/dev/led_controller")
        print("[OK] LED controller started")
    except Exception as e:
        print(f"[WARN] LED controller not available: {e}")

    def handle_exit(sig, frame):
        print("\n[EXIT] Caught interrupt, cleaning up...")
        if led is not None:
            led.off()
            led.close()
        if IR is not None:
            try:
                IR.stop()
            except Exception:
                pass
        pygame.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    await display_and_wait(screen, claw_ctl, INTRO)

    if not SKIP_TRAINING:
        await display_and_wait(screen, claw_ctl, PRE_TRAINING)

    all_records = []
    # NUM_TRAINING_ROUNDS rounds; in each round every claw appears once, in random order
    training_types = []
    for _ in range(NUM_TRAINING_ROUNDS):
        round_types = list(grip_distribution_types.keys())
        random.shuffle(round_types)
        training_types.extend(round_types)
    if SKIP_TRAINING:
        print("[INFO] SKIP_TRAINING is on: going straight to the experiment")
        training_types = []
    total_training = len(training_types)
    training_results = {g: {"attempts": 0, "successes": 0} for g in grip_distribution_types}

    # Training forces: one pre-made list per claw, as long as the number of times
    # that claw appears in training, then taken one by one (pop) in the loop below
    training_forces = {}
    for g, p in grip_distribution_types.items():
        n_g = training_types.count(g)
        if STRATIFIED_TRAINING:
            training_forces[g] = generate_stratified_forces(p["force"]["mu"], p["force"]["sigma"], n_g)
        else:
            training_forces[g] = [random.gauss(p["force"]["mu"], p["force"]["sigma"]) for _ in range(n_g)]
        print(f"[TRAINING FORCES] {g}: " + ", ".join(f"{f:.2f}" for f in sorted(training_forces[g])))

    participant_id = generate_participant_id()
    session_dir = make_session_dir(participant_id, base_dir='claw_data')
    os.makedirs(session_dir, exist_ok=True)

    # Combined summaries: fixed file names for the whole session, rewritten
    # after every trial so they are complete even if the session is stopped early
    session_ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    train_path = os.path.join(session_dir, f'combined_all_training_{participant_id}_{session_ts}.json')
    exp_path   = os.path.join(session_dir, f'combined_all_trials_{participant_id}_{session_ts}.json')
    print(f"[DATA] Summaries will be updated after every trial in {session_dir}")

    # TRAINING TRIALS
    for i, grip_type in enumerate(training_types, start=1):
        if i == 1:
            # one start screen before the first training claw only
            claw_ctl.ctl.axis_enabled = False
            await display_trial_start(
                screen, claw_ctl,
                trial_number=i,
                is_training=True,
                total_training_trials=total_training,
                total_trials=total_trials,
                automatic_mode=True,
                delay=2000
            )
            claw_ctl.ctl.axis_enabled = True
        else:
            # no screen in between: just clear and go straight to the next claw
            screen.fill((255, 255, 255))

        grip_force = training_forces[grip_type].pop()   # next pre-made force for this claw

        grip_value  = grip_force
        speed_value = grip_distribution_types[grip_type]["rate"]

        trial_record = await run_trial(
            claw_ctl,
            screen,
            grip_type=grip_type,
            speed=speed_value,
            grip=grip_value,
            trial_number=i,
            training_mode=True,
            total_trials=total_training,
            led=led,
            IR=IR,
            training_results=training_results,
        )

        training_results[grip_type]["attempts"] += 1
        if trial_record.get('ir_success'):
            training_results[grip_type]["successes"] += 1

        ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        _, ev_path, _ = save_event_data(trial_record,
                                       participant_id=participant_id,
                                       timestamp=ts,
                                       save_locally=True,
                                       out_dir=session_dir,
                                       filename_prefix=f"train_trial_{i}",
                                       include_samples=False)
        if ev_path:
            print(f"[DATA] Saved training events JSON to {ev_path}")
        all_records.append(trial_record)
        save_summaries(all_records, participant_id, train_path, exp_path)

    # TRAINING COMPLETE
    claw_ctl.ctl.axis_enabled = False

    # LED off while the "training completed" and main-experiment
    # instructions screens are shown
    if led is not None:
        led.off()

    if not SKIP_TRAINING:
        await display_and_wait(
            screen,
            claw_ctl,
            title="Training completed. Press fire to begin the experiment."
        )

    for text in POST_TRAINING_SCREENS:
        await show_instruction_screen_with_text(screen, claw_ctl, text)

    # EXPERIMENT TRIALS
    for trial in range(1, total_trials + 1):
        claw_ctl.ctl.axis_enabled = False

        # LED off on the trial-start screen and during claw selection
        if led is not None:
            led.off()

        await display_trial_start(
            screen, claw_ctl,
            trial_number=trial,
            is_training=False,
            total_training_trials=total_trials,
            total_trials=total_trials,
            automatic_mode=True,
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

        # Light the LEDs in the color of the chosen claw as soon as the choice
        # is confirmed; they stay on through the ratings and the pick-and-drop,
        # and run_trial turns them off after the drop.
        if led is not None:
            led.set_grip(choice)

        selection_confidence = await display_rating(
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
            question_type="selection",
            trial=trial
        )

        outcome_confidence = await display_rating(
            screen,
            claw_ctl,
            question_lines=[
                "How sure are you",
                "that you will drop the ball in the box?"
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

        with claw_ctl.ctl.ui_nav_queue.mutex:
            claw_ctl.ctl.ui_nav_queue.queue.clear()
        screen.fill((255, 255, 255))
        pygame.display.flip()

        grip_params = grip_distribution_types[choice]
        grip_force = random.gauss(
            grip_params["force"]["mu"],
            grip_params["force"]["sigma"]
        )
        grip_value  = grip_force
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
            total_reward=current_total_reward,
            led=led,
            IR=IR,
        )

        trial_record['choice'] = choice
        trial_record['selection_confidence'] = selection_confidence
        trial_record['outcome_confidence'] = outcome_confidence
        trial_record['chosen_reward'] = rewards.get(choice)
        trial_record['options_displayed'] = (first_type, second_type)

        choice_path = save_choice_data(trial_record,
                                    participant_id=participant_id,
                                    out_dir=session_dir,
                                    filename_prefix=f"choice_trial_{trial}")
        print(f"[DATA] Saved choice summary to {choice_path}")

        ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        _, ev_path, _ = save_event_data(trial_record,
                                       participant_id=participant_id,
                                       timestamp=ts,
                                       save_locally=True,
                                       out_dir=session_dir,
                                       filename_prefix=f"exp_trial_{trial}",
                                       include_samples=False)
        if ev_path:
            print(f"[DATA] Saved experiment events JSON to {ev_path}")
        all_records.append(trial_record)
        save_summaries(all_records, participant_id, train_path, exp_path)

        claw_ctl.ctl.axis_enabled = False

    # Final save (already up to date after the last trial; kept as a safety net)
    save_summaries(all_records, participant_id, train_path, exp_path)
    print(f"[DATA] Final summaries saved to {train_path} and {exp_path}")

    claw_ctl.ctl.axis_enabled = False

    # LED off on the final screen
    if led is not None:
        led.off()

    await display_and_wait(
        screen,
        claw_ctl,
        title="Experiment completed",
        total_reward=10
    )

    # Cleanup
    if led is not None:
        led.off()
        led.close()

    if IR is not None:
        try:
            IR.stop()
        except Exception:
            pass