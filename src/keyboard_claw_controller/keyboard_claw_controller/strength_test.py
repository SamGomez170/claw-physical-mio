# strength_test.py
import asyncio
import time
import json
import csv
import datetime
import pygame
import os 

from .claw_lib import ClawCtl
from .read_rfid import RFIDReader
from .read_IR import IRDetector
import random

# Test configuration
# 0.080 kg mass whole ball 
# Each entry: {"mu": <nominal force>, "sigma": <spread>}
TEST_FORCE_PARAMS = [
    {"mu": 192,   "sigma": 0.7},
    {"mu": 192, "sigma": 1.2},
    {"mu": 185,   "sigma": 0.7},
    {"mu": 185,   "sigma": 1.2}
]
TRIALS_PER_FORCE = 10
SPEED = 255
COOLDOWN_SECONDS = 8#600  # 10 min cooldown between force levels

# The start positions the claw will be tested from
START_POSITIONS = [
    (445, 385),
    (625, 150),
    (750, 450)
]

# CSV column order
CSV_FIELDS = [
    'trial_number',
    'force_mu',
    'force_sigma',
    'force',          # actual drawn value (random.gauss result)
    'start_position_x',
    'start_position_y',
    'success_rfid',
    'success_ir',
    'success_combined',
    'tag',
    'ir_detection',
    'duration',
    'timestamp',
]

CSV_SAVE_INTERVAL = 4  # save to CSV every N completed trials


def _append_to_csv(filepath, rows):
    """Append a list of result dicts to the CSV, writing header only if file is new."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


async def _run_blocking(func, *args, **kwargs):
    """Run a blocking function in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def _pump_pygame_events(pump_interval=0.05):
    """Keep pygame responsive by regularly pumping events."""
    try:
        while True:
            pygame.event.pump()
            await asyncio.sleep(pump_interval)
    except asyncio.CancelledError:
        return


def _show_ir_popup(screen, ir_detection, duration=2.0):
    """Display a popup overlay when the IR sensor detects a ball."""
    screen_w, screen_h = screen.get_size()

    # Popup dimensions
    popup_w, popup_h = 420, 160
    popup_x = (screen_w - popup_w) // 2
    popup_y = (screen_h - popup_h) // 2

    # Draw dimmed overlay
    overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    screen.blit(overlay, (0, 0))

    # Draw popup box
    popup_rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
    pygame.draw.rect(screen, (220, 255, 220), popup_rect, border_radius=12)
    pygame.draw.rect(screen, (0, 180, 0), popup_rect, width=3, border_radius=12)

    # Title
    title_font = pygame.font.Font(None, 46)
    title_surf = title_font.render("✅ IR Ball Detected!", True, (0, 140, 0))
    title_rect = title_surf.get_rect(centerx=popup_rect.centerx, top=popup_rect.top + 20)
    screen.blit(title_surf, title_rect)

    # Detection detail
    detail_font = pygame.font.Font(None, 28)
    detail_text = str(ir_detection)
    if len(detail_text) > 50:
        detail_text = detail_text[:47] + "..."
    detail_surf = detail_font.render(detail_text, True, (30, 30, 30))
    detail_rect = detail_surf.get_rect(centerx=popup_rect.centerx, top=title_rect.bottom + 14)
    screen.blit(detail_surf, detail_rect)

    pygame.display.flip()
    pygame.time.wait(int(duration * 1000))


async def _cooldown_countdown(screen, seconds, next_force):
    """Show a live countdown on screen while the motor cools down."""
    print(f"\n[COOLDOWN] Waiting {seconds}s before force {next_force}...")
    font_big   = pygame.font.Font(None, 72)
    font_small = pygame.font.Font(None, 32)
    for remaining in range(seconds, 0, -1):
        pygame.event.pump()
        screen.fill((255, 255, 255))
        screen.blit(font_small.render("Motor cooldown — next force level:", True, (80, 80, 80)), (200, 200))
        screen.blit(font_big.render(f"Force {next_force}", True, (0, 0, 180)), (310, 250))
        screen.blit(font_small.render("Resuming in", True, (80, 80, 80)), (320, 340))
        mins, secs = divmod(remaining, 60)
        screen.blit(font_big.render(f"{mins:02d}:{secs:02d}", True, (200, 60, 60)), (295, 375))
        pygame.display.flip()
        await asyncio.sleep(1.0)
    print("[COOLDOWN] Done — resuming test.")


async def test_single_grip(claw_ctl, force_mu, force_sigma, force_value, trial_num,
                           rfid_reader, ir_detector, screen, start_pos):
    """Run one grip test from a given (x, y) start position."""
    pos_x, pos_y = start_pos
    print(f"\n[TEST {trial_num}] Force drawn: {force_value:.3f}  (μ={force_mu}, σ={force_sigma})  Start pos: ({pos_x}, {pos_y})")

    # UI update
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 36)
    screen.blit(font.render(f"Force: {force_value:.3f}  Trial: {trial_num}", True, (0, 0, 0)), (50, 260))
    screen.blit(font.render(f"μ={force_mu}  σ={force_sigma}  pos: ({pos_x}, {pos_y})", True, (0, 0, 0)), (50, 310))
    pygame.display.flip()

    await asyncio.sleep(1.0)  # let operator prepare / visually confirm

    # --- Clear sensors before the trial ---
    if rfid_reader:
        try:
            with rfid_reader._lock:
                rfid_reader.last_tag = None
                rfid_reader.last_tag_ts = None
        except Exception:
            pass

    if ir_detector:
        try:
            ir_detector.clear_detection()
        except Exception:
            pass

    start_time = time.time()
    try:
        # Move claw to the designated start position first
        if hasattr(claw_ctl, 'move_home'):
            await _run_blocking(claw_ctl.move_home, target=(pos_x, pos_y))
            await asyncio.sleep(0.5)  # wait for claw to settle at start position

        await _run_blocking(claw_ctl.grab_sequence, SPEED, force_value)

        if hasattr(claw_ctl, 'move_to_box'):
            await _run_blocking(claw_ctl.move_to_box, x=0.0, y=150.0)

        await _run_blocking(claw_ctl.open_claw)
        await asyncio.sleep(0.1)

        # Return to the same start position for consistency
        #if hasattr(claw_ctl, 'move_home'):
        #    await _run_blocking(claw_ctl.move_home, target=(pos_x, pos_y))

        #await asyncio.sleep(1.0)

    except Exception as e:
        print(f"[ERROR] Trial failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    duration = time.time() - start_time

    # --- RFID check ---
    success_rfid = False
    tag = None
    try:
        if rfid_reader:
            await asyncio.sleep(0.5)  # give RFID hardware a moment
            tag = await _run_blocking(rfid_reader.read_tag, 3.0, False)
            if not tag and hasattr(rfid_reader, 'get_last_tag'):
                tag = await _run_blocking(rfid_reader.get_last_tag, 2.0)

            if tag:
                tag_str = str(tag).upper()
                if "NO BALL" not in tag_str and "NO SUCCESS" not in tag_str:
                    success_rfid = True
                    print(f"✅ RFID success! Tag: {tag}")
                else:
                    print("❌ RFID failed - no ball string in tag")
            else:
                print("❌ RFID failed - no tag detected")
    except Exception as e:
        print(f"[WARN] RFID read error: {e}")

    # --- IR check ---
    success_ir = False
    ir_detection = None
    try:
        if ir_detector:
            ir_detection = ir_detector.get_last_detection(max_age=None)
            if ir_detection:
                success_ir = True
                print(f"✅ IR success! Detection: {ir_detection}")
                _show_ir_popup(screen, ir_detection)
            else:
                print("❌ IR failed - no detection")
    except Exception as e:
        print(f"[WARN] IR read error: {e}")

    success_combined = success_rfid and success_ir

    return {
        'trial_number':     trial_num,
        'force_mu':         force_mu,
        'force_sigma':      force_sigma,
        'force':            round(force_value, 4),  # actual drawn value
        'start_position_x': pos_x,
        'start_position_y': pos_y,
        'success_rfid':     success_rfid,
        'success_ir':       success_ir,
        'success_combined': success_combined,
        'tag':              str(tag) if tag else None,
        'ir_detection':     str(ir_detection) if ir_detection else None,
        'duration':         round(duration, 3),
        'timestamp':        datetime.datetime.utcnow().isoformat() + 'Z',
    }


async def run_grip_strength_test(screen, claw_ctl, ir_detector=None):
    """Main test loop (async)."""

    # --- Sensors ---
    rfid = RFIDReader(port=None, baud=115200, timeout=0.1, verbose=True)
    rfid.start()

    # Use passed-in IR instance or create a new one
    owns_ir = ir_detector is None
    if owns_ir:
        ir_detector = IRDetector(port="/dev/ir_detector", baud=115200, verbose=True)
        ir_detector.start()

    # Keep pygame responsive
    pump_task = asyncio.create_task(_pump_pygame_events())

    # CSV output in claw_data/ folder
    timestamp_str = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    csv_path = os.path.join("claw_data", f"grip_strength_test_{timestamp_str}.csv")
    os.makedirs("claw_data", exist_ok=True)
    print(f"[INFO] CSV will be saved to: {os.path.abspath(csv_path)}")

    pending_csv_rows = []
    total_completed = 0
    all_results = []

    # Build schedule: each param dict repeated TRIALS_PER_FORCE times, then shuffled
    force_schedule = TEST_FORCE_PARAMS * TRIALS_PER_FORCE
    random.shuffle(force_schedule)

    total_trials = len(force_schedule) * len(START_POSITIONS)

    # Show initial screen
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 32)
    screen.blit(font.render("Grip Strength Test Starting...", True, (0, 0, 0)), (200, 300))
    pygame.display.flip()
    await asyncio.sleep(2.0)

    try:
        trial_num = 0

        for force_idx, force_params in enumerate(force_schedule, 1):
            force_mu    = force_params["mu"]
            force_sigma = force_params["sigma"]

            print("\n" + "=" * 60)
            print(f"FORCE GROUP: mu={force_mu} sigma={force_sigma}  "
                  f"(randomized order {force_idx}/{len(force_schedule)})")
            print("=" * 60)

            force_results = []

            # Each trial draws its own force independently
            for pos in START_POSITIONS:
                trial_num += 1
                force_value = random.gauss(force_mu, force_sigma)  # fresh draw every trial

                result = await test_single_grip(
                    claw_ctl, force_mu, force_sigma, force_value, trial_num,
                    rfid, ir_detector, screen, pos
                )

                if result:
                    force_results.append(result)
                    all_results.append(result)
                    pending_csv_rows.append(result)
                    total_completed += 1

                    if total_completed % CSV_SAVE_INTERVAL == 0:
                        _append_to_csv(csv_path, pending_csv_rows)
                        pending_csv_rows.clear()
                        print(f"[CSV] Saved {total_completed} trials so far → {csv_path}")
                
                # Inter-trial interval
                print(f"[ITI] Waiting 18s (answering questions time) before next trial...")
                await asyncio.sleep(18.0)

                # Cooldown every 15 trials
            #if trial_num % 15 == 0 and trial_num < total_trials:
               # print(f"[COOLDOWN] Reached {trial_num} trials → starting cooldown")
               # await _cooldown_countdown(screen, COOLDOWN_SECONDS, force)

            sr = sum(1 for r in force_results if r['success_rfid'])
            si = sum(1 for r in force_results if r['success_ir'])
            sb = sum(1 for r in force_results if r['success_combined'])
            n = len(force_results)
            print(f"\n[SUMMARY] μ={force_mu} σ={force_sigma}  drawn={force_value:.3f}: "
                  f"RFID {sr}/{n}  IR {si}/{n}  Both {sb}/{n}")

        # Flush remaining rows
        if pending_csv_rows:
            _append_to_csv(csv_path, pending_csv_rows)
            pending_csv_rows.clear()
            print(f"[CSV] Final flush → {csv_path}")

        # JSON backup
        json_path = os.path.join("claw_data", f"grip_strength_test_{timestamp_str}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"[COMPLETE] JSON backup saved → {json_path}")

        # Terminal summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY  (RFID / IR / Both)")
        print("=" * 60)
        for fp in TEST_FORCE_PARAMS:
            mu = fp["mu"]
            sigma = fp["sigma"]
            for pos in START_POSITIONS:
                fd = [r for r in all_results
                      if r['force_mu'] == mu
                      and r['start_position_x'] == pos[0]
                      and r['start_position_y'] == pos[1]]
                if not fd:
                    continue
                n = len(fd)
                print(f"  μ={mu} σ={sigma}  pos ({pos[0]:3d},{pos[1]:3d}): "
                      f"RFID {sum(r['success_rfid'] for r in fd)}/{n}  "
                      f"IR {sum(r['success_ir'] for r in fd)}/{n}  "
                      f"Both {sum(r['success_combined'] for r in fd)}/{n}")

        # Pygame summary screen
        screen.fill((255, 255, 255))
        title_font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)
        y = 40
        screen.blit(title_font.render("TEST COMPLETE — check terminal / CSV for results",
                                      True, (0, 128, 0)), (60, y))
        y += 40
        for fp in TEST_FORCE_PARAMS:
            mu = fp["mu"]
            sigma = fp["sigma"]
            fd = [r for r in all_results if r['force_mu'] == mu]
            sb = sum(1 for r in fd if r['success_combined'])
            n = len(fd)
            rate = (sb / n * 100) if n else 0
            screen.blit(small_font.render(
                f"μ={mu} σ={sigma}: {sb}/{n} both sensors ({rate:.0f}%)",
                True, (0, 0, 0)), (60, y))
            y += 26
        pygame.display.flip()
        await asyncio.sleep(10.0)

    finally:
        # Emergency flush on crash/exit
        if pending_csv_rows:
            try:
                _append_to_csv(csv_path, pending_csv_rows)
                print(f"[CSV] Emergency flush on exit → {csv_path}")
            except Exception as e:
                print(f"[WARN] Emergency flush failed: {e}")

        try:
            rfid.stop()
        except Exception:
            pass
        if owns_ir:
            try:
                ir_detector.stop()
            except Exception:
                pass

        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass





'''
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
from .read_rfid import RFIDReader
from .read_IR import IRDetector

rfid = RFIDReader(port="/dev/rfid_reader", baud=115200, timeout=0.1, verbose=False)
rfid.start()
IR = IRDetector(port="/dev/ir_detector", baud=115200, verbose=True)
IR.start()

SELECTION_COLOR = (0, 128, 0)   # e.g. a green for "selection" 
OUTCOME_COLOR   = (128, 0, 0)   # e.g. a red for "outcome"
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
        "force": {"mu": 192, "sigma": 1.2}, #0.3
        "rate": 255
    }, #yellow
    "narrow_high": {
        "force": {"mu": 192, "sigma": 0.7}, #0.12
        "rate": 255
    },#red
    "wide_low": {
        "force": {"mu": 188, "sigma": 1.2},
        "rate": 255
    }, #green
    "narrow_low": {
        "force": {"mu": 188, "sigma": 0.7},
        "rate": 255
    }#blue
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
#LED
class LEDController:
    def __init__(self, port="/dev/ttyUSB0", baud=9600, timeout=2):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baud, timeout=timeout)
            time.sleep(2)  # wait for Arduino reset after serial connect
            print(f"[LED] Connected to Arduino on {port}")
        except Exception as e:
            print(f"[LED] Could not connect to Arduino: {e}")

    def set_grip(self, grip_type: str):
        """Send grip type name, e.g. 'narrow_low'"""
        if self.ser and self.ser.is_open:
            try:
                cmd = f"SET_GRIP:{grip_type}\n"
                self.ser.write(cmd.encode())
            except Exception as e:
                print(f"[LED] Serial write failed: {e}")

    def off(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"OFF\n")
            except Exception as e:
                print(f"[LED] Serial write failed: {e}")

    def close(self):
        if self.ser:
            self.off()
            self.ser.close()

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
    total_trials = 1,
    delay=2000
):
    # clear out any old events, prep font
    pygame.event.clear()
    pygame.font.init()
    screen.fill((255, 255, 255))
    font = pygame.font.SysFont(None, 36)

    # choose message
    if is_training:
        msg = f"Training Trial {trial_number}/{total_training_trials}. Press Red Button to continue"
        draw_legend(screen)
    else:
        if automatic_mode:
            msg = f"Trial {trial_number}/{total_trials}. Press Red Button to continue"
        else:
            msg = f"Action Selection Trial {trial_number}/{total_training_trials}. Red Button to continue"

    # render & blit
    surf = font.render(msg, True, (0, 0, 0))
    rect = surf.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
    screen.blit(surf, rect)

    pygame.display.flip()

    # wait until the fire button is pressed, but keep UI responsive
    while not claw_ctl.ctl.red_button_event.is_set():
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)
        pygame.event.pump()
        await asyncio.sleep(0.01)
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
            paragraph   = item.get('text', '')
            text_color  = item.get('color', color)
            draw_icon   = item.get('icon', False)

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

                    icon_y = y + (font.get_height() - (15 + 4 + 12)) // 2
                    claw_w = draw_small_claw(screen, crgb, (x, icon_y))
                    x += claw_w + 8

                    s1 = font.render(cname, True, crgb)
                    screen.blit(s1, (x, y))
                    x += s1.get_width()

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

            y_offset += font.get_linesize() * 1.2

    pygame.display.flip()

    while not claw_ctl.ctl.red_button_event.is_set():
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)
        pygame.event.pump()
        await asyncio.sleep(0.01)
    claw_ctl.ctl.red_button_event.clear()

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


def draw_grip_debug(screen, grip_sampled=None, force_mu=None, force_sigma=None):
    """Draw grip force parameters (mu, sigma, sampled value) in the top-left corner."""
    font = pygame.font.Font(None, 26)
    debug_x = 10
    debug_y = 10
    debug_color = (30, 30, 200)
    line_h = 24

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
    font = pygame.font.Font(None, 24)
    reward_x = WIDTH - 300
    reward_y = 10

    large_font_size = 36
    large_font = pygame.font.Font(None, large_font_size)

    claw_text = "You are playing now with this claw:"
    claw_surface = large_font.render(claw_text, True, (200, 200, 200))
    claw_rect = claw_surface.get_rect()
    claw_rect.centerx = WIDTH // 2
    claw_rect.top = 150
    screen.blit(claw_surface, claw_rect)

    if not training_mode:
        potential_reward = rewards[grip_distribution_type] if grip_distribution_type in rewards else 0
        potential_text = f"Potential Reward in This Trial: {potential_reward:.1f}"
        total_text = f"Total Reward: {total_reward:.1f}"
        potential_surface = font.render(potential_text, True, (0, 0, 0))
        total_surface = font.render(total_text, True, (0, 0, 0))
        screen.blit(potential_surface, (reward_x, reward_y))
        screen.blit(total_surface, (reward_x, reward_y + 25))

    # Always draw grip debug info
    #draw_grip_debug(screen, grip_sampled=grip_sampled, force_mu=force_mu, force_sigma=force_sigma)


def draw_training_claw_counter(screen, current_claw, total_claws,
                                grip_sampled=None, force_mu=None, force_sigma=None):
    font = pygame.font.Font(None, 24)
    large_font_size = 36
    large_font = pygame.font.Font(None, large_font_size)

    claw_text = "You are playing now with this claw:"
    claw_surface = large_font.render(claw_text, True, (200, 200, 200))
    claw_rect = claw_surface.get_rect()
    claw_rect.centerx = WIDTH // 2
    claw_rect.top = 150
    screen.blit(claw_surface, claw_rect)

    instruction1_text = "Move <- and -> the joystick to control the claw horizontally"
    instruction1_surface = font.render(instruction1_text, True, (0, 0, 0))
    instruction1_rect = instruction1_surface.get_rect(topright=(WIDTH - 10, 10))
    screen.blit(instruction1_surface, instruction1_rect)

    instruction2_text = "Press red button for the claw to go down and pick the object"
    instruction2_surface = font.render(instruction2_text, True, (0, 0, 0))
    instruction2_rect = instruction2_surface.get_rect(topright=(WIDTH - 10, 40))
    screen.blit(instruction2_surface, instruction2_rect)

    counter_text = f"Claws: {current_claw}/{total_claws}"
    text_surface = font.render(counter_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(topright=(WIDTH - 10, 70))
    screen.blit(text_surface, text_rect)

    # Draw grip debug info in the top-left
    #draw_grip_debug(screen, grip_sampled=grip_sampled, force_mu=force_mu, force_sigma=force_sigma)

# pop-up drawing helper
def draw_tag_popup(screen, text, padding=20, alpha=200, font_size=28):
    if screen is None:
        return
    font = pygame.font.Font(None, font_size)

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

    claw_ctl.ctl.red_button_event.clear()
    while not claw_ctl.ctl.ui_nav_queue.empty():
        claw_ctl.ctl.ui_nav_queue.get_nowait()

    current_selection = None
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

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
        base_w, arm_l, arm_w, cord_l = 60, 50, 10, 100

        pygame.draw.line(surface, (0,0,0),
                         (cx, cy - cord_l),
                         (cx, cy - arm_w//2), 2)
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
            pygame.draw.circle(surface, color, (x, ey), 4)

    while True:
        screen.fill((255,255,255))
        w, h = screen.get_size()

        normal, selected = (200,200), (240,240)
        lx, rx, by = 100, 500, 200

        if current_selection == 0:
            left  = pygame.Rect(lx - 20, by - 20, *selected)
            right = pygame.Rect(rx, by, *normal)
        elif current_selection == 1:
            left  = pygame.Rect(lx, by, *normal)
            right = pygame.Rect(rx - 20, by - 20, *selected)
        else:  # Nothing selected yet
            left  = pygame.Rect(lx, by, *normal)
            right = pygame.Rect(rx, by, *normal)

        highlight = (50,255,255)
        pygame.draw.rect(screen,
                        highlight if current_selection == 0 else (0, 0, 0),
                        left,  4 if current_selection == 0 else 2)
        pygame.draw.rect(screen,
                        highlight if current_selection == 1 else (0, 0, 0),
                        right, 4 if current_selection == 1 else 2)

        draw_claw(screen, left.centerx, left.centery+30, first_type)
        draw_claw(screen, right.centerx, right.centery+30, second_type)

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

        instr = font.render("Choose a claw (Left/Right to choose, Red button to confirm)", True, (0,0,0))
        screen.blit(instr, (w//2 - instr.get_width()//2, 100))

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
        
        if claw_ctl.ctl.red_button_event.is_set():
            claw_ctl.ctl.red_button_event.clear()
            if current_selection is not None:  # <-- add this guard
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
    claw_ctl.ctl.red_button_event.clear()
    while not claw_ctl.ctl.ui_nav_queue.empty():
        claw_ctl.ctl.ui_nav_queue.get_nowait()

    header_text  = ("BALL SELECTION CONFIDENCE"
                    if question_type == "selection"
                    else "ACTION OUTCOME CONFIDENCE")
    header_color = (0, 128, 255) if question_type == "selection" else (255, 100, 0)
    trial_text   = f"Trial {trial}."
    header_font  = pygame.font.SysFont(None, 40)
    font         = pygame.font.SysFont(None, 32)
    label_font   = pygame.font.SysFont(None, 28)
    clock        = pygame.time.Clock()
    current      = random.randint(0, len(scale_texts) - 1)  # random start
    locked_in    = False
    locked_value = None

    def draw():
        nonlocal locked_in, locked_value
        screen.fill((255, 255, 255))
        w, h = screen.get_size()
        y = 50

        surf = header_font.render(header_text, True, header_color)
        screen.blit(surf, surf.get_rect(center=(w // 2, y)))
        y += header_font.get_linesize() + 10

        surf = font.render(trial_text, True, (0, 0, 0))
        screen.blit(surf, surf.get_rect(center=(w // 2, y)))
        y += font.get_linesize() + 10

        for line in question_lines:
            surf = font.render(line, True, (0, 0, 0))
            screen.blit(surf, surf.get_rect(center=(w // 2, y)))
            y += font.get_linesize() + 5
        y += 20

        for idx, label in enumerate(scale_texts):
            col = (0, 0, 255) if idx == current else (0, 0, 0)
            surf = label_font.render(label, True, col)
            screen.blit(surf, surf.get_rect(center=(w // 2, y)))
            y += label_font.get_linesize() + 15

        prompt = "Navigate with Up/Down, then press Red Button" if not locked_in else "Press Red again to continue"
        instr = label_font.render(prompt, True, (0, 0, 0))
        screen.blit(instr, instr.get_rect(center=(w // 2, h - 40)))
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

        if claw_ctl.ctl.red_button_event.is_set():
            claw_ctl.ctl.red_button_event.clear()
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
        base_w, arm_l, arm_w, cord_l = 60, 50, 10, 100

        pygame.draw.line(surface, (0,0,0),
                         (cx, cy - cord_l),
                         (cx, cy - arm_w//2), 2)
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
            pygame.draw.circle(surface, color, (x, ey), 4)

def generate_participant_id(prefix='P', length=8):
    """Return a random hex id like 'P8b3a2f1c' (length is number of chars after prefix)."""
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
    IR.clear_detection()
    if rfid_reader:
        rfid_reader.clear_tag()
    
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

    # Extract mu/sigma for this grip type to display on screen
    grip_params = grip_distribution_types.get(grip_type, {})
    force_mu    = grip_params.get("force", {}).get("mu")
    force_sigma = grip_params.get("force", {}).get("sigma")

    # show UI overlays depending on mode
    if training_mode:
        if trial_number is not None and total_trials is not None:
            draw_legend(screen)
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

    claw_ctl.ctl.red_button_event.clear()  # flush any stale button press from nav screens
    claw_ctl.enable_joystick()


    while not claw_ctl.ctl.red_button_event.is_set():
        rclpy.spin_once(claw_ctl.ctl, timeout_sec=0.0)
        pygame.event.pump()
        await asyncio.sleep(0.01)
    claw_ctl.ctl.red_button_event.clear()

    logger.log_event('fire_pressed')

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
        await asyncio.sleep(open_delay)

        random_home_x = random.randint(440, 755)
        random_home_y = random.randint(150, 500)
        claw_ctl.move_home(target=(random_home_x, random_home_y))

        await asyncio.sleep(0.5)
        claw_ctl.enable_joystick()
    except Exception as e:
        logger.log_event('run_sequence_exception', {'exc': str(e)})
 

    if orig_grab:
        setattr(claw_ctl, 'grab_sequence', orig_grab)
    if orig_move_home:
        setattr(claw_ctl, 'move_home', orig_move_home)
    if orig_open_claw:
        setattr(claw_ctl, 'open_claw', orig_open_claw)

    info_value = None
    ir_value = None
    
    try:
        import time as _time
        with IR._lock:
            last_det = IR.last_detection
            last_ts = IR.last_detection_ts
        
        if last_ts:
            age = _time.time() - last_ts
            print(f"[DEBUG] IR last detection: {last_det}")
            print(f"[DEBUG] IR detection age: {age:.3f}s ago")
        else:
            print(f"[DEBUG] No IR detection timestamp available")
        
        ir_detection = IR.get_last_detection(max_age=None)
        if ir_detection:
            ir_value = ir_detection
            print(f"✅ IR detected: {ir_detection}")
            try:
                await show_popup(screen, f"IR Ball Detected!\n{ir_detection}", duration=2.0, fps=30, claw_ctl=claw_ctl)
            except Exception as e:
                print(f"[WARN] failed to show IR popup: {e}")
        else:
            ir_value = "No IR detection"
            print("❌ No IR detection")
    except Exception as e:
        print(f"[WARN] IR check failed: {e}")
        ir_value = "IR check failed"
    
    try:
        if rfid_reader:
            import time as _time
            with rfid_reader._lock:
                last_tag = rfid_reader.last_tag
                last_tag_ts = rfid_reader.last_tag_ts
            
            if last_tag_ts:
                age = _time.time() - last_tag_ts
                print(f"[DEBUG] RFID last tag: {last_tag}")
                print(f"[DEBUG] RFID tag age: {age:.3f}s ago")
            else:
                print(f"[DEBUG] No RFID tag timestamp available")
            
            tag = rfid_reader.get_last_tag(max_age=None)
            if tag:
                info_value = f"SUCCESSFUL BALL, tag: {tag}"
                print(f"✅ Tag detected: {tag}")

                popup_text = f"Tag detected!\n{tag}"
                try:
                    await show_popup(screen, popup_text, duration=2.0, fps=30, claw_ctl=claw_ctl)
                except Exception as e:
                    print(f"[WARN] failed to show popup: {e}")
            else:
                info_value = "NO BALL SUCCESS"
                print("❌ No RFID tag detected")
    except Exception as e:
        print(f"[WARN] RFID check failed: {e}")
        info_value = "RFID check failed"


    if not info_value:
        info_value = "NO BALL SUCCESS"
        logger.log_event('trial_end', info_value)
        try:
            await show_popup(
                screen,
                "NO TAG\nDETECTED",
                duration=2.0,
                fps=30,
                claw_ctl=claw_ctl
            )
        except Exception:
            pass

    logger.stop()

    try:
        if hasattr(claw_ctl, 'ctl') and hasattr(claw_ctl.ctl, 'on_movement'):
            claw_ctl.ctl.on_movement = None
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
        'rfid': info_value,
        'ir_detection': ir_value
    }

    return trial_record

#30 min - 50 trials a mano
async def run_game(screen, claw_ctl, total_trials=50, training_trials=1):
    await display_and_wait(screen, claw_ctl, INTRO)
    await display_and_wait(screen, claw_ctl, PRE_TRAINING)

    rfid = RFIDReader(port="/dev/rfid_reader", baud=115200, timeout=0.1, verbose=True)
    rfid.start()
    led = LEDController(port="/dev/ttyUSB0")


    all_records = []
    training_types = list(grip_distribution_types.keys())
    random.shuffle(training_types)

    participant_id = generate_participant_id()
    session_dir = make_session_dir(participant_id, base_dir='claw_data')

    # TRAINING TRIALS
    for i, grip_type in enumerate(training_types, start=1):
        claw_ctl.ctl.axis_enabled = False
        await display_trial_start(
            screen, claw_ctl,
            trial_number=i,
            is_training=True,
            total_training_trials=len(training_types),
            total_trials = total_trials,
            automatic_mode=True,
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
        led.set_grip(grip_type)

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
            total_trials = total_trials,
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
        grip_value = grip_force
        speed_value = grip_distribution_types[choice]["rate"]

        claw_ctl.ctl.axis_enabled = True
        current_total_reward = 0
        led.set_grip(choice)

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
        all_records.append(trial_record)

        claw_ctl.ctl.axis_enabled = False

    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')

    def _is_success(rfid_val):
        if not rfid_val:
            return False
        s = str(rfid_val).upper()
        if 'NO' in s and 'NO BALL' in s:
            return False
        if 'NO BALL SUCCESS' in s or 'NO SUCCESS' in s:
            return False
        if 'SUCCESS' in s or 'TAG' in s or 'SUCCESSFUL' in s:
            return True
        return True

    training_trials_list = [t for t in all_records if t.get('training_mode')]
    experiment_trials_list = [t for t in all_records if not t.get('training_mode')]

    training_summary = []
    for t in training_trials_list:
        _gp = t.get('grip_params', {}).get('force', {})
        training_summary.append({
            'participant_id': participant_id,
            'trial_number': t.get('trial_number'),
            'grip_type': t.get('grip_type'),
            'force_mu': _gp.get('mu'),
            'force_sigma': _gp.get('sigma'),
            'grip_force': t.get('grip_sampled_value'),
            'rfid': t.get('rfid'),
            'ir_detection': t.get('ir_detection'),
            'success': _is_success(t.get('rfid')),
        })

    experiment_summary = []
    for t in experiment_trials_list:
        _gp = t.get('grip_params', {}).get('force', {})
        experiment_summary.append({
            'participant_id': participant_id,
            'trial_number': t.get('trial_number'),
            'grip_type': t.get('grip_type'),
            'force_mu': _gp.get('mu'),
            'force_sigma': _gp.get('sigma'),
            'grip_force': t.get('grip_sampled_value'),
            'rfid': t.get('rfid'),
            'ir_detection': t.get('ir_detection'),
            'success': _is_success(t.get('rfid')),
            'selection_confidence': t.get('selection_confidence'),
            'outcome_confidence': t.get('outcome_confidence'),
        })

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
    led.off()
    led.close()
'''