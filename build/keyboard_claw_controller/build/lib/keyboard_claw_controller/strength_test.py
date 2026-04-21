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
    {"mu": 192,   "sigma": 1},
    {"mu": 192, "sigma": 2},
    {"mu": 189,   "sigma": 1},
    {"mu": 189,   "sigma": 2}
]
TRIALS_PER_FORCE = 5
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