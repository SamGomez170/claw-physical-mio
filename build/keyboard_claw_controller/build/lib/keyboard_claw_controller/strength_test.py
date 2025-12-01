# strength_test.py
import asyncio
import time
import json
import datetime
import pygame
import os

from .claw_lib import ClawCtl
from .read_rfid import RFIDReader

# Test configuration
# 0.080 kg mass
TEST_FORCES = [174, 174.5, 175, 175.5]
TRIALS_PER_FORCE = 5
SPEED = 255

async def _run_blocking(func, *args, **kwargs):
    """Run a blocking function in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)

async def _pump_pygame_events(pump_interval=0.05):
    """Keep pygame responsive by regularly pumping events."""
    try:
        while True:
            # pump events so OS doesn't think the window is frozen
            pygame.event.pump()
            await asyncio.sleep(pump_interval)
    except asyncio.CancelledError:
        # Clean exit
        return

async def test_single_grip(claw_ctl, force_value, trial_num, rfid_reader, screen):
    """Run one grip test at center position."""
    print(f"\n[TEST {trial_num}] Testing force: {force_value}")
    # UI update (non-blocking portions)
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 36)
    text = font.render(f"Testing Force: {force_value} - Trial {trial_num}", True, (0, 0, 0))
    screen.blit(text, (50, 300))
    pygame.display.flip()

    await asyncio.sleep(1.0)  # let operator prepare / visually confirm

    start_time = time.time()
    try:
        if rfid_reader:
            try:
                with rfid_reader._lock:
                    rfid_reader.last_tag = None
                    rfid_reader.last_tag_ts = None
            except Exception:
                # safe fallback if internals change
                pass
        # run hardware calls in a thread to avoid blocking asyncio
        await _run_blocking(claw_ctl.grab_sequence, SPEED, force_value)

        if hasattr(claw_ctl, 'move_to_box'):
            await _run_blocking(claw_ctl.move_to_box, x=0.0, y=150.0)
        
        await _run_blocking(claw_ctl.close_claw, 255)

        await _run_blocking(claw_ctl.open_claw)
        await asyncio.sleep(2.0)

        if hasattr(claw_ctl, 'move_home'):
            await _run_blocking(claw_ctl.move_home, target=(700, 450))

        await asyncio.sleep(1)

    except Exception as e:
        print(f"[ERROR] Trial failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    duration = time.time() - start_time

    # RFID reading (also run in thread if blocking)
    success = False
    tag = None
    try:
        if rfid_reader:
            await asyncio.sleep(0.5)  # give RFID hardware a moment
            tag = await _run_blocking(rfid_reader.read_tag, 3.0, False)  # adapt args depending on your API
            if not tag and hasattr(rfid_reader, 'get_last_tag'):
                tag = await _run_blocking(rfid_reader.get_last_tag, 2.0)

            if tag:
                tag_str = str(tag).upper()
                if "NO BALL" not in tag_str and "NO SUCCESS" not in tag_str:
                    success = True
                    print(f"✅ Success! Tag: {tag}")
                else:
                    print("❌ Failed - No ball detected")
            else:
                print("❌ Failed - No tag detected")
    except Exception as e:
        print(f"[WARN] RFID read error: {e}")

    return {
        'trial_number': trial_num,
        'force': force_value,
        'success': success,
        'tag': str(tag) if tag else None,
        'duration': duration,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }

async def run_grip_strength_test(screen, claw_ctl):
    """Main test loop (async)."""
    # Start an RFID reader (assumes RFIDReader has start/stop)
    rfid = RFIDReader(port=None, baud=115200, timeout=0.1, verbose=True)
    rfid.start()

    # Start pygame event pump task so window stays responsive
    pump_task = asyncio.create_task(_pump_pygame_events())

    all_results = []

    # show initial screen
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 32)
    text = font.render("Grip Strength Test Starting...", True, (0, 0, 0))
    screen.blit(text, (200, 300))
    pygame.display.flip()
    await asyncio.sleep(2.0)

    try:
        for force_idx, force in enumerate(TEST_FORCES, 1):
            print("\n" + "="*60)
            print(f"TESTING FORCE: {force} ({force_idx}/{len(TEST_FORCES)})")
            print("="*60)

            force_results = []
            for trial in range(1, TRIALS_PER_FORCE + 1):
                result = await test_single_grip(claw_ctl, force, trial, rfid, screen)
                if result:
                    force_results.append(result)
                    all_results.append(result)

                # short delay between trials
                await asyncio.sleep(3.0)

            successes = sum(1 for r in force_results if r['success'])
            success_rate = (successes / len(force_results)) * 100 if force_results else 0
            print(f"\n[SUMMARY] Force {force}: {successes}/{len(force_results)} successful ({success_rate:.1f}%)")

        # Save results
        os.makedirs("claw_data", exist_ok=True)

        timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        output_file = os.path.join("claw_data", f'grip_strength_test_{timestamp}.json')

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n[COMPLETE] Results saved to {output_file}")

        # print overall summary
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        for force in TEST_FORCES:
            force_data = [r for r in all_results if r['force'] == force]
            successes = sum(1 for r in force_data if r['success'])
            total = len(force_data)
            rate = (successes/total)*100 if total > 0 else 0
            print(f"Force {force:6.2f}: {successes:2d}/{total:2d} ({rate:5.1f}%)")

        # show final summary on the pygame screen
        screen.fill((255, 255, 255))
        font = pygame.font.Font(None, 28)
        y = 50
        text = font.render("TEST COMPLETE - Check terminal for results", True, (0, 128, 0))
        screen.blit(text, (100, y))
        y += 40

        small_font = pygame.font.Font(None, 24)
        for force in TEST_FORCES:
            force_data = [r for r in all_results if r['force'] == force]
            successes = sum(1 for r in force_data if r['success'])
            total = len(force_data)
            rate = (successes/total)*100 if total > 0 else 0
            summary_text = small_font.render(f"Force {force:.1f}: {successes}/{total} ({rate:.0f}%)", True, (0, 0, 0))
            screen.blit(summary_text, (100, y))
            y += 30
        pygame.display.flip()
        await asyncio.sleep(10.0)

    finally:
        # cleanup: stop RFID and cancel pump task
        try:
            rfid.stop()
        except Exception:
            pass

        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
