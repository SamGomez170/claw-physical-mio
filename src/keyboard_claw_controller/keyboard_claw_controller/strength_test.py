# strength_test.py
#
# Automatic grip test that maps success as a function of BOTH grip force and
# aiming offset. Each trial:
#   1. picks a ball position, an offset (distance + direction) and a force
#   2. moves the claw to ball + offset and does the grab
#   3. drops in the box and records RFID / IR success
#
# Balls are tested in a fixed cycle (1 -> 2 -> 3 -> 1 ...) so the operator
# always knows where to put the ball; each trial starts right away - no buttons,
# no waiting.
# Force: drawn uniformly over FORCE_RANGE ("uniform") or like the game from a claw
# distribution ("claws").
# Offset: OFFSET_MODE = "rings" -> balanced over OFFSET_RADII with a random direction
#         OFFSET_MODE = "aim_area" -> a random point INSIDE the participants' aim area
#         of that ball (ellipse from aim_areas.json, made by goal_heatmap.py / offsets.py)
#
# After the drop the claw moves to a random spot (as in the game), so every
# trial starts from a random position and approaches its target from there.
#
# Saved per trial (CSV, appended after every trial, + JSON backup at the end):
#   ball position, requested offset, target position, MEASURED position where the
#   claw went down, measured end position, force, success (RFID / IR / both).
import asyncio
import time
import json
import csv
import datetime
import math
import os
import random

import pygame

from .claw_lib import ClawCtl
from .read_rfid import RFIDReader
from .read_IR import IRDetector

# ---------------------------------------------------------------- settings
# 0.080 kg mass whole ball
BALL_POSITIONS = [          # (x, y) of each ball - measure these once with the claw!
    (445, 385),
    (625, 150),
    (750, 450),
]
# TEST_MODE:
#   "uniform" - force drawn uniformly from FORCE_RANGE (maps the whole curve)
#   "claws"   - force drawn like the game: random.gauss(mu, sigma) of a claw
#               from CLAWS, so every trial is "claw X at offset Y"
TEST_MODE = "claws"

FORCE_RANGE = (184.0, 196.0)                         # used in "uniform" mode

# used in "claws" mode - same format as grip_distribution_types in the game
grip_distribution_types = {
    "wide_high": {
        "force": {"mu": 199, "sigma": 2},
        "rate": 255
    },
    "narrow_high": {
        "force": {"mu": 199, "sigma": 0.5},
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
CLAWS = {name: {"mu": c["force"]["mu"], "sigma": c["force"]["sigma"],
                "rate": c.get("rate", 255)}
         for name, c in grip_distribution_types.items()}
# Safety limits: a drawn force outside these is clipped (and flagged in the CSV).
# Set FORCE_SAFE_MAX to the highest force you know is safe for claw and ball.
FORCE_SAFE_MIN = 10.0
FORCE_SAFE_MAX = 1000.0

# OFFSET_MODE:
#   "rings"    - offsets on OFFSET_RADII x DIRECTIONS (as before)
#   "aim_area" - random offset inside the participants' aim area of each ball
OFFSET_MODE = "aim_area"
AIM_AREAS_FILE = "/home/clawMachine/claw_data/offset_data/aim_areas.json"
AREA_COVERAGE = 0.90           # area holding this share of participant grabs (90 %)
# AREA_SAMPLING:
#   "participants" - offsets drawn like the participants aim (more often near their
#                    usual aim point, fewer near the edge) -> the plain success rate
#                    is what participants would get inside the area
#   "uniform"      - every point inside the ellipse equally likely
AREA_SAMPLING = "participants"

OFFSET_RADII = [0, 10, 20, 30]                           # arena units from the ball (rings mode)
DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]    # degrees, 0 = +x, 90 = +y
TRIALS_PER_POSITION = 84                             # 3 balls x 84 = 252 trials
# claws mode: 4 claws x 3 offsets = 12 combinations -> 7 per ball, 21 in total each

# Machine travel limits (GRBL error:15 if a target is outside them).
# The joystick jogs to X780 / Y780, so keep a small margin.
X_LIMITS = (0.0, 775.0)
Y_LIMITS = (0.0, 775.0)

# After each drop the claw goes to a random spot, like in the game, so the
# next trial approaches its target from a random direction.
RANDOM_REPOSITION = True
ARENA_X = (300.0, 750.0)       # same limits as game_flow.py
ARENA_Y = (100.0, 480.0)


# RFID reader: there is no RFID device in the udev rules, and RFIDReader(port=None)
# searches the serial ports itself - it can grab /dev/xcarve or /dev/ir_detector
# and break them. Off by default; success is then judged by the IR sensor alone.
USE_RFID = False
RFID_PORT = "/dev/rfid_reader"     # only used if USE_RFID = True

SPEED = 255
INTER_TRIAL_SECONDS = 15      # motor rest between trials (was 18 s for participants)
COOLDOWN_EVERY = 0           # longer rest every N trials (0 = never)
COOLDOWN_SECONDS = 120
SEED = None                    # set an int to make the schedule reproducible

CSV_FIELDS = [
    'trial_number',
    'ball_index', 'ball_x', 'ball_y',
    'offset_radius', 'offset_angle', 'offset_dx', 'offset_dy',
    'offset_dist', 'offset_mode', 'aim_area_dist',     # exact distance, how the offset was picked,
                                                     # ellipse distance (0 = centre, 1 = edge)
    'target_x', 'target_y',
    'grab_measured_x', 'grab_measured_y',          # where the claw actually went down
    'grab_measured_dx', 'grab_measured_dy',        # measured position - ball
    'grab_measured_dist',
    'end_measured_x', 'end_measured_y',            # claw position at the end of the trial
    'next_start_x', 'next_start_y',                # random spot the claw went to afterwards
    'claw', 'claw_mu', 'claw_sigma', 'rate', 'force_drawn', 'force_clipped',
    'force',
    'success_rfid', 'success_ir', 'success_combined',
    'tag', 'ir_detection',
    'duration', 'timestamp',
]


# ---------------------------------------------------------------- helpers
GOAL_NAMES = {(750, 450): "Goal 1", (625, 150): "Goal 2", (445, 385): "Goal 3"}


def load_aim_areas():
    """Ellipse per ball from aim_areas.json (mean + covariance of participant grab
    offsets). The size is set here from AREA_COVERAGE, so a file made with 80 %
    can be used for a 90 % area."""
    with open(AIM_AREAS_FILE) as f:
        areas = json.load(f)
    r2 = -2 * math.log(1 - AREA_COVERAGE)
    out = {}
    for b_idx, (bx, by) in enumerate(BALL_POSITIONS):
        name = min(GOAL_NAMES.items(), key=lambda g: math.hypot(bx - g[0][0], by - g[0][1]))[1]
        if name not in areas:
            raise SystemExit(f"{AIM_AREAS_FILE} has no area for {name} (ball {b_idx+1})")
        a = areas[name]
        cov = a["cov"]
        # Cholesky of the 2x2 covariance: offset = mean + L @ z
        l11 = math.sqrt(cov[0][0])
        l21 = cov[1][0] / l11
        l22 = math.sqrt(max(cov[1][1] - l21 ** 2, 1e-9))
        out[b_idx] = {"name": name, "mean": a["mean"], "L": (l11, l21, l22), "r2": r2}
    return out


def sample_area_offset(rng, area):
    """Random (dx, dy) inside the ellipse, plus its ellipse distance (0 centre, 1 edge)."""
    k = math.sqrt(area["r2"])
    if AREA_SAMPLING == "uniform":
        rad, ang = k * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        z1, z2 = rad * math.cos(ang), rad * math.sin(ang)
    else:                                            # like the participants, cut at the edge
        while True:
            z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
            if z1 * z1 + z2 * z2 <= area["r2"]:
                break
    l11, l21, l22 = area["L"]
    dx = area["mean"][0] + l11 * z1
    dy = area["mean"][1] + l21 * z1 + l22 * z2
    return dx, dy, math.hypot(z1, z2) / k


def build_schedule():
    """Balls in a FIXED cycle: ball 1, ball 2, ball 3, ball 1, ...
    Within each ball, the offset (radius + direction) and force are random.
    Targets outside the machine limits get a different direction."""
    rng = random.Random(SEED)
    claw_names = list(CLAWS)
    areas = load_aim_areas() if OFFSET_MODE == "aim_area" else None
    if OFFSET_MODE == "aim_area":                      # offset is random -> balance claws only
        combos = [(c, None) for c in claw_names] if TEST_MODE == "claws" else [(None, None)]
    elif TEST_MODE == "claws":
        combos = [(c, r) for c in claw_names for r in OFFSET_RADII]
    else:
        combos = [(None, r) for r in OFFSET_RADII]
    per_ball = []
    for b_idx, (bx, by) in enumerate(BALL_POSITIONS):
        sched = []
        for i in range(TRIALS_PER_POSITION):
            claw, r = combos[i % len(combos)]          # balanced claw (x offset)
            area_d = None
            if OFFSET_MODE == "aim_area":
                for _ in range(200):
                    dx, dy, area_d = sample_area_offset(rng, areas[b_idx])
                    tx, ty = bx + dx, by + dy
                    if X_LIMITS[0] <= tx <= X_LIMITS[1] and Y_LIMITS[0] <= ty <= Y_LIMITS[1]:
                        break
                else:
                    print(f"[SCHEDULE] no valid offset for ball {b_idx+1} - skipped")
                    continue
                dist = math.hypot(dx, dy)
                r = int(round(dist / 10) * 10)          # nearest 10, so old plots still work
                ang = round(math.degrees(math.atan2(dy, dx)) % 360, 1)
            else:
                dirs = [0] if r == 0 else rng.sample(DIRECTIONS, len(DIRECTIONS))
                for ang in dirs:
                    dx = r * math.cos(math.radians(ang))
                    dy = r * math.sin(math.radians(ang))
                    tx, ty = bx + dx, by + dy
                    if X_LIMITS[0] <= tx <= X_LIMITS[1] and Y_LIMITS[0] <= ty <= Y_LIMITS[1]:
                        break
                else:
                    print(f"[SCHEDULE] no valid direction for ball {b_idx+1}, r={r} - skipped")
                    continue
                dist = float(r)
            if claw is None:
                drawn = rng.uniform(*FORCE_RANGE)
                mu = sd = None
                rate = SPEED
            else:
                rate = CLAWS[claw]["rate"]
                mu, sd = CLAWS[claw]["mu"], CLAWS[claw]["sigma"]
                drawn = rng.gauss(mu, sd)              # same as the game
            force = min(max(drawn, FORCE_SAFE_MIN), FORCE_SAFE_MAX)
            sched.append({
                "ball_index": b_idx + 1, "ball": (bx, by),
                "radius": r, "angle": ang, "dx": round(dx, 3), "dy": round(dy, 3),
                "dist": round(dist, 3),
                "area_d": None if area_d is None else round(area_d, 3),
                "target": (round(tx, 3), round(ty, 3)),
                "claw": claw, "claw_mu": mu, "claw_sigma": sd,
                "force_drawn": round(drawn, 4), "force_clipped": force != drawn,
                "force": force, "rate": rate,
            })
        rng.shuffle(sched)            # random offset/force order for this ball
        per_ball.append(sched)

    # interleave: 1, 2, 3, 1, 2, 3, ...
    out = []
    for i in range(max(len(b) for b in per_ball)):
        for b in per_ball:
            if i < len(b):
                out.append(b[i])
    return out


def _read_position(claw_ctl):
    """Latest measured X-Carve position (needs the xcarve_position_callback
    change in claw_lib that stores current_x / current_y).
    Only READS the stored value - it does not spin the ROS node itself, so it
    never competes with claw_lib's own waiting loops."""
    node = getattr(claw_ctl, 'ctl', None)
    if node is None:
        return None, None
    x, y = getattr(node, 'current_x', None), getattr(node, 'current_y', None)
    if x is None or y is None:
        return None, None
    return float(x), float(y)


def _append_to_csv(filepath, rows):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


async def _run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _pump_pygame_events(pump_interval=0.05):
    try:
        while True:
            pygame.event.pump()
            await asyncio.sleep(pump_interval)
    except asyncio.CancelledError:
        return


def _show_ir_popup(screen, ir_detection, duration=1.5):
    screen_w, screen_h = screen.get_size()
    popup_w, popup_h = 420, 160
    popup_x = (screen_w - popup_w) // 2
    popup_y = (screen_h - popup_h) // 2

    overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 120))
    screen.blit(overlay, (0, 0))

    popup_rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
    pygame.draw.rect(screen, (220, 255, 220), popup_rect, border_radius=12)
    pygame.draw.rect(screen, (0, 180, 0), popup_rect, width=3, border_radius=12)

    title_font = pygame.font.Font(None, 46)
    title_surf = title_font.render("IR Ball Detected!", True, (0, 140, 0))
    title_rect = title_surf.get_rect(centerx=popup_rect.centerx, top=popup_rect.top + 20)
    screen.blit(title_surf, title_rect)

    detail_font = pygame.font.Font(None, 28)
    detail_text = str(ir_detection)
    if len(detail_text) > 50:
        detail_text = detail_text[:47] + "..."
    detail_surf = detail_font.render(detail_text, True, (30, 30, 30))
    detail_rect = detail_surf.get_rect(centerx=popup_rect.centerx, top=title_rect.bottom + 14)
    screen.blit(detail_surf, detail_rect)

    pygame.display.flip()
    pygame.time.wait(int(duration * 1000))


async def _countdown(screen, seconds, title):
    print(f"\n[REST] {title}: {seconds}s")
    font_big = pygame.font.Font(None, 72)
    font_small = pygame.font.Font(None, 32)
    for remaining in range(int(seconds), 0, -1):
        pygame.event.pump()
        screen.fill((255, 255, 255))
        screen.blit(font_small.render(title, True, (80, 80, 80)), (200, 220))
        mins, secs = divmod(remaining, 60)
        screen.blit(font_big.render(f"{mins:02d}:{secs:02d}", True, (200, 60, 60)), (300, 300))
        pygame.display.flip()
        await asyncio.sleep(1.0)


def _draw_trial_screen(screen, trial_num, total, t):
    screen.fill((255, 255, 255))
    big = pygame.font.Font(None, 44)
    font = pygame.font.Font(None, 36)
    small = pygame.font.Font(None, 30)
    bx, by = t["ball"]
    if t.get("claw"):
        dist_txt = f"{t['claw']}   (mu = {t['claw_mu']}, sigma = {t['claw_sigma']})"
    else:
        dist_txt = f"uniform {FORCE_RANGE[0]:g} - {FORCE_RANGE[1]:g}"
    force_txt = f"Force  {t['force']:.2f}"
    if t.get("force_clipped"):
        force_txt += f"   (drawn {t['force_drawn']:.2f}, clipped)"
    if t.get("area_d") is not None:
        where = f"inside aim area ({AREA_COVERAGE:.0%}): {t['area_d']:.0%} of the way to the edge"
    else:
        where = f"ring r = {t['radius']}"
    lines = [
        (big,   (0, 0, 0),      f"Trial {trial_num}/{total}"),
        (font,  (13, 54, 107),  force_txt),
        (small, (60, 60, 60),   f"Distribution:  {dist_txt}"),
        (font,  (13, 54, 107),  f"Offset  dx = {t['dx']:+.1f}   dy = {t['dy']:+.1f}   "
                                f"(distance {t['dist']:.1f})"),
        (small, (60, 60, 60),   where),
        (small, (60, 60, 60),   f"Ball {t['ball_index']} at ({bx}, {by})   ->   "
                                f"target ({t['target'][0]:.1f}, {t['target'][1]:.1f})"),
    ]
    y = 160
    for f, col, txt in lines:
        screen.blit(f.render(txt, True, col), (50, y))
        y += 55
    pygame.display.flip()


# ---------------------------------------------------------------- one trial
async def test_single_grip(claw_ctl, t, trial_num, total, rfid_reader, ir_detector, screen):
    bx, by = t["ball"]
    tx, ty = t["target"]
    force_value = t["force"]
    print(f"\n[TEST {trial_num}/{total}] ball {t['ball_index']} ({bx},{by})  "
          f"offset dx={t['dx']:+.1f} dy={t['dy']:+.1f} (dist {t['dist']:.1f})  "
          f"target ({tx:.1f},{ty:.1f})  force {force_value:.3f}  "
          + (f"from {t['claw']} N({t['claw_mu']}, {t['claw_sigma']})" if t.get('claw')
             else f"from uniform {FORCE_RANGE}"))

    _draw_trial_screen(screen, trial_num, total, t)

    # clear sensors
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

    grab_x = grab_y = end_x = end_y = None
    next_x = next_y = None
    start_time = time.time()
    try:
        # 1. go to ball + offset
        await _run_blocking(claw_ctl.move_home, target=(tx, ty))
        await asyncio.sleep(0.5)                       # settle
        grab_x, grab_y = _read_position(claw_ctl)      # where it will go down
        if grab_x is not None:
            print(f"[POS] measured grab position ({grab_x:.2f}, {grab_y:.2f})")

        # 2. grab, carry to the box, release
        await _run_blocking(claw_ctl.grab_sequence, t.get("rate", SPEED), force_value)
        if hasattr(claw_ctl, 'move_to_box'):
            await _run_blocking(claw_ctl.move_to_box, x=0.0, y=150.0)
        await _run_blocking(claw_ctl.open_claw)
        await asyncio.sleep(0.3)
        end_x, end_y = _read_position(claw_ctl)

        # 3. like the game: move to a random spot before the next trial
        if RANDOM_REPOSITION:
            next_x = round(random.uniform(*ARENA_X), 3)
            next_y = round(random.uniform(*ARENA_Y), 3)
            print(f"[POS] random reposition to ({next_x:.1f}, {next_y:.1f})")
            await _run_blocking(claw_ctl.move_home, target=(next_x, next_y))
    except Exception as e:
        print(f"[ERROR] Trial failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    duration = time.time() - start_time

    # RFID
    success_rfid = False
    tag = None
    try:
        if rfid_reader:
            await asyncio.sleep(0.5)
            tag = await _run_blocking(rfid_reader.read_tag, 3.0, False)
            if not tag and hasattr(rfid_reader, 'get_last_tag'):
                tag = await _run_blocking(rfid_reader.get_last_tag, 2.0)
            if tag:
                tag_str = str(tag).upper()
                success_rfid = "NO BALL" not in tag_str and "NO SUCCESS" not in tag_str
            print(("✅" if success_rfid else "❌") + f" RFID: {tag}")
    except Exception as e:
        print(f"[WARN] RFID read error: {e}")

    # IR
    success_ir = False
    ir_detection = None
    try:
        if ir_detector:
            ir_detection = ir_detector.get_last_detection(max_age=None)
            if ir_detection:
                success_ir = True
                print(f"✅ IR: {ir_detection}")
                _show_ir_popup(screen, ir_detection)
            else:
                print("❌ IR: no detection")
    except Exception as e:
        print(f"[WARN] IR read error: {e}")

    gdx = gdy = gdist = None
    if grab_x is not None:
        gdx, gdy = grab_x - bx, grab_y - by
        gdist = math.hypot(gdx, gdy)

    def r3(v):
        return None if v is None else round(v, 3)

    return {
        'trial_number': trial_num,
        'ball_index': t['ball_index'], 'ball_x': bx, 'ball_y': by,
        'offset_radius': t['radius'], 'offset_angle': t['angle'],
        'offset_dx': t['dx'], 'offset_dy': t['dy'],
        'offset_dist': t['dist'], 'offset_mode': OFFSET_MODE, 'aim_area_dist': t.get('area_d'),
        'target_x': tx, 'target_y': ty,
        'grab_measured_x': r3(grab_x), 'grab_measured_y': r3(grab_y),
        'grab_measured_dx': r3(gdx), 'grab_measured_dy': r3(gdy),
        'grab_measured_dist': r3(gdist),
        'end_measured_x': r3(end_x), 'end_measured_y': r3(end_y),
        'next_start_x': next_x, 'next_start_y': next_y,
        'claw': t.get('claw'), 'claw_mu': t.get('claw_mu'), 'claw_sigma': t.get('claw_sigma'),
        'rate': t.get('rate'),
        'force_drawn': t.get('force_drawn'), 'force_clipped': t.get('force_clipped'),
        'force': round(force_value, 4),
        'success_rfid': success_rfid,
        'success_ir': success_ir,
        'success_combined': (success_rfid and success_ir) if USE_RFID else success_ir,
        'tag': str(tag) if tag else None,
        'ir_detection': str(ir_detection) if ir_detection else None,
        'duration': round(duration, 3),
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    }


# ---------------------------------------------------------------- main loop
async def run_grip_strength_test(screen, claw_ctl, ir_detector=None):
    rfid = None
    if USE_RFID:
        rfid = RFIDReader(port=RFID_PORT, baud=115200, timeout=0.1, verbose=True)
        rfid.start()
    else:
        print("[INFO] RFID disabled - success = IR detection")

    owns_ir = ir_detector is None
    if owns_ir:
        ir_detector = IRDetector(port="/dev/ir_detector", baud=115200, verbose=True)
        ir_detector.start()

    pump_task = asyncio.create_task(_pump_pygame_events())

    timestamp_str = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    os.makedirs("claw_data", exist_ok=True)
    csv_path = os.path.join("claw_data", f"grip_offset_test_{timestamp_str}.csv")
    json_path = os.path.join("claw_data", f"grip_offset_test_{timestamp_str}.json")
    print(f"[INFO] CSV: {os.path.abspath(csv_path)}")

    schedule = build_schedule()
    total = len(schedule)
    offs = (f"aim area {AREA_COVERAGE:.0%} ({AREA_SAMPLING}) from {AIM_AREAS_FILE}"
            if OFFSET_MODE == "aim_area" else f"rings {OFFSET_RADII}")
    print(f"[INFO] {total} trials: {len(BALL_POSITIONS)} balls x {TRIALS_PER_POSITION}, "
          f"force mode={TEST_MODE}, offsets: {offs}")
    if TEST_MODE == "claws":
        for name, c in CLAWS.items():
            print(f"        {name:12s} mu={c['mu']}  sigma={c['sigma']}  rate={c['rate']}")

    # check that position logging works before spending 2 hours
    px, py = _read_position(claw_ctl)
    if px is None:
        print("[WARN] No measured claw position available (claw_ctl.ctl.current_x missing). "
              "Trials will still run, but grab_measured_* will be empty - "
              "check the xcarve_position_callback change in claw_lib.py.")
    else:
        print(f"[INFO] Position logging OK, claw at ({px:.1f}, {py:.1f})")

    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 32)
    screen.blit(font.render(f"Grip + offset test: {total} trials", True, (0, 0, 0)), (200, 300))
    pygame.display.flip()
    await asyncio.sleep(2.0)

    all_results = []
    try:
        for trial_num, t in enumerate(schedule, 1):
            result = await test_single_grip(claw_ctl, t, trial_num, total,
                                            rfid, ir_detector, screen)
            if result:
                all_results.append(result)
                _append_to_csv(csv_path, [result])   # saved after EVERY trial
                try:                                 # JSON rewritten after every trial too
                    tmp = json_path + ".tmp"
                    with open(tmp, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    os.replace(tmp, json_path)
                except Exception as e:
                    print(f"[WARN] JSON save failed: {e}")
                print(f"[SAVE] trial {trial_num} saved ({len(all_results)} total)")

            if trial_num < total:
                if COOLDOWN_EVERY and trial_num % COOLDOWN_EVERY == 0:
                    await _countdown(screen, COOLDOWN_SECONDS, "Motor cooldown")
                else:
                    await asyncio.sleep(INTER_TRIAL_SECONDS)

        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"[COMPLETE] JSON backup → {json_path}")

        # terminal summary: success by ball x radius
        radii = sorted({x['offset_radius'] for x in all_results})
        print("\n" + "=" * 60)
        if TEST_MODE == "claws":
            print("SUMMARY   rows = claw, columns = offset radius")
            print("claw          " + "".join(f"r={r:<8}" for r in radii) + "all")
            for name in CLAWS:
                cells = []
                for r in radii + [None]:
                    sub = [x for x in all_results if x['claw'] == name
                           and (r is None or x['offset_radius'] == r)]
                    k = sum(x['success_combined'] for x in sub)
                    cells.append(f"{k}/{len(sub)} {100*k/len(sub):3.0f}%" if sub else "-")
                print(f"{name:13s} " + "  ".join(f"{c:<9}" for c in cells))
            print()
        print("SUMMARY  (both sensors)   rows = ball, columns = offset radius")
        print("=" * 60)
        header = "ball      " + "".join(f"r={r:<7}" for r in radii)
        print(header)
        for b_idx in range(1, len(BALL_POSITIONS) + 1):
            cells = []
            for r in radii:
                sub = [x for x in all_results if x['ball_index'] == b_idx and x['offset_radius'] == r]
                k = sum(x['success_combined'] for x in sub)
                cells.append(f"{k}/{len(sub):<6}" if sub else "-      ")
            print(f"ball {b_idx}    " + " ".join(cells))

        screen.fill((255, 255, 255))
        title_font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 24)
        screen.blit(title_font.render("TEST COMPLETE - see terminal / CSV", True, (0, 128, 0)), (60, 40))
        y = 90
        for r in radii:
            sub = [x for x in all_results if x['offset_radius'] == r]
            k = sum(x['success_combined'] for x in sub)
            rate = 100 * k / len(sub) if sub else 0
            screen.blit(small_font.render(f"offset {r:>2}: {k}/{len(sub)} ({rate:.0f}%)",
                                          True, (0, 0, 0)), (60, y))
            y += 28
        pygame.display.flip()
        await asyncio.sleep(10.0)

    finally:
        if all_results and not os.path.exists(json_path):
            try:
                with open(json_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"[EXIT] partial JSON saved → {json_path}")
            except Exception as e:
                print(f"[WARN] JSON save failed: {e}")
        if rfid is not None:
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