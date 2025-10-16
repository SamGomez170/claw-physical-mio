# data_logger.py
import threading
import time as _time
import datetime
import json
import os
import queue
import numpy as _np

# ----------------- DataLogger ------------------------------------------------
class DataLogger:
    """
    Background sampler + event logger for ClawCtl/RosClawCtl.
    API:
      logger = DataLogger(claw_ctl, sample_interval=0.01)
      logger.start()
      logger.log_event('name', info={'k':v})
      logger.stop()
      data = logger.as_dict()
    """
    def __init__(self, claw_ctl, sample_interval=0.01):
        self.claw_ctl = claw_ctl
        self.sample_interval = float(sample_interval)
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

        self.samples = []   # [{'t': float, 'state': {...}}, ...]
        self.events  = []   # [{'t': float, 'name': str, 'info': {...}}, ...]

        # Candidate attribute/callable paths to probe.
        # Add or remove strings here to match your RosClawCtl internals.
        self._candidates = [
            'get_pose', 'get_axes', 'position', 'pose', 'current_pose', 'current_position',
            'last_axes', 'last_joy', 'joy_axes', 'axes',
            'ctl.get_pose', 'ctl.current_pose', 'ctl.current_position',
            'ctl.last_axes', 'ctl.ui_nav_queue', 'ctl.red_button_event',
            'ctl.claw_status_event', 'ctl.home_event', 'latest_pose', 'latest_position'
        ]

    def _now(self):
        return _time.monotonic() - self._t0

    def _get_by_path(self, root, path):
        """Walk dotted path and call if callable. Return None on any failure."""
        cur = root
        for part in path.split('.'):
            if cur is None:
                return None
            try:
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    cur = getattr(cur, part)
            except Exception:
                return None
        # if callable attempt to call it (safe)
        try:
            if callable(cur):
                return cur()
        except Exception:
            return None
        return cur

    def _serialize(self, v):
        """Try sensible serializations for numpy, ROS-like messages, queues, objects."""
        # numpy
        try:
            if isinstance(v, _np.generic):
                return v.item()
            if isinstance(v, _np.ndarray):
                return v.tolist()
        except Exception:
            pass
        # ROS-style Position with x,y
        try:
            if hasattr(v, 'x') and hasattr(v, 'y'):
                return {'x': float(v.x), 'y': float(v.y)}
        except Exception:
            pass
        # queue: report size and try to peek last item (non-destructive)
        try:
            if isinstance(v, queue.Queue):
                info = {'size': v.qsize()}
                if hasattr(v, 'queue'):
                    try:
                        arr = list(v.queue)
                        if arr:
                            info['last'] = str(arr[-1])
                    except Exception:
                        pass
                return info
        except Exception:
            pass
        # object with __dict__
        try:
            if hasattr(v, '__dict__'):
                d = {}
                for k, val in vars(v).items():
                    try:
                        if isinstance(val, _np.generic):
                            d[k] = val.item()
                        elif isinstance(val, _np.ndarray):
                            d[k] = val.tolist()
                        else:
                            d[k] = val
                    except Exception:
                        d[k] = str(val)
                return d
        except Exception:
            pass
        # fallback: try jsonable, else string
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    def _snapshot(self):
        """Probe each candidate and return a dict of serializable values."""
        state = {}
        for path in self._candidates:
            try:
                val = self._get_by_path(self.claw_ctl, path)
                if val is None:
                    continue
                state[path] = self._serialize(val)
            except Exception:
                # swallow probe errors
                continue

        # explicit extra probe of ctl.* attributes if present
        try:
            ctl = getattr(self.claw_ctl, 'ctl', None)
            if ctl:
                for name in ('home_event', 'red_button_event', 'claw_status_event', 'last_axes', 'current_pose'):
                    try:
                        v = getattr(ctl, name, None)
                        if v is None:
                            continue
                        if isinstance(v, queue.Queue):
                            state[f'ctl.{name}'] = {'size': v.qsize()}
                        else:
                            state[f'ctl.{name}'] = self._serialize(v)
                    except Exception:
                        pass
        except Exception:
            pass

        return state

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._t0 = _time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._t0 = None

    def _run(self):
        while not self._stop.is_set():
            t = self._now()
            try:
                st = self._snapshot()
            except Exception:
                st = {}
            self.samples.append({'t': t, 'state': st})
            _time.sleep(self.sample_interval)

    def log_event(self, name, info=None):
        t = self._now() if self._t0 is not None else 0.0
        self.events.append({'t': t, 'name': name, 'info': info})

    def as_dict(self):
        return {'samples': self.samples, 'events': self.events}


# ----------------- Saving helpers ------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _json_fallback(o):
    try:
        if isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)

def make_session_dir(participant_id, base_dir='claw_data'):
    """
    Create directory structure:
       base_dir/
         <participant_id>/
           session_<YYYYmmddTHHMMSS>/
    Returns the new session_dir path.
    """
    pid = str(participant_id) if participant_id is not None else 'anonymous'
    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    participant_dir = os.path.join(base_dir, pid)
    session_dir = os.path.join(participant_dir, f"session_{ts}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def save_session_metadata(session_dir, participant_id=None, extra=None):
    """Write a small session_meta.json into session_dir with participant and extra info."""
    meta = {
        'participant_id': participant_id,
        #'created_utc': datetime.datetime.utcnow().isoformat() + 'Z'
    }
    if extra:
        meta['extra'] = extra
    path = os.path.join(session_dir, 'session_meta.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=_json_fallback)
    return path

def save_trial_json(trial_record, out_dir='claw_data', pretty=True):
    """
    Save one trial JSON into out_dir. If out_dir doesn't exist it will be created.
    Returns path to saved pretty JSON file.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S.%fZ')
    trial_num = trial_record.get('trial_number', 'unknown')
    fname = f"trial_{trial_num}_{ts}.json"
    path = os.path.join(out_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(trial_record, f, indent=2, default=_json_fallback)
        else:
            json.dump(trial_record, f, default=_json_fallback)
    # append to per-session jsonl for easy streaming
    jsonl_path = os.path.join(out_dir, 'all_trials.jsonl')
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(trial_record, default=_json_fallback) + '\n')
    return path

def _serialize_for_json(v):
    """Recursively convert v into JSON-serializable Python primitives."""
    # primitives
    if v is None or isinstance(v, (str, bool, int, float)):
        return v

    # numpy scalar / array
    try:
        if isinstance(v, _np.generic):
            return v.item()
        if isinstance(v, _np.ndarray):
            return v.tolist()
    except Exception:
        pass

    # built-in containers
    if isinstance(v, (list, tuple, set)):
        return [_serialize_for_json(x) for x in v]

    if isinstance(v, dict):
        return {str(k): _serialize_for_json(val) for k, val in v.items()}

    # queue: return size and a (stringified) peek of last item if possible
    try:
        if isinstance(v, queue.Queue):
            info = {'size': v.qsize()}
            if hasattr(v, 'queue'):
                try:
                    arr = list(v.queue)
                    if arr:
                        info['last'] = _serialize_for_json(arr[-1])
                except Exception:
                    pass
            return info
    except Exception:
        pass

    # ROS- or message-like objects with x,y fields (Position)
    try:
        if hasattr(v, 'x') and hasattr(v, 'y'):
            return {'x': float(getattr(v, 'x')), 'y': float(getattr(v, 'y'))}
    except Exception:
        pass

    # objects with simple attributes
    try:
        if hasattr(v, '__dict__'):
            d = {}
            for k, val in vars(v).items():
                try:
                    d[k] = _serialize_for_json(val)
                except Exception:
                    d[k] = str(val)
            return d
    except Exception:
        pass

    # fallback: try to JSON dump directly, else str()
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)

def save_event_data(trial_data, participant_id=None, timestamp=None,
                    save_locally=False, out_dir=None, filename_prefix=None,
                    include_samples=False):
    """
    Clean events (and optionally samples) and return JSON string + optionally save to disk.

    The returned/serialized JSON will place `trial_info` as the first/top-level key,
    then `meta`, then `events` and `samples`. This makes trial metadata appear "on top".
    """
    # choose the logger dict (support being passed either a trial_record or a logger dict)
    if 'events' not in trial_data and 'logger' in trial_data:
        logger = trial_data['logger']
    elif 'events' in trial_data:
        logger = trial_data
    else:
        logger = trial_data.get('logger', {})

    # --- clean events ---
    events = logger.get('events', [])
    cleaned_events = []
    for ev in events:
        if not isinstance(ev, dict):
            try:
                if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                    ev = {'t': ev[0], 'name': ev[1], 'info': ev[2] if len(ev) > 2 else None}
                else:
                    ev = {'raw': ev}
            except Exception:
                ev = {'raw': str(ev)}

        clean_ev = {}
        for k, v in ev.items():
            try:
                # --- NEW: drop 'info' from the trial_init event ---
                if ev.get('name') == 'trial_init' and k == 'info':
                    continue
                clean_ev[str(k)] = _serialize_for_json(v)
            except Exception:
                clean_ev[str(k)] = str(v)

        cleaned_events.append(clean_ev)

    # --- clean samples if requested ---
    cleaned_samples = None
    if include_samples:
        samples = logger.get('samples', [])
        cleaned_samples = []
        for s in samples:
            if isinstance(s, dict) and 't' in s and 'state' in s:
                cleaned_samples.append({
                    't': _serialize_for_json(s.get('t')),
                    'state': _serialize_for_json(s.get('state'))
                })
            else:
                cleaned_samples.append(_serialize_for_json(s))

    # --- meta (file-level) ---
    meta = {}
    meta['participant_id'] = participant_id
    if timestamp is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S.%fZ')
    meta['timestamp'] = timestamp

    # --- trial_info: collect trial-level info (if present in trial_data) ---
    trial_info = {}
    try:
        if isinstance(trial_data, dict):
            for k, v in trial_data.items():
                if k in ('logger', 'events', 'samples'):
                    continue
                try:
                    trial_info[str(k)] = _serialize_for_json(v)
                except Exception:
                    trial_info[str(k)] = str(v)
    except Exception:
        trial_info = {'error': 'failed to collect trial_info'}

    # --- Build final cleaned dict in the desired order ---
    cleaned = {}
    if trial_info:
        cleaned['trial_info'] = trial_info
    cleaned['meta'] = meta
    cleaned['events'] = cleaned_events
    if cleaned_samples is not None:
        cleaned['samples'] = cleaned_samples

    # dump to JSON string
    try:
        event_json = json.dumps(cleaned, indent=2, ensure_ascii=False)
    except TypeError:
        event_json = json.dumps(cleaned, default=str, indent=2, ensure_ascii=False)

    # optionally save to disk
    saved_path = None
    if save_locally:
        if out_dir is None:
            raise ValueError("out_dir must be provided when save_locally=True (use make_session_dir).")
        os.makedirs(out_dir, exist_ok=True)
        prefix = filename_prefix or "events"
        pid = f"_P{participant_id}" if participant_id is not None else ""
        fname = f"{prefix}{pid}_{timestamp}.json"
        saved_path = os.path.join(out_dir, fname)
        with open(saved_path, 'w', encoding='utf-8') as f:
            f.write(event_json)

    return event_json, saved_path, cleaned

def save_choice_data(trial_record, participant_id=None, timestamp=None,
                     out_dir=None, filename_prefix="choice"):

    if timestamp is None:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')

    # Extract RFID info if present
    rfid_info = trial_record.get('rfid') if trial_record is not None else None

    summary = {
        "trial_info": {
            "trial_number": trial_record.get("trial_number"),
            "total_trials": trial_record.get("total_trials"),
            "training_mode": trial_record.get("training_mode"),
            "timestamp_utc": trial_record.get("timestamp_utc"),
            "grip_type": trial_record.get("grip_type"),
        },
        "participant_id": participant_id,
        "options_displayed": trial_record.get("options_displayed"),  # we'll set this below
        "choice": trial_record.get("choice"),
        "selection_confidence": trial_record.get("selection_confidence"),
        "action_confidence": trial_record.get("outcome_confidence"),
        "chosen_reward": trial_record.get("chosen_reward"),

        # NEW: original claw position (tuple or None)
        "original_claw_position": trial_record.get("original_claw_position"),

        # NEW: RFID summary dict (raw string + detected/tag fields) or None
        "rfid": rfid_info,
    }

    # The rest of your save logic (create filename, write JSON) goes here.
    # For example:
    if out_dir is None:
        out_dir = '.'
    filename = f"{filename_prefix}_{participant_id or 'unknown'}_{timestamp}.json"
    out_path = os.path.join(out_dir, filename)
    try:
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] failed saving choice summary to {out_path}: {e}")
        return None

    return out_path

