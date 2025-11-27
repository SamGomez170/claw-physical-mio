import serial
import serial.tools.list_ports
import threading
import time
import re

JSON_HEX_RE = re.compile(rb'"hex"\s*:\s*"\s*([0-9A-Fa-f:]{6,})\s*"')
JSON_DEC_RE = re.compile(rb'"dec"\s*:\s*(\d+)')
GENERIC_HEX_BYTES = re.compile(rb'([0-9A-Fa-f]{1,2}(?:[:\-\s]?[0-9A-Fa-f]{1,2}){3,})')

def _bytes_to_printable_line(b):
    # try to decode; fall back to latin1 to preserve bytes
    try:
        s = b.decode('utf-8', errors='replace')
    except Exception:
        s = b.decode('latin1', errors='replace')
    # remove excessive control chars except common whitespace
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]+', ' ', s)

def _normalize_hex_string_from_str(s):
    s = re.sub(r'[^0-9A-Fa-f]', '', s).upper()
    if len(s) % 2 == 1:
        s = '0' + s
    pairs = [s[i:i+2] for i in range(0, len(s), 2)]
    return ':'.join(pairs)

def _dec_to_hex_colon_from_str(dec):
    try:
        n = int(dec)
    except Exception:
        return None
    h = format(n, 'X')
    if len(h) % 2 == 1:
        h = '0' + h
    pairs = [h[i:i+2] for i in range(0, len(h), 2)]
    return ':'.join(pairs)

class RFIDReader:
    """
    Background reader that keeps the serial port open and continually
    parses lines for tag UIDs. Use start() to begin the background thread.
    """
    def __init__(self, port=None, baud=115200, timeout=0.1, verbose=False):
        self.baud = baud
        self.timeout = timeout
        self.port = port or self._find_port()
        self.verbose = verbose

        self.ser = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self.last_tag = None           # last parsed normalized hex (e.g. "AA:BB:CC")
        self.last_tag_ts = None
        self.last_raw_line = None

        if self.port:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                # give a moment for board to settle
                time.sleep(0.1)
                # flush any startup text (avoid reading stale boot lines)
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
            except Exception as e:
                if self.verbose:
                    print(f"[RFID] failed to open {self.port} @ {self.baud}: {e}")
                self.ser = None
        else:
            if self.verbose:
                print("[RFID] no serial port found")

    def _find_port(self, keyword="FT232"):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if p.description and keyword.lower() in p.description.lower():
                return p.device
        for p in ports:
            if "/dev/ttyUSB" in p.device or "/dev/ttyACM" in p.device or p.device.startswith("COM"):
                return p.device
        return None

    def start(self):
        if not self.ser:
            if self.verbose:
                print("[RFID] start() called but serial not available")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def _reader_loop(self):
        """
        Robust loop: read raw bytes, append to a bytearray buffer,
        search for JSON hex/dec first, then generic hex sequences.
        """
        if not self.ser:
            return

        buf = bytearray()
        while self._running:
            try:
                # read whatever is available (non-blocking-ish)
                n = max(1, self.ser.in_waiting) if hasattr(self.ser, 'in_waiting') else 1
                data = self.ser.read(n)
            except Exception:
                data = b''
            if not data:
                time.sleep(0.01)
                continue

            # append to buffer
            buf.extend(data)

            # debug: store last raw bytes (printable)
            try:
                self.last_raw_line = _bytes_to_printable_line(buf[-512:])  # last chunk preview
            except Exception:
                self.last_raw_line = None

            # Try JSON "hex" pattern (strongest)
            m = JSON_HEX_RE.search(buf)
            if m:
                raw_hex = m.group(1).decode('ascii', errors='ignore')
                tag = _normalize_hex_string_from_str(raw_hex)
                with self._lock:
                    self.last_tag = tag
                    self.last_tag_ts = time.time()
                # consume up to end of match to avoid reprocessing
                end = m.end()
                buf = buf[end:]
                if self.verbose:
                    print(f"[RFID PARSE] JSON_HEX -> {tag}")
                continue

            # Try JSON "dec" pattern
            m2 = JSON_DEC_RE.search(buf)
            if m2:
                raw_dec = m2.group(1).decode('ascii', errors='ignore')
                tag = _dec_to_hex_colon_from_str(raw_dec)
                if tag:
                    with self._lock:
                        self.last_tag = tag
                        self.last_tag_ts = time.time()
                    end = m2.end()
                    buf = buf[end:]
                    if self.verbose:
                        print(f"[RFID PARSE] JSON_DEC -> {tag}")
                    continue

            # Try generic hex-like sequences in the buffer (byte-level)
            m3 = GENERIC_HEX_BYTES.search(buf)
            if m3:
                raw = m3.group(1).decode('ascii', errors='ignore')
                tag = _normalize_hex_string_from_str(raw)
                # Heuristic: require at least 4 bytes (8 hex chars) to reduce false positives
                if len(tag.replace(':','')) >= 8:
                    with self._lock:
                        self.last_tag = tag
                        self.last_tag_ts = time.time()
                    # consume up to end of match
                    end = m3.end()
                    buf = buf[end:]
                    if self.verbose:
                        print(f"[RFID PARSE] GENERIC_HEX -> {tag}")
                    continue
                else:
                    # too-short candidate; don't consume (wait for more bytes)
                    pass

            # Keep buffer from growing unbounded. If it's huge and nothing matched,
            # trim to the last 1024 bytes (might be mid-line characters).
            if len(buf) > 4096:
                buf = buf[-1024:]
                if self.verbose:
                    print("[RFID WARN] buffer trimmed due to excessive length")

            # small sleep to avoid busy loop
            time.sleep(0.005)

    def get_last_tag(self, max_age=None):
        """Return last tag string (e.g. 'AA:BB:CC') or None. If max_age given (seconds),
           return None if tag is older than max_age."""
        with self._lock:
            t = self.last_tag
            ts = self.last_tag_ts
        if t is None:
            return None
        if max_age is not None and ts is not None:
            if time.time() - ts > max_age:
                return None
        return t

    def read_tag(self, timeout=3.0, require_new=False):
        """
        Blocking wait up to timeout seconds for a tag. If require_new is True,
        it waits for a tag that arrives after the call (useful to avoid returning
        a leftover tag from earlier).
        """
        deadline = time.time() + timeout
        start_ts = None
        if require_new:
            start_ts = self.last_tag_ts

        while time.time() < deadline:
            t = self.get_last_tag(max_age=None)
            ts = None
            with self._lock:
                ts = self.last_tag_ts
            if t:
                if require_new:
                    if ts is None or start_ts is None:
                        # weird; treat as new
                        return t
                    if ts > start_ts:
                        return t
                else:
                    return t
            time.sleep(0.02)
        return None
