import serial
import serial.tools.list_ports
import threading
import time
import re

class RFIDReader:
    """Minimal RFID reader that parses tag UIDs from serial data."""
    
    def __init__(self, port=None, baud=115200, timeout=0.1, verbose=False):
        self.baud = baud
        self.timeout = timeout
        self.verbose = verbose
        self.port = port or self._find_port()
        
        self.ser = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        
        self.last_tag = None
        self.last_tag_ts = None
        self.last_raw_line = None
        
        if self.port:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                time.sleep(0.1)
                self.ser.reset_input_buffer()
                if self.verbose:
                    print(f"[RFID] Connected to {self.port}")
            except Exception as e:
                if self.verbose:
                    print(f"[RFID] Failed to open {self.port}: {e}")
                self.ser = None
        else:
            if self.verbose:
                print("[RFID] No serial port found")
    
    def _find_port(self):
        """Find likely RFID reader port."""
        ports = list(serial.tools.list_ports.comports())
        # Try FT232 first
        for p in ports:
            if p.description and "FT232" in p.description.upper():
                return p.device
        # Try common serial ports
        for p in ports:
            dev = p.device
            if "/dev/ttyUSB" in dev or "/dev/ttyACM" in dev or dev.startswith("COM"):
                return p.device
        return None
    
    def start(self):
        """Start background reading thread."""
        if not self.ser:
            if self.verbose:
                print("[RFID] Cannot start - no serial connection")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop reading and close serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except:
                pass
    
    def _parse_tag(self, line):
        """Extract and normalize tag UID from line."""
        # Try JSON format: "hex": "AA:BB:CC:DD"
        m = re.search(r'"hex"\s*:\s*"([0-9A-Fa-f:]+)"', line)
        if m:
            return self._normalize_hex(m.group(1))
        
        # Try JSON format: "dec": 123456789
        m = re.search(r'"dec"\s*:\s*(\d+)', line)
        if m:
            return self._dec_to_hex(m.group(1))
        
        # Try raw hex bytes (at least 4 bytes = 8 hex chars)
        m = re.search(r'([0-9A-Fa-f]{2}[:\-\s]?){3,}[0-9A-Fa-f]{2}', line)
        if m:
            return self._normalize_hex(m.group(0))
        
        return None
    
    def _normalize_hex(self, s):
        """Clean and format hex string as AA:BB:CC:DD."""
        s = re.sub(r'[^0-9A-Fa-f]', '', s).upper()
        if len(s) < 8:  # At least 4 bytes
            return None
        if len(s) % 2 == 1:
            s = '0' + s
        return ':'.join([s[i:i+2] for i in range(0, len(s), 2)])
    
    def _dec_to_hex(self, dec_str):
        """Convert decimal string to hex format."""
        try:
            n = int(dec_str)
            h = format(n, 'X')
            if len(h) % 2 == 1:
                h = '0' + h
            return ':'.join([h[i:i+2] for i in range(0, len(h), 2)])
        except:
            return None
    
    def _reader_loop(self):
        """Background loop reading and parsing serial data."""
        if not self.ser:
            return
        
        while self._running:
            try:
                line = self.ser.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                
                # Store raw line for debugging
                try:
                    self.last_raw_line = line.decode('utf-8', errors='ignore').strip()
                except:
                    self.last_raw_line = str(line)
                
                # Parse tag
                tag = self._parse_tag(self.last_raw_line)
                if tag:
                    with self._lock:
                        self.last_tag = tag
                        self.last_tag_ts = time.time()
                    if self.verbose:
                        print(f"[RFID] Tag detected: {tag}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"[RFID] Read error: {e}")
                time.sleep(0.1)
    
    def get_last_tag(self, max_age=None):
        """Get last detected tag. If max_age specified, returns None if tag is older."""
        with self._lock:
            tag = self.last_tag
            ts = self.last_tag_ts
        
        if tag is None:
            return None
        if max_age is not None and ts is not None:
            if time.time() - ts > max_age:
                return None
        return tag
    
    def read_tag(self, timeout=3.0, require_new=False):
        """
        Wait for a tag up to timeout seconds.
        If require_new=True, only returns tags detected after this call.
        """
        deadline = time.time() + timeout
        start_ts = self.last_tag_ts if require_new else None
        
        while time.time() < deadline:
            with self._lock:
                tag = self.last_tag
                ts = self.last_tag_ts
            
            if tag:
                if require_new:
                    if start_ts is None or (ts and ts > start_ts):
                        return tag
                else:
                    return tag
            
            time.sleep(0.02)
        return None
    
    def clear_tag(self):
        """Clear the stored tag. Call this at the start of each trial."""
        with self._lock:
            self.last_tag = None
            self.last_tag_ts = None