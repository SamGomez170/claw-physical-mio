import serial
import serial.tools.list_ports
import threading
import time

class IRDetector:
    """Minimal IR ball detector that parses Arduino timestamp lines."""
    def __init__(self, port=None, baud=115200, timeout=0.1, verbose=False):
        self.baud = baud
        self.timeout = timeout
        self.verbose = verbose
        self.port = port or self._find_port()
        self.ser = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.last_detection = None  # stores the last timestamp line
        self.last_detection_ts = None  # stores when the detection happened

        if self.port:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                time.sleep(2)  # wait for Arduino reset
                self.ser.reset_input_buffer()
                if self.verbose:
                    print(f"[Ball] Connected to {self.port}")
            except Exception as e:
                if self.verbose:
                    print(f"[Ball] Failed to open {self.port}: {e}")
                self.ser = None
        else:
            if self.verbose:
                print("[Ball] No serial port found")

    def _find_port(self):
        """Try to auto-detect Arduino port."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "Arduino" in p.description or "USB" in p.description:
                return p.device
        return None

    def start(self):
        """Start background thread to read serial data."""
        if not self.ser:
            if self.verbose:
                print("[Ball] Cannot start - no serial connection")
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
            self.ser.close()

    def _reader_loop(self):
        """Background loop to read serial lines."""
        if not self.ser:
            return
        while self._running:
            try:
                line = self.ser.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str.startswith("Detected at"):
                    with self._lock:
                        self.last_detection = line_str
                        self.last_detection_ts = time.time()  # record when we got it
                    if self.verbose:
                        print(f"[Ball] {line_str}")
            except Exception as e:
                if self.verbose:
                    print(f"[Ball] Read error: {e}")
                time.sleep(0.1)

    def read_detection(self, require_new=False):
        """
        Return last detection line.
        If require_new=True, only returns if it's newer than last call.
        """
        with self._lock:
            detection = self.last_detection
        if detection and require_new:
            self.last_detection = None  # mark it read
        return detection
    
    def get_last_detection(self, max_age=None):
        """
        Get last detected IR signal. If max_age specified (in seconds), 
        returns None if detection is older than max_age.
        This works exactly like RFIDReader.get_last_tag()
        """
        with self._lock:
            detection = self.last_detection
            ts = self.last_detection_ts
        
        if detection is None:
            return None
        if max_age is not None and ts is not None:
            if time.time() - ts > max_age:
                return None  # too old
        return detection
    
    def clear_detection(self):
        """Clear the stored detection. Call this at the start of each trial."""
        with self._lock:
            self.last_detection = None
            self.last_detection_ts = None