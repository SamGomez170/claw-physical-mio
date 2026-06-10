const byte sensorPin = 2;
bool prevDetected = false;

void setup() {
  pinMode(sensorPin, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);   
  while (!Serial) { /* wait for USB serial on some boards */ }
  Serial.println("IR timestamp test");
  prevDetected = (digitalRead(sensorPin) == LOW); // avoid immediate trigger
}

void loop() {
  bool detected = (digitalRead(sensorPin) == LOW); // FC-51 often LOW on detection
  digitalWrite(LED_BUILTIN, detected ? HIGH : LOW);

  // print only when detection *starts*
  if (detected && !prevDetected) {
    unsigned long ms = millis();
    Serial.print("Detected at ");
    printTime(ms);                       // HH:MM:SS.mmm
    Serial.print("  (");
    Serial.print(ms);
    Serial.println(" ms since start)");
  }
  prevDetected = detected;
}

// prints HH:MM:SS.mmm for a millis() value
void printTime(unsigned long ms) {
  unsigned long s = ms / 1000;
  unsigned int msec = ms % 1000;
  unsigned int sec = s % 60;
  unsigned int min = (s / 60) % 60;
  unsigned int hour = (s / 3600) % 24;

  if (hour < 10) Serial.print('0'); Serial.print(hour); Serial.print(':');
  if (min  < 10) Serial.print('0'); Serial.print(min);  Serial.print(':');
  if (sec  < 10) Serial.print('0'); Serial.print(sec);  Serial.print('.');
  if (msec < 100) Serial.print('0');
  if (msec < 10)  Serial.print('0');
  Serial.print(msec);
}
