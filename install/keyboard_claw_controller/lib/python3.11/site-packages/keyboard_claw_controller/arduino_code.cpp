// MFRC522: aggressive polling + retry + presence hysteresis
#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 10
#define RST_PIN 9

MFRC522 mfrc522(SS_PIN, RST_PIN);

const unsigned long POLL_INTERVAL = 5;     // ms: poll frequently to catch fast contacts
const unsigned long ABSENCE_TIMEOUT_MS = 150; // ms: treat tag as still present for short dropouts
unsigned long lastPoll = 0;
unsigned long lastSeenMillis = 0;
String lastUidStr = "";

// convert UID to hex string (lowercase)
String uidToString(const MFRC522::Uid &uid) {
  String s = "";
  for (byte i = 0; i < uid.size; i++) {
    if (uid.uidByte[i] < 0x10) s += "0";
    s += String(uid.uidByte[i], HEX);
    if (i < uid.size - 1) s += ":";
  }
  s.toLowerCase();
  return s;
}

void printUid(const MFRC522::Uid &uid) {
  String h = uidToString(uid);
  unsigned long dec = 0;
  for (byte i = 0; i < uid.size; i++) dec = (dec << 8) | uid.uidByte[i];
  Serial.print("{\"hex\":\""); Serial.print(h); Serial.print("\",\"dec\":"); Serial.print(dec); Serial.println("}");
}

// Try reading card serial with a few very short retries
bool tryReadCardSerial(int retries = 5, int retryDelayMs = 1) {
  for (int i = 0; i <= retries; ++i) {
    if (mfrc522.PICC_ReadCardSerial()) return true;
    if (i < retries) delay(retryDelayMs);
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }
  SPI.begin();
  mfrc522.PCD_Init();
  // detect any tag already in the field at startup
  if (mfrc522.PICC_IsNewCardPresent() && tryReadCardSerial()) {
    printUid(mfrc522.uid);
    mfrc522.PCD_StopCrypto1();
    lastUidStr = uidToString(mfrc522.uid);
    lastSeenMillis = millis();
  }

  //delay(50);
  // Increase antenna gain to max
  mfrc522.PCD_SetAntennaGain(mfrc522.RxGain_max);
  Serial.println("Reader ready (aggressive polling + hysteresis)");
}

void loop() {
  unsigned long now = millis();
  if (now - lastPoll < POLL_INTERVAL) return;
  lastPoll = now;

  // If a card appears in the field
  if (mfrc522.PICC_IsNewCardPresent()) {
    if (tryReadCardSerial()) {
      String thisUid = uidToString(mfrc522.uid);
      lastSeenMillis = now;

      // If it's a newly seen UID, print it
      if (thisUid != lastUidStr) {
        for (int i = 0; i < 3; ++i) {
          printUid(mfrc522.uid);
          delay(6); // tiny gap so serial lines separate (tune if necessary)
        }
        lastUidStr = thisUid;
      }

      // stop crypto if any
      mfrc522.PCD_StopCrypto1();
    } else {
      // transient read failure: update lastSeenMillis so short dropouts count as present
      lastSeenMillis = now;
    }
    return;
  }

  // No card currently reported by PICC_IsNewCardPresent()
  // If last seen recently (within ABSENCE_TIMEOUT_MS), assume it's a short dropout: do nothing
  if (lastUidStr != "" && (now - lastSeenMillis) < ABSENCE_TIMEOUT_MS) {
    // still consider the tag present (do nothing)
    return;
  }

  // If here, tag has been absent long enough — clear it
  if (lastUidStr != "") {
    lastUidStr = "";
    // optionally print removal event if you want
    // Serial.println("{\"event\":\"tag_removed\"}");
  }
}