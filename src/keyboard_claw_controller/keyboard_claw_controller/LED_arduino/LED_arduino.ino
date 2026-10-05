#include <FastLED.h>

#define LED_PIN     6
#define NUM_LEDS    100
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];
String inputLine = "";

// Same colors as the claws drawn on screen
void setAll(const CRGB &c) {
  fill_solid(leds, NUM_LEDS, c);
  FastLED.show();
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd == "OFF") {
    setAll(CRGB::Black);
    return;
  }

  if (cmd.startsWith("SET_GRIP:")) {
    String grip = cmd.substring(9);
    grip.trim();

    if (grip == "narrow_low") {          // BLUE
      setAll(CRGB(30, 144, 255));
    } else if (grip == "wide_low") {     // GREEN
      setAll(CRGB(50, 205, 50));
    } else if (grip == "wide_high") {    // YELLOW
      setAll(CRGB(255, 215, 0));
    } else if (grip == "narrow_high") {  // RED
      setAll(CRGB(220, 20, 60));
    } else {
      setAll(CRGB::Black);               // unknown grip -> off
    }
  }
}

void setup() {
  Serial.begin(9600);
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(255);
  // Cap total current so a full strip can't brown out the board and
  // knock it off USB. Raise this to match your 5V supply's rating
  // (e.g. 4000 for a 4A supply). Keep ~500 mA if powering from USB only.
  FastLED.setMaxPowerInVoltsAndMilliamps(5, 500);   // safe for USB power
  setAll(CRGB::Black);  // start with lights off
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputLine);
      inputLine = "";
    } else if (c != '\r') {
      inputLine += c;
      if (inputLine.length() > 64) inputLine = "";  // guard against garbage
    }
  }
}
