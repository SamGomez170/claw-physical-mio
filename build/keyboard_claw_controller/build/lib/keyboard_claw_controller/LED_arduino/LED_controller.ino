#include <FastLED.h>

#define LED_PIN     3
#define NUM_LEDS    100
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];

void setColor(CRGB color) {
  for (int i = 0; i < NUM_LEDS; i++) leds[i] = color;
  FastLED.show();
}

void setup() {
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(50);
  Serial.begin(9600);
  Serial.println("READY");
  setColor(CRGB::Black);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.replace("\r", "");
    Serial.print("GOT: ");
    Serial.println(cmd);

    if (cmd == "OFF") {
      setColor(CRGB::Black);
      Serial.println("LED OFF");
    } else if (cmd == "SET_GRIP:narrow_low") {
      setColor(CRGB(30, 144, 255));
      Serial.println("LED BLUE");
    } else if (cmd == "SET_GRIP:wide_low") {
      setColor(CRGB(50, 205, 50));
      Serial.println("LED GREEN");
    } else if (cmd == "SET_GRIP:wide_high") {
      setColor(CRGB(255, 215, 0));
      Serial.println("LED YELLOW");
    } else if (cmd == "SET_GRIP:narrow_high") {
      setColor(CRGB(220, 20, 60));
      Serial.println("LED RED");
    } else {
      Serial.println("UNKNOWN CMD");
    }
  }
}
