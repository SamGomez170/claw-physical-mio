#include <FastLED.h>

#define LED_PIN   6
#define NUM_LEDS  100
#define LED_TYPE  WS2812B
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];

void setup() {
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(255);
  
  // Just turn everything red immediately, no serial needed
  for (int i = 0; i < NUM_LEDS; i++) {
    leds[i] = CRGB(255, 0, 0);
  }
  FastLED.show();
}

void loop() {
  // nothing
}
