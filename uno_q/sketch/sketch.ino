#include <Arduino_Modulino.h>
#include <Arduino_RouterBridge.h>

ModulinoThermo thermo;

unsigned long previousMillis = 0;
const long interval = 1000;

void setup() {
  Bridge.begin();
  Modulino.begin(Wire1);
  thermo.begin();
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    float celsius = thermo.getTemperature();
    float humidity = thermo.getHumidity();
    Bridge.notify("record_sensor_samples", celsius, humidity);
  }
}
