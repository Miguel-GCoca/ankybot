const int NUM_SERVOS = 6;
const int analogPins[NUM_SERVOS] = {A0, A1, A2, A3, A4, A5};

void setup() {
  Serial.begin(115200);
}

void loop() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    analogRead(analogPins[i]);
    int raw = analogRead(analogPins[i]);
    Serial.print(raw);
    Serial.print(" ");
  }
  Serial.println();

  delay(200);
}


