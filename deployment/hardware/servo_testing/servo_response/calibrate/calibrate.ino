// Pot calibration helper — drives the servo to a commanded angle and
// continuously streams the RAW (uncalibrated) ADC reading so you can
// find RAW_MIN/RAW_MAX for the other sketches.
//
// Pin 9 (OC1A) → servo signal
// A1            → pot wiper (via 100kΩ/100kΩ voltage divider to GND)
//
// Protocol:
//   Send "STEP:<target_deg>\n" → moves servo there, streams raw ADC continuously
//   Send "STOP\n"              → stops streaming

#define PIN_POT    A1
#define PULSE_MIN  1000      // 500µs × 2 ticks (prescaler 8, 16MHz)
#define PULSE_MAX  5000      // 2500µs × 2 ticks
#define TIMER_TOP  6667      // 300 Hz PWM (3333µs period)

#define PRINT_INTERVAL_MS 200

bool running = false;
unsigned long lastPrintMs = 0;
String inputBuffer = "";

void servoWriteAngle(float angle) {
    angle = constrain(angle, 0.0f, 180.0f);
    OCR1A = (int)map((long)(angle * 10), 0, 1800, PULSE_MIN, PULSE_MAX);
}

int readRaw() {
    analogRead(PIN_POT);  // throwaway read — lets S&H cap settle at 50kΩ source impedance
    return analogRead(PIN_POT);
}

void processSerial() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (inputBuffer.startsWith("STEP:")) {
                float target = inputBuffer.substring(5).toFloat();
                servoWriteAngle(target);
                running = true;
                Serial.print("Moving to ");
                Serial.print(target, 1);
                Serial.println(" deg. Streaming raw_adc...");
            } else if (inputBuffer == "STOP") {
                running = false;
                Serial.println("STOPPED");
            }
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }
}

void setup() {
    TCCR1A = (1 << COM1A1) | (1 << WGM11);
    TCCR1B = (1 << WGM13) | (1 << WGM12) | (1 << CS11);
    ICR1   = TIMER_TOP;
    OCR1A  = (PULSE_MIN + PULSE_MAX) / 2;
    pinMode(9, OUTPUT);

    delay(500);
    Serial.begin(115200);
    Serial.println("READY. Send STEP:<angle> to move, STOP to halt printing.");
}

void loop() {
    processSerial();

    if (!running) return;

    unsigned long now = millis();
    if (now - lastPrintMs >= PRINT_INTERVAL_MS) {
        lastPrintMs = now;
        Serial.println(readRaw());
    }
}
