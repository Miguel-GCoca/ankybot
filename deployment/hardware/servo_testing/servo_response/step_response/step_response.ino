// Step response data collection for natural frequency / damping ratio
// Pin 9 (OC1A) → servo signal
// A1            → pot wiper (via 100kΩ/100kΩ voltage divider to GND)
//
// Protocol:
//   Send "STEP:<target_deg>\n" → applies step immediately, streams CSV
//   Send "STOP\n"              → stops streaming, sends "STOPPED\n"
//
// Note: STOP does not move the servo — the host commands the return
// step explicitly (e.g. STEP:90) so the return transient is recorded too.

#define PIN_POT    A1
#define PULSE_MIN  1000      // 500µs × 2 ticks (prescaler 8, 16MHz)
#define PULSE_MAX  5000      // 2500µs × 2 ticks
#define TIMER_TOP  6667      // 300 Hz PWM (3333µs period)

#define RAW_MIN    124       // pot raw at 180°
#define RAW_MAX    515       // pot raw at 0°
#define SAMPLE_US  2500UL    // 400 Hz sample rate

float targetDeg    = 90.0f;
bool  running      = false;
unsigned long stepStartUs  = 0;
unsigned long lastSampleUs = 0;
String inputBuffer = "";

void servoWriteAngle(float angle) {
    angle = constrain(angle, 0.0f, 180.0f);
    OCR1A = (int)map((long)(angle * 10), 0, 1800, PULSE_MIN, PULSE_MAX);
}

float readAngle() {
    analogRead(PIN_POT);  // throwaway read — lets S&H cap settle at 50kΩ source impedance
    int raw = constrain(analogRead(PIN_POT), RAW_MIN, RAW_MAX);
    return (float)(RAW_MAX - raw) / (RAW_MAX - RAW_MIN) * 180.0f;
}

void processSerial() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (inputBuffer.startsWith("STEP:")) {
                float target = inputBuffer.substring(5).toFloat();
                targetDeg    = target;
                running      = true;
                stepStartUs  = micros();
                lastSampleUs = stepStartUs;
                servoWriteAngle(targetDeg);   // apply the step immediately
                Serial.println("t_us,cmd_deg,meas_deg");
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
    // Timer1: Fast PWM mode 14, prescaler 8, non-inverting OC1A (pin 9)
    TCCR1A = (1 << COM1A1) | (1 << WGM11);
    TCCR1B = (1 << WGM13) | (1 << WGM12) | (1 << CS11);
    ICR1   = TIMER_TOP;
    OCR1A  = (PULSE_MIN + PULSE_MAX) / 2;
    pinMode(9, OUTPUT);

    delay(500);
    Serial.begin(115200);
    Serial.println("READY");
}

void loop() {
    processSerial();

    if (!running) return;

    unsigned long now = micros();
    if (now - lastSampleUs >= SAMPLE_US) {
        lastSampleUs = now;
        unsigned long elapsed = now - stepStartUs;
        float meas = readAngle();

        Serial.print(elapsed);
        Serial.print(',');
        Serial.print(targetDeg, 3);
        Serial.print(',');
        Serial.println(meas, 3);
    }
}
