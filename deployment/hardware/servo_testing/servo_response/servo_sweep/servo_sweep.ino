// Sine sweep frequency response measurement
// Pin 9 (OC1A) → servo signal
// A1            → pot wiper (via 100kΩ/100kΩ voltage divider to GND)
//
// Protocol:
//   Send "FREQ:X.X:A.AA\n" → streams CSV at given freq and amplitude
//   Send "STOP\n"          → returns to center, sends "STOPPED\n"
//
// Amplitude is set by the host so peak velocity = 10% of rated speed.

#define PIN_POT    A1
#define PULSE_MIN  1000      // 500µs × 2 ticks (prescaler 8, 16MHz)
#define PULSE_MAX  5000      // 2500µs × 2 ticks
#define TIMER_TOP  6667      // 300 Hz PWM (3333µs period)

#define RAW_MIN    124       // pot raw at 180°
#define RAW_MAX    515       // pot raw at 0°
#define CENTER_DEG 90.0f
#define SAMPLE_US  2500UL   // 400 Hz sample rate

float currentFreq  = 0;
float currentAmp   = 0;
bool  running      = false;
unsigned long sweepStartUs = 0;
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
            if (inputBuffer.startsWith("FREQ:")) {
                // Format: FREQ:<freq>:<amplitude>
                String params = inputBuffer.substring(5);
                int sep = params.indexOf(':');
                if (sep > 0) {
                    float freq = params.substring(0, sep).toFloat();
                    float amp  = params.substring(sep + 1).toFloat();
                    if (freq > 0 && amp > 0) {
                        currentFreq  = freq;
                        currentAmp   = amp;
                        running      = true;
                        sweepStartUs = micros();
                        lastSampleUs = sweepStartUs;
                        Serial.println("t_us,cmd_deg,meas_deg");
                    }
                }
            } else if (inputBuffer == "STOP") {
                running = false;
                servoWriteAngle(CENTER_DEG);
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
        unsigned long elapsed = now - sweepStartUs;
        float t   = elapsed / 1e6f;
        float cmd = CENTER_DEG + currentAmp * sin(2.0f * PI * currentFreq * t);
        servoWriteAngle(cmd);
        float meas = readAngle();

        Serial.print(elapsed);
        Serial.print(',');
        Serial.print(cmd, 3);
        Serial.print(',');
        Serial.println(meas, 3);
    }
}
