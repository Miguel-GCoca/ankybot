#include <Wire.h>

// I2C slave, CALIBRATION variant of arduino_mega_i2c_slave.ino - same wire
// protocol (address 0x08, chunk-select-then-read-24-bytes, 2 chunks of 6
// floats = 12 channels total) but skips the RAW_MIN/RAW_MAX mapping and
// JOINT_ORIENTATION flip entirely, exposing raw ADC counts directly (as
// floats, purely for wire-format compatibility with the same Python reader
// used against the production firmware - see i2c_servo_common.py's
// MegaFeedback).
//
// Flash this onto the Mega for BOTH scripts in the current bench-test
// workflow - find_calibration_extremes.py AND run_step_response_targets.py
// both read raw ADC, so NO reflash is needed between them (2026-07-17).
// This firmware is also compatible with the older manual calibrate_i2c.py.
//
// The ONE script that needs the OTHER firmware is run_step_response_sweep.py
// (the older, repeated ±45/±30/±15 multi-rep sweep) - that one reads
// calibrated angle, not raw ADC, so reflash arduino_mega_i2c_slave.ino
// (arduino_ws/arduino_mega_final/arduino_mega_i2c_slave/) - the real
// production feedback firmware - before running that script, or before
// deploying to the robot. Do NOT reflash to arduino_mega_i2c_slave.ino
// before run_step_response_targets.py - it needs raw ADC, and feeding it
// already-converted radians (tiny values, ~-1.57..1.57) instead of raw
// ADC counts (~100-900) will silently break its linear calibration math.
//
// See arduino_mega_i2c_slave.ino for the full protocol rationale (ISR
// safety, chunk sizing vs. the AVR Wire library's 32-byte buffer, why
// analogRead() never happens in the ISR, etc.) - not repeated here, only
// the calibration-specific difference (no RAW_MIN/RAW_MAX/orientation math)
// is called out.

const int NUM_SERVOS = 12;

// Same corrected pins as arduino_mega_i2c_slave.ino (fixed 2026-07-17 -
// see that file's comment for the full story): A0=BR_Hip, A1=BR_Thigh,
// A2=BR_Foot, A3=FR_Hip, A4=FR_Thigh, A7=FR_Foot, A8=FL_Hip, A9=FL_Thigh,
// A10=FL_Foot, A11=BL_Hip, A12=BL_Thigh, A13=BL_Foot (A5/A6 skipped - dead
// pins on this board). Index i is JOINT_ORDER[i]; each entry points at that
// joint's real pin.
//
// 2026-07-22: BR_Hip/BR_Foot pins SWAPPED AGAIN (A0<->A2), same fix and
// same reasoning as arduino_mega_i2c_slave.ino's copy of this array - see
// that file's comment for the full root-cause writeup. Feedback-only
// change; keep this array in sync with that file's.
const int analogPins[NUM_SERVOS] = {
  A8, A9, A10,   // FL: hip, thigh, foot
  A3, A4, A7,    // FR: hip, thigh, foot
  A11, A12, A13, // BL: hip, thigh, foot
  A2, A1, A0     // BR: hip, thigh, foot (hip/foot pins swapped 2026-07-22 - see comment above)
};

const int NUM_AVG_SAMPLES = 4;

volatile float rawCounts[NUM_SERVOS];

const uint8_t I2C_SLAVE_ADDR = 0x08;
const int FLOATS_PER_CHUNK = 6;
volatile uint8_t requestedChunk = 0;

void onI2CReceive(int numBytes) {
  if (numBytes >= 1) {
    uint8_t chunk = Wire.read();
    if (chunk < 2) {
      requestedChunk = chunk;
    }
  }
  while (Wire.available()) Wire.read();  // drain anything extra
}

void onI2CRequest() {
  int startIdx = requestedChunk * FLOATS_PER_CHUNK;
  float chunk[FLOATS_PER_CHUNK];
  for (int i = 0; i < FLOATS_PER_CHUNK; i++) {
    chunk[i] = rawCounts[startIdx + i];
  }
  Wire.write((uint8_t *)chunk, sizeof(chunk));
}

void setup() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    rawCounts[i] = 0.0f;
  }

  Wire.begin(I2C_SLAVE_ADDR);
  Wire.onReceive(onI2CReceive);
  Wire.onRequest(onI2CRequest);
}

void loop() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    analogRead(analogPins[i]);  // throwaway - lets S&H cap settle
    long sum = 0;
    for (int s = 0; s < NUM_AVG_SAMPLES; s++) {
      sum += analogRead(analogPins[i]);
    }
    rawCounts[i] = (float)(sum / NUM_AVG_SAMPLES);
  }
}
