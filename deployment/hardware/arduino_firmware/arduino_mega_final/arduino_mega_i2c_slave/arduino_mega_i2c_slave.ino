#include <Wire.h>


// Protocol: master selects which 6-float (24-byte) chunk it wants with a
// single-byte write, then reads 24 bytes back in the same transaction
//   chunk 0 = joints 0-5, chunk 1 = joints 6-11
// Joint order (matches ankybot_i2c_bridge/pca9685_commander_node.cpp's

const int NUM_SERVOS = 12;

// 2026-07-22: BR_Hip/BR_Foot pins SWAPPED (A0<->A2)
const int analogPins[NUM_SERVOS] = {
  A8, A9, A10,   // FL: hip, thigh, foot
  A3, A4, A7,    // FR: hip, thigh, foot
  A11, A12, A13, // BL: hip, thigh, foot
  A2, A1, A0     // BR: hip, thigh, foot 
};

// RAW_MIN[i]/RAW_MAX[i] are the raw ADC readings measured, via
// find_calibration_extremes.py, at this channel's logical
// JOINT_RANGE_MIN_DEG[i]/JOINT_RANGE_MAX_DEG[i]
int RAW_MIN[NUM_SERVOS] = {
  174, 387, 244,   // FL: hip, thigh, foot
  152, 210, 357,   // FR: hip, thigh, foot
  436, 219, 348,   // BL: hip, thigh, foot
  413, 383, 235    // BR: hip, thigh, foot
};
int RAW_MAX[NUM_SERVOS] = {
  451, 127, 491,   // FL: hip, thigh, foot
  433, 476, 130,   // FR: hip, thigh, foot
  175, 483, 122,   // BL: hip, thigh, foot
  188, 127, 477    // BR: hip, thigh, foot
};

const float DEG_TO_RAD_F = 0.01745329252f;

// per-channel reachable angle range (degrees), must match
// pca9685_commander_node.cpp's JOINT_RANGE_MIN_DEG/JOINT_RANGE_MAX_DEG
// exactly (that copy is the write-side safety clamp
float JOINT_RANGE_MIN_DEG[NUM_SERVOS] = {
  -70.0f, -40.0f, -30.0f,  //FL
  -70.0f, -40.0f, -30.0f,  //FR
  -65.0f, -40.0f, -30.0f,  //BL
  -60.0f, -40.0f, -30.0f   //BR
};
float JOINT_RANGE_MAX_DEG[NUM_SERVOS] = {
  70.0f, 90.0f, 90.0f,  //FL
  70.0f, 90.0f, 90.0f,  //FR
  65.0f, 90.0f, 90.0f,  //BL
  60.0f, 90.0f, 90.0f   //BR
};

const int NUM_AVG_SAMPLES = 4;

volatile float feedbackAngles[NUM_SERVOS];

const uint8_t I2C_SLAVE_ADDR = 0x08;
const int FLOATS_PER_CHUNK = 6;
volatile uint8_t requestedChunk = 0;

float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

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
    float rangeMinRad = JOINT_RANGE_MIN_DEG[i] * DEG_TO_RAD_F;
    float rangeMaxRad = JOINT_RANGE_MAX_DEG[i] * DEG_
    chunk[i] = feedbackAngles[startIdx + i];
  }
  Wire.write((uint8_t *)chunk, sizeof(chunk));
}

void setup() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    feedbackAngles[i] = 0.0f;
  }

  Wire.begin(I2C_SLAVE_ADDR);
  Wire.onReceive(onI2CReceive);
  Wire.onRequest(onI2CRequest);
}

void loop() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    analogRead(analogPins[i]);  // analogread settling
    long sum = 0;
    for (int s = 0; s < NUM_AVG_SAMPLES; s++) {
      sum += analogRead(analogPins[i]);
    }
    int raw = sum / NUM_AVG_SAMPLES;

    float rangeMinRad = JOINT_RANGE_MIN_DEG[i] * DEG_TO_RAD_F;
    float rangeMaxRad = JOINT_RANGE_MAX_DEG[i] * DEG_TO_RAD_F;
    float angle = mapFloat((float)raw, (float)RAW_MIN[i], (float)RAW_MAX[i], rangeMinRad, rangeMaxRad);
    feedbackAngles[i] = constrain(angle, rangeMinRad, rangeMaxRad);
  }
}
