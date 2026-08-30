#include <Servo.h>
#include <Wire.h>

// =====================================================================
//  DINO HEAD - 2 DOF (Yaw + Pitch)
//  4 modes, switched over I2C by a speech-to-text host:
//    1 = Idle             (slow random look-around) - DEFAULT ON POWER-UP
//    2 = Heard Command     (nod acknowledgment x2, then back to idle)
//    3 = Walking            (pitch bobs 0 <-> 30 continuously)
//    4 = Turning             (yaw sweeps -20 <-> +20 continuously)
//  Whatever mode is active keeps running until a different command byte
//  arrives - including going back to idle, which is its own command (1).
// =====================================================================

// ---------------------------------------------------------------
// PIN ASSIGNMENTS
// ---------------------------------------------------------------
const int YAW_PIN   = 9;   // servo mounted vertically, rotates head left/right
const int PITCH_PIN = 10;  // servo mounted on its side, tilts head up/down

// ---------------------------------------------------------------
// SERVO HARDWARE SPEC (both servos are identical 270 deg / 25kg units)
// Do not need to touch this unless you change servo hardware.
// ---------------------------------------------------------------
const int   SERVO_PULSE_MIN     = 500;   // microseconds -> mechanical -135 deg
const int   SERVO_PULSE_MAX     = 2500;  // microseconds -> mechanical +135 deg
const int   SERVO_PULSE_CENTER  = 1500;  // microseconds -> 0 deg (your zero point)
const float SERVO_HALF_RANGE_DEG = 135.0; // matches your +/-135 deg spec

// =====================================================================
//  >>> EASY-EDIT ZONE: MODE 1 - IDLE (random look-around range) <<<
// =====================================================================
float YAW_MIN_ANGLE = -60;   // how far the head can turn toward its LEFT
float YAW_MAX_ANGLE =  60;   // how far the head can turn toward its RIGHT

float PITCH_MIN_ANGLE = -5;  // how far the head can dip DOWN
float PITCH_MAX_ANGLE =  35; // how far the head can lift UP

const unsigned long MOVE_DURATION_MIN  = 1200;  // ms - fastest a look-move happens
const unsigned long MOVE_DURATION_MAX  = 2000;  // ms - slowest a look-move happens
const unsigned long PAUSE_DURATION_MIN = 800;   // ms - shortest "hold and stare"
const unsigned long PAUSE_DURATION_MAX = 2000;  // ms - longest "hold and stare"

// ---- Motion feel, applies to ALL modes ----
// Low-pass filter (0-1): lower = smoother/more lag, higher = snappier/more jitter-prone
const float POSITION_SMOOTHING = 0.5;
// Curve softness (0-1): 1.0 = full ease-in-out, 0.0 = constant speed.
// Lower this if motion looks choppy/stuttery (coarse-resolution servos).
const float EASE_SOFTNESS = 1.0;

// =====================================================================
//  >>> EASY-EDIT ZONE: I2C <<<
// =====================================================================
const byte I2C_ADDRESS = 0x5B; // slave address the dino listens on

// Command byte values the speech-to-text host sends to select a mode.
const byte CMD_IDLE          = 1;
const byte CMD_HEARD_COMMAND = 2;
const byte CMD_WALKING       = 3;
const byte CMD_TURNING       = 4;

// =====================================================================
//  >>> EASY-EDIT ZONE: MODE 2 - HEARD COMMAND (double-nod sequence) <<<
//  Waypoints run in order, each fully finishing before the next starts.
//  The actual list is assigned in setup() below (search "i2cSequence[").
// =====================================================================
const unsigned long SEQUENCE_STEP_HOLD_MS = 250; // pause held at each waypoint, ms

// =====================================================================
//  >>> EASY-EDIT ZONE: MODE 3 - WALKING (pitch bob) <<<
// =====================================================================
float WALK_DOWN_ANGLE = 0;    // pitch angle at the "down" point of the bob
float WALK_UP_ANGLE   = 30;   // pitch angle at the "up" point of the bob
const unsigned long WALK_STEP_DURATION = 800; // ms per up or down movement (bob speed)
const unsigned long WALK_HOLD_MS       = 0;   // pause at each extreme, ms (0 = continuous)

// =====================================================================
//  >>> EASY-EDIT ZONE: MODE 4 - TURNING (yaw sweep) <<<
// =====================================================================
float TURN_LEFT_ANGLE  = -20;
float TURN_RIGHT_ANGLE =  20;
const unsigned long TURN_STEP_DURATION = 1500; // ms per left or right sweep
const unsigned long TURN_HOLD_MS       = 300;  // pause at each extreme, ms

// Time given to smoothly zero the "other" axis when entering walking/turning
// (e.g. yaw centers before walking bob starts; pitch centers before turn starts)
const unsigned long MODE_SETTLE_DURATION = 500;

// ---------------------------------------------------------------
// Internal state - no need to edit below this line
// ---------------------------------------------------------------
Servo yawServo;
Servo pitchServo;

enum MotionState { MOVING, PAUSED };
enum Mode { MODE_IDLE, MODE_HEARD_COMMAND, MODE_WALKING, MODE_TURNING };
enum PhaseState { PHASE_SETTLING, PHASE_MOVING, PHASE_HOLDING };

struct Axis {
  Servo* servo;
  float minAngle;
  float maxAngle;
  float startAngle;
  float targetAngle;
  float currentAngle;    // ideal trajectory position (eased, unfiltered)
  float commandedAngle;  // actual position sent to servo (filtered/smoothed)
  unsigned long moveStart;
  unsigned long moveDuration;
  unsigned long pauseStart;
  unsigned long pauseDuration;
  MotionState state;
};

Axis yawAxis;
Axis pitchAxis;

Mode currentMode = MODE_IDLE;

volatile byte i2cCommandValue = 0;   // set inside the I2C interrupt, read in loop()
volatile bool i2cCommandPending = false;

// ---- Mode 2: heard-command sequence state ----
struct SequenceStep {
  Axis* axis;
  float targetAngle;
  unsigned long duration;
};
SequenceStep i2cSequence[6];
const int I2C_SEQUENCE_LENGTH = 6;
int sequenceIndex = 0;
PhaseState seqPhase = PHASE_MOVING;
unsigned long seqHoldStart = 0;

// ---- Modes 3 & 4: shared cyclic (walking/turning) state ----
struct CyclicModeConfig {
  Axis* primaryAxis;   // the axis that actually bobs/sweeps
  Axis* otherAxis;     // the axis that centers and holds still
  float boundA;
  float boundB;
  unsigned long stepDuration;
  unsigned long holdMs;
};
CyclicModeConfig walkConfig = { &pitchAxis, &yawAxis, WALK_DOWN_ANGLE, WALK_UP_ANGLE, WALK_STEP_DURATION, WALK_HOLD_MS };
CyclicModeConfig turnConfig = { &yawAxis, &pitchAxis, TURN_LEFT_ANGLE, TURN_RIGHT_ANGLE, TURN_STEP_DURATION, TURN_HOLD_MS };
PhaseState cyclePhase = PHASE_SETTLING;
unsigned long cycleHoldStart = 0;
bool cycleAtBoundB = false; // false = currently at/heading to boundA, true = boundB

// Smooth accel/decel curve (ease-in-out) so moves feel organic, not mechanical
float easeInOutSine(float t) {
  return -(cos(PI * t) - 1.0) / 2.0;
}

// Blends the eased curve with constant-speed motion, per EASE_SOFTNESS.
float blendedEase(float t) {
  float eased = easeInOutSine(t);
  return eased * EASE_SOFTNESS + t * (1.0 - EASE_SOFTNESS);
}

// Converts an angle (relative to center) into a servo pulse width
int angleToPulse(float angle) {
  angle = constrain(angle, -SERVO_HALF_RANGE_DEG, SERVO_HALF_RANGE_DEG);
  float pulse = SERVO_PULSE_CENTER +
                (angle / SERVO_HALF_RANGE_DEG) * (SERVO_PULSE_MAX - SERVO_PULSE_CENTER);
  return (int)pulse;
}

// Random float angle between two limits (2 decimal precision)
float randomAngle(float minA, float maxA) {
  long minL = (long)(minA * 100);
  long maxL = (long)(maxA * 100);
  return random(minL, maxL + 1) / 100.0;
}

void initAxis(Axis &axis, Servo* servo, float minAngle, float maxAngle) {
  axis.servo = servo;
  axis.minAngle = minAngle;
  axis.maxAngle = maxAngle;
  axis.currentAngle = 0;
  axis.commandedAngle = 0;
  axis.startAngle = 0;
  axis.targetAngle = randomAngle(minAngle, maxAngle);
  axis.moveStart = millis();
  axis.moveDuration = random(MOVE_DURATION_MIN, MOVE_DURATION_MAX);
  axis.state = MOVING;
}

// Points an axis toward a new target over a given duration, starting from
// wherever it currently is - this is what makes every mode transition smooth.
void startAxisMove(Axis &axis, float target, unsigned long duration) {
  axis.startAngle = axis.commandedAngle;
  axis.targetAngle = target;
  axis.moveStart = millis();
  axis.moveDuration = duration;
}

// Drives one axis toward axis.targetAngle over axis.moveDuration, through the
// filter, and reports back whether it has arrived (t >= 1.0).
bool driveAxisTrajectory(Axis &axis) {
  unsigned long now = millis();
  unsigned long elapsed = now - axis.moveStart;
  float t = (float)elapsed / (float)axis.moveDuration;
  bool arrived = false;
  if (t >= 1.0) {
    t = 1.0;
    arrived = true;
  }
  float eased = blendedEase(t);
  axis.currentAngle = axis.startAngle + (axis.targetAngle - axis.startAngle) * eased;
  axis.commandedAngle += (axis.currentAngle - axis.commandedAngle) * POSITION_SMOOTHING;
  axis.servo->writeMicroseconds(angleToPulse(axis.commandedAngle));
  return arrived;
}

// ---------------------------------------------------------------
// MODE 1: IDLE - slow, random "looking around" behavior
// ---------------------------------------------------------------
void updateStandbyAxis(Axis &axis) {
  unsigned long now = millis();

  if (axis.state == MOVING) {
    if (driveAxisTrajectory(axis)) {
      axis.state = PAUSED;
      axis.pauseStart = now;
      axis.pauseDuration = random(PAUSE_DURATION_MIN, PAUSE_DURATION_MAX);
    }
  } else { // PAUSED
    if (now - axis.pauseStart >= axis.pauseDuration) {
      axis.startAngle = axis.commandedAngle;
      axis.targetAngle = randomAngle(axis.minAngle, axis.maxAngle);
      axis.moveStart = now;
      axis.moveDuration = random(MOVE_DURATION_MIN, MOVE_DURATION_MAX);
      axis.state = MOVING;
    }
  }
}

// Restarts idle motion for an axis starting from wherever it currently is.
void resumeStandby(Axis &axis) {
  axis.startAngle = axis.commandedAngle;
  axis.currentAngle = axis.commandedAngle;
  axis.targetAngle = randomAngle(axis.minAngle, axis.maxAngle);
  axis.moveStart = millis();
  axis.moveDuration = random(MOVE_DURATION_MIN, MOVE_DURATION_MAX);
  axis.state = MOVING;
}

// ---------------------------------------------------------------
// MODE 2: HEARD COMMAND - double-nod sequence, then back to idle
// ---------------------------------------------------------------
void startSequenceStep(int idx) {
  SequenceStep &step = i2cSequence[idx];
  startAxisMove(*step.axis, step.targetAngle, step.duration);
  seqPhase = PHASE_MOVING;
}

void updateHeardCommandSequence() {
  SequenceStep &step = i2cSequence[sequenceIndex];
  Axis &axis = *step.axis;

  if (seqPhase == PHASE_MOVING) {
    if (driveAxisTrajectory(axis)) {
      seqPhase = PHASE_HOLDING;
      seqHoldStart = millis();
    }
  } else { // PHASE_HOLDING
    if (millis() - seqHoldStart >= SEQUENCE_STEP_HOLD_MS) {
      sequenceIndex++;
      if (sequenceIndex >= I2C_SEQUENCE_LENGTH) {
        enterMode(MODE_IDLE); // sequence finished -> resume idle automatically
      } else {
        startSequenceStep(sequenceIndex);
      }
    }
  }
}

// ---------------------------------------------------------------
// MODES 3 & 4: WALKING / TURNING - continuous cyclic motion.
// Shared logic: settle the "other" axis to 0, then ping-pong the primary
// axis between boundA and boundB until the mode is changed externally.
// ---------------------------------------------------------------
void enterCyclicMode(CyclicModeConfig &cfg) {
  cyclePhase = PHASE_SETTLING;
  startAxisMove(*cfg.otherAxis, 0, MODE_SETTLE_DURATION);
}

void updateCyclicMode(CyclicModeConfig &cfg) {
  if (cyclePhase == PHASE_SETTLING) {
    if (driveAxisTrajectory(*cfg.otherAxis)) {
      // pick whichever bound is closer to start the cycle smoothly
      float distA = fabs(cfg.primaryAxis->commandedAngle - cfg.boundA);
      float distB = fabs(cfg.primaryAxis->commandedAngle - cfg.boundB);
      cycleAtBoundB = (distB < distA);
      startAxisMove(*cfg.primaryAxis, cycleAtBoundB ? cfg.boundB : cfg.boundA, cfg.stepDuration);
      cyclePhase = PHASE_MOVING;
    }
  } else if (cyclePhase == PHASE_MOVING) {
    if (driveAxisTrajectory(*cfg.primaryAxis)) {
      if (cfg.holdMs > 0) {
        cyclePhase = PHASE_HOLDING;
        cycleHoldStart = millis();
      } else {
        cycleAtBoundB = !cycleAtBoundB;
        startAxisMove(*cfg.primaryAxis, cycleAtBoundB ? cfg.boundB : cfg.boundA, cfg.stepDuration);
      }
    }
  } else { // PHASE_HOLDING
    if (millis() - cycleHoldStart >= cfg.holdMs) {
      cycleAtBoundB = !cycleAtBoundB;
      startAxisMove(*cfg.primaryAxis, cycleAtBoundB ? cfg.boundB : cfg.boundA, cfg.stepDuration);
      cyclePhase = PHASE_MOVING;
    }
  }
}

// ---------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------
void enterMode(Mode m) {
  currentMode = m;
  switch (m) {
    case MODE_IDLE:
      resumeStandby(yawAxis);
      resumeStandby(pitchAxis);
      break;
    case MODE_HEARD_COMMAND:
      sequenceIndex = 0;
      startSequenceStep(0);
      break;
    case MODE_WALKING:
      enterCyclicMode(walkConfig);
      break;
    case MODE_TURNING:
      enterCyclicMode(turnConfig);
      break;
  }
}

// Called by the Wire library when data arrives at I2C_ADDRESS.
// Keep this minimal - it runs in an interrupt context.
void onI2CReceive(int numBytes) {
  if (numBytes > 0) {
    i2cCommandValue = Wire.read();
    while (Wire.available()) Wire.read(); // drain any extra bytes
  }
  i2cCommandPending = true;
}

void setup() {
  yawServo.attach(YAW_PIN, SERVO_PULSE_MIN, SERVO_PULSE_MAX);
  pitchServo.attach(PITCH_PIN, SERVO_PULSE_MIN, SERVO_PULSE_MAX);

  randomSeed(analogRead(A0)); // leave A0 floating/unconnected for good randomness

  initAxis(yawAxis, &yawServo, YAW_MIN_ANGLE, YAW_MAX_ANGLE);
  initAxis(pitchAxis, &pitchServo, PITCH_MIN_ANGLE, PITCH_MAX_ANGLE);

  // Mode 2 sequence: zero the yaw, then nod pitch (+45 -> -10) twice, then settle to 0.
  i2cSequence[0] = { &yawAxis,   0,   1000 };
  i2cSequence[1] = { &pitchAxis, 45,  900  };
  i2cSequence[2] = { &pitchAxis, -10, 900  };
  i2cSequence[3] = { &pitchAxis, 45,  900  };
  i2cSequence[4] = { &pitchAxis, -10, 900  };
  i2cSequence[5] = { &pitchAxis, 0,   900  };

  Wire.begin(I2C_ADDRESS);
  Wire.onReceive(onI2CReceive);

  // currentMode starts as MODE_IDLE by default - initAxis() above already put
  // both axes into a live idle "looking around" move, so no extra call needed.
}

void loop() {
  // Check for a new mode command from the speech-to-text host.
  if (i2cCommandPending) {
    i2cCommandPending = false;
    Mode newMode = currentMode;
    bool valid = true;
    switch (i2cCommandValue) {
      case CMD_IDLE:          newMode = MODE_IDLE;          break;
      case CMD_HEARD_COMMAND: newMode = MODE_HEARD_COMMAND; break;
      case CMD_WALKING:       newMode = MODE_WALKING;       break;
      case CMD_TURNING:       newMode = MODE_TURNING;       break;
      default: valid = false; break; // unrecognized byte, ignore
    }
    if (valid && newMode != currentMode) {
      enterMode(newMode);
    }
  }

  switch (currentMode) {
    case MODE_IDLE:
      updateStandbyAxis(yawAxis);
      updateStandbyAxis(pitchAxis);
      break;
    case MODE_HEARD_COMMAND:
      updateHeardCommandSequence();
      break;
    case MODE_WALKING:
      updateCyclicMode(walkConfig);
      break;
    case MODE_TURNING:
      updateCyclicMode(turnConfig);
      break;
  }

  delay(10); // ~100Hz refresh
}
