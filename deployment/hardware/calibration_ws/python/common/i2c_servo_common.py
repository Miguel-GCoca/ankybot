#!/usr/bin/env python3
"""
Shared I2C driver code for the servo-characterization scripts
(calibrate_i2c.py, run_step_response_sweep.py, find_calibration_extremes.py,
run_step_response_targets.py). Drives/reads a single joint through the same
physical path as the real robot: PCA9685 servo driver (commanded directly
over I2C, no ROS here) + Arduino Mega I2C slave (feedback only).

PCA9685 driver, angle_to_pulse(), JOINT_ORDER, JOINT_ORIENTATION, and
JOINT_ZERO_TRIM_DEG (2026-07-20, per-channel physical zero-position trim)
are ported verbatim from the actual deployed nodes so a channel commanded
here behaves identically to how the real pipeline commands it:
  - ankybot_i2c_bridge/pca9685_commander_node.cpp (register sequence,
    angleToPulse formula, JOINT_ORIENTATION - this is the sole commander;
    the Python serial_communication_py/pca9685_commander.py version was
    deleted 2026-07-20)
  - arduino_mega_i2c_slave.ino (matching JOINT_ORIENTATION on the feedback
    side)
Mega feedback chunk-read protocol matches mega_feedback_reader.py /
arduino_mega_i2c_slave.ino: 2 chunks of 6 floats, one-byte chunk-select
write followed by a 24-byte read in the same transaction (repeated START).
That wire format is shared by both Mega firmware variants used here -
arduino_mega_i2c_slave.ino (production, floats are calibrated radians) and
arduino_mega_i2c_calibrate.ino (raw ADC counts as floats) - only the
meaning of the floats differs, so the same read code works against either.
"""
import struct
import time

from smbus2 import SMBus, i2c_msg

JOINT_ORDER = [
    'FL_Hip_Joint', 'FL_Thigh_Joint', 'FL_Foot_Joint',
    'FR_Hip_Joint', 'FR_Thigh_Joint', 'FR_Foot_Joint',
    'BL_Hip_Joint', 'BL_Thigh_Joint', 'BL_Foot_Joint',
    'BR_Hip_Joint', 'BR_Thigh_Joint', 'BR_Foot_Joint',
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_ORDER)}
NUM_SERVOS = 12

# Matches arduino_mega_i2c_slave.ino / pca9685_commander_node.cpp exactly - do not edit here without also updating those, or bench-test behavior will no longer match real deployment behavior.
JOINT_ORIENTATION = [
    -1, 1, -1,   # FL
    -1, -1, 1,   # FR
    1, -1, 1,   # BL
    1, 1, -1,   # BR
]

# per-channel physical zero-position trim (degrees) - must match pca9685_commander_node.cpp's JOINT_ZERO_TRIM_DEG exactly. Hardware-measured 2026-07-22 (see CLAUDE_HISTORY.md).
JOINT_ZERO_TRIM_DEG = [
    0.0, 2.8648, 2.8648,   # FL
    8.5944, 8.5944, 5.7296,   # FR
    2.8648, 5.7296, 8.5944,   # BL
    5.7296, 2.8648, 8.5944,   # BR
]

PI = 3.14159265
HALF_PI = 1.57079632679

PCA9685_ADDRESS = 0x60
MEGA_ADDRESS = 0x08
I2C_BUS = 1       # PCA9685 (and BNO085) stay on the Pi's original bus
# Mega moved to a second Pi I2C controller (i2c2-pi5 overlay, GPIO4/5) to
# isolate its 200Hz polling from BNO085 bus contention - see
# ankybot_i2c_bridge/launch/i2c_bridge.launch.py's mega_i2c_bus arg (same
# value, same reason). PCA9685_ADDRESS/MEGA_ADDRESS are unchanged - only
# the bus differs. Bench scripts must open a separate SMBus(MEGA_I2C_BUS)
# for MegaFeedback rather than sharing the PCA9685's bus object.
MEGA_I2C_BUS = 2
PWM_FREQ_HZ = 300.0
SERVO_MIN = 614
SERVO_MAX = 3072

FLOATS_PER_CHUNK = 6
NUM_CHUNKS = 2

# PCA9685 register map (matches Adafruit_PWMServoDriver.h)
PCA9685_MODE1 = 0x00
PCA9685_PRESCALE = 0xFE
PCA9685_LED0_ON_L = 0x06


class PCA9685:
    """Same register addresses/init sequence as Adafruit_PWMServoDriver -
    ported verbatim from pca9685_commander.py / pca9685.cpp."""

    def __init__(self, bus: SMBus, address: int = PCA9685_ADDRESS, freq_hz: float = PWM_FREQ_HZ):
        self.bus = bus
        self.address = address
        self._reset()
        self._set_pwm_freq(freq_hz)

    def _write8(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def _read8(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def _reset(self):
        self._write8(PCA9685_MODE1, 0x00)
        time.sleep(0.01)

    def _set_pwm_freq(self, freq_hz):
        prescale_val = 25000000.0
        prescale_val /= 4096.0
        prescale_val /= freq_hz
        prescale_val -= 1.0
        prescale = int(prescale_val + 0.5)

        old_mode = self._read8(PCA9685_MODE1)
        new_mode = (old_mode & 0x7F) | 0x10  # sleep - required to change prescale
        self._write8(PCA9685_MODE1, new_mode)
        self._write8(PCA9685_PRESCALE, prescale)
        self._write8(PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self._write8(PCA9685_MODE1, old_mode | 0xA1)  # restart + auto-increment

    def set_pwm(self, channel, on, off):
        base = PCA9685_LED0_ON_L + 4 * channel
        self._write8(base, on & 0xFF)
        self._write8(base + 1, (on >> 8) & 0xFF)
        self._write8(base + 2, off & 0xFF)
        self._write8(base + 3, (off >> 8) & 0xFF)


def angle_to_pulse(angle_rad, servo_min=SERVO_MIN, servo_max=SERVO_MAX):
    """Same formula as arduino_mega_final.ino / pca9685_commander_node.cpp."""
    shifted = angle_rad + HALF_PI  # remap to [0, PI]
    shifted = max(0.0, min(PI, shifted))
    return servo_min + int((shifted / PI) * (servo_max - servo_min) + 0.5)


def resolve_joint(name_or_index):
    """Accepts a JOINT_ORDER name (e.g. 'FL_Thigh_Joint') or a bare channel
    index (0-11, e.g. '4')."""
    try:
        return int(name_or_index)
    except ValueError:
        pass
    idx = JOINT_INDEX.get(name_or_index)
    if idx is None:
        raise ValueError(
            f"'{name_or_index}' is not a valid joint name or channel index 0-11. "
            f"Valid names: {JOINT_ORDER}"
        )
    return idx


class ServoChannel:
    """Commands one joint's PCA9685 channel with the same orientation/
    neutral convention as pca9685_commander_node.cpp (servo_neutral=0.0 by
    default - that node's actual deployed parameters, no override yaml
    exists for ankybot_i2c_bridge as of 2026-07-17)."""

    def __init__(self, pca: PCA9685, index: int, servo_neutral: float = 0.0):
        self.pca = pca
        self.index = index
        self.orientation = JOINT_ORIENTATION[index]
        self.zero_trim_rad = JOINT_ZERO_TRIM_DEG[index] * PI / 180.0
        self.servo_neutral = servo_neutral

    def command_deg(self, signed_deg):
        angle_rad = signed_deg * PI / 180.0
        angle_rad = max(-HALF_PI, min(HALF_PI, angle_rad))
        target = self.orientation * (self.servo_neutral + angle_rad) + self.zero_trim_rad
        pulse = angle_to_pulse(target)
        self.pca.set_pwm(self.index, 0, pulse)


class MegaFeedback:
    """Reads channels from the Mega I2C slave (2 chunks of 6 floats,
    repeated-START read per chunk). Works against either Mega firmware
    variant - see module docstring."""

    def __init__(self, bus: SMBus, address: int = MEGA_ADDRESS):
        self.bus = bus
        self.address = address

    def _read_chunk(self, chunk_idx):
        write = i2c_msg.write(self.address, [chunk_idx])
        read = i2c_msg.read(self.address, FLOATS_PER_CHUNK * 4)
        self.bus.i2c_rdwr(write, read)
        return struct.unpack('<6f', bytes(read))

    def read_all(self):
        values = []
        for chunk_idx in range(NUM_CHUNKS):
            values.extend(self._read_chunk(chunk_idx))
        return values

    def read_channel(self, index):
        chunk_idx = index // FLOATS_PER_CHUNK
        offset = index % FLOATS_PER_CHUNK
        return self._read_chunk(chunk_idx)[offset]
