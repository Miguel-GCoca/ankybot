#pragma once
#include <cstdint>

#include "ankybot_i2c_bridge/i2c_device.hpp"

namespace ankybot_i2c_bridge
{
class PCA9685
{
public:
  PCA9685(int bus_number, uint8_t address, float freq_hz);

  // Same semantics as Adafruit_PWMServoDriver::setPWM(channel, on, off).
  void setPWM(uint8_t channel, uint16_t on, uint16_t off);

private:
  void reset();
  void setPWMFreq(float freq_hz);
  void write8(uint8_t reg, uint8_t value);
  uint8_t read8(uint8_t reg);

  I2CDevice device_;
};

}  // namespace ankybot_i2c_bridge
