#include "ankybot_i2c_bridge/pca9685.hpp"

#include <chrono>
#include <thread>

namespace ankybot_i2c_bridge
{

namespace
{
// Same register map as Adafruit_PWMServoDriver.h
constexpr uint8_t MODE1 = 0x00;
constexpr uint8_t PRESCALE = 0xFE;
constexpr uint8_t LED0_ON_L = 0x06;
}  // namespace

PCA9685::PCA9685(int bus_number, uint8_t address, float freq_hz)
: device_(bus_number, address)
{
  reset();
  setPWMFreq(freq_hz);
}

void PCA9685::write8(uint8_t reg, uint8_t value)
{
  device_.write({reg, value});
}

uint8_t PCA9685::read8(uint8_t reg)
{
  auto result = device_.writeThenRead({reg}, 1);
  return result[0];
}

void PCA9685::reset()
{
  write8(MODE1, 0x00);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void PCA9685::setPWMFreq(float freq_hz)
{
  // Same prescale formula as Adafruit_PWMServoDriver::setPWMFreq().
  float prescale_val = 25000000.0f;
  prescale_val /= 4096.0f;
  prescale_val /= freq_hz;
  prescale_val -= 1.0f;
  uint8_t prescale = static_cast<uint8_t>(prescale_val + 0.5f);

  uint8_t old_mode = read8(MODE1);
  uint8_t new_mode = static_cast<uint8_t>((old_mode & 0x7F) | 0x10);  // sleep - required to set prescale
  write8(MODE1, new_mode);
  write8(PRESCALE, prescale);
  write8(MODE1, old_mode);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  write8(MODE1, static_cast<uint8_t>(old_mode | 0xA1));  // restart + auto-increment
}

void PCA9685::setPWM(uint8_t channel, uint16_t on, uint16_t off)
{
  // Deliberately 4 separate register writes (not one auto-increment burst,
  // even though the AI bit is set above) to match Adafruit_PWMServoDriver's
  // own setPWM() exactly - keeps this a line-for-line port of behavior
  // already proven on this robot, rather than an untested "optimization".
  uint8_t base = static_cast<uint8_t>(LED0_ON_L + 4 * channel);
  write8(base, static_cast<uint8_t>(on & 0xFF));
  write8(static_cast<uint8_t>(base + 1), static_cast<uint8_t>((on >> 8) & 0xFF));
  write8(static_cast<uint8_t>(base + 2), static_cast<uint8_t>(off & 0xFF));
  write8(static_cast<uint8_t>(base + 3), static_cast<uint8_t>((off >> 8) & 0xFF));
}

}  // namespace ankybot_i2c_bridge
