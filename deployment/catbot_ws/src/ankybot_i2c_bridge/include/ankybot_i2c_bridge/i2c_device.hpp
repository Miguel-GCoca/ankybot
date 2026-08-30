#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace ankybot_i2c_bridge
{

// Thin wrapper around a Linux i2c-dev bus handle bound to one target
// address - opens /dev/i2c-<bus>, binds via ioctl(I2C_SLAVE), and exposes
// raw write()/read() plus a combined write-then-read (repeated START)
// helper for the "select register/chunk, then read" pattern used by both
// the PCA9685 and the Mega feedback slave.
class I2CDevice
{
public:
  I2CDevice(int bus_number, uint8_t address);
  ~I2CDevice();

  I2CDevice(const I2CDevice &) = delete;
  I2CDevice & operator=(const I2CDevice &) = delete;

  // Plain write of `data` in one I2C transaction.
  void write(const std::vector<uint8_t> & data);

  // Plain read of `length` bytes in one I2C transaction.
  std::vector<uint8_t> read(std::size_t length);

  // Writes `write_data` then reads `read_length` bytes back in a single
  // combined I2C transaction (repeated START, no STOP in between).
  std::vector<uint8_t> writeThenRead(
    const std::vector<uint8_t> & write_data, std::size_t read_length);

private:
  int fd_;
  uint8_t address_;
};

}  // namespace ankybot_i2c_bridge
