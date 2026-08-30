#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace ankybot_i2c_bridge
{
class I2CDevice
{
public:
  I2CDevice(int bus_number, uint8_t address);
  ~I2CDevice();

  I2CDevice(const I2CDevice &) = delete;
  I2CDevice & operator=(const I2CDevice &) = delete;

  void write(const std::vector<uint8_t> & data);

  std::vector<uint8_t> read(std::size_t length);

  std::vector<uint8_t> writeThenRead(
    const std::vector<uint8_t> & write_data, std::size_t read_length);

private:
  int fd_;
  uint8_t address_;
};

}  // namespace ankybot_i2c_bridge
