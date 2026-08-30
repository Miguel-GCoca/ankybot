#include "ankybot_i2c_bridge/i2c_device.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ankybot_i2c_bridge
{

namespace
{
std::string addressHex(uint8_t address)
{
  char buf[8];
  std::snprintf(buf, sizeof(buf), "0x%02x", address);
  return std::string(buf);
}

// ENXIO/EREMOTEIO = nothing acked the address; matches the wording the Python IMU stack already uses for the same condition.
std::string describeI2CError(const char * op, uint8_t address, int err)
{
  if (err == ENXIO || err == EREMOTEIO) {
    return "No I2C device found at address " + addressHex(address);
  }
  return std::string("I2C ") + op + " failed at address " + addressHex(address) +
    ": " + std::strerror(err) + " (errno " + std::to_string(err) + ")";
}
}  // namespace

I2CDevice::I2CDevice(int bus_number, uint8_t address)
: fd_(-1), address_(address)
{
  std::string path = "/dev/i2c-" + std::to_string(bus_number);
  fd_ = ::open(path.c_str(), O_RDWR);
  if (fd_ < 0) {
    int err = errno;
    throw std::runtime_error(
      "Failed to open " + path + ": " + std::strerror(err) + " (errno " +
      std::to_string(err) + ")");
  }
  if (::ioctl(fd_, I2C_SLAVE, address_) < 0) {
    int err = errno;
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error(
      "Failed to bind I2C slave address " + addressHex(address_) + ": " +
      std::strerror(err) + " (errno " + std::to_string(err) + ")");
  }
}

I2CDevice::~I2CDevice()
{
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

void I2CDevice::write(const std::vector<uint8_t> & data)
{
  ssize_t written = ::write(fd_, data.data(), data.size());
  if (written < 0) {
    throw std::runtime_error(describeI2CError("write", address_, errno));
  }
  if (static_cast<std::size_t>(written) != data.size()) {
    throw std::runtime_error(
      "Incomplete I2C write to address " + addressHex(address_) + ": wrote " +
      std::to_string(written) + " of " + std::to_string(data.size()) + " bytes");
  }
}

std::vector<uint8_t> I2CDevice::read(std::size_t length)
{
  std::vector<uint8_t> buf(length);
  ssize_t got = ::read(fd_, buf.data(), length);
  if (got < 0) {
    throw std::runtime_error(describeI2CError("read", address_, errno));
  }
  if (static_cast<std::size_t>(got) != length) {
    throw std::runtime_error(
      "Incomplete I2C read from address " + addressHex(address_) + ": got " +
      std::to_string(got) + " of " + std::to_string(length) + " bytes");
  }
  return buf;
}

std::vector<uint8_t> I2CDevice::writeThenRead(
  const std::vector<uint8_t> & write_data, std::size_t read_length)
{
  std::vector<uint8_t> read_buf(read_length);
  std::vector<uint8_t> write_copy(write_data);  // i2c_msg wants a non-const buf pointer

  i2c_msg msgs[2];
  msgs[0].addr = address_;
  msgs[0].flags = 0;  // write
  msgs[0].len = static_cast<uint16_t>(write_copy.size());
  msgs[0].buf = write_copy.data();

  msgs[1].addr = address_;
  msgs[1].flags = I2C_M_RD;  // read
  msgs[1].len = static_cast<uint16_t>(read_buf.size());
  msgs[1].buf = read_buf.data();

  i2c_rdwr_ioctl_data rdwr;
  rdwr.msgs = msgs;
  rdwr.nmsgs = 2;

  if (::ioctl(fd_, I2C_RDWR, &rdwr) < 0) {
    throw std::runtime_error(describeI2CError("combined write/read transaction", address_, errno));
  }

  return read_buf;
}

}  // namespace ankybot_i2c_bridge
