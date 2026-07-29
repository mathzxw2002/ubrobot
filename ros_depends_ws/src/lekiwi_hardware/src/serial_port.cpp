#include "lekiwi_hardware/serial_port.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

namespace lekiwi_hardware
{
namespace
{

speed_t baud_constant(const int baud_rate)
{
  switch (baud_rate) {
    case 115200:
      return B115200;
    case 500000:
      return B500000;
    case 1000000:
      return B1000000;
    default:
      throw std::invalid_argument("supported baud rates are 115200, 500000, and 1000000");
  }
}

std::runtime_error system_error(const std::string & operation)
{
  return std::runtime_error(operation + ": " + std::strerror(errno));
}

}  // namespace

SerialPort::~SerialPort()
{
  close();
}

void SerialPort::open(const std::string & device, const int baud_rate)
{
  close();
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC | O_NONBLOCK);
  if (fd_ < 0) {
    throw system_error("failed to open " + device);
  }

  try {
    termios config{};
    if (tcgetattr(fd_, &config) != 0) {
      throw system_error("tcgetattr failed");
    }
    cfmakeraw(&config);
    const speed_t speed = baud_constant(baud_rate);
    if (cfsetispeed(&config, speed) != 0 || cfsetospeed(&config, speed) != 0) {
      throw system_error("failed to set serial baud rate");
    }
    config.c_cflag |= CLOCAL | CREAD;
    config.c_cflag &= static_cast<tcflag_t>(~(PARENB | CSTOPB | CRTSCTS));
    config.c_cflag = static_cast<tcflag_t>((config.c_cflag & ~CSIZE) | CS8);
    config.c_cc[VMIN] = 0;
    config.c_cc[VTIME] = 0;
    if (tcsetattr(fd_, TCSANOW, &config) != 0) {
      throw system_error("tcsetattr failed");
    }
    const int flags = fcntl(fd_, F_GETFL, 0);
    if (flags < 0 || fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK) != 0) {
      throw system_error("failed to make serial port blocking");
    }
    flush();
  } catch (...) {
    close();
    throw;
  }
}

void SerialPort::close() noexcept
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialPort::is_open() const noexcept
{
  return fd_ >= 0;
}

void SerialPort::flush()
{
  if (!is_open() || tcflush(fd_, TCIOFLUSH) != 0) {
    throw system_error("failed to flush serial port");
  }
}

void SerialPort::write_all(const std::vector<uint8_t> & data)
{
  std::size_t written = 0;
  while (written < data.size()) {
    const ssize_t result = ::write(fd_, data.data() + written, data.size() - written);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw system_error("serial write failed");
    }
    written += static_cast<std::size_t>(result);
  }
  if (tcdrain(fd_) != 0) {
    throw system_error("serial drain failed");
  }
}

std::vector<uint8_t> SerialPort::read_exact(
  const std::size_t size, const std::chrono::milliseconds timeout)
{
  std::vector<uint8_t> data(size);
  std::size_t received = 0;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (received < size) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      throw std::runtime_error("serial read timed out");
    }
    pollfd descriptor{fd_, POLLIN, 0};
    const int poll_result = poll(&descriptor, 1, static_cast<int>(remaining.count()));
    if (poll_result == 0) {
      throw std::runtime_error("serial read timed out");
    }
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw system_error("serial poll failed");
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      throw std::runtime_error("serial device disconnected");
    }
    const ssize_t result = ::read(fd_, data.data() + received, size - received);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw system_error("serial read failed");
    }
    received += static_cast<std::size_t>(result);
  }
  return data;
}

}  // namespace lekiwi_hardware
