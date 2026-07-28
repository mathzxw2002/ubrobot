#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lekiwi_hardware
{

class SerialPort
{
public:
  SerialPort() = default;
  ~SerialPort();

  SerialPort(const SerialPort &) = delete;
  SerialPort & operator=(const SerialPort &) = delete;

  void open(const std::string & device, int baud_rate);
  void close() noexcept;
  bool is_open() const noexcept;
  void flush();
  void write_all(const std::vector<uint8_t> & data);
  std::vector<uint8_t> read_exact(std::size_t size, std::chrono::milliseconds timeout);

private:
  int fd_{-1};
};

}  // namespace lekiwi_hardware
