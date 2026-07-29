#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lekiwi_hardware::feetech
{

constexpr uint8_t kBroadcastId = 0xFE;
constexpr uint8_t kInstructionRead = 0x02;
constexpr uint8_t kInstructionWrite = 0x03;
constexpr uint8_t kInstructionSyncRead = 0x82;
constexpr uint8_t kInstructionSyncWrite = 0x83;

constexpr uint8_t kModelNumberAddress = 3;
constexpr uint8_t kReturnDelayAddress = 7;
constexpr uint8_t kOperatingModeAddress = 33;
constexpr uint8_t kTorqueEnableAddress = 40;
constexpr uint8_t kAccelerationAddress = 41;
constexpr uint8_t kGoalVelocityAddress = 46;
constexpr uint8_t kLockAddress = 55;
constexpr uint8_t kPresentVelocityAddress = 58;
constexpr uint8_t kMaximumAccelerationAddress = 85;
constexpr uint16_t kSts3215ModelNumber = 777;
constexpr uint8_t kVelocityMode = 1;
constexpr double kRawUnitsPerRevolution = 4096.0;
constexpr double kPi = 3.14159265358979323846;
constexpr uint16_t kSignBit = 0x8000;
constexpr uint16_t kMagnitudeMask = 0x7FFF;

inline uint16_t encode_sign_magnitude(const int value)
{
  const auto magnitude = static_cast<uint16_t>(
    std::min<int64_t>(std::llabs(static_cast<int64_t>(value)), kMagnitudeMask));
  return value < 0 ? static_cast<uint16_t>(magnitude | kSignBit) : magnitude;
}

inline int decode_sign_magnitude(const uint16_t value)
{
  const int magnitude = static_cast<int>(value & kMagnitudeMask);
  return (value & kSignBit) != 0U ? -magnitude : magnitude;
}

inline uint16_t radians_per_second_to_raw(const double radians_per_second, const int max_raw)
{
  if (!std::isfinite(radians_per_second)) {
    throw std::invalid_argument("wheel velocity must be finite");
  }
  if (max_raw <= 0 || max_raw > static_cast<int>(kMagnitudeMask)) {
    throw std::invalid_argument("max_raw must be in [1, 32767]");
  }
  const double raw = radians_per_second * kRawUnitsPerRevolution / (2.0 * kPi);
  const auto clipped = static_cast<int>(std::clamp(
    std::llround(raw), static_cast<long long>(-max_raw), static_cast<long long>(max_raw)));
  return encode_sign_magnitude(clipped);
}

inline double raw_to_radians_per_second(const uint16_t raw)
{
  return static_cast<double>(decode_sign_magnitude(raw)) * (2.0 * kPi) /
         kRawUnitsPerRevolution;
}

inline std::vector<uint8_t> make_packet(
  const uint8_t id, const uint8_t instruction, const std::vector<uint8_t> & parameters)
{
  if (parameters.size() > 251U) {
    throw std::invalid_argument("Feetech packet has too many parameters");
  }
  std::vector<uint8_t> packet{0xFF, 0xFF, id,
    static_cast<uint8_t>(parameters.size() + 2U), instruction};
  packet.insert(packet.end(), parameters.begin(), parameters.end());
  uint8_t sum = 0;
  for (std::size_t index = 2; index < packet.size(); ++index) {
    sum = static_cast<uint8_t>(sum + packet[index]);
  }
  packet.push_back(static_cast<uint8_t>(~sum));
  return packet;
}

inline uint16_t little_endian_u16(const std::vector<uint8_t> & data)
{
  if (data.size() != 2U) {
    throw std::invalid_argument("expected exactly two bytes");
  }
  return static_cast<uint16_t>(data[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8U);
}

inline std::array<uint8_t, 2> to_little_endian(const uint16_t value)
{
  return {static_cast<uint8_t>(value & 0xFFU), static_cast<uint8_t>(value >> 8U)};
}

}  // namespace lekiwi_hardware::feetech
