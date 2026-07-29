#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lekiwi_hardware/feetech_protocol.hpp"

namespace ft = lekiwi_hardware::feetech;

TEST(FeetechProtocol, EncodesSignMagnitudeRatherThanTwosComplement)
{
  EXPECT_EQ(ft::encode_sign_magnitude(123), 123U);
  EXPECT_EQ(ft::encode_sign_magnitude(-123), 0x807BU);
  EXPECT_EQ(ft::decode_sign_magnitude(0x807B), -123);
}

TEST(FeetechProtocol, ConvertsRadiansPerSecondAndClamps)
{
  EXPECT_EQ(ft::radians_per_second_to_raw(2.0 * ft::kPi, 5000), 4096U);
  EXPECT_EQ(ft::radians_per_second_to_raw(-2.0 * ft::kPi, 5000), 0x9000U);
  EXPECT_EQ(ft::radians_per_second_to_raw(100.0, 3000), 3000U);
  EXPECT_NEAR(ft::raw_to_radians_per_second(0x9000), -2.0 * ft::kPi, 1e-12);
}

TEST(FeetechProtocol, BuildsProtocolZeroPacketWithChecksum)
{
  const auto packet = ft::make_packet(7, ft::kInstructionRead, {58, 2});
  const std::vector<uint8_t> expected{0xFF, 0xFF, 7, 4, 2, 58, 2, 0xB6};
  EXPECT_EQ(packet, expected);
}

TEST(FeetechProtocol, RejectsNonFiniteVelocity)
{
  EXPECT_THROW(
    ft::radians_per_second_to_raw(std::numeric_limits<double>::quiet_NaN(), 3000),
    std::invalid_argument);
}
