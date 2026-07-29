#include "lekiwi_hardware/feetech_bus.hpp"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

#include "lekiwi_hardware/feetech_protocol.hpp"

namespace lekiwi_hardware
{
namespace ft = feetech;

void FeetechBus::connect(
  const std::string & device, const int baud_rate, const std::chrono::milliseconds timeout,
  const MotorIds & motor_ids)
{
  if (timeout.count() <= 0) {
    throw std::invalid_argument("serial timeout must be greater than zero");
  }
  motor_ids_ = motor_ids;
  timeout_ = timeout;
  serial_.open(device, baud_rate);
}

void FeetechBus::disconnect() noexcept
{
  serial_.close();
}

bool FeetechBus::is_connected() const noexcept
{
  return serial_.is_open();
}

void FeetechBus::send_instruction(
  const uint8_t id, const uint8_t instruction, const std::vector<uint8_t> & parameters)
{
  serial_.write_all(ft::make_packet(id, instruction, parameters));
}

FeetechBus::StatusPacket FeetechBus::read_status(const uint8_t expected_id)
{
  uint8_t previous = 0;
  for (;;) {
    const uint8_t current = serial_.read_exact(1, timeout_)[0];
    if (previous == 0xFF && current == 0xFF) {
      break;
    }
    previous = current;
  }
  const auto prefix = serial_.read_exact(3, timeout_);
  const uint8_t id = prefix[0];
  const uint8_t length = prefix[1];
  const uint8_t error = prefix[2];
  if (id != expected_id || length < 2U) {
    throw std::runtime_error("unexpected Feetech status packet header");
  }
  const auto tail = serial_.read_exact(static_cast<std::size_t>(length - 1U), timeout_);
  std::vector<uint8_t> parameters(tail.begin(), tail.end() - 1);
  uint8_t sum = static_cast<uint8_t>(id + length + error);
  for (const auto value : parameters) {
    sum = static_cast<uint8_t>(sum + value);
  }
  const uint8_t expected_checksum = static_cast<uint8_t>(~sum);
  if (tail.back() != expected_checksum) {
    throw std::runtime_error("invalid Feetech status checksum");
  }
  if (error != 0U) {
    std::ostringstream message;
    message << "Feetech motor " << static_cast<int>(id)
            << " returned error 0x" << std::hex << static_cast<int>(error);
    throw std::runtime_error(message.str());
  }
  return {id, error, parameters};
}

std::vector<uint8_t> FeetechBus::read_register(
  const uint8_t id, const uint8_t address, const uint8_t length)
{
  send_instruction(id, ft::kInstructionRead, {address, length});
  auto status = read_status(id);
  if (status.parameters.size() != length) {
    throw std::runtime_error("Feetech register response has unexpected length");
  }
  return status.parameters;
}

void FeetechBus::write_register(
  const uint8_t id, const uint8_t address, const std::vector<uint8_t> & data)
{
  std::vector<uint8_t> parameters{address};
  parameters.insert(parameters.end(), data.begin(), data.end());
  send_instruction(id, ft::kInstructionWrite, parameters);
  static_cast<void>(read_status(id));
}

void FeetechBus::verify_sts3215_motors()
{
  for (const auto id : motor_ids_) {
    const uint16_t model = ft::little_endian_u16(read_register(id, ft::kModelNumberAddress, 2));
    if (model != ft::kSts3215ModelNumber) {
      throw std::runtime_error(
              "motor " + std::to_string(id) + " is not an STS3215 (model 777)");
    }
  }
}

void FeetechBus::configure_velocity_mode()
{
  for (const auto id : motor_ids_) {
    write_register(id, ft::kTorqueEnableAddress, {0});
    write_register(id, ft::kLockAddress, {0});
    write_register(id, ft::kReturnDelayAddress, {0});
    write_register(id, ft::kMaximumAccelerationAddress, {254});
    write_register(id, ft::kAccelerationAddress, {254});
    write_register(id, ft::kOperatingModeAddress, {ft::kVelocityMode});
  }
  zero_goal_registers_verified();
}

void FeetechBus::enable_torque()
{
  // Zero the goal registers with acknowledged writes before applying torque.
  // A stale goal left by a previous session would otherwise run immediately;
  // the earlier fire-and-forget sync write could not prove the registers
  // actually cleared.
  zero_goal_registers_verified();
  write_velocities({0, 0, 0});
  for (const auto id : motor_ids_) {
    write_register(id, ft::kTorqueEnableAddress, {1});
    write_register(id, ft::kLockAddress, {1});
  }
  // 150 steps/s is well above the observed ±50 steps/s idle quantization
  // jitter and far below the 300 steps/s first-test command clamp.
  assert_wheels_stationary(150);
}

void FeetechBus::stop_and_disable() noexcept
{
  if (!is_connected()) {
    return;
  }
  try {
    // Leave the goal registers at a proven zero so a later torque enable can
    // never run a stale goal, even if this shutdown path raced a failure.
    zero_goal_registers_verified();
    write_velocities({0, 0, 0});
    for (const auto id : motor_ids_) {
      write_register(id, ft::kTorqueEnableAddress, {0});
      write_register(id, ft::kLockAddress, {0});
    }
  } catch (...) {
    // Best effort during shutdown; disconnect still releases the serial bus.
  }
}

void FeetechBus::write_velocities(const RawVelocities & raw_velocities)
{
  std::vector<uint8_t> parameters{ft::kGoalVelocityAddress, 2};
  for (std::size_t index = 0; index < motor_ids_.size(); ++index) {
    const auto bytes = ft::to_little_endian(raw_velocities[index]);
    parameters.push_back(motor_ids_[index]);
    parameters.insert(parameters.end(), bytes.begin(), bytes.end());
  }
  send_instruction(ft::kBroadcastId, ft::kInstructionSyncWrite, parameters);
}

FeetechBus::RawVelocities FeetechBus::read_velocities()
{
  std::vector<uint8_t> parameters{ft::kPresentVelocityAddress, 2};
  parameters.insert(parameters.end(), motor_ids_.begin(), motor_ids_.end());
  send_instruction(ft::kBroadcastId, ft::kInstructionSyncRead, parameters);

  RawVelocities velocities{};
  for (std::size_t index = 0; index < motor_ids_.size(); ++index) {
    velocities[index] = ft::little_endian_u16(read_status(motor_ids_[index]).parameters);
  }
  return velocities;
}

void FeetechBus::zero_goal_registers_verified()
{
  for (const auto id : motor_ids_) {
    write_register(id, ft::kGoalVelocityAddress, {0, 0});
    const auto goal = ft::little_endian_u16(read_register(id, ft::kGoalVelocityAddress, 2));
    if (goal != 0U) {
      throw std::runtime_error(
              "motor " + std::to_string(id) +
              " goal velocity register did not clear (raw " + std::to_string(goal) + ")");
    }
  }
}

void FeetechBus::assert_wheels_stationary(const uint16_t max_steps)
{
  for (const auto id : motor_ids_) {
    const auto raw = ft::little_endian_u16(read_register(id, ft::kPresentVelocityAddress, 2));
    const int steps = ft::decode_sign_magnitude(raw);
    if (std::abs(steps) > static_cast<int>(max_steps)) {
      throw std::runtime_error(
              "motor " + std::to_string(id) + " moving unexpectedly at " +
              std::to_string(steps) + " steps/s right after torque enable");
    }
  }
}

}  // namespace lekiwi_hardware
