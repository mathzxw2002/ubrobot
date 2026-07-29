#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "lekiwi_hardware/serial_port.hpp"

namespace lekiwi_hardware
{

class FeetechBus
{
public:
  using MotorIds = std::array<uint8_t, 3>;
  using RawVelocities = std::array<uint16_t, 3>;

  void connect(
    const std::string & device, int baud_rate, std::chrono::milliseconds timeout,
    const MotorIds & motor_ids);
  void disconnect() noexcept;
  bool is_connected() const noexcept;

  void verify_sts3215_motors();
  void configure_velocity_mode();
  void enable_torque();
  void stop_and_disable() noexcept;
  void write_velocities(const RawVelocities & raw_velocities);
  RawVelocities read_velocities();

  // Write goal velocity zero to every motor with individually acknowledged
  // writes and read the register back. Sync writes are fire-and-forget, so a
  // silent failure would leave a stale goal that runs as soon as torque is
  // enabled. Throws unless every goal register reads back zero.
  void zero_goal_registers_verified();

  // Throw if any motor reports |present velocity| above max_steps. Used right
  // after torque enable to catch unexpected motion before the first control
  // cycle. max_steps is in raw Feetech steps/second (sign-magnitude coded).
  void assert_wheels_stationary(uint16_t max_steps);

private:
  struct StatusPacket
  {
    uint8_t id;
    uint8_t error;
    std::vector<uint8_t> parameters;
  };

  void send_instruction(
    uint8_t id, uint8_t instruction, const std::vector<uint8_t> & parameters);
  StatusPacket read_status(uint8_t expected_id);
  std::vector<uint8_t> read_register(uint8_t id, uint8_t address, uint8_t length);
  void write_register(uint8_t id, uint8_t address, const std::vector<uint8_t> & data);

  SerialPort serial_;
  MotorIds motor_ids_{};
  std::chrono::milliseconds timeout_{20};
};

}  // namespace lekiwi_hardware
