#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "lekiwi_hardware/feetech_bus.hpp"

namespace lekiwi_hardware
{

class LeKiwiSystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(LeKiwiSystemHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_error(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void stop_and_disconnect() noexcept;

  static constexpr std::size_t kWheelCount = 3;
  std::array<double, kWheelCount> velocity_commands_{};
  std::array<double, kWheelCount> position_states_{};
  std::array<double, kWheelCount> velocity_states_{};
  std::array<int, kWheelCount> direction_{{1, 1, 1}};
  FeetechBus::MotorIds motor_ids_{{8, 9, 7}};  // back, right, left
  FeetechBus bus_;
  std::string device_{"/dev/lekiwi-base"};
  int baud_rate_{1000000};
  int max_raw_velocity_{300};
  bool enable_motor_torque_{false};
  std::chrono::milliseconds serial_timeout_{20};
};

}  // namespace lekiwi_hardware
