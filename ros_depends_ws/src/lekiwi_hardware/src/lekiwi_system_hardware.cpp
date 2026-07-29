#include "lekiwi_hardware/lekiwi_system_hardware.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <unordered_map>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

#include "lekiwi_hardware/feetech_protocol.hpp"

namespace lekiwi_hardware
{
namespace
{

const std::array<std::string, 3> kExpectedJoints{
  "base_back_wheel_joint", "base_right_wheel_joint", "base_left_wheel_joint"};

std::string required_parameter(
  const std::unordered_map<std::string, std::string> & parameters, const std::string & name)
{
  const auto found = parameters.find(name);
  if (found == parameters.end() || found->second.empty()) {
    throw std::invalid_argument("missing hardware parameter: " + name);
  }
  return found->second;
}

int integer_parameter(
  const std::unordered_map<std::string, std::string> & parameters, const std::string & name)
{
  const std::string text = required_parameter(parameters, name);
  std::size_t parsed = 0;
  const int value = std::stoi(text, &parsed);
  if (parsed != text.size()) {
    throw std::invalid_argument("hardware parameter is not an integer: " + name);
  }
  return value;
}

bool boolean_parameter(
  const std::unordered_map<std::string, std::string> & parameters, const std::string & name)
{
  const std::string text = required_parameter(parameters, name);
  if (text == "true") {
    return true;
  }
  if (text == "false") {
    return false;
  }
  throw std::invalid_argument("hardware parameter must be 'true' or 'false': " + name);
}

uint8_t motor_id_parameter(
  const std::unordered_map<std::string, std::string> & parameters, const std::string & name)
{
  const int value = integer_parameter(parameters, name);
  if (value < 1 || value >= static_cast<int>(feetech::kBroadcastId)) {
    throw std::invalid_argument("motor ID must be in [1, 253]: " + name);
  }
  return static_cast<uint8_t>(value);
}

}  // namespace

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  try {
    if (info_.joints.size() != kWheelCount) {
      throw std::invalid_argument("LeKiwi hardware requires exactly three wheel joints");
    }
    for (std::size_t index = 0; index < kWheelCount; ++index) {
      const auto & joint = info_.joints[index];
      if (joint.name != kExpectedJoints[index]) {
        throw std::invalid_argument(
                "wheel joints must be ordered back, right, left; got " + joint.name);
      }
      if (joint.command_interfaces.size() != 1U ||
        joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY ||
        joint.state_interfaces.size() != 2U ||
        joint.state_interfaces[0].name != hardware_interface::HW_IF_POSITION ||
        joint.state_interfaces[1].name != hardware_interface::HW_IF_VELOCITY)
      {
        throw std::invalid_argument(
                "each wheel must expose a velocity command plus position and velocity states");
      }
    }

    const auto & parameters = info_.hardware_parameters;
    device_ = required_parameter(parameters, "device");
    baud_rate_ = integer_parameter(parameters, "baud_rate");
    serial_timeout_ = std::chrono::milliseconds(integer_parameter(parameters, "serial_timeout_ms"));
    max_raw_velocity_ = integer_parameter(parameters, "max_raw_velocity");
    enable_motor_torque_ = boolean_parameter(parameters, "enable_motor_torque");
    motor_ids_ = {
      motor_id_parameter(parameters, "back_motor_id"),
      motor_id_parameter(parameters, "right_motor_id"),
      motor_id_parameter(parameters, "left_motor_id")};
    direction_ = {
      integer_parameter(parameters, "back_direction"),
      integer_parameter(parameters, "right_direction"),
      integer_parameter(parameters, "left_direction")};

    if (serial_timeout_.count() <= 0 || max_raw_velocity_ <= 0 || max_raw_velocity_ > 32767) {
      throw std::invalid_argument("invalid timeout or max_raw_velocity hardware parameter");
    }
    for (std::size_t index = 0; index < kWheelCount; ++index) {
      if (direction_[index] != -1 && direction_[index] != 1) {
        throw std::invalid_argument("wheel direction must be -1 or 1");
      }
    }
    if (motor_ids_[0] == motor_ids_[1] || motor_ids_[0] == motor_ids_[2] ||
      motor_ids_[1] == motor_ids_[2])
    {
      throw std::invalid_argument("wheel motor IDs must be unique");
    }
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "Invalid LeKiwi hardware configuration: %s", exception.what());
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
LeKiwiSystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  interfaces.reserve(2U * kWheelCount);
  for (std::size_t index = 0; index < kWheelCount; ++index) {
    interfaces.emplace_back(
      info_.joints[index].name, hardware_interface::HW_IF_POSITION, &position_states_[index]);
    interfaces.emplace_back(
      info_.joints[index].name, hardware_interface::HW_IF_VELOCITY, &velocity_states_[index]);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
LeKiwiSystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  interfaces.reserve(kWheelCount);
  for (std::size_t index = 0; index < kWheelCount; ++index) {
    interfaces.emplace_back(
      info_.joints[index].name, hardware_interface::HW_IF_VELOCITY, &velocity_commands_[index]);
  }
  return interfaces;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_configure(
  const rclcpp_lifecycle::State &)
{
  try {
    std::fill(velocity_commands_.begin(), velocity_commands_.end(), 0.0);
    std::fill(position_states_.begin(), position_states_.end(), 0.0);
    std::fill(velocity_states_.begin(), velocity_states_.end(), 0.0);
    bus_.connect(device_, baud_rate_, serial_timeout_, motor_ids_);
    bus_.verify_sts3215_motors();
    bus_.configure_velocity_mode();
    RCLCPP_INFO(get_logger(), "Configured LeKiwi STS3215 bus on %s", device_.c_str());
    return hardware_interface::CallbackReturn::SUCCESS;
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "Failed to configure LeKiwi hardware: %s", exception.what());
    stop_and_disconnect();
    return hardware_interface::CallbackReturn::ERROR;
  }
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_activate(
  const rclcpp_lifecycle::State &)
{
  try {
    std::fill(velocity_commands_.begin(), velocity_commands_.end(), 0.0);
    if (enable_motor_torque_) {
      bus_.enable_torque();
      RCLCPP_WARN(get_logger(), "LeKiwi motor torque ENABLED with zero command");
    } else {
      bus_.stop_and_disable();
      RCLCPP_INFO(get_logger(), "LeKiwi bus active in torque-disabled preflight mode");
    }
    return hardware_interface::CallbackReturn::SUCCESS;
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "Failed to activate LeKiwi hardware: %s", exception.what());
    stop_and_disconnect();
    return hardware_interface::CallbackReturn::ERROR;
  }
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  bus_.stop_and_disable();
  std::fill(velocity_commands_.begin(), velocity_commands_.end(), 0.0);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  stop_and_disconnect();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  stop_and_disconnect();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn LeKiwiSystemHardware::on_error(
  const rclcpp_lifecycle::State &)
{
  stop_and_disconnect();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type LeKiwiSystemHardware::read(
  const rclcpp::Time &, const rclcpp::Duration & period)
{
  try {
    const auto raw_velocities = bus_.read_velocities();
    for (std::size_t index = 0; index < kWheelCount; ++index) {
      velocity_states_[index] = static_cast<double>(direction_[index]) *
        feetech::raw_to_radians_per_second(raw_velocities[index]);
      if (std::isfinite(period.seconds()) && period.seconds() > 0.0) {
        position_states_[index] += velocity_states_[index] * period.seconds();
      }
    }
    return hardware_interface::return_type::OK;
  } catch (const std::exception & exception) {
    bus_.stop_and_disable();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "LeKiwi feedback failed; torque disabled: %s", exception.what());
    return hardware_interface::return_type::ERROR;
  }
}

hardware_interface::return_type LeKiwiSystemHardware::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  try {
    if (!enable_motor_torque_) {
      bus_.write_velocities({0, 0, 0});
      return hardware_interface::return_type::OK;
    }
    FeetechBus::RawVelocities raw_velocities{};
    for (std::size_t index = 0; index < kWheelCount; ++index) {
      if (!std::isfinite(velocity_commands_[index])) {
        throw std::runtime_error("controller produced a non-finite wheel velocity");
      }
      raw_velocities[index] = feetech::radians_per_second_to_raw(
        static_cast<double>(direction_[index]) * velocity_commands_[index], max_raw_velocity_);
    }
    bus_.write_velocities(raw_velocities);
    return hardware_interface::return_type::OK;
  } catch (const std::exception & exception) {
    bus_.stop_and_disable();
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000, "LeKiwi write failed; torque disabled: %s", exception.what());
    return hardware_interface::return_type::ERROR;
  }
}

void LeKiwiSystemHardware::stop_and_disconnect() noexcept
{
  bus_.stop_and_disable();
  bus_.disconnect();
}

}  // namespace lekiwi_hardware

PLUGINLIB_EXPORT_CLASS(
  lekiwi_hardware::LeKiwiSystemHardware, hardware_interface::SystemInterface)
