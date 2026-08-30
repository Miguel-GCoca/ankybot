#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ankybot_i2c_bridge/i2c_device.hpp"

namespace
{
constexpr int NUM_JOINTS = 12;
constexpr int FLOATS_PER_CHUNK = 6;
constexpr int NUM_CHUNKS = 2;  // 6 + 6 = 12 positions total
constexpr double DEG_TO_RAD = 0.017453292519943295;

// velocity smoothing (wider-baseline diff + EMA)
constexpr int VEL_BASELINE_TICKS = 4;   // diff against this many polls back, not just 1
constexpr double VEL_EMA_ALPHA = 0.6;   // extra smoothing on the resulting velocity signal

const std::array<std::string, NUM_JOINTS> JOINT_ORDER = {
  "FL_Hip_Joint", "FL_Thigh_Joint", "FL_Foot_Joint",
  "FR_Hip_Joint", "FR_Thigh_Joint", "FR_Foot_Joint",
  "BL_Hip_Joint", "BL_Thigh_Joint", "BL_Foot_Joint",
  "BR_Hip_Joint", "BR_Thigh_Joint", "BR_Foot_Joint",
};

// must match pca9685_commander_node.cpp's JOINT_ORIENTATION exactly, needed
// here only to apply joint_zero_trim_deg with the correct sign (the Mega no
// longer applies any trim itself, see arduino_mega_i2c_slave.ino).
const std::array<int, NUM_JOINTS> JOINT_ORIENTATION = {
  -1, 1, -1,  // FL
  -1, -1, 1,  // FR
  1, -1, 1,   // BL
  1, 1, -1,   // BR
};
}  // namespace

class MegaFeedbackReaderNode : public rclcpp::Node
{
public:
  MegaFeedbackReaderNode()
  : Node("mega_feedback_reader"), have_prev_(false)
  {
    declare_parameter("i2c_bus", 1);
    declare_parameter("mega_address", 0x08);
    declare_parameter("poll_rate_hz", 200.0);  // poll rate, hardware-tested up to 200Hz
    declare_parameter("joint_states_topic", std::string("/joint_states"));
    // deadband: republish the held value instead of noise within this many radians of the last reading (covers ~0.02-0.08 rad ADC jitter).
    declare_parameter("deadband_rad", 0.04);
    // per-channel physical zero-position trim (degrees), applied here (not on the Mega, see arduino_mega_i2c_slave.ino) so it's tunable live via `ros2 param set` with no rebuild/reflash/restart. Must match pca9685_commander_node.cpp's JOINT_ZERO_TRIM_DEG for the round-trip to self-cancel. Default below is from hardware measurement (see that file's comment for derivation), override live only when re-measuring.
    declare_parameter(
      "joint_zero_trim_deg",
      std::vector<double>{
        0.0, 2.8648, 2.8648,   // FL
        8.5944, 8.5944, 5.7296,  // FR
        2.8648, 5.7296, 8.5944,  // BL
        5.7296, 2.8648, 8.5944   // BR
      });

    int bus_number = static_cast<int>(get_parameter("i2c_bus").as_int());
    int address = static_cast<int>(get_parameter("mega_address").as_int());
    double rate_hz = get_parameter("poll_rate_hz").as_double();
    std::string topic = get_parameter("joint_states_topic").as_string();
    deadband_rad_ = get_parameter("deadband_rad").as_double();
    setZeroTrimDeg(get_parameter("joint_zero_trim_deg").as_double_array());

    param_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&MegaFeedbackReaderNode::onParamChange, this, std::placeholders::_1));

    device_ = std::make_unique<ankybot_i2c_bridge::I2CDevice>(
      bus_number, static_cast<uint8_t>(address));

    publisher_ = create_publisher<sensor_msgs::msg::JointState>(topic, 10);

    prev_pos_.fill(0.0);

    RCLCPP_INFO(
      get_logger(), "Polling Mega feedback slave on i2c-%d @ 0x%02X at %.1fHz",
      bus_number, address, rate_hz);

    auto period = std::chrono::duration<double>(1.0 / rate_hz);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&MegaFeedbackReaderNode::poll, this));
  }

private:
  // selects chunk_idx via a 1-byte write, then reads 6 floats back in the same transaction (repeated START).
  std::array<float, FLOATS_PER_CHUNK> readChunk(uint8_t chunk_idx)
  {
    auto raw = device_->writeThenRead({chunk_idx}, FLOATS_PER_CHUNK * sizeof(float));
    std::array<float, FLOATS_PER_CHUNK> values{};
    std::memcpy(values.data(), raw.data(), raw.size());
    return values;
  }

  bool setZeroTrimDeg(const std::vector<double> & values)
  {
    if (values.size() != NUM_JOINTS) {
      return false;
    }
    std::copy(values.begin(), values.end(), zero_trim_deg_.begin());
    return true;
  }

  rcl_interfaces::msg::SetParametersResult onParamChange(
    const std::vector<rclcpp::Parameter> & params)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto & param : params) {
      if (param.get_name() == "joint_zero_trim_deg") {
        if (!setZeroTrimDeg(param.as_double_array())) {
          result.successful = false;
          result.reason = "joint_zero_trim_deg must have exactly 12 entries";
          return result;
        }
        RCLCPP_INFO(get_logger(), "joint_zero_trim_deg updated live");
      }
    }
    return result;
  }

  void poll()
  {
    std::array<double, NUM_JOINTS> raw{};
    try {
      for (int c = 0; c < NUM_CHUNKS; ++c) {
        auto chunk = readChunk(static_cast<uint8_t>(c));
        for (int i = 0; i < FLOATS_PER_CHUNK; ++i) {
          raw[c * FLOATS_PER_CHUNK + i] = static_cast<double>(chunk[i]);
        }
      }
    } catch (const std::exception & e) {
      // retries every tick forever, never crashes, reports immediately then throttles to once/3s while it persists.
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "I2C read from Mega failed: %s", e.what());
      i2c_healthy_ = false;
      return;
    }
    if (!i2c_healthy_) {
      RCLCPP_INFO(get_logger(), "I2C read from Mega recovered");
      i2c_healthy_ = true;
    }

    // undo the Mega's JOINT_ORIENTATION flip to apply trim with the correct
    // sign, matching pca9685_commander_node.cpp's physical = O*L + T convention.
    for (int i = 0; i < NUM_JOINTS; ++i) {
      raw[i] -= JOINT_ORIENTATION[i] * zero_trim_deg_[i] * DEG_TO_RAD;
    }

    // deadband against the last published value, reject changes smaller than deadband_rad_.
    std::array<double, NUM_JOINTS> pos{};
    if (have_prev_) {
      for (int i = 0; i < NUM_JOINTS; ++i) {
        pos[i] = (std::abs(raw[i] - prev_pos_[i]) >= deadband_rad_) ? raw[i] : prev_pos_[i];
      }
    } else {
      pos = raw;
    }

    rclcpp::Time now = this->now();

    // wider-baseline finite difference (VEL_BASELINE_TICKS polls back,
    // not just 1). pos_history_ holds exactly the last VEL_BASELINE_TICKS
    // ticks *not including* this one, so once full, front() is always
    // exactly VEL_BASELINE_TICKS ticks old.
    std::array<double, NUM_JOINTS> vel{};
    vel.fill(0.0);
    bool have_raw_vel = false;
    if (static_cast<int>(pos_history_.size()) == VEL_BASELINE_TICKS) {
      const rclcpp::Time & base_time = pos_history_.front().first;
      const std::array<double, NUM_JOINTS> & base_pos = pos_history_.front().second;
      double dt = (now - base_time).seconds();
      if (dt > 0.0) {
        for (int i = 0; i < NUM_JOINTS; ++i) {
          vel[i] = (pos[i] - base_pos[i]) / dt;
        }
        have_raw_vel = true;
      }
      pos_history_.pop_front();
    }

    // extra EMA smoothing on top of the wider baseline above.
    if (have_raw_vel) {
      if (!have_vel_ema_) {
        vel_ema_ = vel;
        have_vel_ema_ = true;
      } else {
        for (int i = 0; i < NUM_JOINTS; ++i) {
          vel_ema_[i] = VEL_EMA_ALPHA * vel_ema_[i] + (1.0 - VEL_EMA_ALPHA) * vel[i];
        }
      }
    }

    pos_history_.emplace_back(now, pos);

    have_prev_ = true;
    prev_pos_ = pos;

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now;
    msg.name.assign(JOINT_ORDER.begin(), JOINT_ORDER.end());
    msg.position.assign(pos.begin(), pos.end());
    msg.velocity.assign(vel_ema_.begin(), vel_ema_.end());
    publisher_->publish(msg);
  }

  std::unique_ptr<ankybot_i2c_bridge::I2CDevice> device_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  std::array<double, NUM_JOINTS> zero_trim_deg_{};
  std::array<double, NUM_JOINTS> prev_pos_{};
  bool have_prev_;
  double deadband_rad_{0.04};
  bool i2c_healthy_{true};

  // velocity smoothing state
  std::deque<std::pair<rclcpp::Time, std::array<double, NUM_JOINTS>>> pos_history_;
  std::array<double, NUM_JOINTS> vel_ema_{};
  bool have_vel_ema_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MegaFeedbackReaderNode>());
  rclcpp::shutdown();
  return 0;
}
