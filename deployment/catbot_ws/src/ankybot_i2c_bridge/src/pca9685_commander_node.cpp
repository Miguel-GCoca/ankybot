#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ankybot_i2c_bridge/pca9685.hpp"

namespace
{
constexpr int NUM_SERVOS = 12;
constexpr float PI_F = 3.14159265f;
constexpr float HALF_PI_F = 1.57079632679f;

const std::array<std::string, NUM_SERVOS> JOINT_ORDER = {
  "FL_Hip_Joint", "FL_Thigh_Joint", "FL_Foot_Joint",
  "FR_Hip_Joint", "FR_Thigh_Joint", "FR_Foot_Joint",
  "BL_Hip_Joint", "BL_Thigh_Joint", "BL_Foot_Joint",
  "BR_Hip_Joint", "BR_Thigh_Joint", "BR_Foot_Joint",
};

int JOINT_ORIENTATION[NUM_SERVOS] = {
  -1, 1, -1, //FL
  -1, -1, 1, //FR
  1, -1, 1, //BL
  1, 1, -1  //BR
};

constexpr float DEG_TO_RAD_F = 0.01745329252f;

// per-channel physical zero-position trim (degrees), corrects for gear-tooth assembly slop, applied in raw servo-angle space, after JOINT_ORIENTATION.
// Hardware-measured via `ros2 topic pub /joint_commands` (data[i] found by hand that reaches true physical zero with trim=0, back-solved as T_deg[i] = JOINT_ORIENTATION[i] * data[i] * (180/pi)).
float JOINT_ZERO_TRIM_DEG[NUM_SERVOS] = {
  0.0f, 2.8648f, 2.8648f,  //FL
  8.5944f, 8.5944f, 5.7296f,  //FR
  2.8648f, 5.7296f, 8.5944f,  //BL
  5.7296f, 2.8648f, 8.5944f   //BR
};

float JOINT_RANGE_MIN_DEG[NUM_SERVOS] = {
  -70.0f, -40.0f, -30.0f,  //FL
  -70.0f, -40.0f, -30.0f,  //FR
  -65.0f, -40.0f, -30.0f,  //BL
  -60.0f, -40.0f, -30.0f   //BR
};
float JOINT_RANGE_MAX_DEG[NUM_SERVOS] = {
  70.0f, 90.0f, 90.0f,  //FL
  70.0f, 90.0f, 90.0f,  //FR
  65.0f, 90.0f, 90.0f,  //BL
  60.0f, 90.0f, 90.0f   //BR
};

double clampToJointRange(int i, double logical_angle_rad)
{
  double lo = static_cast<double>(JOINT_RANGE_MIN_DEG[i]) * DEG_TO_RAD_F;
  double hi = static_cast<double>(JOINT_RANGE_MAX_DEG[i]) * DEG_TO_RAD_F;
  return std::max(lo, std::min(hi, logical_angle_rad));
}

int angleToPulse(float angle_rad, int servo_min, int servo_max)
{
  float shifted = angle_rad + HALF_PI_F;  // remap to [0, PI]
  shifted = std::max(0.0f, std::min(PI_F, shifted));
  return servo_min +
    static_cast<int>((shifted / PI_F) * static_cast<float>(servo_max - servo_min) + 0.5f);
}
}  // namespace

class PCA9685CommanderNode : public rclcpp::Node
{
public:
  PCA9685CommanderNode()
  : Node("pca9685_commander")
  {
    declare_parameter("i2c_bus", 1);
    declare_parameter("pca9685_address", 0x60);
    declare_parameter("pwm_freq_hz", 300.0);
    declare_parameter("servo_min", 614);
    declare_parameter("servo_max", 3072);
    // identity-maps index i -> channel i by default; known wiring defect (not fixable here): BR_Foot (11) inert, FL_Foot (2) drives two legs.
    declare_parameter(
      "servo_channels", std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    declare_parameter("servo_neutral", std::vector<double>(NUM_SERVOS, 0.0));
    declare_parameter("joint_commands_topic", std::string("/joint_commands"));
    // deadband: drop a re-command within this many radians of the last driven value, to avoid servo dither from upstream jitter.
    declare_parameter("deadband_rad", 0.01);
    deadband_rad_ = get_parameter("deadband_rad").as_double();

    for (int i = 0; i < NUM_SERVOS; ++i) {
      joint_index_[JOINT_ORDER[i]] = i;
    }

    servo_min_ = static_cast<int>(get_parameter("servo_min").as_int());
    servo_max_ = static_cast<int>(get_parameter("servo_max").as_int());

    auto channels_param = get_parameter("servo_channels").as_integer_array();
    servo_channels_.assign(channels_param.begin(), channels_param.end());

    auto neutral_param = get_parameter("servo_neutral").as_double_array();
    servo_neutral_.assign(neutral_param.begin(), neutral_param.end());

    if (static_cast<int>(servo_channels_.size()) != NUM_SERVOS ||
      static_cast<int>(servo_neutral_.size()) != NUM_SERVOS)
    {
      // config error, not connectivity, still fatal (caught in main()), unlike the I2C failures handled below.
      throw std::runtime_error("servo_channels/servo_neutral must have exactly 12 entries");
    }

    bus_number_ = static_cast<int>(get_parameter("i2c_bus").as_int());
    address_ = static_cast<int>(get_parameter("pca9685_address").as_int());
    freq_ = get_parameter("pwm_freq_hz").as_double();

    std::string topic = get_parameter("joint_commands_topic").as_string();
    subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      topic, 10,
      std::bind(&PCA9685CommanderNode::callback, this, std::placeholders::_1));

    // first connect attempt (reports immediately if it fails, see tryConnect()); the timer below retries forever after this.
    tryConnect();
    reconnect_timer_ = create_wall_timer(
      std::chrono::seconds(3), std::bind(&PCA9685CommanderNode::tryConnect, this));
  }

private:
  // (re)connects if needed and re-drives every channel to neutral on success; runs every 3s so a plain WARN per failed attempt already reports at the right cadence.
  void tryConnect()
  {
    if (pca_) {
      return;
    }
    try {
      auto pca = std::make_unique<ankybot_i2c_bridge::PCA9685>(
        bus_number_, static_cast<uint8_t>(address_), static_cast<float>(freq_));
      for (int i = 0; i < NUM_SERVOS; ++i) {
        double logical = clampToJointRange(i, servo_neutral_[i]);
        double target = JOINT_ORIENTATION[i] * logical + JOINT_ZERO_TRIM_DEG[i] * DEG_TO_RAD_F;
        float clamped = std::max(-HALF_PI_F, std::min(HALF_PI_F, static_cast<float>(target)));
        int pulse = angleToPulse(clamped, servo_min_, servo_max_);
        pca->setPWM(static_cast<uint8_t>(servo_channels_[i]), 0, static_cast<uint16_t>(pulse));
        last_target_[i] = target;
      }
      pca_ = std::move(pca);
      RCLCPP_INFO(
        get_logger(), "PCA9685 %s on i2c-%d @ 0x%02X, %.1fHz",
        reported_failure_ ? "reconnected" : "initialized", bus_number_, address_, freq_);
      reported_failure_ = false;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "PCA9685 not connected: %s", e.what());
      reported_failure_ = true;
    }
  }

  void driveChannel(int i, double angle_rad)
  {
    float clamped = std::max(-HALF_PI_F, std::min(HALF_PI_F, static_cast<float>(angle_rad)));
    int pulse = angleToPulse(clamped, servo_min_, servo_max_);
    pca_->setPWM(static_cast<uint8_t>(servo_channels_[i]), 0, static_cast<uint16_t>(pulse));
  }

  void callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (!pca_) {
      // not connected, drop the command, the reconnect timer already retries/reports on its own schedule.
      return;
    }

    if (msg->name.size() != static_cast<std::size_t>(NUM_SERVOS) ||
      msg->position.size() != static_cast<std::size_t>(NUM_SERVOS))
    {
      RCLCPP_WARN(
        get_logger(), "Expected %d names and positions, got %zu names / %zu positions",
        NUM_SERVOS, msg->name.size(), msg->position.size());
      return;
    }

    std::array<double, NUM_SERVOS> data{};
    std::array<bool, NUM_SERVOS> filled{};
    filled.fill(false);

    for (std::size_t k = 0; k < msg->name.size(); ++k) {
      auto it = joint_index_.find(msg->name[k]);
      if (it == joint_index_.end()) {
        RCLCPP_WARN(
          get_logger(), "Unknown joint name '%s' in JointState message", msg->name[k].c_str());
        return;
      }
      data[it->second] = msg->position[k];
      filled[it->second] = true;
    }

    for (int i = 0; i < NUM_SERVOS; ++i) {
      if (!filled[i]) {
        RCLCPP_WARN(get_logger(), "Missing joint in message: %s", JOINT_ORDER[i].c_str());
        return;
      }
    }

    for (int i = 0; i < NUM_SERVOS; ++i) {
      double logical = clampToJointRange(i, servo_neutral_[i] + data[i]);
      double target = JOINT_ORIENTATION[i] * logical + JOINT_ZERO_TRIM_DEG[i] * DEG_TO_RAD_F;
      if (std::abs(target - last_target_[i]) >= deadband_rad_) {
        try {
          driveChannel(i, target);
          last_target_[i] = target;
        } catch (const std::exception &) {
          // mid-operation failure, force a full reinit via the reconnect timer rather than just retrying the write.
          pca_.reset();
          return;
        }
      }
    }
  }

  std::unordered_map<std::string, int> joint_index_;
  std::vector<int64_t> servo_channels_;
  std::vector<double> servo_neutral_;
  int servo_min_{614};
  int servo_max_{3072};
  int bus_number_{1};
  int address_{0x60};
  double freq_{300.0};
  double deadband_rad_{0.01};
  bool reported_failure_{false};
  std::array<double, NUM_SERVOS> last_target_{};
  std::unique_ptr<ankybot_i2c_bridge::PCA9685> pca_;
  rclcpp::TimerBase::SharedPtr reconnect_timer_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // only genuine config errors reach here now, all I2C connectivity failures are handled internally above.
  try {
    rclcpp::spin(std::make_shared<PCA9685CommanderNode>());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pca9685_commander"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
