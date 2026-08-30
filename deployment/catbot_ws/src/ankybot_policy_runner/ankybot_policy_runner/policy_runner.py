
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import onnxruntime as ort
import numpy as np


class PolicyRunner(Node):
    """
    Open-loop (no encoder feedback) ONNX policy runner.

    Observation order (must match training ObservationsCfg.PolicyCfg, 48 total):
        imu_ang_vel         (3)  - from IMU angular_velocity
        projected_gravity   (3)  - from IMU orientation quaternion
        imu_lin_accel       (3)  - from IMU linear_acceleration
        velocity_commands   (3)  - from /cmd_vel
        joint_pos           (n)  - absolute joint position from /joint_states
        joint_vel           (n)  - joint velocity from /joint_states
        last_action         (n)  - raw clipped policy output from prev step

    Action pipeline (matches ActionsCfg: scale=0.4, use_default_offset=True,
    clip=(-1.4923, 1.4923)):
        raw_action = policy(obs)                          [runs at control_dt, 50Hz]
        raw_action = clip(raw_action, -1.4923, 1.4923)
        last_action <- raw_action          (used in NEXT obs)
        filtered = alpha*filtered + (1-alpha)*raw_action
        desired_pos = default_pos + 0.4 * filtered     (rad)
        target_pos = target_pos + clip(desired_pos - target_pos, -max_delta, max_delta)
        (target_pos is the rate-limited goal for this control period; publish_step
         below interpolates toward it and does the actual publishing)

    max_delta = max_joint_vel_rad_s * control_dt, applied per joint on top of
    the existing EMA filter - hard caps how far the goal can move per control
    tick regardless of what the (filtered) policy output requests. Added
    2026-07-21: on hardware, the EMA filter alone did not prevent large
    single-tick jumps (e.g. on a command-direction reversal), which drove the
    servos at/near their max slew rate and looked/felt too fast and too abrupt.

    max_joint_vel_rad_s raised 3.0 -> 8.0 same day, after user testing showed
    walking gait amplitude was shrunk (roughly 2x as many smaller steps as
    expected) - the cap was clipping legitimate swing-phase velocity, not
    just abrupt jumps. A real trot-like swing (~2Hz gait, ~80-100ms swing
    phase sweeping ~0.3-0.5 rad of thigh/foot travel) needs on the order of
    4-6 rad/s, above the old 3.0 rad/s cap, so target_pos permanently lagged
    desired_pos and the leg never reached full stride amplitude before the
    policy moved to the next phase. 8.0 rad/s stays under the servo's rated
    no-load 10.4719755 rad/s (600 deg/s) while clearing normal gait swing
    speed - now mainly a backstop against pathological single-tick jumps
    (the interpolated publish_step above already handles routine smoothing),
    not a routine amplitude limiter. Still an estimate, not measured on
    hardware - retune if gait amplitude still looks short, or if it's judged
    too close to the servo's real (loaded, lower than no-load) max speed.

    Sub-tick interpolated publishing (added 2026-07-21, same day): a real
    servo has no compliance like the sim's spring-damper actuator model - it
    snaps to whatever position it's told, fast. Publishing target_pos
    directly at only 50Hz meant each ~0.06 rad (at max_joint_vel_rad_s=3.0)
    step was small individually, but the servo fully completed each one well
    within the 20ms before the next arrived, then sat idle - a rapid
    snap-then-pause sequence, not a continuous ramp like sim shows (sim's
    smoothness comes from physics integrating continuously between control
    ticks, not from anything about the 50Hz action rate itself). A second,
    faster timer (publish_rate_hz, default 200Hz) now linearly interpolates
    between each control tick's old and new target_pos and publishes those
    smaller, more frequent sub-steps instead - shrinks the settle-then-idle
    gap without changing the policy's own 50Hz observation/action cadence
    (must stay matched to training) or the overall max_joint_vel_rad_s cap.
    """

    def __init__(self):
        super().__init__('policy_runner')

        # ---- parameters ----
        self.declare_parameter('policy_path', '')
        self.declare_parameter('joint_names', [
            'BL_Hip_Joint', 'BR_Hip_Joint', 'FL_Hip_Joint', 'FR_Hip_Joint',
            'BL_Thigh_Joint', 'BR_Thigh_Joint', 'FL_Thigh_Joint', 'FR_Thigh_Joint',
            'BL_Foot_Joint', 'BR_Foot_Joint', 'FL_Foot_Joint', 'FR_Foot_Joint',
        ])
        self.declare_parameter('default_joint_pos_rad',
                                [0.0, 0.0, 0.0, 0.0,
                                 0.5236, 0.5236, 0.5236, 0.5236,
                                 0.3840, 0.3840, 0.3840, 0.3840])
        self.declare_parameter('action_scale', 0.4)
        self.declare_parameter('action_clip', 1.4923)
        self.declare_parameter('control_dt', 0.02)
        self.declare_parameter('action_filter_alpha', 0.5)
        self.declare_parameter('max_joint_vel_rad_s', 8.0)
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('servo_cmd_topic', '/joint_commands')
        self.declare_parameter('cmd_vel_timeout', 1.0)  # zero cmd_vel if stale
        self.declare_parameter('joint_states_topic', '/joint_states')

        p = self.get_parameter
        self.joint_names = list(p('joint_names').value)
        self.n = len(self.joint_names)
        self.default_pos = np.array(p('default_joint_pos_rad').value, dtype=np.float32)
        self.action_scale = float(p('action_scale').value)
        self.action_clip = float(p('action_clip').value)
        self.dt = float(p('control_dt').value)
        self.alpha = float(p('action_filter_alpha').value)
        self.max_joint_vel_rad_s = float(p('max_joint_vel_rad_s').value)
        self.max_delta = self.max_joint_vel_rad_s * self.dt
        self.publish_rate_hz = float(p('publish_rate_hz').value)
        self.publish_dt = 1.0 / self.publish_rate_hz
        self.cmd_vel_timeout = float(p('cmd_vel_timeout').value)

        assert len(self.default_pos) == self.n

        policy_path = os.path.expanduser(p('policy_path').value)
        if not policy_path:
            raise RuntimeError('policy_path parameter is required')

        self.session = ort.InferenceSession(policy_path,
                                             providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(
            f'Loaded policy "{policy_path}", input "{self.input_name}", '
            f'{self.n} joints, control_dt={self.dt}s'
        )

        # ---- internal state ----
        self.target_pos = self.default_pos.copy()        # rate-limited goal, updated at control_dt
        self._interp_start = self.default_pos.copy()      # target_pos at the start of the current control period
        self._interp_progress = 1.0                        # fraction of current control period covered [0,1]
        self._interp_step_frac = self.publish_dt / self.dt
        self.published_pos = self.default_pos.copy()      # what's actually sent on /joint_commands
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.filtered_action = np.zeros(self.n, dtype=np.float32)

        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.lin_accel = np.zeros(3, dtype=np.float32)
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.last_cmd_vel_time = self.get_clock().now()
        self._last_step_time = None  # for backwards-time detection
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.encoder_pos = self.default_pos.copy()
        self.encoder_vel = np.zeros(self.n, dtype=np.float32)

        # ---- ROS interfaces ----
        self.create_subscription(Imu, p('imu_topic').value, self.imu_cb, 10)
        self.create_subscription(Twist, p('cmd_vel_topic').value, self.cmd_cb, 10)
        self.create_subscription(JointState, p('joint_states_topic').value, self.joint_state_cb, 10)
        self.pub = self.create_publisher(JointState, p('servo_cmd_topic').value, 10)

        self.create_timer(self.dt, self.step)
        self.create_timer(self.publish_dt, self.publish_step)

    # ------------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        self.ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float32)

        q = msg.orientation
        # gravity vector expressed in body frame, from quaternion (x,y,z,w)
        gx = 2.0 * (q.x * q.z - q.w * q.y)
        gy = 2.0 * (q.y * q.z + q.w * q.x)
        gz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.proj_gravity = -np.array([gx, gy, gz], dtype=np.float32)

        self.lin_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float32)

    def cmd_cb(self, msg: Twist):
        self.cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z],
                                 dtype=np.float32)
        self.last_cmd_vel_time = self.get_clock().now()

    def joint_state_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.joint_index:
                j = self.joint_index[name]
                self.encoder_pos[j] = msg.position[i]
                if i < len(msg.velocity):
                    self.encoder_vel[j] = msg.velocity[i]

    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        # zero out cmd_vel if stale (safety)
        age = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds * 1e-9
        cmd = self.cmd_vel if age < self.cmd_vel_timeout else np.zeros(3, dtype=np.float32)

        parts = [
            self.ang_vel,            # 3  imu_ang_vel
            self.proj_gravity,       # 3  projected_gravity
            self.lin_accel,          # 3  imu_lin_accel
            cmd,                     # 3  velocity_commands
            self.encoder_pos,        # n  joint_pos (absolute)
            self.encoder_vel,        # n  joint_vel
            self.last_action,        # n  last_action
        ]

        return np.concatenate(parts).astype(np.float32)[None, :]

    def step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_step_time is not None and now < self._last_step_time:
            self.get_logger().warn('Time jumped backwards — resetting filter and action state.')
            self.filtered_action = np.zeros(self.n, dtype=np.float32)
            self.last_action = np.zeros(self.n, dtype=np.float32)
        self._last_step_time = now

        obs = self._build_observation()
        raw = self.session.run(None, {self.input_name: obs})[0].squeeze().astype(np.float32)

        # clip raw policy output (matches ActionsCfg clip)
        raw = np.clip(raw, -self.action_clip, self.action_clip)
        self.last_action = raw.copy()

        # exponential smoothing on action before applying
        self.filtered_action = self.alpha * self.filtered_action + (1.0 - self.alpha) * raw

        desired_pos = self.default_pos + self.action_scale * self.filtered_action
        step = np.clip(desired_pos - self.target_pos, -self.max_delta, self.max_delta)
        self._interp_start = self.target_pos.copy()
        self.target_pos = self.target_pos + step
        self._interp_progress = 0.0

    def publish_step(self):
        self._interp_progress = min(1.0, self._interp_progress + self._interp_step_frac)
        self.published_pos = (
            self._interp_start
            + self._interp_progress * (self.target_pos - self._interp_start)
        )

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.published_pos.tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
