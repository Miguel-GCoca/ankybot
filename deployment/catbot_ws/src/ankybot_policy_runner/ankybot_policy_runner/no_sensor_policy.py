
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import onnxruntime as ort
import numpy as np


class NoSensorPolicy(Node):
    """
    Closed-loop (joint feedback only, no IMU) ONNX policy runner for the
    Ankybot-Static-v3 env_cfg (AnkybotV3StaticEnvCfg.StaticObservationsCfg
    in ankybot_v3_env_cfg_no_sensors.py).

    That env_cfg has no velocity/height command and no IMU term. The policy's
    observation is:

        joint_pos    (12) - absolute joint position from /joint_states
        joint_vel    (12) - joint velocity from /joint_states
        last_action  (12) - raw clipped policy output from prev step

    i.e. 36 obs, sourced entirely from /joint_states (no IMU, no command).
    This node validates joint/actuator sanity end-to-end on hardware: real
    encoder feedback in, /joint_commands out at the correct rate.

    Action pipeline (matches ActionsCfg: scale=0.4, use_default_offset=True,
    clip=(-1.4923, 1.4923)), identical to policy_runner.py (including the
    max_joint_vel_rad_s slew-rate cap and the sub-tick interpolated
    publish_step added 2026-07-21 - see that file's docstring for why):
        raw_action = policy(obs)                          [runs at control_dt, 50Hz]
        raw_action = clip(raw_action, -1.4923, 1.4923)
        last_action <- raw_action          (used in NEXT obs)
        filtered = alpha*filtered + (1-alpha)*raw_action
        desired_pos = default_pos + 0.4 * filtered     (rad)
        target_pos = target_pos + clip(desired_pos - target_pos, -max_delta, max_delta)
        (target_pos is the rate-limited goal for this control period; publish_step,
         running at publish_rate_hz, linearly interpolates toward it and publishes)
    """

    def __init__(self):
        super().__init__('no_sensor_policy')

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
        self.declare_parameter('action_filter_alpha', 0.8)
        self.declare_parameter('max_joint_vel_rad_s', 8.0)
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('servo_cmd_topic', '/joint_commands')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('diag_log_period', 1.0)  # seconds

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
        self.diag_log_period = float(p('diag_log_period').value)

        assert len(self.default_pos) == self.n

        policy_path = os.path.expanduser(p('policy_path').value)
        if not policy_path:
            raise RuntimeError('policy_path parameter is required')

        self.session = ort.InferenceSession(policy_path,
                                             providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(
            f'Loaded no-sensor policy "{policy_path}", input "{self.input_name}", '
            f'{self.n} joints, control_dt={self.dt}s (joint feedback only, no IMU/command)'
        )

        # ---- internal state ----
        self.target_pos = self.default_pos.copy()          # rate-limited goal, updated at control_dt
        self._interp_start = self.default_pos.copy()        # target_pos at the start of the current control period
        self._interp_progress = 1.0                          # fraction of current control period covered [0,1]
        self._interp_step_frac = self.publish_dt / self.dt
        self.published_pos = self.default_pos.copy()        # what's actually sent on /joint_commands
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.filtered_action = np.zeros(self.n, dtype=np.float32)
        self.encoder_pos = self.default_pos.copy()
        self.encoder_vel = np.zeros(self.n, dtype=np.float32)
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}

        # comms diagnostics (message rate, last sample)
        self._js_msgs_in_window = 0
        self._js_total_msgs = 0
        self._last_js_sample = None
        self._cmd_pubs_in_window = 0

        # ---- ROS interfaces ----
        self.create_subscription(JointState, p('joint_states_topic').value,
                                  self.joint_state_cb, 10)
        self.pub = self.create_publisher(JointState, p('servo_cmd_topic').value, 10)

        self.create_timer(self.dt, self.step)
        self.create_timer(self.publish_dt, self.publish_step)
        self.create_timer(self.diag_log_period, self.log_diagnostics)

    # ------------------------------------------------------------------
    def joint_state_cb(self, msg: JointState):
        """Feeds joint_pos/joint_vel into the policy observation and tracks comms diagnostics."""
        self._js_msgs_in_window += 1
        self._js_total_msgs += 1
        sample = []
        for i, name in enumerate(msg.name):
            if name in self.joint_index and i < len(msg.position):
                j = self.joint_index[name]
                self.encoder_pos[j] = msg.position[i]
                if i < len(msg.velocity):
                    self.encoder_vel[j] = msg.velocity[i]
                sample.append((name, msg.position[i]))
        if sample:
            self._last_js_sample = sample[:3]

    def log_diagnostics(self):
        js_hz = self._js_msgs_in_window / self.diag_log_period
        cmd_hz = self._cmd_pubs_in_window / self.diag_log_period
        sample_str = (', '.join(f'{n}={v:.3f}' for n, v in self._last_js_sample)
                      if self._last_js_sample else 'none received yet')
        self.get_logger().info(
            f'[comms] joint_commands published: {cmd_hz:.1f} Hz | '
            f'joint_states received: {js_hz:.1f} Hz (total={self._js_total_msgs}) | '
            f'sample: {sample_str}'
        )
        self._js_msgs_in_window = 0
        self._cmd_pubs_in_window = 0

    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        return np.concatenate([
            self.encoder_pos,   # 12  joint_pos (absolute)
            self.encoder_vel,   # 12  joint_vel
            self.last_action,   # 12  last_action
        ]).astype(np.float32)[None, :]

    def step(self):
        obs = self._build_observation()
        raw = self.session.run(None, {self.input_name: obs})[0].squeeze().astype(np.float32)

        raw = np.clip(raw, -self.action_clip, self.action_clip)
        self.last_action = raw.copy()

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
        self._cmd_pubs_in_window += 1


def main(args=None):
    rclpy.init(args=args)
    node = NoSensorPolicy()
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
