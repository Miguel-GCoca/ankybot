
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import onnxruntime as ort
import numpy as np


class NoSensorPolicy(Node):
    """
    Open-loop ONNX policy runner for the no_pos_no_imu/v1 stand-only env_cfg.

    This env_cfg (Ankybot-v3, edited for a comms-capacity test) has every
    sensor observation term commented out and the velocity command pinned to
    zero (rel_standing_envs=1.0, all ranges (0,0)). The policy's observation
    is therefore just:

        velocity_commands  (3)  - always zero, no /cmd_vel involved
        last_action         (12) - raw clipped policy output from prev step

    i.e. 15 obs, no IMU or joint-state feedback into the policy at all. This
    node exists to validate that the Pi can run inference and publish
    /joint_commands at the correct rate and that the Arduino executes and
    echoes /joint_states back reliably - not to run a real controller.

    Action pipeline (matches ActionsCfg: scale=0.4, use_default_offset=True,
    clip=(-1.4923, 1.4923)), identical to policy_runner.py:
        raw_action = policy(obs)
        raw_action = clip(raw_action, -1.4923, 1.4923)
        last_action <- raw_action          (used in NEXT obs)
        filtered = alpha*filtered + (1-alpha)*raw_action
        target_pos = default_pos + 0.4 * filtered     (rad)
        publish target_pos directly on /joint_commands

    /joint_states is subscribed only for comms diagnostics (message rate,
    last sample) - it is never fed into the policy observation.
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
            f'{self.n} joints, control_dt={self.dt}s (open-loop, no sensor feedback)'
        )

        # ---- internal state ----
        self.target_pos = self.default_pos.copy()
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.filtered_action = np.zeros(self.n, dtype=np.float32)
        self.zero_cmd = np.zeros(3, dtype=np.float32)  # velocity_commands, always zero
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}

        # comms diagnostics only - not used by the policy
        self._js_msgs_in_window = 0
        self._js_total_msgs = 0
        self._last_js_sample = None
        self._cmd_pubs_in_window = 0

        # ---- ROS interfaces ----
        self.create_subscription(JointState, p('joint_states_topic').value,
                                  self.joint_state_cb, 10)
        self.pub = self.create_publisher(JointState, p('servo_cmd_topic').value, 10)

        self.create_timer(self.dt, self.step)
        self.create_timer(self.diag_log_period, self.log_diagnostics)

    # ------------------------------------------------------------------
    def joint_state_cb(self, msg: JointState):
        """Diagnostics only - confirms the Arduino/Pi link is alive and echoing."""
        self._js_msgs_in_window += 1
        self._js_total_msgs += 1
        sample = []
        for i, name in enumerate(msg.name):
            if name in self.joint_index and i < len(msg.position):
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
        return np.concatenate([self.zero_cmd, self.last_action]).astype(np.float32)[None, :]

    def step(self):
        obs = self._build_observation()
        raw = self.session.run(None, {self.input_name: obs})[0].squeeze().astype(np.float32)

        raw = np.clip(raw, -self.action_clip, self.action_clip)
        self.last_action = raw.copy()

        self.filtered_action = self.alpha * self.filtered_action + (1.0 - self.alpha) * raw
        self.target_pos = self.default_pos + self.action_scale * self.filtered_action

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.target_pos.tolist()
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
