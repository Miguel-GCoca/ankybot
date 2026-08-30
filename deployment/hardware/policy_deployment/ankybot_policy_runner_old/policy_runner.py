
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
        joint_pos           (n)  - absolute commanded target (rad); proxy for sim joint_pos
        joint_vel           (n)  - (last_target - prev_target) / dt; proxy for sim joint_vel
        last_action         (n)  - raw clipped policy output from prev step

    Action pipeline (matches ActionsCfg: scale=0.4, use_default_offset=True,
    clip=(-1.4923, 1.4923)):
        raw_action = policy(obs)
        raw_action = clip(raw_action, -1.4923, 1.4923)
        last_action <- raw_action          (used in NEXT obs)
        filtered = alpha*filtered + (1-alpha)*raw_action
        target_pos = default_pos + 0.4 * filtered     (rad)
        publish target_pos directly on /joint_commands
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
        self.declare_parameter('action_filter_alpha', 0.8)
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
        self.cmd_vel_timeout = float(p('cmd_vel_timeout').value)

        assert len(self.default_pos) == self.n

        policy_path = p('policy_path').value
        if not policy_path:
            raise RuntimeError('policy_path parameter is required')

        self.session = ort.InferenceSession(policy_path,
                                             providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(
            f'Loaded policy "{policy_path}", input "{self.input_name}", '
            f'{self.n} joints, control_dt={self.dt}s'
        )

        # ---- internal state (open-loop: target == "measured") ----
        self.target_pos = self.default_pos.copy()       # last commanded (rad)
        self.prev_target_pos = self.default_pos.copy()
        self.last_action = np.zeros(self.n, dtype=np.float32)
        self.filtered_action = np.zeros(self.n, dtype=np.float32)

        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.proj_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.lin_accel = np.zeros(3, dtype=np.float32)
        self.cmd_vel = np.zeros(3, dtype=np.float32)
        self.last_cmd_vel_time = self.get_clock().now()
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.encoder_pos = self.default_pos.copy()
        self.encoder_vel = np.zeros(self.n, dtype=np.float32)

        # ---- ROS interfaces ----
        self.create_subscription(Imu, p('imu_topic').value, self.imu_cb, 10)
        self.create_subscription(Twist, p('cmd_vel_topic').value, self.cmd_cb, 10)
        self.create_subscription(JointState, p('joint_states_topic').value, self.joint_state_cb, 10)
        self.pub = self.create_publisher(JointState, p('servo_cmd_topic').value, 10)

        self.create_timer(self.dt, self.step)

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
                self.encoder_vel[j] = msg.velocity[i]
    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        # zero out cmd_vel if stale (safety)
        age = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds * 1e-9
        cmd = self.cmd_vel if age < self.cmd_vel_timeout else np.zeros(3, dtype=np.float32)

        # absolute commanded position — matches training's joint_pos (asset.data.joint_pos)
        joint_pos = self.encoder_pos
        # commanded velocity proxy — matches training's joint_vel (asset.data.joint_vel)
        joint_vel = self.encoder_vel

        obs = np.concatenate([
            self.ang_vel,            # 3  imu_ang_vel
            self.proj_gravity,       # 3  projected_gravity
            self.lin_accel,          # 3  imu_lin_accel
            cmd,                     # 3  velocity_commands
            joint_pos,               # n  joint_pos (absolute)
            joint_vel,               # n  joint_vel (commanded proxy)
            self.last_action,        # n  last_action
        ]).astype(np.float32)

        return obs[None, :]

    def step(self):
        obs = self._build_observation()
        raw = self.session.run(None, {self.input_name: obs})[0].squeeze().astype(np.float32)

        # clip raw policy output (matches ActionsCfg clip)
        raw = np.clip(raw, -self.action_clip, self.action_clip)
        self.last_action = raw.copy()

        # exponential smoothing on action before applying
        self.filtered_action = self.alpha * self.filtered_action + (1.0 - self.alpha) * raw

        # update commanded target history
        self.prev_target_pos = self.target_pos.copy()
        self.target_pos = self.default_pos + self.action_scale * self.filtered_action

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.target_pos.tolist()
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
