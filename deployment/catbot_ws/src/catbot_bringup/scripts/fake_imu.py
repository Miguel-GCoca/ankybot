#!/usr/bin/env python3
"""
Publishes synthetic IMU data matching a quadruped walking forward.
Simulates realistic body oscillations: roll/pitch sway, vertical bobbing,
and corresponding angular velocities.

Copy of topic_pub_scripts/fake_imu.py, installed here so
ankybot_bringup.launch.py can run it when launched with no_imu:=True.
Keep both copies in sync if either changes.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math

# --- Gait parameters ---
GAIT_FREQ    = 2.0   # Hz, trot frequency
ROLL_AMP     = math.radians(3.0)   # body roll oscillation (±3 deg)
PITCH_AMP    = math.radians(2.0)   # body pitch oscillation (±2 deg)
YAW_AMP      = math.radians(1.0)   # slight yaw sway (±1 deg)
Z_ACCEL_AMP  = 0.3   # m/s² vertical bobbing amplitude (2x gait freq)
X_ACCEL_AMP  = 0.15  # m/s² fore-aft oscillation
Y_ACCEL_AMP  = 0.10  # m/s² lateral oscillation
RATE_HZ      = 100   # IMU publish rate


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return (
        sr*cp*cy - cr*sp*sy,  # x
        cr*sp*cy + sr*cp*sy,  # y
        cr*cp*sy - sr*sp*cy,  # z
        cr*cp*cy + sr*sp*sy,  # w
    )


class FakeImu(Node):
    def __init__(self):
        super().__init__('fake_imu')
        self.pub = self.create_publisher(Imu, '/imu/data', 10)
        self.t = 0.0
        self.dt = 1.0 / RATE_HZ
        self.create_timer(self.dt, self.publish)
        self.get_logger().info(
            f'Publishing fake walking IMU at {RATE_HZ}Hz '
            f'(gait {GAIT_FREQ}Hz, roll±{math.degrees(ROLL_AMP):.0f}°, '
            f'pitch±{math.degrees(PITCH_AMP):.0f}°). Ctrl+C to stop.'
        )

    def publish(self):
        w = 2 * math.pi * GAIT_FREQ
        t = self.t

        # Euler angles oscillating at gait frequency
        roll  =  ROLL_AMP  * math.sin(w * t)
        pitch =  PITCH_AMP * math.sin(w * t + math.pi / 4)
        yaw   =  YAW_AMP   * math.sin(w * t * 0.5)

        # Angular velocity = derivative of euler angles
        ang_x =  ROLL_AMP  * w * math.cos(w * t)
        ang_y =  PITCH_AMP * w * math.cos(w * t + math.pi / 4)
        ang_z =  YAW_AMP   * (w * 0.5) * math.cos(w * t * 0.5)

        # Linear acceleration: gravity + body oscillations
        # Body bobs at 2x gait frequency (two steps per cycle)
        lin_x =  X_ACCEL_AMP * math.sin(w * t)
        lin_y =  Y_ACCEL_AMP * math.sin(w * t + math.pi / 2)
        lin_z =  9.81 + Z_ACCEL_AMP * math.sin(2 * w * t)

        qx, qy, qz, qw = euler_to_quat(roll, pitch, yaw)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_imu'
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.angular_velocity.x = ang_x
        msg.angular_velocity.y = ang_y
        msg.angular_velocity.z = ang_z
        msg.linear_acceleration.x = lin_x
        msg.linear_acceleration.y = lin_y
        msg.linear_acceleration.z = lin_z
        self.pub.publish(msg)

        self.t += self.dt


def main():
    rclpy.init()
    node = FakeImu()
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
