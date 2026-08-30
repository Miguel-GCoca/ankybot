import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

import board
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C


class Bno085Publisher(Node):
    """
    Publishes sensor_msgs/Imu from a BNO085 wired directly to the Pi's I2C
    pins (no Arduino involved).

    Report selection matches the sim training convention, not the sensor's
    "cleanest" output:
      - BNO_REPORT_ACCELEROMETER (gravity-inclusive "specific force", ~9.81
        m/s^2 on the up axis at rest) - NOT BNO_REPORT_LINEAR_ACCELERATION
        (gravity-compensated). Ankybot-v3's ImuCfg uses the Isaac Lab default
        gravity_bias=(0,0,9.81), so the real sensor must report the same
        gravity-inclusive convention it was trained against.
      - BNO_REPORT_GAME_ROTATION_VECTOR (gyro+accel fusion only, no
        magnetometer) rather than BNO_REPORT_ROTATION_VECTOR. The policy's
        only use of orientation is imu_projected_gravity, which is invariant
        to yaw - so the magnetometer's absolute heading reference buys
        nothing here, while dropping it avoids magnetic interference from
        the nearby servos/PCA9685.
      - BNO_REPORT_GYROSCOPE (calibrated angular velocity, rad/s).

    2026-07-20: retries forever instead of dying on connect/read failure; reports immediately then throttles to once/3s (see _try_connect()/step()).
    """

    def __init__(self):
        super().__init__('bno085_publisher')

        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('i2c_address', 0x4A)
        self.declare_parameter('reset_pin', '')   # e.g. 'D17' - '' disables explicit reset
        self.declare_parameter('publish_rate_hz', 50.0)

        p = self.get_parameter
        self.frame_id = p('frame_id').value
        self.address = int(p('i2c_address').value)
        self.reset_pin_name = p('reset_pin').value
        self.rate_hz = float(p('publish_rate_hz').value)
        self.imu_topic = p('imu_topic').value

        self.bno = None
        self.reported_failure = False

        self.pub = self.create_publisher(Imu, self.imu_topic, 10)

        # first connect attempt (reports immediately if it fails, see _try_connect())
        self._try_connect()

        self.create_timer(1.0 / self.rate_hz, self.step)

    def _try_connect(self):
        if self.bno is not None:
            return
        try:
            i2c = board.I2C()

            reset_pin = None
            if self.reset_pin_name:
                import digitalio
                reset_pin = digitalio.DigitalInOut(getattr(board, self.reset_pin_name))

            bno = BNO08X_I2C(i2c, address=self.address, reset=reset_pin)
            report_interval_us = int(1_000_000 / self.rate_hz)
            bno.enable_feature(BNO_REPORT_ACCELEROMETER, report_interval_us)
            bno.enable_feature(BNO_REPORT_GYROSCOPE, report_interval_us)
            bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR, report_interval_us)

            self.bno = bno
            self.get_logger().info(
                f'BNO085 {"reconnected" if self.reported_failure else "online"} at '
                f'I2C address {hex(self.address)}, publishing "{self.imu_topic}" '
                f'at {self.rate_hz}Hz'
            )
            self.reported_failure = False
        except Exception as e:
            self.get_logger().warn(
                f'BNO085 not connected: {e}', throttle_duration_sec=3.0)
            self.reported_failure = True
            self.bno = None

    def step(self):
        if self.bno is None:
            self._try_connect()
            return

        try:
            self.bno._process_available_packets()
            accel_x, accel_y, accel_z = self.bno._readings[BNO_REPORT_ACCELEROMETER]
            gyro_x, gyro_y, gyro_z = self.bno._readings[BNO_REPORT_GYROSCOPE]
            quat_i, quat_j, quat_k, quat_real = self.bno._readings[BNO_REPORT_GAME_ROTATION_VECTOR]
        except Exception as e:
            self.get_logger().warn(
                f'BNO085 read failed: {e}', throttle_duration_sec=3.0)
            self.reported_failure = True
            # force a full reconnect - this sensor needs feature-enable redone after any real comm fault
            self.bno = None
            return

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.orientation.x = quat_i
        msg.orientation.y = quat_j
        msg.orientation.z = quat_k
        msg.orientation.w = quat_real

        msg.angular_velocity.x = gyro_x
        msg.angular_velocity.y = gyro_y
        msg.angular_velocity.z = gyro_z

        msg.linear_acceleration.x = accel_x
        msg.linear_acceleration.y = accel_y
        msg.linear_acceleration.z = accel_z

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Bno085Publisher()
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
