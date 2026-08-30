from collections import namedtuple
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

Vec3 = namedtuple('Vec3', 'x y z')
Vec4 = namedtuple('Vec4', 'i j k real')


class Bno085Publisher(Node):
    """
    Publishes imu msg type from BNO085 to ROS2 network over i2c 

    my isaac sim imu was trained with gravity present and not filtered out, this must be present here as well.
    if your isaac sim training filters out gravity you must filter it out here as well.

    important links:
    https://docs.circuitpython.org/projects/bno08x/en/latest/ // lines 6-11: reading the imu over i2c
    https://github.com/adafruit/Adafruit_Blinka  // line 5: "import board" translating circuitpython to linux
    """

    def __init__(self):
        super().__init__('bno085_publisher')

        ## the following "declare_parameter"s can be changed when running through CLI with --ros-args 
        ## or at the following file for launch: catbot_ws/src/ankybot_imu/config/bno085.yaml

        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('i2c_address', 0x4A)
        self.declare_parameter('reset_pin', '')   # was not used, but can be used to control poll rate
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('cutoff_freq', 3)
        
        p = self.get_parameter
        self.frame_id = p('frame_id').value
        self.address = int(p('i2c_address').value)
        self.reset_pin_name = p('reset_pin').value
        self.rate_hz = float(p('publish_rate_hz').value)
        self.imu_topic = p('imu_topic').value
        self.cutoff_freq = float(p('cutoff_freq').value)
        
        self.bno = None
        self.reported_failure = False

        self.pub = self.create_publisher(Imu, self.imu_topic, 10)

        # first connect attempt 
        self._try_connect()

        #set imu x(i-1) value to null for low pass filter first pass
        self.filter_reset()

        self.create_timer(1.0 / self.rate_hz, self.step)

    #function whos only purpose is to establish connection between pi and imu
    def _try_connect(self):
        if self.bno is not None:
            return
        try:
            i2c = board.I2C()

            reset_pin = None #remains unused
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
            accel_raw = Vec3(*self.bno.acceleration)
            gyro_raw = Vec3(*self.bno.gyro)
            quat_raw = Vec4(*self.bno.game_quaternion)
        except Exception as e:
            self.get_logger().warn(
                f'BNO085 read failed: {e}', throttle_duration_sec=3.0)
            self.reported_failure = True
            self.bno = None # triggers a "_try_connect()" during the next step()
            self.filter_reset() # reset low pass filter
            return

        accel = Vec3(*(self.Eds_low_pass_filter(r, p) for r, p in zip(accel_raw, self.accel_prev)))
        gyro = Vec3(*(self.Eds_low_pass_filter(r, p) for r, p in zip(gyro_raw, self.gyro_prev)))
        quat = Vec4(*(self.Eds_low_pass_filter(r, p) for r, p in zip(quat_raw, self.quat_prev)))

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.orientation.x = quat.i
        msg.orientation.y = quat.j
        msg.orientation.z = quat.k
        msg.orientation.w = quat.real

        msg.angular_velocity.x = gyro.x
        msg.angular_velocity.y = gyro.y
        msg.angular_velocity.z = gyro.z

        msg.linear_acceleration.x = accel.x
        msg.linear_acceleration.y = accel.y
        msg.linear_acceleration.z = accel.z

        self.pub.publish(msg)

        self.accel_prev = accel
        self.gyro_prev = gyro
        self.quat_prev = quat

    def Eds_low_pass_filter(self, x1, x0):
        if x0 is None:
            return x1
        else:
            return x0 + (1/self.rate_hz) * self.cutoff_freq * (x1 - x0)

    def filter_reset(self):
        self.accel_prev = Vec3(None, None, None)
        self.gyro_prev = Vec3(None, None, None)
        self.quat_prev = Vec4(None, None, None, None)

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
