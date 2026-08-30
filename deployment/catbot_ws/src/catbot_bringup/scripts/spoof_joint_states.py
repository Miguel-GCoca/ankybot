#!/usr/bin/env python3
"""
Spoofs /joint_states from /joint_commands for bench-testing without the Mega
wired up. Not an instant echo, still a placeholder for real hardware
feedback, not a substitute for it.

Copy of topic_pub_scripts/spoof_joint_states.py, installed here so
ankybot_bringup.launch.py can run it when launched with no_mega:=True.
Keep both copies in sync if either changes.

Simulates a slew-rate-limited follower (converges toward each commanded
target at MAX_VEL_RAD_S, not instantly) with velocity via finite
difference, plus a fixed onset delay (DELAY_S), so feedback resembles what
training expects from real PD actuator dynamics instead of an
out-of-distribution instant, zero-velocity echo. MAX_VEL_RAD_S and
DELAY_S are hardware-measured (see actuation_delay/measure_actuation_delay.py
and step_response_sweep/), not guesses.
"""
import collections

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

MAX_VEL_RAD_S = 8.4  # rad/s, from hardware step-response measurement
DELAY_S = 0.0095     # s, from hardware actuation-delay measurement


class SpoofJointStates(Node):
    def __init__(self):
        super().__init__('spoof_joint_states')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.sub = self.create_subscription(
            JointState, '/joint_commands', self.command_cb, 10
        )
        self.cmd_buffer = collections.deque()  # (timestamp_s, position_list, names)
        self.names = None
        self.pos = None
        self.last_time = None
        self.get_logger().info(
            'Spoofing /joint_states from /joint_commands '
            f'(slew-limited follower, max_vel={MAX_VEL_RAD_S} rad/s, '
            f'onset delay={DELAY_S * 1000:.1f}ms). Ctrl+C to stop.'
        )

    def command_cb(self, msg):
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        self.cmd_buffer.append((now_s, list(msg.position), list(msg.name)))

        # keep popping until the front is the most recent command that's
        # still at least DELAY_S old, that's the one actually in effect now
        while len(self.cmd_buffer) > 1 and self.cmd_buffer[1][0] <= now_s - DELAY_S:
            self.cmd_buffer.popleft()
        delayed_time, target, names = self.cmd_buffer[0]
        if delayed_time > now_s - DELAY_S:
            return  # still within the startup delay window, nothing has "arrived" yet

        if self.pos is None:
            self.names = names
            self.pos = list(target)
            vel = [0.0] * len(self.pos)
        else:
            dt = (now - self.last_time).nanoseconds * 1e-9
            max_delta = MAX_VEL_RAD_S * dt if dt > 0.0 else 0.0
            vel = []
            for i, t in enumerate(target):
                delta = t - self.pos[i]
                delta = max(-max_delta, min(max_delta, delta))
                self.pos[i] += delta
                vel.append(delta / dt if dt > 0.0 else 0.0)
            self.names = names
        self.last_time = now

        out = JointState()
        out.header.stamp = now.to_msg()
        out.name = list(self.names)
        out.position = list(self.pos)
        out.velocity = vel
        self.pub.publish(out)


def main():
    rclpy.init()
    node = SpoofJointStates()
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
