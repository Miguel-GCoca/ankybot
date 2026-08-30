#!/usr/bin/env python3
"""
Publishes a repeating /cmd_vel test pattern on a loop: forward, stop,
backward, stop, turn left, stop, turn right, stop. Each leg is held for
HOLD_S seconds, published continuously at PUBLISH_RATE_HZ (matching
speech_detection's command_interpreter.py convention of republishing
continuously rather than once).

Built 2026-07-21 to test policy_runner's new max_joint_vel_rad_s slew-rate
limiter - run this alongside a no_mega:=True/no_imu:=True ("full-spoof")
bringup to watch leg motion without needing the Mega or IMU wired.

2026-07-21, corrected same day: HOLD_S raised 3.0 -> 15.0 and STOP_HOLD_S
1.0 -> 3.0 - the actual reported symptom is per-tick joint-angle changes
being too fast WITHIN a single steady /cmd_vel command (i.e. during normal
gait cycling, not command transitions), not the /cmd_vel transitions
themselves. Cycling commands quickly was muddying that signal by mixing
transition transients into every observation window. Slow, long holds
isolate it: most of each phase is spent under a constant /cmd_vel, so
whatever joint-target jitter you see there is the steady-state gait
behavior the slew-rate limiter needs to be judged against - the
forward/backward/turn transitions themselves are now a small fraction of
total runtime instead of the dominant event.

2026-07-23: LIN_X capped 0.5 -> 0.2 (user's explicit request, "for now").
0.5 was under the old project-wide lin_vel_x cap of 0.6, but that's the
wide range every *existing* checkpoint was actually trained under - the
newer "slow meandering pace" command range (lin_vel_x=(0.1,0.25), see
CLAUDE.md "Task Registration") is what any checkpoint trained after
2026-07-2x will actually expect, and 0.2 sits inside that range instead of
near the old range's upper end. ang_z=1.0 rad/s is unchanged, still an
untuned guess for a visible turn-in-place rate, not verified against any
training command range.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

PUBLISH_RATE_HZ = 10.0
HOLD_S = 15.0       # seconds to hold each motion command (long - see note above)
STOP_HOLD_S = 3.0   # seconds to hold zero between commands

LIN_X = 0.2   # m/s, capped 2026-07-23 - see note above
ANG_Z = 1.0   # rad/s, untuned guess for a visible turn rate

# (lin_x, lin_y, ang_z, hold_s)
CYCLE = [
    (LIN_X, 0.0, 0.0, HOLD_S),      # forward
    (0.0, 0.0, 0.0, STOP_HOLD_S),
    (-LIN_X, 0.0, 0.0, HOLD_S),     # backward
    (0.0, 0.0, 0.0, STOP_HOLD_S),
    (0.0, 0.0, ANG_Z, HOLD_S),      # turn left
    (0.0, 0.0, 0.0, STOP_HOLD_S),
    (0.0, 0.0, -ANG_Z, HOLD_S),     # turn right
    (0.0, 0.0, 0.0, STOP_HOLD_S),
]


class TestCmdVel(Node):
    def __init__(self):
        super().__init__('test_cmd_vel')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.dt = 1.0 / PUBLISH_RATE_HZ
        self.step_idx = 0
        self.step_elapsed = 0.0
        self.create_timer(self.dt, self.publish)
        self.get_logger().info(
            f'Publishing /cmd_vel test cycle (forward/backward/turn, '
            f'hold={HOLD_S}s) at {PUBLISH_RATE_HZ}Hz. Ctrl+C to stop.'
        )

    def publish(self):
        lin_x, lin_y, ang_z, hold_s = CYCLE[self.step_idx]

        if self.step_elapsed == 0.0:
            self.get_logger().info(
                f'-> lin_x={lin_x:+.2f} lin_y={lin_y:+.2f} ang_z={ang_z:+.2f} '
                f'for {hold_s:.1f}s'
            )

        msg = Twist()
        msg.linear.x = lin_x
        msg.linear.y = lin_y
        msg.angular.z = ang_z
        self.pub.publish(msg)

        self.step_elapsed += self.dt
        if self.step_elapsed >= hold_s:
            self.step_elapsed = 0.0
            self.step_idx = (self.step_idx + 1) % len(CYCLE)


def main():
    rclpy.init()
    node = TestCmdVel()
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
