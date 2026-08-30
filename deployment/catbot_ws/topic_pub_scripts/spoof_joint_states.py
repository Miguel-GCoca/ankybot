#!/usr/bin/env python3
"""
Spoofs /joint_states from /joint_commands for bench-testing without the Mega
wired up. NOT an instant echo - see 2026-07-21 note below. Still a
placeholder for real hardware feedback, not a substitute for it.

2026-07-21: rewritten from a plain instant-echo (position = command,
velocity = 0 always) after that was flagged as a likely cause of overly
aggressive/jerky policy output during no-Mega bench testing. Reasoning: the
policy was trained on joint_pos/joint_vel from real PhysX actuator dynamics
(finite PD stiffness/damping + DelayedPDActuatorCfg's 0-30ms delay), so real
feedback always lags the commanded target and joint_vel is a physically real,
nonzero signal. An instant, zero-velocity echo is out-of-distribution input -
the policy never sees any consequence of its own actions and has no signal to
temper the next one. This version instead simulates a simple slew-rate-limited
follower (converges toward each commanded target at MAX_VEL_RAD_S, not
instantly) and computes velocity via finite difference, so feedback at least
resembles what training expects.

2026-07-21, corrected same day: MAX_VEL_RAD_S raised 3.0 -> 10.0, and no
longer meant to mirror policy_runner's max_joint_vel_rad_s. That parameter
caps how far apart consecutive *commanded* targets can be (a rate limit on
the command sequence); it does not slow the real servo's own tracking speed
once it has a target, which is bounded only by the servo's rated physical
speed (~10.4719755 rad/s no-load, 600 deg/s). Setting this to match the
actuation-side cap exactly (was 3.0, matching the old default) gave the
spoof zero margin and made the feedback path itself an extra, redundant
bottleneck on top of the real one - understating how far/fast the legs were
actually being told to move, which (in a reactive closed loop) can push the
policy toward faster, smaller-amplitude corrections. 10.0 stays just under
the rated no-load ceiling so this never binds ahead of the real
policy_runner-side cap, whatever that's currently set to.

2026-07-22: MAX_VEL_RAD_S halved 10.0 -> 5.0 (diagnostic, doubling the
spoofer's lag) after a full-spoof run showed the policy commanding gait
motion far faster than it looks in sim. Same day's other config change
(action_filter_alpha 0.8->0.15, max_joint_vel_rad_s 3.0->8.0 - see
ankybot_policy.yaml) is the leading suspect, but this node feeds the
closed-loop obs (joint_pos/joint_vel) the policy reacts to - testing
whether an encoder that converges too fast is itself part of the problem
(policy sees its target reached almost immediately, has less lag signal
than training's real PD dynamics provided, and may compensate by pushing
harder/faster). Not yet confirmed which change is at fault; revert to
10.0 if this isn't it.

2026-07-23: both constants replaced with real hardware-measured values
instead of guesses, now that the data exists.

MAX_VEL_RAD_S set to 8.4 rad/s, the median of (step_size/rise_time) over
every range10/20/30 trial in the 2026-07-23 step-response sweep
(calibration_ws/python/step_response_sweep/servo_measurement/), pooled
across all 12 joints and both directions (n=360, per-type medians 7.7-9.2
rad/s, all close to this pooled figure) - the smaller range4 steps were
excluded as too noise-floor-limited for a reliable rise-time read. This
lands close to both prior guesses (10.0, then 5.0) without needing either -
turns out the original 10.0-near-the-no-load-ceiling reasoning wasn't far
off, and the 2026-07-22 halving to 5.0 moved further from the real
figure, not closer.

A genuine onset delay was also added, which this spoofer never had before
(the old version started slewing toward a new target the instant it
arrived) - real actuation has a measured 4.04-14.96ms command-to-motion
delay (actuation_delay/measure_actuation_delay.py, same day as the
DelayedPDActuatorCfg min_delay/max_delay bake-in - see the main project
CLAUDE.md's Mechanical Values). DELAY_S=0.0095 is the mean of that range
(a single fixed representative delay, not sim's per-episode-randomized
min_delay/max_delay - this is one deterministic bench node, not a batch of
training envs). Implemented as a small deque-based delay line: incoming
commands are buffered and the follower only starts chasing a command once
DELAY_S has actually elapsed since it arrived, rather than reacting
immediately.
"""
import collections

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

MAX_VEL_RAD_S = 8.4  # data-derived 2026-07-23 (was 5.0) - see note above
DELAY_S = 0.0095     # data-derived 2026-07-23, new (was 0.0) - see note above


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
        # still at least DELAY_S old - that's the one actually in effect now
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
