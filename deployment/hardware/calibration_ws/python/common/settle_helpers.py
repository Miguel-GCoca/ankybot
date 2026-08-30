#!/usr/bin/env python3
"""Shared settle-detection / measurement-uncertainty helpers for
find_calibration_extremes.py and run_step_response_targets.py."""
import statistics
import time


def sample_stats(samples):
    """Returns (mean, margin) where margin = 2 * sample stdev - a simple,
    consistent way to report measurement uncertainty from fluctuation."""
    mean = statistics.fmean(samples)
    margin = 2.0 * statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, margin


def wait_for_self_settle(read_fn, poll_period_s, window_size, tol, hold_s, timeout_s, log_fn=None):
    """Polls read_fn() and waits until a rolling window of readings has a
    spread (max-min) under tol, continuously for hold_s. Used when there's
    no known target yet (establishing a calibration point, not testing
    against one). If log_fn is given, it's called as log_fn(elapsed_s,
    value) for every sample - lets the caller build a full CSV trace
    without this module needing to know about file formats. Returns
    (settled: bool, elapsed_s, last_window)."""
    t_start = time.time()
    window = []
    stable_since = None

    while True:
        now = time.time()
        elapsed = now - t_start
        if elapsed > timeout_s:
            return False, elapsed, window

        value = read_fn()
        if log_fn is not None:
            log_fn(elapsed, value)

        window.append(value)
        if len(window) > window_size:
            window.pop(0)

        if len(window) == window_size and (max(window) - min(window)) <= tol:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= hold_s:
                return True, elapsed, window
        else:
            stable_since = None

        time.sleep(poll_period_s)


def wait_for_target_settle(read_fn, target, margin, poll_period_s, hold_s, timeout_s, log_fn=None):
    """Polls read_fn() (assumed already in the same units as target/margin)
    until the reading stays within [target-margin, target+margin]
    continuously for hold_s. Call this immediately after issuing a command
    and elapsed_s is the step response time. If log_fn is given, it's
    called as log_fn(elapsed_s, value) for every sample. Returns
    (settled: bool, elapsed_s, last_value)."""
    t_start = time.time()
    stable_since = None
    last_value = None

    while True:
        now = time.time()
        elapsed = now - t_start
        if elapsed > timeout_s:
            return False, elapsed, last_value

        last_value = read_fn()
        if log_fn is not None:
            log_fn(elapsed, last_value)

        if abs(last_value - target) <= margin:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= hold_s:
                return True, elapsed, last_value
        else:
            stable_since = None

        time.sleep(poll_period_s)
