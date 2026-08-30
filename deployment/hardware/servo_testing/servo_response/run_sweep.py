import argparse
import serial
import time
import csv
import os
import math
import sys

sys.stdout.reconfigure(line_buffering=True)

PORT           = '/dev/ttyACM0'   # change to your Arduino port
BAUD           = 115200
FREQ_MIN       = 0.1              # Hz
FREQ_MAX       = 10.0             # Hz
# Deadband search region — densely sampled to locate the response onset
DEADBAND_FREQ_MAX = 0.7           # Hz
DEADBAND_POINTS   = 10            # points between FREQ_MIN and DEADBAND_FREQ_MAX
UPPER_POINTS      = 6             # points between DEADBAND_FREQ_MAX and FREQ_MAX
RATED_SPEED    = 600.0            # deg/sec (0.1 sec / 60 deg)
MAX_AMPLITUDE  = 90.0              # deg — servo hard stops are 0-180, center is 90
# Peak velocity is set so the lowest test frequency reaches exactly
# MAX_AMPLITUDE without exceeding it. Higher frequencies use the same
# peak velocity, which naturally requires smaller amplitude.
PEAK_VELOCITY  = 2.0 * math.pi * FREQ_MIN * MAX_AMPLITUDE   # deg/sec
SPEED_FRACTION = PEAK_VELOCITY / RATED_SPEED
SETTLE_CYCLES  = 5               # unrecorded cycles per frequency
MEASURE_CYCLES = 5               # recorded cycles per frequency
SETTLE_BETWEEN = 2.0             # seconds between frequency runs
PRINT_INTERVAL_S = 0.3           # live readout cadence while collecting
BASE_OUT_DIR   = os.path.expanduser('~/Desktop/servo_measurement/sweep/')


def logspace(start, stop, n):
    """Return n logarithmically spaced values from start to stop."""
    log_start = math.log10(start)
    log_stop  = math.log10(stop)
    step = (log_stop - log_start) / (n - 1)
    return [10 ** (log_start + i * step) for i in range(n)]


def build_frequency_list():
    """
    Two-band log spacing: dense in the deadband search region
    (FREQ_MIN to DEADBAND_FREQ_MAX), sparser above it.
    """
    dense  = logspace(FREQ_MIN, DEADBAND_FREQ_MAX, DEADBAND_POINTS)
    sparse = logspace(DEADBAND_FREQ_MAX, FREQ_MAX, UPPER_POINTS + 1)[1:]  # drop duplicate boundary
    return dense + sparse


def amplitude_for(freq):
    """Amplitude (deg) giving PEAK_VELOCITY at this frequency."""
    return PEAK_VELOCITY / (2.0 * math.pi * freq)


def wait_for(ser, target, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == target:
            return
    raise TimeoutError(f"Timed out waiting for '{target}'")


def compute_frequency_response(rows, freq):
    """
    Fit a sinusoid at the known frequency using Fourier projection.
    Returns (amplitude_ratio, amplitude_db, phase_lag_deg).
    Phase lag > 0 means output lags input.
    """
    omega = 2.0 * math.pi * freq
    n     = len(rows)

    t_sec   = [r[0] / 1e6 for r in rows]
    mean_cmd  = sum(r[1] for r in rows) / n
    mean_meas = sum(r[2] for r in rows) / n
    cmd_ac  = [r[1] - mean_cmd  for r in rows]
    meas_ac = [r[2] - mean_meas for r in rows]

    def fourier_fit(t_list, y_list):
        s = sum(y * math.sin(omega * t) for t, y in zip(t_list, y_list)) / len(t_list)
        c = sum(y * math.cos(omega * t) for t, y in zip(t_list, y_list)) / len(t_list)
        return 2.0 * math.sqrt(s**2 + c**2), math.atan2(c, s)

    amp_cmd,  phi_cmd  = fourier_fit(t_sec, cmd_ac)
    amp_meas, phi_meas = fourier_fit(t_sec, meas_ac)

    amp_ratio     = amp_meas / amp_cmd if amp_cmd > 0 else 0
    amp_db        = 20.0 * math.log10(amp_ratio) if amp_ratio > 0 else float('-inf')
    phase_lag_deg = math.degrees(phi_cmd - phi_meas)
    phase_lag_deg = (phase_lag_deg + 180) % 360 - 180   # wrap to (-180, 180]

    return amp_ratio, amp_db, phase_lag_deg


def parse_args():
    parser = argparse.ArgumentParser(description="Servo sine sweep frequency response test")
    parser.add_argument('--servo_num', required=True, help="Servo identifier/number being tested")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(BASE_OUT_DIR, f"servo_{args.servo_num}")
    os.makedirs(out_dir, exist_ok=True)

    freqs = build_frequency_list()

    print(f"Connecting to {PORT}...")
    ser = serial.Serial(PORT, BAUD, timeout=5)
    time.sleep(2)

    wait_for(ser, 'READY')
    print("Arduino ready.")
    print(f"Peak velocity: {PEAK_VELOCITY:.1f} deg/sec "
          f"({SPEED_FRACTION*100:.0f}% of {RATED_SPEED:.0f} deg/sec rated)")
    print(f"Frequencies:   {', '.join(f'{f:.3f}' for f in freqs)} Hz\n")

    summary_rows = []

    for freq in freqs:
        amp          = amplitude_for(freq)
        settle_secs  = SETTLE_CYCLES  / freq
        measure_secs = MEASURE_CYCLES / freq
        total_secs   = settle_secs + measure_secs

        print(f"[{freq:.3f} Hz]  amp={amp:.3f}°  "
              f"duration={total_secs:.1f}s "
              f"({settle_secs:.1f}s settle + {measure_secs:.1f}s measure)...")

        ser.reset_input_buffer()
        ser.write(f"FREQ:{freq:.4f}:{amp:.4f}\n".encode())

        wait_for(ser, 't_us,cmd_deg,meas_deg', timeout=int(total_secs) + 10)

        deadline = time.time() + total_secs
        rows = []
        last_print = 0.0
        while time.time() < deadline:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) == 3:
                try:
                    t_us     = int(parts[0])
                    cmd_deg  = float(parts[1])
                    meas_deg = float(parts[2])
                    if t_us >= settle_secs * 1e6:
                        rows.append([t_us, cmd_deg, meas_deg])
                    now = time.time()
                    if now - last_print >= PRINT_INTERVAL_S:
                        last_print = now
                        print(f"    t={t_us/1e6:7.3f}s  cmd={cmd_deg:7.2f}  meas={meas_deg:7.2f}")
                except ValueError:
                    pass

        ser.write(b"STOP\n")
        wait_for(ser, 'STOPPED')
        print(f"  Servo stopped. Settling {SETTLE_BETWEEN:.0f}s...")
        time.sleep(SETTLE_BETWEEN)

        # Save raw data — filename uses rounded frequency
        raw_file = os.path.join(out_dir, f'sine_{freq:.3f}hz.csv')
        with open(raw_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t_us', 'cmd_deg', 'meas_deg'])
            writer.writerows(rows)

        if len(rows) > 10:
            amp_ratio, amp_db, phase_lag = compute_frequency_response(rows, freq)
            summary_rows.append([freq, amp, amp_ratio, amp_db, phase_lag])
            print(f"  {len(rows)} samples | "
                  f"amp ratio={amp_ratio:.3f} ({amp_db:.2f} dB) | "
                  f"phase lag={phase_lag:.1f}°")
        else:
            print(f"  Warning: only {len(rows)} samples collected")

        print(f"  Saved → {raw_file}\n")

    # Frequency response summary (Bode plot data)
    summary_file = os.path.join(out_dir, 'frequency_response.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['freq_hz', 'amplitude_deg', 'amp_ratio', 'amp_db', 'phase_lag_deg'])
        writer.writerows(summary_rows)
    print(f"Frequency response summary → {summary_file}")

    ser.close()
    print("All done.")


if __name__ == '__main__':
    main()
