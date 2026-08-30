import argparse
import serial
import time
import csv
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

PORT            = '/dev/ttyACM0'   # change to your Arduino port
BAUD            = 115200

NEUTRAL_DEG     = 90.0             # center position
STEP_TARGET_DEG = 150.0            # full-scale step target (60 deg step, per guide example)
RECORD_SECS     = 1.5              # recording window per step (rise + overshoot + settle)
REPEAT_COUNT    = 5                # number of up/down step pairs
SETTLE_BETWEEN  = 1.0              # pause between steps
BASE_OUT_DIR    = os.path.expanduser('~/Desktop/servo_measurement/step/')


def wait_for(ser, target, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == target:
            return
    raise TimeoutError(f"Timed out waiting for '{target}'")


def run_step(ser, target_deg):
    ser.reset_input_buffer()
    ser.write(f"STEP:{target_deg:.2f}\n".encode())
    wait_for(ser, 't_us,cmd_deg,meas_deg')

    deadline = time.time() + RECORD_SECS
    rows = []
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(',')
        if len(parts) == 3:
            try:
                rows.append([int(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                pass

    ser.write(b"STOP\n")
    wait_for(ser, 'STOPPED')
    return rows


def save_csv(out_dir, name, rows):
    path = os.path.join(out_dir, name)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_us', 'cmd_deg', 'meas_deg'])
        writer.writerows(rows)
    print(f"  {len(rows)} samples saved → {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Servo step response data collection")
    parser.add_argument('--servo_num', required=True, help="Servo identifier/number being tested")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(BASE_OUT_DIR, f"servo_{args.servo_num}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Connecting to {PORT}...")
    ser = serial.Serial(PORT, BAUD, timeout=5)
    time.sleep(2)

    wait_for(ser, 'READY')
    print("Arduino ready.")
    print(f"Step: {NEUTRAL_DEG}° <-> {STEP_TARGET_DEG}°, "
          f"{REPEAT_COUNT} repetitions, {RECORD_SECS}s per step\n")

    # Start at neutral before the first step
    run_step(ser, NEUTRAL_DEG)
    time.sleep(SETTLE_BETWEEN)

    for rep in range(1, REPEAT_COUNT + 1):
        print(f"[Rep {rep}] Step up {NEUTRAL_DEG}° -> {STEP_TARGET_DEG}°...")
        rows_up = run_step(ser, STEP_TARGET_DEG)
        save_csv(out_dir, f'step_up_rep{rep}.csv', rows_up)
        time.sleep(SETTLE_BETWEEN)

        print(f"[Rep {rep}] Step down {STEP_TARGET_DEG}° -> {NEUTRAL_DEG}°...")
        rows_down = run_step(ser, NEUTRAL_DEG)
        save_csv(out_dir, f'step_down_rep{rep}.csv', rows_down)
        time.sleep(SETTLE_BETWEEN)
        print()

    ser.close()
    print("All done.")


if __name__ == '__main__':
    main()
