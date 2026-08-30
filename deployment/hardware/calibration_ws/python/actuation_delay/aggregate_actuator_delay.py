#!/usr/bin/env python3
"""
Pools several joints' actuation_delay_<joint>.json summaries (from
measure_actuation_delay.py) into one overall min_delay/max_delay
recommendation. DelayedPDActuatorCfg.min_delay/max_delay (see
isaaclab.actuators.actuator_pd_cfg) are a single physics-step range applied
to every joint in ANKYBOT_LEG_ACTUATOR_CFG, not a per-joint-type dict like
stiffness/damping - so before trusting a global number it's worth measuring
more than just one joint (the existing stiffness/damping characterization
only ever measured the BL leg and extrapolated; this at least lets you
check whether hip/thigh/foot or left/right actually differ before doing
the same here).

Takes the min across all joints' lower bounds and the max across all
joints' upper bounds - the widest range any tested joint actually showed,
which is the conservative choice for a shared delay model.

Usage:
    python3 aggregate_actuator_delay.py actuation_delay_BL_Hip_Joint.json \
        actuation_delay_BL_Thigh_Joint.json actuation_delay_BL_Foot_Joint.json
"""
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Pool multiple joints' actuation-delay summaries")
    parser.add_argument('summaries', nargs='+', help="One or more actuation_delay_<joint>.json files")
    return parser.parse_args()


def main():
    args = parse_args()

    entries = []
    sim_dt_s = None
    for path in args.summaries:
        with open(path) as f:
            data = json.load(f)
        if sim_dt_s is None:
            sim_dt_s = data['sim_dt_s']
        elif data['sim_dt_s'] != sim_dt_s:
            raise SystemExit(
                f"{path} was measured with sim_dt_s={data['sim_dt_s']}, but an earlier file used "
                f"{sim_dt_s} - re-run measure_actuation_delay.py for all joints with the same "
                f"SIM_DT_S before pooling (it must match ankybot_v3_env_cfg_base.py's self.sim.dt)."
            )
        entries.append(data)

    print(f"{'Joint':<16} {'lower min (ms)':>15} {'upper max (ms)':>15} {'n_valid':>8}")
    for e in entries:
        print(f"{e['joint']:<16} {e['delay_lower_s']['min']*1000:>15.2f} "
              f"{e['delay_upper_s']['max']*1000:>15.2f} {e['n_valid']:>8}")

    overall_min_s = max(0.0, min(e['delay_lower_s']['min'] for e in entries))
    overall_max_s = max(e['delay_upper_s']['max'] for e in entries)
    min_delay_steps = int(overall_min_s // sim_dt_s)
    max_delay_steps = -(-int(round(overall_max_s / sim_dt_s)))  # ceil

    print(f"\nPooled across {len(entries)} joint(s), sim_dt_s={sim_dt_s}:")
    print(f"    overall lower bound: {overall_min_s*1000:.2f}ms -> min_delay={min_delay_steps} steps")
    print(f"    overall upper bound: {overall_max_s*1000:.2f}ms -> max_delay={max_delay_steps} steps")
    print(f"\nPaste into ankybot_v3.py's ANKYBOT_LEG_ACTUATOR_CFG:")
    print(f"    min_delay={min_delay_steps},")
    print(f"    max_delay={max_delay_steps},")


if __name__ == '__main__':
    main()
