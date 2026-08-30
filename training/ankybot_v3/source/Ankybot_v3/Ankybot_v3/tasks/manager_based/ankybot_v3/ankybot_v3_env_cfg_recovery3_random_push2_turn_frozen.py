# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 2026-07-24: one-off continuation config, NOT part of the regular Base->Rec1->Rec2->Rec3
# lineage (which has since moved on to the joint_vel-removed v5 gauntlet). Reconstructed from
# low_gait/v3_recovery3_rand_push2_turn's saved params/env.yaml (its exact training config,
# through model_5994.pt) -- this is the checkpoint the user confirmed is working well deployed
# on the ros2_copy policy_runner branch with joint-state spoofing. That live checkpoint predates
# essentially every 2026-07-21 through 2026-07-24 project change: this file restores all of them
# to match so continued training stays on the same distribution, rather than resuming into the
# current (heavily drifted) ankybot_v3_env_cfg_recovery3_random_push2_turn.py, which would hand
# this checkpoint a different pose/actuator/friction/command-range/observation-noise/obs-count
# than it ever trained under.
#
# Differences from today's live ankybot_v3_env_cfg_recovery3_random_push2_turn.py, all reverted
# here to match the snapshot:
#   - joint_vel added BACK into observations (48-obs contract; live file is on the 36-obs
#     no-joint-vel contract) and all observation noise (Unoise) removed -- snapshot predates the
#     2026-07-23 joint_vel-removal/noise-addition change entirely.
#   - init_state.joint_pos: thigh 0.0 -> 0.5236, foot 0.5236 -> 0.384 (this snapshot predates
#     both the mid-project 30 deg thigh experiment AND its 2026-07-24 correction to 0.0).
#   - actuator stiffness/damping/delay reverted to the old datasheet-derived values (hip
#     10.6449->44.6604, thigh 3.5125->9.7725, foot 0.8102->4.1693 N*m/rad stiffness; hip
#     0.2334->1.4291, thigh 0.1040->0.3127, foot 0.0236->0.1334 N*m*s/rad damping;
#     min_delay 2->0, max_delay 7->15) -- snapshot predates the 2026-07-23 step-response
#     recalibration.
#   - foot_physics_material friction range (0.8,1.2) -> (0.6,1.0).
#   - add_base_mass mass_distribution_params (0.95,1.05) -> (1.0,1.6) (this branch's own
#     original range, wider than Rec1-Rand's; predates that lineage's tightening entirely).
#   - randomize_base_com com_range widened back to x:+-0.045/y:+-0.018/z:+-0.006 (from
#     x:+-0.01/y:+-0.00375/z:+-0.00125).
#   - command ranges: lin_vel_x (0.1,0.25)->(0.0,0.6), lin_vel_y (-0.12,0.12)->(-0.4,0.4),
#     ang_vel_z (-0.15,0.15)->(-0.4,0.4) -- snapshot predates the 2026-07-23 universal
#     slow-meandering-pace unification. track_lin_vel_xy/track_ang_vel_z std 0.05/0.10 -> 0.25/0.25
#     to match.
#   - base_orientation weight -8.0 -> -6.0. base_height weight -8.0 -> -6.0, target_height
#     0.235 -> 0.24. lin_vel_z weight -3.0 -> -1.0.
#   - standing_leg_deviation joint_names back to [Thigh_Joint, Foot_Joint] (both -- snapshot
#     predates the 2026-07-24 split that pulled Thigh_Joint out into standing_thigh_target).
#   - rear_foot_min_angle/rear_foot_max_angle/front_foot_max_angle/rear_thigh_min_angle,
#     backward_motion, hip_spread_left/hip_spread_right/standing_thigh_target: all REMOVED --
#     none of these reward terms existed yet when this checkpoint trained.
#
# Deliberately NOT reverted (kept at current behavior, both are code-level correctness fixes
# rather than distributional/pose/observation characteristics, so keeping them doesn't create a
# checkpoint-loading or observation-contract mismatch):
#   - gait/foot_air_time's 2026-07-23 direction-aware real-speed gate (mdp/rewards.py's
#     _forward_progress_scale) stays ACTIVE (asset_cfg=SceneEntityCfg("robot") passed, same as
#     the live file) -- asset_cfg is now a required argument on both functions regardless, and
#     there's no safety reason to fight the exploit-closing fix just to match old behavior
#     exactly. gait's own threshold param was never explicitly set in either snapshot or live
#     file (both use the function default).
#   - foot_clearance_reward's per-leg-averaged formula (a rewards.py function-body fix, not a
#     per-env parameter) -- out of scope, same reasoning as the Rec2-StiffnessIsolated precedent.
#
# 2026-07-24, user's explicit follow-up instructions on top of the historical reconstruction
# above (both deliberate deviations FROM the pure snapshot, not part of "matching what this
# checkpoint trained under"):
#   - Command range switched from the snapshot's own old wide range back to the project's
#     current universal slow-meandering-pace convention: lin_vel_x (0.0,0.6)->(0.1,0.25),
#     lin_vel_y (-0.4,0.4)->(-0.12,0.12), ang_vel_z (-0.4,0.4)->(-0.15,0.15), with
#     track_lin_vel_xy/track_ang_vel_z std tightened to match (0.25/0.25 -> 0.05/0.10), per
#     user request ("make sure it follows slow commands"). Note: CLAUDE.md's own Task
#     Registration section flags this exact range change as "a genuine distributional shift...
#     whichever stage retrains first under it should be a from-scratch run, not a resume" --
#     the user gave this instruction directly and explicitly for this resumed continuation
#     regardless, so proceeding as instructed, flagged here for visibility.
#   - foot_air_time, foot_clearance_reward, and foot_slip (leg lift / air time / foot) split
#     from one aggregate 4-foot term each into 4 explicit per-leg terms (FL/FR/BL/BR), per user
#     request ("leg lift, air time, foot are all rewarded per leg"). Each per-leg term's weight
#     is the original aggregate weight divided by 4, which exactly reproduces the original total
#     for foot_air_time (sums across feet) and foot_slip (sums across feet), and reproduces
#     foot_clearance_reward's swing-phase (moving-env) magnitude exactly (that branch averages
#     across feet, so original_weight * mean == (original_weight/4) * sum-of-4-per-leg-terms).
#     Side effect: foot_clearance_reward's standing-phase branch (-sum(is_airborne), a
#     collapse-detection guard) becomes 4x weaker in aggregate under this split, since that
#     branch itself already sums rather than averages -- accepted as a minor tradeoff since
#     standing_leg_deviation_penalty (weight -2.0, joint-angle-based) already independently
#     guards against standing collapse.
#
# Resume point: low_gait/v3_recovery3_rand_push2_turn/model_5994.pt (this run's own last
# checkpoint), as a new run under this frozen config.
#
# 2026-07-24, later same day: per-leg foot_slip weight bumped -0.875 -> -1.25 each (aggregate
# -3.5 -> -5.0, a ~43% steepening, matching this project's own prior foot_slip steepening step
# size of -2.5->-3.5/40%) -- user's explicit request for a higher foot slip penalty, made after
# the first 3000-iteration run (low_gait/v3_recovery3_rand_push2_turn_frozen) was launched under
# the original -3.5 aggregate weight and had already shown rear feet (bl/br) sliding noticeably
# more than front (fl/fr) via the new per-leg breakdown. This edit does NOT affect that
# already-running process (env cfg is read once at process start); it takes effect only for the
# next resumed run launched from this file, per this project's established
# edit-now-relaunch-later convention.

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from .ankybot_v3 import ANKYBOT_CFG  # isort:skip

# Old datasheet-derived actuator values this checkpoint actually trained under (see file header)
# -- kept local to this file rather than touching ankybot_v3.py's shared, currently-validated
# ANKYBOT_LEG_ACTUATOR_CFG used by every other active lineage file.
FROZEN_LEG_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_Hip_Joint", ".*_Thigh_Joint", ".*_Foot_Joint"],
    effort_limit=3.92266,
    velocity_limit=10.4719755,
    effort_limit_sim=3.92266,
    velocity_limit_sim=10.4719755,
    stiffness={
        ".*_Hip_Joint": 44.6604,
        ".*_Thigh_Joint": 9.7725,
        ".*_Foot_Joint": 4.1693,
    },
    damping={
        ".*_Hip_Joint": 1.4291,
        ".*_Thigh_Joint": 0.3127,
        ".*_Foot_Joint": 0.1334,
    },
    min_delay=0,
    max_delay=15,
)


##
# Scene definition
##


@configclass
class AnkybotV3SceneCfg(InteractiveSceneCfg):
    """Configuration for a ankybot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                friction_combine_mode="average",
                restitution=0.0,
            ),
        ),
    )

    # robot -- init_state.joint_pos and actuators overridden to match this checkpoint's original
    # training snapshot (see file header); everything else about ANKYBOT_CFG (USD, rigid/
    # articulation props, base spawn pos/height) is unchanged and already matches.
    robot: ArticulationCfg = ANKYBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            joint_pos={
                ".*_Hip_Joint": 0.0,
                ".*_Thigh_Joint": 0.5236,
                ".*_Foot_Joint": 0.384,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={"legs": FROZEN_LEG_ACTUATOR_CFG},
    )

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FinalRemakeURDF_SLDASM/.*_Link",
        history_length=3,
        track_air_time=True,
    )
    # imu sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FinalRemakeURDF_SLDASM/Base_Link",
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.TurnInPlaceVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.25,
        rel_turn_in_place_envs=0.2,
        ranges=mdp.TurnInPlaceVelocityCommandCfg.Ranges(
            # 2026-07-24: switched to the project's current universal slow-meandering-pace
            # range per explicit user instruction, NOT the snapshot's own old wide range
            # (0.0,0.6)/(-0.4,0.4) -- see file header for the deliberate-deviation note.
            lin_vel_x=(0.1, 0.25),
            lin_vel_y=(-0.12, 0.12),
            ang_vel_z=(-0.15, 0.15),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*Joint"],
        scale=0.4,
        use_default_offset=True,
        clip={".*": (-1.4923, 1.4923)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP -- 48-obs contract (joint_vel included, no noise),
    matching this checkpoint's original observation layout exactly."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved -- must match the trained network's input layout)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)
        imu_lin_accel = ObsTerm(func=mdp.imu_lin_acc)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization"""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "roll": (-0.2618, 0.2618),
                "pitch": (-0.2618, 0.2618),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (-0.05, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.15, 0.15),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*Joint"],
            ),
            "position_range": (-0.2618, 0.2618),
            "velocity_range": (0.1, 0.1),
        },
    )

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base_Link"),
            "mass_distribution_params": (1.0, 1.6),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base_Link"),
            "com_range": {
                "x": (-0.045, 0.045),
                "y": (-0.018, 0.018),
                "z": (-0.006, 0.006),
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP -- matches v3_recovery3_rand_push2_turn's original training
    config (see file header for the gait/foot_air_time speed-gate exception kept active)."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            # tightened to match the 2026-07-24 slow-command switch (see file header)
            "std": 0.05,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.10,
        },
    )
    # foot_air_time split per-leg 2026-07-24 (user request) -- weight 2.0/4 each, see file header
    foot_air_time_fl = RewTerm(
        func=mdp.foot_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_Foot_Link"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,
        },
    )
    foot_air_time_fr = RewTerm(
        func=mdp.foot_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_Foot_Link"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,
        },
    )
    foot_air_time_bl = RewTerm(
        func=mdp.foot_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BL_Foot_Link"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,
        },
    )
    foot_air_time_br = RewTerm(
        func=mdp.foot_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BR_Foot_Link"]),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,
        },
    )
    gait = RewTerm(
        func=mdp.three_leg_stance_reward,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # foot_clearance_reward ("leg lift") split per-leg 2026-07-24 -- weight 1.0/4 each
    foot_clearance_reward_fl = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_Foot_Link"]),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "std": 0.25,
            "tanh_mult": 2.5,
        },
    )
    foot_clearance_reward_fr = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_Foot_Link"]),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "std": 0.25,
            "tanh_mult": 2.5,
        },
    )
    foot_clearance_reward_bl = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BL_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BL_Foot_Link"]),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "std": 0.25,
            "tanh_mult": 2.5,
        },
    )
    foot_clearance_reward_br = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BR_Foot_Link"]),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "std": 0.25,
            "tanh_mult": 2.5,
        },
    )
    # --penalties
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
        },
    )
    prolonged_airborne = RewTerm(
        func=mdp.prolonged_airborne_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "max_air_time": 0.75,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # foot_slip ("foot") split per-leg 2026-07-24 -- weight -3.5/4 each
    foot_slip_fl = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_Foot_Link"]),
            "threshold": 1.0,
        },
    )
    foot_slip_fr = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_Foot_Link"]),
            "threshold": 1.0,
        },
    )
    foot_slip_bl = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BL_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BL_Foot_Link"]),
            "threshold": 1.0,
        },
    )
    foot_slip_br = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BR_Foot_Link"]),
            "threshold": 1.0,
        },
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*Joint"],
            )
        },
    )
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.5e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*Joint"],
            )
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-6.0,
        params={"target_height": 0.24},
    )
    standing_leg_deviation = RewTerm(
        func=mdp.standing_leg_deviation_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Thigh_Joint", ".*_Foot_Joint"]),
            "command_name": "base_velocity",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    joint_pos_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Joint"],
            ),
            "bounds": (-1.5708, 1.5708),
        },
    )


##
# Environment configuration
##


@configclass
class AnkybotV3EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene = AnkybotV3SceneCfg(num_envs=1024, env_spacing=1.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    # Post initialization
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt * self.decimation
        self.scene.imu.debug_vis = False
        self.commands.base_velocity.debug_vis = False


@configclass
class AnkybotV3EnvCfg_PLAY(AnkybotV3EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 10
        self.events.push_robot.interval_range_s = (5.0, 10.0)
