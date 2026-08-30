# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 2026-07-19: new stage, built from data-driven analysis of the completed
# low_gait/v2_recovery2_rand_push run (4000 additional iterations, resumed
# through Rec1->Rec2->Rec2-Rand->Rec2-Rand-Push). Identical to
# ankybot_v3_env_cfg_recovery2_random_push.py except:
#   - reset_joints.position_range widened from +-15deg (+-0.2618 rad) to
#     +-25deg (+-0.4363 rad)
#   - reset_base.pose_range roll/pitch widened from +-15deg to +-25deg,
#     matching the joint reset widening
#   - foot_slip weight steepened -2.5 -> -3.5
#
# Rationale (from the completed run's TensorBoard data, steady-state window
# step>=4100 of the full 4000-iteration run):
#   - Episode_Termination/joint_pos_out_of_bounds stayed at 0.0-0.4% and
#     time_out stayed ~99.6-100% for the entire run -- the policy is not
#     failing at the current +-15deg reset/tilt difficulty, meaning further
#     training at this same level has diminishing returns. Widening reset
#     ranges is the natural next-difficulty-notch, the same logic used for
#     Rec1->Rec2's own reset widening.
#   - track_lin_vel_xy (+0.22) and track_ang_vel_z (+0.90) both climbed
#     while Metrics/base_velocity/error_vel_xy (-0.047) and error_vel_yaw
#     (-0.241) both dropped -- genuine, still-improving velocity tracking,
#     not a plateau or reward-hacking signature.
#   - foot_slip was the one reward term with a real negative trend across
#     the run (-0.230 -> -0.273) -- plausibly the push disturbances (active
#     in this env) force some slip during recovery, but steepening the
#     penalty is the same fix already validated once before in this
#     project's history (the 2026-07-15 gliding-exploit fix, see
#     /workspace/CLAUDE.md's low_gait family notes) rather than a new,
#     untested idea.
#   - No other reward term showed a concerning trend (all other deltas were
#     small/flat) so nothing else was changed -- action_rate, base_height,
#     air_time_variance, foot_air_time, joint_torques/velocity, lin_vel_z,
#     ang_vel_xy all stayed roughly stable.
# These specific new numbers (25deg, -3.5) are my own judgment call from
# this data, not independently confirmed with the user beyond the general
# instruction to iterate on reward/penalty/reset parameters from the
# gathered metrics -- flag/revert if they don't hold up in the new run.
#
# 2026-07-20: knees-on-ground standing collapse (Rec3-specific, confirmed
# via user playback -- v2_recovery2_rand_push stayed clean under the same
# -6.0/-6.0 height/orientation weights, only Rec3 still showed it) fixed by
# adding standing_leg_deviation_penalty (weight -2.0, thigh+foot joints,
# gated on is_standing_env) -- the winning fix out of 3 variants trained
# fresh from v2_recovery2_rand_push/model_3996.pt. A pure contact-based fix
# (thigh-link contact) was tried and rejected: the actual ground contact
# when kneeling registers on .*_Foot_Link (the proximal/knee-adjacent
# region of that link), not .*_Thigh_Link, so a thigh-contact sensor can't
# see this failure mode at all -- same class of body-level-vs-sub-body
# contact-sensor blind spot as the air_time_variance/foot_clearance_reward
# bugs documented elsewhere in this project's history. A pure
# target-height bump was also tried and rejected: it reduced but did not
# eliminate the collapse, and introduced a new front/back asymmetry (policy
# leaned on the front legs to hit the higher target while leaving the rear
# under-extended) -- an indirect height-based incentive gave the policy
# room to cheat asymmetrically rather than actually standing square.
# v3_recovery3_rand_push_fix3_standingdeviation (final checkpoint model_4995.pt)
# was renamed to v3_recovery3_rand_push by the user -- that run/checkpoint is
# now the canonical, working Rec3 log for this file.
#
# 2026-07-20, later same day: a further round of changes (harder push_robot,
# a dedicated turn-in-place command regime) was judged big enough to warrant
# its own env_cfg/task/run rather than mutating this working file in place --
# see ankybot_v3_env_cfg_recovery3_random_push2_turn.py (task
# Ankybot-Rec3-Rand-Push2-Turn-v3, run v3_recovery3_rand_push2_turn), built
# from this file's post-fix state. This file itself is unchanged by that
# split and continues to reflect the working v3_recovery3_rand_push lineage.

import isaaclab.sim as sim_utils
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
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from .ankybot_v3 import ANKYBOT_CFG  # isort:skip


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

    # robot
    robot: ArticulationCfg = ANKYBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.25,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # universal slow-meandering range, unified across the whole gauntlet 2026-07-23
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
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        imu_lin_accel = ObsTerm(
            func=mdp.imu_lin_acc,
            noise=Unoise(n_min=-0.45, n_max=0.45),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0725, n_max=0.0725),
        )
        actions = ObsTerm(
            func=mdp.last_action
        )

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
                "roll": (-0.4363, 0.4363),
                "pitch": (-0.4363, 0.4363),
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
            "position_range": (-0.4363, 0.4363),
            "velocity_range": (0.1, 0.1),
        },
    )

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base_Link"),
            "mass_distribution_params": (0.95, 1.05),
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
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
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
                "x": (-0.01, 0.01),
                "y": (-0.00375, 0.00375),
                "z": (-0.00125, 0.00125),
            },
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={
          "command_name": "base_velocity",
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
    foot_air_time = RewTerm(
        func=mdp.foot_air_time,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.8,
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
    foot_clearance_reward = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "std": 0.25,
            "tanh_mult": 2.5,
        },
    )
    #--penalties
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
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty,
        weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    rear_foot_min_angle = RewTerm(
        func=mdp.min_joint_angle_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["BL_Foot_Joint", "BR_Foot_Joint"]),
            "min_angle": 0.2618,
        },
    )
    rear_foot_max_angle = RewTerm(
        func=mdp.max_joint_angle_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["BL_Foot_Joint", "BR_Foot_Joint"]),
            "max_angle": 1.0472,
        },
    )
    front_foot_max_angle = RewTerm(
        func=mdp.max_joint_angle_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_Foot_Joint", "FR_Foot_Joint"]),
            "max_angle": 1.0472,
        },
    )
    rear_thigh_min_angle = RewTerm(
        func=mdp.min_joint_angle_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["BL_Thigh_Joint", "BR_Thigh_Joint"]),
            "min_angle": -0.1,
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-3.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 1.0
        },
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-3.0
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5
    )
    backward_motion = RewTerm(
        func=mdp.backward_motion_penalty,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
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
        weight=-8.0,
        params={"target_height": 0.235
        },
    )
    standing_leg_deviation = RewTerm(
        func=mdp.standing_leg_deviation_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Foot_Joint"]),
            "command_name": "base_velocity",
        },
    )
    hip_spread_left = RewTerm(
        func=mdp.standing_joint_target_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_Hip_Joint", "BL_Hip_Joint"]),
            "target": 0.0,
            "command_name": "base_velocity",
        },
    )
    hip_spread_right = RewTerm(
        func=mdp.standing_joint_target_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FR_Hip_Joint", "BR_Hip_Joint"]),
            "target": 0.0,
            "command_name": "base_velocity",
        },
    )
    standing_thigh_target = RewTerm(
        func=mdp.standing_joint_target_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Thigh_Joint"]),
            "target": 0.0,
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
        #self.sim.render_interval = 1
