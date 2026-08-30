# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 2026-07-23: one-off isolation test, NOT part of the regular Base->Rec1->Rec2->Rec3 lineage.
# Reconstructed from v2_recovery2_slowpace's saved params/env.yaml (its exact training config,
# model_3997.pt) to test ONLY the actuator stiffness/damping change in isolation, holding every
# reward/pose parameter fixed at what that checkpoint actually trained under. Differences from
# ankybot_v3_env_cfg_recovery2.py (today's live config), all reverted here to match slowpace's
# snapshot: foot default pose 0.7854->0.384 (override below), foot_clearance_reward std
# 0.01->0.25, lin_vel_z weight -3.0->-1.0, base_height target 0.25->0.23/weight -8.0->-3.0,
# and rear_foot_min_angle/rear_foot_max_angle/rear_thigh_min_angle/standing_leg_deviation removed
# entirely (didn't exist yet in slowpace's era). Command ranges/track_lin_vel_xy std/foot_clearance
# weight/events/actions were already identical between the two snapshots -- left as-is.
# Deliberately NOT reverting: the actuator itself (ankybot_v3.py's ANKYBOT_LEG_ACTUATOR_CFG) --
# that's the one variable this test exists to observe, at its current (new) stiffness/damping/delay
# values. Also NOT reverting foot_clearance_reward's per-leg-averaged formula (a mdp/rewards.py
# function fix, not a per-env parameter) -- out of scope for this isolation.

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

    # robot -- init_state.joint_pos overridden to match v2_recovery2_slowpace's original foot
    # default (0.384 rad, 22deg); current live ANKYBOT_CFG uses 0.7854 (45deg). Actuator
    # (stiffness/damping/delay) intentionally left at its current (new) values -- the one
    # variable under test.
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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.25,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
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
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
        )
        imu_lin_accel = ObsTerm(
            func=mdp.imu_lin_acc,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
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

@configclass
class RewardsCfg:
    """Reward terms for the MDP -- matches v2_recovery2_slowpace's original training config."""

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
            "threshold": 0.5,
        },
    )
    gait = RewTerm(
        func=mdp.three_leg_stance_reward,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
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
    # diagnostic only (weight negligible, does not meaningfully affect training) -- front/rear
    # split of foot_clearance_reward, carried over from the current config for bonus data even
    # though slowpace's own era didn't have these. See rear_foot_max_angle skip-caveat note in
    # ankybot_v3_env_cfg_recovery2.py for why weight can't be exactly 0.0.
    front_foot_clearance_diag = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_Foot_Link", "FR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_Foot_Link", "FR_Foot_Link"]),
            "command_name": "base_velocity",
            "target_height": 0.06,
            "std": 0.25,
            "tanh_mult": 2.5,
        },
    )
    rear_foot_clearance_diag = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BL_Foot_Link", "BR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BL_Foot_Link", "BR_Foot_Link"]),
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
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 1.0
        },
    )
    # diagnostic only, see foot_clearance diagnostic note above
    front_foot_slip_diag = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_Foot_Link", "FR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_Foot_Link", "FR_Foot_Link"]),
            "threshold": 1.0
        },
    )
    rear_foot_slip_diag = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["BL_Foot_Link", "BR_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BL_Foot_Link", "BR_Foot_Link"]),
            "threshold": 1.0
        },
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5
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
        weight=-3.0,
        params={"target_height": 0.23
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
        self.episode_length_s = 3
        #self.sim.render_interval = 1
