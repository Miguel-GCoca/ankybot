# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height-command variant of the Ankybot V3 locomotion environment.

Identical to the baseline_full configuration with three additions:
- A scalar `desired_height` command (0.15–0.24 m) resampled each interval alongside velocity.
- A `height_command` observation term (obs dim 48 → 49).
- Z-offset randomisation in `reset_base` to vary starting height at each episode reset.

All reward weights, event ranges, and terminations match the baseline_full run exactly.
"""

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

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

from . import mdp

##
# Pre-defined configs
##

from .ankybot_v3 import ANKYBOT_CFG  # isort:skip


##
# Scene definition
##


@configclass
class AnkybotV3HeightSceneCfg(InteractiveSceneCfg):
    """Configuration for the Ankybot V3 height-command scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                friction_combine_mode="max",
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
        rel_standing_envs=0.3,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.25),
            lin_vel_y=(-0.12, 0.12),
            ang_vel_z=(-0.15, 0.15),
        ),
    )

    desired_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        ranges=mdp.UniformHeightCommandCfg.Ranges(height=(0.15, 0.24)),
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
        """Observations for policy group. Total: 49 (48 base + 1 height command)."""

        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)
        imu_lin_accel = ObsTerm(func=mdp.imu_lin_acc)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "desired_height"},
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
    """Configuration for randomization — matches baseline_full with z-offset added to pose_range."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
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
            "position_range": (-0.8, 0.8),
            "velocity_range": (0.0, 0.0),
        },
    )

    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms — weights match baseline_full; base_height tracks the desired_height command."""

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
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 0.5,
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
    #gait = RewTerm(
    #    func=mdp.GaitReward,
    #    weight=0.5,
    #    params={
    #        "std": 0.5,
    #        "max_err": 0.4,
    #        "velocity_threshold": 0.5,
    #        "synced_feet_pair_names": (("FL_Foot_Link", "BR_Foot_Link"), ("FR_Foot_Link", "BL_Foot_Link")),
    #        "asset_cfg": SceneEntityCfg("robot"),
    #        "sensor_cfg": SceneEntityCfg("contact_forces"),
    #    },
    #)

    # -- penalties
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.75,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
        },
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 1.0,
        },
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
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
        func=mdp.base_height_command_l2,
        weight=-5.0,
        params={"command_name": "desired_height"},
    )
    #hip_pos_deviation = RewTerm(
    #    func=mdp.joint_pos_soft_limit_penalty,
    #    weight=-0.25,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Joint"]),
    #        "soft_limit": 0.175,
    #        "hard_limit": 0.25,
    #    },
    #)
    #thigh_pos_deviation = RewTerm(
    #    func=mdp.joint_pos_soft_limit_penalty,
    #    weight=-0.25,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Thigh_Joint"]),
    #        "soft_limit": 0.3,
    #        "hard_limit": 0.65,
    #    },
    #)
    foot_pos_deviation = RewTerm(
        func=mdp.joint_pos_soft_limit_penalty,
        weight=-0.25e-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Foot_Joint"]),
            "soft_limit": 0.7,
            "hard_limit": 1.1,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms — matches baseline_full (base_contact active)."""

    #time_out = DoneTerm(func=mdp.time_out, time_out=True)
#
    #base_contact = DoneTerm(
    #    func=velocity_mdp.illegal_contact,
    #    params={
    #        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Base_Link"),
    #        "threshold": 1.0,
    #    },
    #)

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
class AnkybotV3HeightEnvCfg(ManagerBasedRLEnvCfg):
    """Height-command locomotion environment for Ankybot V3."""

    # Scene settings
    scene = AnkybotV3HeightSceneCfg(num_envs=1024, env_spacing=1.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 10
        self.episode_length_s = 20
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt * self.decimation
        self.scene.imu.debug_vis = False
        self.commands.base_velocity.debug_vis = False
        self.commands.desired_height.debug_vis = False


@configclass
class AnkybotV3HeightEnvCfg_PLAY(AnkybotV3HeightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 3
