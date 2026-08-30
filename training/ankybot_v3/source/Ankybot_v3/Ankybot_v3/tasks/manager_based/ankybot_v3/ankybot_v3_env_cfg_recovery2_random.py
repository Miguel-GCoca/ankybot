# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 2026-07-18: rewritten to apply the same rec1->rec2 transform to
# ankybot_v3_env_cfg_recovery1_random.py (rand1): identical to rand1 except:
#   - reset_joints.position_range narrowed from +-1.2 rad to +-15deg (+-0.2618 rad), matching rec2
#   - reset_base.pose_range gains roll/pitch +-15deg (+-0.2618 rad) tilt, matching rec2
#   - commands.base_velocity.ranges.lin_vel_x capped to (0.0, 0.6), matching the project-wide cap
#   - domain-randomization ranges (add_base_mass, randomize_base_com) widened slightly beyond
#     rand1's values as a small next-difficulty-notch ("slight randomizations" per user request,
#     not independently measured/derived -- mass_distribution_params (1.0,1.5)->(1.0,1.6),
#     com_range widened ~20% on each axis)
# The previous draft of this file (unregistered, structurally divergent from rand1) was discarded
# rather than patched, for the same reason as recovery2.py's rewrite -- see that file's header.

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
            noise=Unoise(n_min=-0.35, n_max=0.35),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0575, n_max=0.0575),
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

    ## interval
    #push_robot = EventTerm(
    #    func=mdp.push_by_setting_velocity,
    #    mode="interval",
    #    interval_range_s=(10.0, 15.0),
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot"),
    #        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    #    },
    #)


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
        weight=-2.5,
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
        self.episode_length_s = 3
        #self.sim.render_interval = 1
