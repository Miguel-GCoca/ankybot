# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Ankybot V2 locomotion environment with straight and turning gait rewards."""


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
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

from . import mdp

##
# Pre-defined configs
##

from .ankybot_v2 import ANKYBOT_CFG  # isort:skip


##
# Scene definition
##


@configclass
class AnkybotV2SceneCfg(InteractiveSceneCfg):
    """Configuration for a ankybot scene."""

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
        prim_path="{ENV_REGEX_NS}/Robot/FinalURDF_SLDASM/.*",
        history_length=3,
        track_air_time=True,
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
        class_type="Ankybot_v2.tasks.manager_based.ankybot_v2.mdp.commands:ExclusiveForwardTurnVelocityCommand",
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0, 0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_Hip_Joint", ".*_Thigh_Joint", ".*_Foot_Joint"],
        scale=0.2,
        use_default_offset=True,
        clip={".*": (-1.4923, 1.4923)},
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        actions = ObsTerm(
            func=mdp.last_action
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy : PolicyCfg = PolicyCfg()


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
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.15, 0.15),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Hip_Joint", ".*_Thigh_Joint", ".*_Foot_Joint"],
            ),
            "position_range": (-0.02, 0.02),
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


#use when robot is walking
#
#  randomize_base_com = EventTerm(
#      func=mdp.randomize_rigid_body_com,
#      mode="startup",
#      params={
#          "asset_cfg": SceneEntityCfg("robot", body_names="Base_Link"),
#          "com_range": {
#              "x": (-0.01, 0.01),
#              "y": (-0.01, 0.01),
#              "z": (-0.005, 0.005),
#          },
#      },
#  )

#   base_external_force_torque = EventTerm(
#       func=mdp.apply_external_force_torque,
#       mode="reset",
#       params={
#           "asset_cfg": SceneEntityCfg("robot", body_names="body"),
#           "force_range": (-0.1, 0.1),
#           "torque_range": (-0.0, 0.0),
#       },
#   )
#   push_robot = EventTerm(
#       func=mdp.push_by_setting_velocity,
#       mode="interval",
#       interval_range_s=(10.0, 15.0),
#       params={
#           "asset_cfg": SceneEntityCfg("robot"),
#           "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
#       },
#   )


#remember to add the manager at the bottom when uncommenting.


#    """PhysX-only startup mass randomization."""

#    add_base_mass = EventTerm(
#           func=mdp.randomize_rigid_body_mass,
#      mode="startup",
#      params={
#          "asset_cfg": SceneEntityCfg("robot", body_names="Base_Link"),
#          "mass_distribution_params": (0.9, 1.1),
#          "operation": "scale",
#          "distribution": "uniform",
#          "recompute_inertia": True,
#      },
#    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={
          "command_name": "base_velocity",
          "std": 1.0,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={
          "command_name": "base_velocity",
          "std": 0.5,
        },
    )
    foot_air_time = RewTerm(
        func=velocity_mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 0.6,
        },
    )
    turn_gait = RewTerm(
        func=mdp.TurnGaitReward,
        weight=0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "yaw_threshold": 0.05,
            "synced_feet_pair_names": (("FL_Foot_Link", "BR_Foot_Link"), ("FR_Foot_Link", "BL_Foot_Link")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    turn_foot_air_time = RewTerm(
        func=mdp.turn_feet_air_time,
        weight=0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 0.3,
            "yaw_threshold": 0.05,
        },
    )

    #--penalties
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-1.0
    )
    air_time_variance = RewTerm(
        func=mdp.straight_air_time_variance_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "linear_threshold": 0.1,
            "yaw_threshold": 0.05,
        },
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold": 1.0
        }
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.05
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05
    )
    joint_acc = RewTerm(
        func=mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint")},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint")},
    )
    joint_vel = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Joint")},
    )
    #--my params
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Base_Link", ".*_Hip_Link"],
            ),
            "threshold": 1.0,
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-3.0,
        params={"target_height": 0.225}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Base contact, matching the stock velocity tasks.
    base_contact = DoneTerm(
        func=velocity_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Base_Link"),
            "threshold": 1.25,
        },
    )

    joint_pos_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Hip_Joint", ".*_Thigh_Joint", ".*_Foot_Joint"],
            ),
            "bounds": (-1.5708, 1.5708),
        },
    )


##
# Environment configuration
##


@configclass
class AnkybotV2EnvCfg(ManagerBasedRLEnvCfg):
    scene = AnkybotV2SceneCfg(num_envs=1024, env_spacing=1.0)
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
        # viewer settings
        self.viewer.eye = (1,1,1)
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class AnkybotV2EnvCfg_PLAY(AnkybotV2EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.25
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.actions.joint_position.debug_vis = True
        self.episode_length_s = 5
