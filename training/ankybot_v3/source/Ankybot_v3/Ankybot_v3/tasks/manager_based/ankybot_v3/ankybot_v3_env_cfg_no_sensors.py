# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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
from .mdp import rewards_static

##
# Pre-defined configs
##

from .ankybot_v3 import ANKYBOT_CFG  # isort:skip

# Static-base variant: fixes the root link to the world frame via a non-USD schema
# property (no USD re-export needed) for joint/actuator sanity checks without the
# robot falling over.
ANKYBOT_STATIC_CFG = ANKYBOT_CFG.replace(
    spawn=ANKYBOT_CFG.spawn.replace(
        articulation_props=ANKYBOT_CFG.spawn.articulation_props.replace(fix_root_link=True)
    )
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
    #imu = ImuCfg(
    #    prim_path="{ENV_REGEX_NS}/Robot/FinalRemakeURDF_SLDASM/Base_Link",
    #)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


#@configclass
#class CommandsCfg:
#    """Command specifications for the MDP."""
#
#    base_velocity = mdp.UniformVelocityCommandCfg(
#        asset_name="robot",
#        resampling_time_range=(10.0, 10.0),
#        rel_standing_envs=0.3,
#        ranges=mdp.UniformVelocityCommandCfg.Ranges(
#            lin_vel_x=(-0.5, 0.5),
#            lin_vel_y=(-0.3, 0.3),
#            ang_vel_z=(-0.0, 0.0),
#        ),
#    )


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
        #imu_ang_vel = ObsTerm(
        #    func=mdp.imu_ang_vel,
        #)
        #projected_gravity = ObsTerm(
        #    func=mdp.imu_projected_gravity,
        #)
        #imu_lin_accel = ObsTerm(
        #    func=mdp.imu_lin_acc,
        #)
        #velocity_commands = ObsTerm(
        #    func=mdp.generated_commands,
        #    params={"command_name": "base_velocity"}
        #)
        #joint_pos = ObsTerm(
        #    func=mdp.joint_pos,
        #)
        #joint_vel = ObsTerm(
        #    func=mdp.joint_vel,
        #)
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
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (-0.0, 0.),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
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
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    #foot_physics_material = EventTerm(
    #    func=mdp.randomize_rigid_body_material,
    #    mode="startup",
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
    #        "static_friction_range": (0.8, 1.0),
    #        "dynamic_friction_range": (0.8, 1.0),
    #        "restitution_range": (0.0, 0.0),
    #        "num_buckets": 1,
    #        "make_consistent": True,
    #    },
    #)


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
#@configclass
#class StartupEventCfg:
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

    #track_lin_vel_xy = RewTerm(
    #    func=mdp.track_lin_vel_xy_exp,
    #    weight=2.0,
    #    params={
    #      "command_name": "base_velocity",
    #      "std": 0.25,
    #    },
    #)
    #track_ang_vel_z = RewTerm(
    #    func=mdp.track_ang_vel_z_exp,
    #    weight=2.0,
    #    params={
    #      "command_name": "base_velocity",
    #      "std": 0.25,
    #    },
    #)
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
        func=mdp.GaitReward,
        weight=2.0,
        params={
            "std": 1.0,
            "max_err": 0.4,
            "command_name": "base_velocity",
            "synced_feet_pair_names": (("FL_Foot_Link", "BR_Foot_Link"), ("FR_Foot_Link", "BL_Foot_Link")),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    foot_clearance_reward = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "target_height": 0.04,
            "std": 1.0,
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
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05
    )
    #base_orientation = RewTerm(
    #    func=mdp.base_orientation_penalty,
    #    weight=-3.0,
    #    params={"asset_cfg": SceneEntityCfg("robot")}
    #)
    #foot_slip = RewTerm(
    #    func=mdp.foot_slip_penalty,
    #    weight=-2.5,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
    #        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
    #        "threshold": 1.0
    #    },
    #)
    #lin_vel_z = RewTerm(
    #    func=mdp.lin_vel_z_l2,
    #    weight=-1.0
    #)
    #ang_vel_xy = RewTerm(
    #    func=mdp.ang_vel_xy_l2,
    #    weight=-0.5
    #)
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
    #base_height = RewTerm(
    #    func=mdp.base_height_l2,
    #    weight=-3.0,
    #    params={"target_height": 0.23
    #    },
    #)
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
    #foot_pos_deviation = RewTerm(
    #    func=mdp.joint_pos_soft_limit_penalty,
    #    weight=-0.05,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Foot_Joint"]),
    #        "soft_limit": 0.7,
    #        "hard_limit": 1.1,
    #    },
    #)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    #(2) Base contact — disabled to allow recovery training from collapsed starting poses.
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
class AnkybotV3EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene = AnkybotV3SceneCfg(num_envs=1024, env_spacing=1.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    #commands = CommandsCfg()
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
        #self.scene.imu.update_period = self.sim.dt * self.decimation
        #self.scene.imu.debug_vis = False
        #self.commands.base_velocity.debug_vis = False



@configclass
class AnkybotV3EnvCfg_PLAY(AnkybotV3EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 3
        #self.sim.render_interval = 1


@configclass
class AnkybotV3StaticEnvCfg(AnkybotV3EnvCfg):
    """Static-base variant of the no-sensor env — root link fixed to the world.

    Same USD asset, no re-import; fix is applied purely via
    ``ArticulationRootPropertiesCfg.fix_root_link`` on ``ANKYBOT_STATIC_CFG``.

    Goal here is different from the plain no-sensor env: with the base welded to the
    world there is no balance/locomotion task to speak of, so this overrides
    ``observations`` and ``rewards`` (only — events/terminations/scene are otherwise
    inherited unchanged) to train a continuous, command-free alternating foot-lift
    ("marching in place") gait using diagonal-pair contact-timing rewards from
    :mod:`.mdp.rewards_static`. There is intentionally no velocity/height command and
    no standing-still mode — the reward is unconditionally active every step, so the
    policy should never settle into a static pose.
    """

    @configclass
    class StaticObservationsCfg:
        """Joint feedback + last action only — no IMU, no command (there isn't one)."""

        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=mdp.joint_pos)
            joint_vel = ObsTerm(func=mdp.joint_vel)
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class StaticRewardsCfg:
        """Command-free rewards driving a continuous diagonal foot-lift gait."""

        foot_air_time = RewTerm(
            func=rewards_static.air_time_reward_static,
            weight=3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
                "mode_time": 0.25,
            },
        )
        gait = RewTerm(
            func=rewards_static.GaitRewardStatic,
            weight=1.5,
            params={
                "std": 1.0,
                "max_err": 0.4,
                "synced_feet_pair_names": (("FL_Foot_Link", "BR_Foot_Link"), ("FR_Foot_Link", "BL_Foot_Link")),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
            },
        )
        foot_clearance = RewTerm(
            func=rewards_static.foot_clearance_reward_static,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
                "target_height": 0.04,
                "std": 0.25,
                "tanh_mult": 2.5,
            },
        )
        # -- penalties
        air_time_variance = RewTerm(
            func=mdp.air_time_variance_penalty,
            weight=-0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link")},
        )
        foot_slip = RewTerm(
            func=mdp.foot_slip_penalty,
            weight=-0.3,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
                "threshold": 1.0,
            },
        )
        joint_torques = RewTerm(func=mdp.joint_torques_penalty, weight=-2.5e-5, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_acceleration = RewTerm(
            func=mdp.joint_acceleration_penalty, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
        joint_limits = RewTerm(
            func=mdp.joint_pos_soft_limit_penalty,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Joint"]),
                "soft_limit": 1.3,
                "hard_limit": 1.5708,
            },
        )

    observations: StaticObservationsCfg = StaticObservationsCfg()
    rewards: StaticRewardsCfg = StaticRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANKYBOT_STATIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")