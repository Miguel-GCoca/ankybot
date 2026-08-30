# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom Ankybot quadruped robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Assets.
##

ANKYBOT_USD_PATH = (
    "/workspace/custom_assets/Ankybot_v2_description/FinalURDF.SLDASM/usd_v3/World0.usd"
)
"""Path to the Ankybot USD asset."""

##
# Configuration - Actuators.
##

ANKYBOT_LEG_ACTUATOR_CFG = IdealPDActuatorCfg(
    joint_names_expr=[".*_Hip_Joint", ".*_Thigh_Joint", ".*_Foot_Joint"],
    effort_limit=3.92266,
    velocity_limit=10.4719755,
    effort_limit_sim=3.92266,
    velocity_limit_sim=10.4719755,
    stiffness={
        ".*_Hip_Joint": 20.0,
        ".*_Thigh_Joint": 20.0,
        ".*_Foot_Joint": 18.0,
    },
    damping={
        ".*_Hip_Joint": 0.8,
        ".*_Thigh_Joint": 0.8,
        ".*_Foot_Joint": 0.8,
    },
)
"""Conservative position-servo PD configuration for the Ankybot leg joints."""

##
# Configuration - Articulation.
##

ANKYBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ANKYBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            ".*_Hip_Joint": 0.0,
            ".*_Thigh_Joint": 0.5236,
            ".*_Foot_Joint": 0.3840,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": ANKYBOT_LEG_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of the custom Ankybot quadruped robot."""
