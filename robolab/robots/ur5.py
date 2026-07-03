# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR5E_URDF_PATH = os.path.join(ROBOTS_DIR, "ur5e", "ur5e_mesh.urdf")
UR5E_USD_CACHE_DIR = "/tmp/robolab_ur5e_mesh_v2_usd"

# The mesh URDF keeps the working kinematic chain from ur5e_primitive.urdf while using
# official Universal Robots UR5e DAE meshes for visuals. Collisions remain primitive for
# stable contact behavior in the static-scene smoke test.


@configclass
class UR5eCfg:
    """Cfg class that adds a UR5e arm articulation to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=UR5E_URDF_PATH,
            usd_dir=UR5E_USD_CACHE_DIR,
            fix_base=True,
            root_link_name="base_link",
            merge_fixed_joints=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=800.0, damping=40.0)
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.2,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )


UR5Cfg = UR5eCfg


def _joint_indices(robot, joint_names: Sequence[str]) -> list[int]:
    return [robot.data.joint_names.index(name) for name in joint_names]


def _body_index(robot, body_names: Sequence[str]) -> int:
    for body_name in body_names:
        if body_name in robot.data.body_names:
            return robot.data.body_names.index(body_name)
    raise ValueError(f"None of {list(body_names)} found in robot bodies: {robot.data.body_names}")


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, _joint_indices(robot, ARM_JOINT_NAMES)]


def ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the UR tool frame position in the env-local frame."""
    robot = env.scene[asset_cfg.name]
    body_idx = _body_index(robot, ["tool0", "flange", "wrist_3_link"])
    return robot.data.body_pos_w[:, body_idx, :] - env.scene.env_origins[:, 0:3]


def ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the UR tool frame orientation as quaternion (w, x, y, z)."""
    robot = env.scene[asset_cfg.name]
    body_idx = _body_index(robot, ["tool0", "flange", "wrist_3_link"])
    return robot.data.body_quat_w[:, body_idx, :]


@configclass
class UR5eJointPositionActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


UR5JointPositionActionCfg = UR5eJointPositionActionCfg


@configclass
class ProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/wrist_3_link"}
