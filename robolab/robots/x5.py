# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass, noise

from robolab.constants import ROBOTS_DIR

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS = 0.044
GRIPPER_CLOSED_POS = 0.0
ROBOT_ROOT = "{ENV_REGEX_NS}/robot/root_joint"

# Midpoint between the two finger joint origins in link6 coordinates.
EEF_OFFSET_POS: tuple[float, float, float] = (0.08657, -0.000002, -0.00024363)
EEF_OFFSET_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
EEF_OFFSET_ROT_CFG = (*EEF_OFFSET_ROT[1:], EEF_OFFSET_ROT[0])

_frame_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TF")
_frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

_WRIST_CAM = TiledCameraCfg(
    prim_path=f"{ROBOT_ROOT}/link6/wrist_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.09, 0.0, 0.04), rot=(-0.5, 0.5, -0.5, 0.5), convention="opengl"
    ),
)


@configclass
class X5Cfg:
    """Cfg class that adds the ARX X5 articulation to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ROBOTS_DIR, "x5/usd/x5_2023.usd"),
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
            rot=(0, 0, 0, 1),
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": GRIPPER_OPEN_POS,
                "joint8": GRIPPER_OPEN_POS,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit=100.0,
                velocity_limit=10.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit=100.0,
                velocity_limit=0.2,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    wrist_cam = _WRIST_CAM

    frames = FrameTransformerCfg(
        prim_path=f"{ROBOT_ROOT}/base_link",
        debug_vis=False,
        visualizer_cfg=_frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{ROBOT_ROOT}/base_link",
                name="base_link",
            )
        ] + [
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{ROBOT_ROOT}/link{i}",
                name=f"link{i}",
            )
            for i in range(1, 9)
        ] + [
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{ROBOT_ROOT}/link6",
                name="eef_frame",
                offset=OffsetCfg(pos=EEF_OFFSET_POS, rot=EEF_OFFSET_ROT_CFG),
            ),
        ],
    )


@configclass
class WristCameraCfg:
    """Introspection wrapper so the wrist camera can be passed to generate_image_obs_from_cameras.
    The scene's wrist_cam is still sourced from X5Cfg; this wrapper only exposes the name.
    """

    wrist_cam = _WRIST_CAM


########################################################
# Contact gripper
########################################################

contact_gripper = {"gripper": f"{ROBOT_ROOT}/link7"}

########################################################
# Definitions
########################################################


def _joint_indices(robot, joint_names: Sequence[str]) -> list[int]:
    return [robot.data.joint_names.index(name) for name in joint_names]


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, _joint_indices(robot, ARM_JOINT_NAMES)]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns gripper position as 0 for open and 1 for closed."""
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, _joint_indices(robot, GRIPPER_JOINT_NAMES)]
    closed_pos = 1.0 - joint_pos.mean(dim=1, keepdim=True) / GRIPPER_OPEN_POS
    return torch.clamp(closed_pos, 0.0, 1.0)


def ee_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns the end effector position (x, y, z) in the env-local frame."""
    robot = env.scene[asset_cfg.name]
    ee_body_name = "link6"
    body_idx = robot.data.body_names.index(ee_body_name)
    return robot.data.body_pos_w[:, body_idx, :] - env.scene.env_origins[:, 0:3]


def ee_quat(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns the end effector orientation as quaternion (x, y, z, w) in the world frame."""
    robot = env.scene[asset_cfg.name]
    ee_body_name = "link6"
    body_idx = robot.data.body_names.index(ee_body_name)
    return robot.data.body_quat_w[:, body_idx, :]


def eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """Returns the eef_frame position (x, y, z) in the env-local frame."""
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return frames.data.target_pos_w[:, idx, :] - env.scene.env_origins[:, 0:3]


def eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """Returns the eef_frame orientation as quaternion (x, y, z, w) in the world frame."""
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return frames.data.target_quat_w[:, idx, :]


########################################################
# Actions
########################################################


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction


@configclass
class X5JointPositionActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={joint_name: GRIPPER_OPEN_POS for joint_name in GRIPPER_JOINT_NAMES},
        close_command_expr={joint_name: GRIPPER_CLOSED_POS for joint_name in GRIPPER_JOINT_NAMES},
    )


@configclass
class X5IKActionCfg:
    """Absolute end-effector pose control via differential IK."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link6",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=EEF_OFFSET_POS),
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={joint_name: GRIPPER_OPEN_POS for joint_name in GRIPPER_JOINT_NAMES},
        close_command_expr={joint_name: GRIPPER_CLOSED_POS for joint_name in GRIPPER_JOINT_NAMES},
    )


@configclass
class X5RelIKActionCfg:
    """Relative end-effector pose control via differential IK."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link6",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=EEF_OFFSET_POS),
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={joint_name: GRIPPER_OPEN_POS for joint_name in GRIPPER_JOINT_NAMES},
        close_command_expr={joint_name: GRIPPER_CLOSED_POS for joint_name in GRIPPER_JOINT_NAMES},
    )


########################################################
# Observations
########################################################


@configclass
class ProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(
        func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    )
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)
    eef_pos = ObsTerm(func=eef_pos)
    eef_quat = ObsTerm(func=eef_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False # must include
        self.concatenate_terms = False # must include
