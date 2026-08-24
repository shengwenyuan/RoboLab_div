# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.robots.ur5_profile import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES, UR5E_URDF_PATH
from robolab.robots.ur5_spawn import spawn_ur5e_robotiq_2f85

UR5E_USD_CACHE_DIR = "/tmp/robolab_ur5e_robotiq_2f_85_menagerie_v4_v5_usd"
GRIPPER_MIMIC_JOINT_REGEX = (
    "^(right_outer_knuckle_joint|left_inner_knuckle_joint|right_inner_knuckle_joint|"
    "left_inner_finger_joint|right_inner_finger_joint)$"
)
GRIPPER_OPEN_POS = 0.0
# Menagerie v4 splits the actuator through two 0.485 tendon coefficients.
# A full 0.8 control target therefore settles at this driver angle, with
# roughly 0.095 mm of nominal pad preload.
GRIPPER_CLOSED_POS = 0.8 / (2.0 * 0.485)
GRIPPER_OPEN_COMMAND = {"finger_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSE_COMMAND = {"finger_joint": GRIPPER_CLOSED_POS}
UR5E_HOME_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57079632679,
    "elbow_joint": 1.57079632679,
    "wrist_1_joint": -1.57079632679,
    "wrist_2_joint": -1.57079632679,
    "wrist_3_joint": 1.57079632679,
    **GRIPPER_OPEN_COMMAND,
}

# The mesh URDF keeps the working UR5e kinematic chain while using Universal Robots
# UR5e visual meshes and an actuated Robotiq 2F-85 gripper.
# Wrist camera in Berkeley-autolab dataset style
_WRIST_CAM = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/tool0/wrist_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=5.3,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.19, 0.0, -0.005),
        rot=(0.6532815, 0.6532815, -0.270598, -0.270598),  # x, y, z, w
        convention="opengl",
    ),
)


@configclass
class UR5eCfg:
    """Cfg class that adds a UR5e arm articulation to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UrdfFileCfg(
            func=spawn_ur5e_robotiq_2f85,
            asset_path=UR5E_URDF_PATH,
            usd_dir=UR5E_USD_CACHE_DIR,
            fix_base=True,
            root_link_name="base_link",
            merge_fixed_joints=False,
            # IsaacLab 2.2 passes this flag to set_parse_mimic despite the
            # misleading field name.  True creates one driven master plus five
            # PhysxMimicJointAPI followers instead of six competing drives.
            convert_mimic_joints_to_normal_joints=True,
            # Isaac Lab 3 removed collider_type; its URDF importer retains the
            # convex-hull default. The gripper URDF supplies simple pad boxes.
            self_collision=True,
            # The importer otherwise creates very soft mimic constraints
            # (25 Hz, damping ratio 0.005), allowing external contact to spread
            # follower joints by more than 0.1 rad. Zero compliance parameters
            # produce a hard PhysX mimic constraint; runtime actuator gains are
            # still supplied below for the arm and master finger joint.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
                    natural_frequency={GRIPPER_MIMIC_JOINT_REGEX: 0.0},
                    damping_ratio={GRIPPER_MIMIC_JOINT_REGEX: 0.0},
                )
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # Keep UR5e self-collision enabled. The custom spawner masks
                # only eight documented gripper-internal/mounting pairs.
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(0, 0, 0, 1),
            joint_pos=UR5E_HOME_JOINT_POS.copy(),
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
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_JOINT_NAMES,
                effort_limit_sim=5.0,
                velocity_limit_sim=2.0,
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )


class UR5eWithWristCameraCfg:
    """UR5e robot cfg variant with a robot-mounted wrist camera sensor."""

    robot = UR5eCfg().robot
    wrist_cam = _WRIST_CAM


UR5Cfg = UR5eCfg


@configclass
class WristCameraCfg:
    """Introspection wrapper for the UR5 wrist camera observation term."""

    wrist_cam = _WRIST_CAM


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


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns gripper position as 0 for open and 1 for closed."""
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, _joint_indices(robot, GRIPPER_JOINT_NAMES)]
    closed_pos = (joint_pos - GRIPPER_OPEN_POS) / (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
    return torch.clamp(closed_pos.mean(dim=1, keepdim=True), 0.0, 1.0)


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


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        close_mask = actions if actions.dtype == torch.bool else actions > 0.5
        self._processed_actions = torch.where(close_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    class_type = BinaryJointPositionZeroToOneAction


@configclass
class UR5eJointPositionActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        # The environment keeps one gripper scalar. PhysX propagates the
        # master command to the five follower joints through mimic schemas.
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND,
        close_command_expr=GRIPPER_CLOSE_COMMAND,
    )


UR5JointPositionActionCfg = UR5eJointPositionActionCfg


@configclass
class ProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(func=gripper_pos, clip=(0, 1))
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


# Filtered ContactSensor force matrices require exactly one source body per
# environment. Preserve the historical left-pad contract used by task code.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/left_inner_finger_pad"}
