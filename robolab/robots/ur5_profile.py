# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight UR5 metadata shared by clients and IsaacLab robot configs."""

from __future__ import annotations

import os

from robolab.constants import ROBOTS_DIR
from robolab.core.motion.eef import FixedOperationalFrame, RobotEEFProfile

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINT_NAMES = ["finger_joint"]
GRIPPER_MIMIC_JOINT_NAMES = (
    "right_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
)
GRIPPER_RESET_JOINT_NAMES = (*GRIPPER_JOINT_NAMES, *GRIPPER_MIMIC_JOINT_NAMES)
GRIPPER_OPEN_POS = 0.0
# Menagerie v4 splits the actuator through two 0.485 tendon coefficients.
GRIPPER_CLOSED_POS = 0.8 / (2.0 * 0.485)


def gripper_reset_joint_positions(close_fraction: float) -> dict[str, float]:
    """Return a mimic-consistent gripper reset from an open-to-closed fraction."""

    if not 0.0 <= close_fraction <= 1.0:
        raise ValueError(f"Expected gripper close fraction in [0, 1], got {close_fraction}")
    target = GRIPPER_OPEN_POS + close_fraction * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
    return {joint_name: target for joint_name in GRIPPER_RESET_JOINT_NAMES}

UR5E_URDF_PATH = os.path.join(ROBOTS_DIR, "ur5e", "ur5e_robotiq_2f_85.urdf")
UR5E_PINOCCHIO_URDF_PATH = os.path.join(ROBOTS_DIR, "ur5e", "ur5e_mesh.urdf")
BERKELEY_TCP_OFFSET_M = 0.170

UR5_TOOL0_EEF_PROFILE = RobotEEFProfile(
    name="ur5e",
    arm_joint_names=tuple(ARM_JOINT_NAMES),
    env_action_dim=6,
    base_link="base_link",
    ee_link="tool0",
    joint_position_key="arm_joint_pos",
    ee_pos_key="ee_pos",
    ee_quat_key="ee_quat",
    gripper_position_key="gripper_pos",
    primary_image_key="over_shoulder_left_camera",
    wrist_image_key="wrist_cam",
    secondary_image_key="over_shoulder_right_camera",
    wrist_missing_policy="zero",
    secondary_missing_policy="zero",
    supports_gripper_action=True,
    gripper_strategy="append_binary",
)

UR5_BERKELEY_EEF_PROFILE = RobotEEFProfile(
    name="ur5e_berkeley_tcp",
    arm_joint_names=tuple(ARM_JOINT_NAMES),
    env_action_dim=6,
    base_link="base_link",
    ee_link="berkeley_tcp",
    fixed_operational_frame=FixedOperationalFrame(
        parent_link="tool0",
        xyz=(0.0, 0.0, BERKELEY_TCP_OFFSET_M),
    ),
    joint_position_key="arm_joint_pos",
    ee_pos_key="ee_pos",
    ee_quat_key="ee_quat",
    gripper_position_key="gripper_pos",
    primary_image_key="over_shoulder_left_camera",
    wrist_image_key="wrist_cam",
    secondary_image_key="over_shoulder_right_camera",
    wrist_missing_policy="zero",
    secondary_missing_policy="zero",
    supports_gripper_action=True,
    gripper_strategy="append_binary",
)

# Keep the historical accessor on manufacturer tool0 semantics. Berkeley/Cosmos
# callers opt into the dataset-calibrated profile explicitly.
UR5_EEF_PROFILE = UR5_TOOL0_EEF_PROFILE


def get_ur5_eef_profile() -> RobotEEFProfile:
    return UR5_EEF_PROFILE


def get_ur5_berkeley_eef_profile() -> RobotEEFProfile:
    """Return the Berkeley UR5 profile whose EEF is the fixed 170 mm TCP."""

    return UR5_BERKELEY_EEF_PROFILE
