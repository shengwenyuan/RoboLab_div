# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import numpy as np
import pytest

from robolab.core.motion.eef import (
    EEFPolicyObservation,
    parse_eef_pose_action,
    quat_wxyz_to_xyzw,
)
from robolab.core.motion.pinocchio import PinocchioIKBridge
from robolab.robots.ur5_profile import UR5E_PINOCCHIO_URDF_PATH, get_ur5_eef_profile


def _make_ur5_bridge() -> PinocchioIKBridge:
    if importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Pinocchio is not installed")
    return PinocchioIKBridge(profile=get_ur5_eef_profile(), urdf_path=UR5E_PINOCCHIO_URDF_PATH)


def _make_observation(bridge: PinocchioIKBridge, joint_position: np.ndarray) -> EEFPolicyObservation:
    ee_pos, ee_quat = bridge.forward_pose(joint_position)
    return EEFPolicyObservation(
        primary_image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
        secondary_image=np.zeros((4, 4, 3), dtype=np.uint8),
        joint_position=joint_position,
        ee_pos=ee_pos,
        ee_quat_wxyz=ee_quat,
        gripper_position=np.zeros(1, dtype=np.float32),
    )


def _pose_action_from_observation(observation: EEFPolicyObservation) -> np.ndarray:
    raw = np.zeros((32, 8), dtype=np.float32)
    raw[:, :3] = observation.ee_pos
    raw[:, 3:7] = quat_wxyz_to_xyzw(observation.ee_quat_wxyz)
    return raw


def test_pinocchio_eef_pose_chunk_holds_current_joint_target():
    bridge = _make_ur5_bridge()
    joint_position = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
    observation = _make_observation(bridge, joint_position)
    raw = _pose_action_from_observation(observation)

    result = bridge.convert_chunk(observation, parse_eef_pose_action(raw), env_id=0)

    assert result.actions.shape == (32, 6)
    np.testing.assert_allclose(result.actions[0], joint_position, atol=1e-6)
    assert result.diagnostics[0]["success"] is True
    assert result.diagnostics[0]["action_kind"] == "eef_pose"


def test_pinocchio_eef_pose_offset_outputs_finite_joint_chunk():
    bridge = _make_ur5_bridge()
    joint_position = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
    observation = _make_observation(bridge, joint_position)
    raw = _pose_action_from_observation(observation)
    raw[:, 2] = 0.02

    result = bridge.convert_chunk(observation, parse_eef_pose_action(raw), env_id=0)

    assert result.actions.shape == (32, 6)
    assert not np.isnan(result.actions).any()
    assert result.diagnostics[0]["success"] is True
