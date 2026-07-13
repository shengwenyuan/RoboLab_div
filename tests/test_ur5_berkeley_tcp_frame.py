# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused regressions for the Berkeley UR5 fixed TCP contract."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from robolab.core.motion.eef import resolve_operational_frame_pose
from robolab.core.motion.pinocchio import PinocchioIKBridge
from robolab.robots.ur5_profile import (
    BERKELEY_TCP_OFFSET_M,
    UR5E_PINOCCHIO_URDF_PATH,
    get_ur5_berkeley_eef_profile,
    get_ur5_eef_profile,
)


def _normalized_canonical_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.where(quat[..., :1] < 0.0, -quat, quat)


def _rotate_vector_wxyz(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors without relying on the production matrix conversion."""

    quat = _normalized_canonical_quat(quat_wxyz)
    quat_vector = quat[..., 1:]
    vectors = np.broadcast_to(np.asarray(vector, dtype=np.float64), quat_vector.shape)
    cross = 2.0 * np.cross(quat_vector, vectors)
    return vectors + quat[..., :1] * cross + np.cross(quat_vector, cross)


def test_legacy_ur5_profile_remains_on_manufacturer_tool0() -> None:
    profile = get_ur5_eef_profile()

    assert profile.ee_link == "tool0"
    assert profile.fixed_operational_frame is None


def test_berkeley_profile_is_fixed_tool0_plus_170_mm() -> None:
    profile = get_ur5_berkeley_eef_profile()
    fixed_frame = profile.fixed_operational_frame

    assert profile is not get_ur5_eef_profile()
    assert profile.ee_link == "berkeley_tcp"
    assert fixed_frame is not None
    assert fixed_frame.parent_link == "tool0"
    assert BERKELEY_TCP_OFFSET_M == pytest.approx(0.170, abs=0.0)
    assert fixed_frame.xyz == pytest.approx((0.0, 0.0, 0.170), abs=0.0)
    assert fixed_frame.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=0.0)


def test_operational_frame_pose_resolves_offset_for_arbitrary_orientations() -> None:
    fixed_frame = get_ur5_berkeley_eef_profile().fixed_operational_frame
    assert fixed_frame is not None

    rng = np.random.default_rng(20260713)
    parent_positions = rng.uniform(-1.0, 1.0, size=(64, 3))
    parent_quats = rng.normal(size=(64, 4))

    resolved_positions, resolved_quats = resolve_operational_frame_pose(
        parent_positions,
        parent_quats,
        fixed_frame,
    )

    expected_quats = _normalized_canonical_quat(parent_quats)
    expected_positions = parent_positions + _rotate_vector_wxyz(parent_quats, fixed_frame.xyz)
    np.testing.assert_allclose(resolved_positions, expected_positions, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(resolved_quats, expected_quats, atol=2e-6, rtol=0.0)


def test_pinocchio_keeps_six_dof_and_resolves_berkeley_tcp_fk() -> None:
    if importlib.util.find_spec("pinocchio") is None:
        pytest.skip("Pinocchio is not installed")

    tool0_bridge = PinocchioIKBridge(
        profile=get_ur5_eef_profile(),
        urdf_path=UR5E_PINOCCHIO_URDF_PATH,
    )
    tcp_bridge = PinocchioIKBridge(
        profile=get_ur5_berkeley_eef_profile(),
        urdf_path=UR5E_PINOCCHIO_URDF_PATH,
    )

    assert tool0_bridge._model.nq == tool0_bridge._model.nv == 6
    assert tcp_bridge._model.nq == tcp_bridge._model.nv == 6
    assert tcp_bridge._model.existFrame("tool0")
    assert tcp_bridge._model.existFrame("berkeley_tcp")

    joint_positions = np.asarray(
        [
            [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
            [0.35, -1.10, 1.25, -1.90, -1.20, 0.40],
            [-0.80, -1.70, 1.10, -0.75, 1.20, -0.55],
            [1.15, -0.90, 0.65, -2.10, -0.35, 1.30],
        ],
        dtype=np.float64,
    )
    expected_offset = np.asarray([0.0, 0.0, BERKELEY_TCP_OFFSET_M])

    for joint_position in joint_positions:
        tool0_position, tool0_quat = tool0_bridge.forward_pose(joint_position)
        tcp_position, tcp_quat = tcp_bridge.forward_pose(joint_position)
        expected_tcp_position = tool0_position + _rotate_vector_wxyz(tool0_quat, expected_offset)

        np.testing.assert_allclose(tcp_position, expected_tcp_position, atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(tcp_quat, tool0_quat, atol=2e-6, rtol=0.0)

        pin = tcp_bridge._pin
        pin.forwardKinematics(tcp_bridge._model, tcp_bridge._data, joint_position)
        pin.updateFramePlacements(tcp_bridge._model, tcp_bridge._data)
        tool0_placement = tcp_bridge._data.oMf[tcp_bridge._model.getFrameId("tool0")]
        tcp_placement = tcp_bridge._data.oMf[tcp_bridge._model.getFrameId("berkeley_tcp")]
        relative_placement = tool0_placement.inverse() * tcp_placement
        np.testing.assert_allclose(relative_placement.translation, expected_offset, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(relative_placement.rotation, np.eye(3), atol=1e-9, rtol=0.0)


def test_cosmos_ur5_observation_converts_tool0_pose_to_berkeley_tcp() -> None:
    import torch

    from policies.cosmos3.client import Cosmos3UR5Client

    client = object.__new__(Cosmos3UR5Client)
    client._profile = get_ur5_berkeley_eef_profile()
    client._image_h = 2
    client._image_w = 2
    client.server_action_format = "eef_pose"
    half_sqrt_2 = np.float32(np.sqrt(0.5))
    raw_observation = {
        "image_obs": {
            "over_shoulder_left_camera": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
            "ee_pos": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            # tool0 rotated +90 degrees around Y: local +Z points along world +X.
            "ee_quat": torch.tensor([[half_sqrt_2, 0.0, half_sqrt_2, 0.0]], dtype=torch.float32),
        },
    }

    extracted = client._extract_observation(raw_observation)

    np.testing.assert_allclose(extracted["eef_pos"], [0.27, 0.2, 0.3], atol=2e-6)
    np.testing.assert_allclose(
        extracted["eef_observation"].ee_quat_wxyz,
        [half_sqrt_2, 0.0, half_sqrt_2, 0.0],
        atol=2e-6,
    )
    assert extracted["eef_observation"].metadata["eef_frame"] == "berkeley_tcp"
    np.testing.assert_allclose(extracted["eef_pose_xyzw"][0:3], [0.27, 0.2, 0.3], atol=2e-6)
