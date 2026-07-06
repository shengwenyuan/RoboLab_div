# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common end-effector action types and pose utilities.

This module is intentionally backend-neutral: it knows nothing about Cosmos,
cuRobo, IsaacLab action managers, or any concrete robot. Policy clients and
motion backends meet here through typed observations, action chunks, and bridge
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

MissingViewPolicy = Literal["mirror", "zero"]


@dataclass(frozen=True)
class RobotEEFProfile:
    """Robot-specific wiring needed by generic EEF policy clients."""

    name: str
    arm_joint_names: tuple[str, ...]
    env_action_dim: int
    base_link: str
    ee_link: str
    joint_position_key: str = "arm_joint_pos"
    ee_pos_key: str = "ee_pos"
    ee_quat_key: str = "ee_quat"
    gripper_position_key: str | None = None
    primary_image_key: str = "over_shoulder_left_camera"
    wrist_image_key: str | None = "wrist_cam"
    secondary_image_key: str | None = "over_shoulder_right_camera"
    wrist_missing_policy: MissingViewPolicy = "mirror"
    secondary_missing_policy: MissingViewPolicy = "zero"
    supports_gripper_action: bool = False
    gripper_strategy: str = "diagnostics_only"


@dataclass(frozen=True)
class EEFPolicyObservation:
    """Flat observation consumed by EEF action bridges."""

    primary_image: np.ndarray
    wrist_image: np.ndarray
    secondary_image: np.ndarray
    joint_position: np.ndarray
    ee_pos: np.ndarray
    ee_quat_wxyz: np.ndarray
    gripper_position: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EEFActionChunk:
    """Single-arm absolute EEF pose chunk in the 8D client contract.

    Raw policy-server responses use ``[position(3), quat_xyzw(4), gripper(1)]``.
    RoboLab motion backends use Isaac-style ``wxyz`` quaternions internally, so
    ``quat_wxyz`` is converted before constructing this chunk.
    """

    position: np.ndarray
    quat_wxyz: np.ndarray
    gripper: np.ndarray
    raw: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.raw.shape[0])


@dataclass(frozen=True)
class MotionBridgeResult:
    """Converted environment action chunk and per-step diagnostics."""

    actions: np.ndarray
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    raw_gripper: np.ndarray | None = None


class EEFActionBridge(Protocol):
    """Converts absolute EEF pose chunks into concrete environment actions."""

    def reset(self, *, env_id: int | None = None) -> None:
        """Clear backend state for one env or all envs."""

    def convert_chunk(
        self,
        observation: EEFPolicyObservation,
        action_chunk: EEFActionChunk,
        *,
        env_id: int = 0,
    ) -> MotionBridgeResult:
        """Convert an EEF pose action chunk into env-native actions."""


def _ensure_action_array(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected EEF pose action shape (H, 8), got {arr.shape}")
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected EEF pose action width 8, got {arr.shape}")
    return arr


def parse_eef_pose_action(action: np.ndarray) -> EEFActionChunk:
    """Validate and slice a single-arm absolute EEF pose action response."""

    raw = np.nan_to_num(_ensure_action_array(action).astype(np.float32, copy=True))
    quat_wxyz = normalize_quat_wxyz(quat_xyzw_to_wxyz(raw[:, 3:7]))
    return EEFActionChunk(
        position=raw[:, :3],
        quat_wxyz=quat_wxyz,
        gripper=raw[:, 7:8],
        raw=raw,
    )


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def normalize_quat_wxyz(quat_wxyz: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("Cannot normalize a zero-length quaternion")
    q = q / norm
    sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0).astype(q.dtype)
    return q * sign


def quat_to_matrix_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` matrices."""

    q = normalize_quat_wxyz(quat_wxyz)
    w, x, y, z = np.moveaxis(q, -1, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = ww + xx - yy - zz
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = ww - xx + yy - zz
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = ww - xx - yy + zz
    return matrix


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3, 3)`` rotation matrices to normalized wxyz quats."""

    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (..., 3, 3), got {mat.shape}")

    flat = mat.reshape(-1, 3, 3)
    quats = np.empty((flat.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(flat):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quats[i, 0] = 0.25 * s
            quats[i, 1] = (m[2, 1] - m[1, 2]) / s
            quats[i, 2] = (m[0, 2] - m[2, 0]) / s
            quats[i, 3] = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quats[i, 0] = (m[2, 1] - m[1, 2]) / s
            quats[i, 1] = 0.25 * s
            quats[i, 2] = (m[0, 1] + m[1, 0]) / s
            quats[i, 3] = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quats[i, 0] = (m[0, 2] - m[2, 0]) / s
            quats[i, 1] = (m[0, 1] + m[1, 0]) / s
            quats[i, 2] = 0.25 * s
            quats[i, 3] = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quats[i, 0] = (m[1, 0] - m[0, 1]) / s
            quats[i, 1] = (m[0, 2] + m[2, 0]) / s
            quats[i, 2] = (m[1, 2] + m[2, 1]) / s
            quats[i, 3] = 0.25 * s

    return normalize_quat_wxyz(quats.reshape((*mat.shape[:-2], 4)))
