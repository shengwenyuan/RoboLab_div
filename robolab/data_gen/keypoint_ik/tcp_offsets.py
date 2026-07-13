# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URDF-derived TCP geometry helpers for keypoint-IK waypoints."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from robolab.core.motion.eef import quat_to_matrix_wxyz

# Robotiq 2F-85 open-pose pad center in the UR5 tool0 frame.
#
# URDF chain at finger_joint = 0:
# tool0 -> robotiq base:             z = 0
# base -> outer knuckle:             z = 0.054904
# outer knuckle -> outer finger:     z = -0.0041
# outer finger -> inner finger:      z = 0.0471
# inner finger -> inner finger pad:  z = 0.03242
ROBOTIQ_2F85_OPEN_PAD_CENTER_FROM_TOOL0_XYZ_M = np.asarray(
    (0.0, 0.0, 0.054904 - 0.0041 + 0.0471 + 0.03242),
    dtype=np.float32,
)

ROBOTIQ_2F85_OPEN_PAD_HALF_SIZE_XYZ_M = np.asarray((0.022 * 0.5, 0.00635 * 0.5, 0.0375 * 0.5), dtype=np.float32)
ROBOTIQ_2F85_OPEN_PAD_TILT_RAD = 0.20


def robotiq_2f85_open_pad_rotation_from_tool0() -> np.ndarray:
    """Return the open-pose left pad rotation matrix in the tool0 frame."""

    return robotiq_2f85_open_pad_rotations_from_tool0()[0]


def robotiq_2f85_open_pad_rotations_from_tool0() -> tuple[np.ndarray, np.ndarray]:
    """Return open-pose left/right pad rotation matrices in the tool0 frame."""

    c = math.cos(ROBOTIQ_2F85_OPEN_PAD_TILT_RAD)
    s = math.sin(ROBOTIQ_2F85_OPEN_PAD_TILT_RAD)
    left = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -c, s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )
    right = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )
    return left, right


def tool0_offset_world_z(tool0_quat_wxyz: Sequence[float], tool0_offset_xyz: Sequence[float]) -> float:
    """Project a tool0-frame offset onto world/env-local z."""

    rotation = quat_to_matrix_wxyz(np.asarray(tool0_quat_wxyz, dtype=np.float32).reshape(4)).reshape(3, 3)
    offset = np.asarray(tool0_offset_xyz, dtype=np.float32).reshape(3)
    return float((rotation @ offset)[2])


def robotiq_2f85_open_pad_half_extent_world_z(tool0_quat_wxyz: Sequence[float]) -> float:
    """Return the open pad collision half-extent along world/env-local z."""

    world_from_tool0 = quat_to_matrix_wxyz(np.asarray(tool0_quat_wxyz, dtype=np.float32).reshape(4)).reshape(3, 3)
    extents = []
    for tool0_from_pad in robotiq_2f85_open_pad_rotations_from_tool0():
        world_from_pad = world_from_tool0 @ tool0_from_pad
        extents.append(float(np.abs(world_from_pad[2, :]) @ ROBOTIQ_2F85_OPEN_PAD_HALF_SIZE_XYZ_M))
    return max(extents)


def table_tool0_z_from_tcp_center_z(
    tcp_center_z: float,
    *,
    tool0_quat_wxyz: Sequence[float],
    tool0_to_tcp_xyz: Sequence[float],
) -> float:
    """Convert a desired TCP/pad-center z into the corresponding tool0 z."""

    return float(tcp_center_z) - tool0_offset_world_z(tool0_quat_wxyz, tool0_to_tcp_xyz)
