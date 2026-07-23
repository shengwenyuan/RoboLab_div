# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from robolab.core.motion.eef import (
    matrix_to_quat_wxyz,
    parse_eef_pose_action,
    quat_to_matrix_wxyz,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


def test_quaternion_wxyz_xyzw_roundtrip():
    quat = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)

    xyzw = quat_wxyz_to_xyzw(quat)
    roundtrip = quat_xyzw_to_wxyz(xyzw)

    np.testing.assert_allclose(roundtrip, quat, atol=1e-6)


def test_matrix_quaternion_roundtrip():
    matrix = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    quat = matrix_to_quat_wxyz(matrix)

    np.testing.assert_allclose(quat_to_matrix_wxyz(quat), matrix, atol=1e-6)


def test_parse_eef_pose_action_converts_xyzw_quaternion_to_wxyz():
    action = np.zeros((32, 8), dtype=np.float32)
    action[:, :3] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    action[:, 3:7] = np.array([0.0, 0.0, 0.70710677, 0.70710677], dtype=np.float32)
    action[:, 7] = 1.0

    chunk = parse_eef_pose_action(action)

    assert chunk.horizon == 32
    assert chunk.raw.shape == (32, 8)
    np.testing.assert_allclose(chunk.position[0], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(chunk.quat_wxyz[0], [0.7071068, 0.0, 0.0, 0.7071068], atol=1e-6)
    np.testing.assert_allclose(chunk.gripper, np.ones((32, 1), dtype=np.float32))


@pytest.mark.parametrize("width", [7, 9, 10])
def test_parse_eef_pose_action_requires_single_arm_8d_contract(width: int):
    with pytest.raises(ValueError, match="width 8"):
        parse_eef_pose_action(np.zeros((32, width), dtype=np.float32))
