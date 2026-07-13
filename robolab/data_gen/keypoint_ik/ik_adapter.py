# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters from EEF keypoint trajectories to env-native UR5 actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from robolab.core.motion.eef import EEFActionBridge, EEFPolicyObservation, parse_eef_pose_action
from robolab.data_gen.keypoint_ik.schemas import EnvActionPlan


class IKConversionError(RuntimeError):
    """Raised when an IK bridge cannot produce a usable env action sequence."""


def append_binary_gripper_action(
    arm_actions: np.ndarray,
    raw_gripper: np.ndarray,
    *,
    close_threshold: float = 0.5,
) -> np.ndarray:
    """Append one 0/1 gripper scalar to arm joint actions."""

    arm = np.asarray(arm_actions, dtype=np.float32)
    if arm.ndim != 2:
        raise ValueError(f"Expected arm action shape (T, D), got {arm.shape}")
    gripper = np.asarray(raw_gripper, dtype=np.float32)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    if gripper.ndim != 2 or gripper.shape[1] != 1:
        raise ValueError(f"Expected gripper shape (T, 1), got {gripper.shape}")
    if gripper.shape[0] != arm.shape[0]:
        raise ValueError(f"Arm/gripper horizon mismatch: {arm.shape[0]} vs {gripper.shape[0]}")

    binary_gripper = (gripper > close_threshold).astype(np.float32)
    return np.concatenate([arm, binary_gripper], axis=1)


def convert_eef_actions_to_env_actions(
    *,
    bridge: EEFActionBridge,
    observation: EEFPolicyObservation,
    eef_actions: np.ndarray,
    expected_arm_dim: int = 6,
    allow_ik_fallback: bool = False,
    env_id: int = 0,
) -> EnvActionPlan:
    """Convert raw EEF actions to UR5 `[6 arm, 1 gripper]` env actions."""

    chunk = parse_eef_pose_action(np.asarray(eef_actions, dtype=np.float32))
    result = bridge.convert_chunk(observation, chunk, env_id=env_id)
    arm_actions = np.asarray(result.actions, dtype=np.float32)
    if arm_actions.shape != (chunk.horizon, expected_arm_dim):
        raise IKConversionError(
            f"Expected IK arm actions shape {(chunk.horizon, expected_arm_dim)}, got {arm_actions.shape}"
        )
    diagnostics = [_jsonable_diag(diag) for diag in result.diagnostics]
    failed = [
        diag
        for diag in diagnostics
        if diag.get("success") is False or diag.get("fallback") not in (None, "None")
    ]
    if failed and not allow_ik_fallback:
        first = failed[0]
        raise IKConversionError(
            "IK failed or used fallback; first failure "
            f"step={first.get('step')} pos_err={first.get('position_error')} "
            f"rot_err={first.get('rotation_error')} fallback={first.get('fallback')}"
        )
    raw_gripper = chunk.gripper if result.raw_gripper is None else result.raw_gripper
    env_actions = append_binary_gripper_action(arm_actions, raw_gripper)
    return EnvActionPlan(
        env_actions=env_actions,
        arm_actions=arm_actions,
        raw_gripper=np.asarray(raw_gripper, dtype=np.float32),
        diagnostics=diagnostics,
    )


def _jsonable_diag(diag: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in diag.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        elif isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        else:
            clean[key] = value
    return clean
