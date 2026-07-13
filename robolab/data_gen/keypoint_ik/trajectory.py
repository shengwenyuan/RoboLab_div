# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python keypoint resolution and EEF trajectory interpolation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import acos, ceil, sin

import numpy as np

from robolab.core.motion.eef import normalize_quat_wxyz, quat_wxyz_to_xyzw
from robolab.data_gen.keypoint_ik.schemas import KeypointSpec, ResolvedKeypoint, ResolvedTrajectory


DEFAULT_PICK_PLACE_KEYPOINTS = [
    KeypointSpec(
        name="pre_grasp",
        anchor="target_object",
        offset_xyz=(0.0, 0.0, 0.18),
        hold_steps=8,
        gripper="open",
    ),
    KeypointSpec(
        name="grasp",
        anchor="target_object",
        offset_xyz=(0.0, 0.0, 0.045),
        hold_steps=8,
        gripper="open",
    ),
    KeypointSpec(
        name="close",
        anchor="target_object",
        offset_xyz=(0.0, 0.0, 0.045),
        hold_steps=12,
        gripper="close",
    ),
    KeypointSpec(
        name="lift",
        anchor="target_object",
        offset_xyz=(0.0, 0.0, 0.24),
        hold_steps=8,
        gripper="close",
    ),
    KeypointSpec(
        name="pre_release",
        anchor="container_object",
        offset_xyz=(0.0, 0.0, 0.24),
        hold_steps=8,
        gripper="close",
    ),
    KeypointSpec(
        name="release",
        anchor="container_object",
        offset_xyz=(0.0, 0.0, 0.18),
        hold_steps=12,
        gripper="open",
    ),
]


def resolve_gripper_command(command: str | float, *, open_value: float = 0.0, close_value: float = 1.0) -> float:
    if isinstance(command, str):
        normalized = command.strip().lower()
        if normalized == "open":
            return float(open_value)
        if normalized in {"close", "closed"}:
            return float(close_value)
        raise ValueError(f"Unsupported gripper command {command!r}; expected open/close or a float")
    return float(command)


def resolve_keypoints(
    keypoints: Sequence[KeypointSpec],
    *,
    anchors_xyz: Mapping[str, Sequence[float]],
    quat_wxyz: Sequence[float],
    open_gripper: float = 0.0,
    close_gripper: float = 1.0,
) -> list[ResolvedKeypoint]:
    """Resolve symbolic object anchors into concrete EEF waypoints."""

    quat = normalize_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))
    resolved: list[ResolvedKeypoint] = []
    for keypoint in keypoints:
        if keypoint.anchor not in anchors_xyz:
            available = ", ".join(sorted(anchors_xyz))
            raise KeyError(f"Unknown keypoint anchor {keypoint.anchor!r}; available anchors: {available}")
        anchor = np.asarray(anchors_xyz[keypoint.anchor], dtype=np.float32).reshape(3)
        offset = np.asarray(keypoint.offset_xyz, dtype=np.float32).reshape(3)
        resolved.append(
            ResolvedKeypoint(
                name=keypoint.name,
                anchor=keypoint.anchor,
                position=anchor + offset,
                quat_wxyz=quat.copy(),
                gripper=resolve_gripper_command(
                    keypoint.gripper,
                    open_value=open_gripper,
                    close_value=close_gripper,
                ),
                hold_steps=keypoint.hold_steps,
            )
        )
    return resolved


def build_eef_action_trajectory(
    *,
    start_pos: Sequence[float],
    start_quat_wxyz: Sequence[float],
    start_gripper: float,
    keypoints: Sequence[ResolvedKeypoint],
    max_cartesian_step_m: float = 0.015,
    start_hold_steps: int = 4,
) -> ResolvedTrajectory:
    """Build a raw `[xyz, quat_xyzw, gripper]` EEF action array."""

    if max_cartesian_step_m <= 0.0:
        raise ValueError("max_cartesian_step_m must be positive")

    cur_pos = np.asarray(start_pos, dtype=np.float32).reshape(3)
    cur_quat = normalize_quat_wxyz(np.asarray(start_quat_wxyz, dtype=np.float32).reshape(4))
    cur_gripper = float(start_gripper)

    rows: list[np.ndarray] = []
    for _ in range(max(0, int(start_hold_steps))):
        rows.append(_raw_row(cur_pos, cur_quat, cur_gripper))

    for keypoint in keypoints:
        target_pos = np.asarray(keypoint.position, dtype=np.float32).reshape(3)
        target_quat = normalize_quat_wxyz(np.asarray(keypoint.quat_wxyz, dtype=np.float32).reshape(4))
        distance = float(np.linalg.norm(target_pos - cur_pos))
        move_steps = max(1, int(ceil(distance / max_cartesian_step_m)))
        for index in range(1, move_steps + 1):
            alpha = index / move_steps
            pos = (1.0 - alpha) * cur_pos + alpha * target_pos
            quat = slerp_quat_wxyz(cur_quat, target_quat, alpha)
            rows.append(_raw_row(pos, quat, keypoint.gripper))
        for _ in range(max(0, int(keypoint.hold_steps))):
            rows.append(_raw_row(target_pos, target_quat, keypoint.gripper))
        cur_pos = target_pos
        cur_quat = target_quat
        cur_gripper = float(keypoint.gripper)

    if not rows:
        raise ValueError("Trajectory is empty; add at least one keypoint or hold step")
    return ResolvedTrajectory(eef_actions=np.stack(rows).astype(np.float32), keypoints=list(keypoints))


def slerp_quat_wxyz(q0: Sequence[float], q1: Sequence[float], alpha: float) -> np.ndarray:
    """Shortest-path quaternion interpolation in wxyz order."""

    qa = normalize_quat_wxyz(np.asarray(q0, dtype=np.float32).reshape(4))
    qb = normalize_quat_wxyz(np.asarray(q1, dtype=np.float32).reshape(4))
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quat_wxyz((1.0 - alpha) * qa + alpha * qb)
    theta = acos(dot)
    sin_theta = sin(theta)
    w0 = sin((1.0 - alpha) * theta) / sin_theta
    w1 = sin(alpha * theta) / sin_theta
    return normalize_quat_wxyz(w0 * qa + w1 * qb)


def _raw_row(pos: np.ndarray, quat_wxyz: np.ndarray, gripper: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float32).reshape(3),
            quat_wxyz_to_xyzw(np.asarray(quat_wxyz, dtype=np.float32).reshape(4)),
            np.asarray([gripper], dtype=np.float32),
        ]
    )
