# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab-facing helpers for keypoint-IK execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robolab.core.motion.eef import EEFPolicyObservation
from robolab.data_gen.keypoint_ik.ik_adapter import convert_eef_actions_to_env_actions


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def build_eef_observation_from_env_obs(obs: Mapping[str, Any], profile: Any, *, env_id: int = 0) -> EEFPolicyObservation:
    """Extract the observation needed by the IK bridge from a RoboLab observation dict."""

    image_obs = obs.get("image_obs", {})
    proprio_obs = obs.get("proprio_obs", {})
    joint_position = _required_obs(proprio_obs, profile.joint_position_key, env_id)
    ee_pos = _required_obs(proprio_obs, profile.ee_pos_key, env_id)
    ee_quat_wxyz = _required_obs(proprio_obs, profile.ee_quat_key, env_id)
    gripper_position = None
    if getattr(profile, "gripper_position_key", None):
        gripper_position = _optional_obs(proprio_obs, profile.gripper_position_key, env_id)

    reference_image = _first_image(image_obs, env_id=env_id)
    return EEFPolicyObservation(
        primary_image=_optional_image(image_obs, profile.primary_image_key, reference_image, env_id=env_id),
        wrist_image=_optional_image(image_obs, profile.wrist_image_key, reference_image, env_id=env_id),
        secondary_image=_optional_image(image_obs, profile.secondary_image_key, reference_image, env_id=env_id),
        joint_position=joint_position.astype(np.float32, copy=False),
        ee_pos=ee_pos.astype(np.float32, copy=False),
        ee_quat_wxyz=ee_quat_wxyz.astype(np.float32, copy=False),
        gripper_position=None if gripper_position is None else gripper_position.astype(np.float32, copy=False),
    )


def create_ur5_ik_bridge(*, backend: str, device: str = "cuda:0"):
    """Create a UR5 IK bridge, trying cuRobo before Pinocchio when backend is auto."""

    from robolab.robots.ur5_profile import UR5E_PINOCCHIO_URDF_PATH, get_ur5_eef_profile

    profile = get_ur5_eef_profile()
    requested = backend.lower()
    errors: list[str] = []
    if requested in {"auto", "curobo"}:
        try:
            from robolab.core.motion.curobo import CuRoboIKBridge

            return CuRoboIKBridge(profile=profile, device=device), "curobo"
        except Exception as exc:
            errors.append(f"curobo: {exc}")
            if requested == "curobo":
                raise
    if requested in {"auto", "pinocchio"}:
        try:
            from robolab.core.motion.pinocchio import PinocchioIKBridge

            return PinocchioIKBridge(profile=profile, urdf_path=UR5E_PINOCCHIO_URDF_PATH), "pinocchio"
        except Exception as exc:
            errors.append(f"pinocchio: {exc}")
            if requested == "pinocchio":
                raise
    raise RuntimeError("No usable IK backend. " + " | ".join(errors))


def convert_trajectory_with_bridge(
    *,
    bridge: Any,
    observation: EEFPolicyObservation,
    eef_actions: np.ndarray,
    allow_ik_fallback: bool = False,
):
    return convert_eef_actions_to_env_actions(
        bridge=bridge,
        observation=observation,
        eef_actions=eef_actions,
        expected_arm_dim=6,
        allow_ik_fallback=allow_ik_fallback,
    )


def _required_obs(group: Mapping[str, Any], key: str, env_id: int) -> np.ndarray:
    value = group.get(key)
    if value is None:
        raise KeyError(f"Missing required observation key {key!r}; available: {sorted(group.keys())}")
    return tensor_to_numpy(value[env_id])


def _optional_obs(group: Mapping[str, Any], key: str | None, env_id: int) -> np.ndarray | None:
    if key is None:
        return None
    value = group.get(key)
    if value is None:
        return None
    return tensor_to_numpy(value[env_id])


def _first_image(image_obs: Mapping[str, Any], *, env_id: int) -> np.ndarray:
    for value in image_obs.values():
        image = tensor_to_numpy(value[env_id])
        if image.ndim == 3:
            return image
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _optional_image(image_obs: Mapping[str, Any], key: str | None, reference: np.ndarray, *, env_id: int) -> np.ndarray:
    if key is None or key not in image_obs:
        return np.zeros_like(reference)
    return tensor_to_numpy(image_obs[key][env_id])
