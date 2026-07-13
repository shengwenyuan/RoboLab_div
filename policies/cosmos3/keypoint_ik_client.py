# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 UR5 client variants for keypoint-IK three-camera policy runs."""

from __future__ import annotations

import numpy as np

from policies.cosmos3.client import Cosmos3Client, Cosmos3UR5Client, ServerActionFormat


KEYPOINT_IK_IMAGE_KEYS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)


class Cosmos3UR5KeypointIKClient(Cosmos3UR5Client):
    """UR5 EEF client whose image contract matches keypoint-IK data generation.

    Keypoint-IK raw data is generated with the physical
    ``wrist_left_right`` preset. Unlike the Berkeley UR5 client path, this
    client requires and uses all three camera observations instead of replacing
    the right over-shoulder image with zeros.
    """

    REQUIRED_IMAGE_KEYS = KEYPOINT_IK_IMAGE_KEYS

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        server_action_format: ServerActionFormat = "eef_pose",
        pinocchio_urdf_path: str | None = None,
        ik_pos_tol: float = 0.005,
        ik_rot_tol: float = 0.05,
    ) -> None:
        super().__init__(
            remote_host=remote_host,
            remote_port=remote_port,
            server_action_format=server_action_format,
            pinocchio_urdf_path=pinocchio_urdf_path,
            ik_pos_tol=ik_pos_tol,
            ik_rot_tol=ik_rot_tol,
        )

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs.get("image_obs", {})
        missing = [key for key in self.REQUIRED_IMAGE_KEYS if key not in image_obs]
        if missing:
            raise KeyError(
                "Cosmos3UR5KeypointIKClient requires keypoint-IK camera observations "
                f"{self.REQUIRED_IMAGE_KEYS}; missing={missing}"
            )
        return super()._extract_observation(raw_obs, env_id=env_id)

    def _compose_canvas(self, left_image: np.ndarray, wrist_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        size = (self._image_h // 2, self._image_w // 2)
        left = self._resize_for_canvas(left_image, size=size, dtype=wrist_image.dtype)
        right = self._resize_for_canvas(right_image, size=size, dtype=wrist_image.dtype)
        return np.concatenate((wrist_image, np.concatenate((left, right), axis=1)), axis=0)


class Cosmos3UR5KeypointIKJointposClient(Cosmos3Client):
    """DROID-style joint-position client for UR5 keypoint-IK validation.

    This client intentionally mirrors :class:`Cosmos3Client`'s native
    joint-position wire format. The only UR5/keypoint-IK-specific behavior is
    requiring the three real keypoint-IK cameras and enforcing the UR5 env
    action width of 7: ``[arm_joint_pos(6), gripper(1)]``.
    """

    REQUIRED_IMAGE_KEYS = KEYPOINT_IK_IMAGE_KEYS
    ACTION_DIM = 7

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs.get("image_obs", {})
        missing = [key for key in self.REQUIRED_IMAGE_KEYS if key not in image_obs]
        if missing:
            raise KeyError(
                "Cosmos3UR5KeypointIKJointposClient requires keypoint-IK camera observations "
                f"{self.REQUIRED_IMAGE_KEYS}; missing={missing}"
            )
        return super()._extract_observation(raw_obs, env_id=env_id)

    def _unpack_response(self, response: dict) -> np.ndarray:
        if not isinstance(response, dict) or "action" not in response:
            raise ValueError(f"Expected Cosmos response dict with an 'action' key, got {type(response).__name__}")
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[-1] != self.ACTION_DIM:
            raise ValueError(
                f"Expected UR5 keypoint-IK joint-pos action chunk shape (H, {self.ACTION_DIM}), got {action.shape}"
            )
        return np.nan_to_num(action.astype(np.float32, copy=True))
