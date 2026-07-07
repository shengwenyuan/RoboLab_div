# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from openpi_client import image_tools, websocket_client_policy

from robolab.core.motion.eef import EEFPolicyObservation, parse_eef_pose_action, quat_wxyz_to_xyzw
from robolab.core.motion.pinocchio import PinocchioIKBridge
from robolab.eval.base_client import InferenceClient
from robolab.robots.ur5_profile import UR5E_PINOCCHIO_URDF_PATH, get_ur5_eef_profile

logger = logging.getLogger(__name__)

ServerActionFormat = Literal["auto", "joint", "eef_pose"]


class Cosmos3Client(InferenceClient):
    """Cosmos3 DROID client."""

    IMAGE_W = 640
    IMAGE_H = 360
    OPEN_LOOP_HORIZON = 32

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000):
        super().__init__()
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._image_w = self.IMAGE_W
        self._image_h = self.IMAGE_H
        self.open_loop_horizon = self.OPEN_LOOP_HORIZON

        display = f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {display}.")

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__,
                    e,
                    attempt + 1,
                    max_retries,
                )
                self.client = self._connect()
                self._chunks.clear()
                self._counters.clear()
        raise RuntimeError("unreachable retry state")

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        left_image = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].cpu().numpy()
        left_image = image_tools.resize_with_pad(left_image, self._image_h, self._image_w)
        right_image = raw_obs["image_obs"]["over_shoulder_right_camera"][env_id].cpu().numpy()
        right_image = image_tools.resize_with_pad(right_image, self._image_h, self._image_w)
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].cpu().numpy()
        wrist_image = image_tools.resize_with_pad(wrist_image, self._image_h, self._image_w)

        joint_position = raw_obs["proprio_obs"]["arm_joint_pos"][env_id].cpu().numpy()
        gripper_position = raw_obs["proprio_obs"]["gripper_pos"][env_id].cpu().numpy()

        return {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        image = self._compose_canvas(
            extracted_obs["left_image"],
            extracted_obs["wrist_image"],
            extracted_obs["right_image"],
        )

        return {
            "observation/image": image,
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        logger.debug(
            "[%s] Querying server: prompt=%r image_shape=%s joint_shape=%s gripper_shape=%s",
            self.__class__.__name__,
            request.get("prompt"),
            getattr(request.get("observation/image"), "shape", None),
            getattr(request.get("observation/joint_position"), "shape", None),
            getattr(request.get("observation/gripper_position"), "shape", None),
        )
        response = self._infer_with_retry(request)
        if logger.isEnabledFor(logging.DEBUG):
            action = response.get("action") if isinstance(response, dict) else None
            server_timing = response.get("server_timing") if isinstance(response, dict) else None
            response_keys = sorted(response.keys()) if isinstance(response, dict) else type(response).__name__
            logger.debug(
                "[%s] Server response: keys=%s action_shape=%s server_timing=%s",
                self.__class__.__name__,
                response_keys,
                getattr(action, "shape", None),
                server_timing,
            )
        return response

    def _unpack_response(self, response: dict) -> np.ndarray:
        return np.asarray(response["action"])

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        left = extracted_obs["left_image"]
        wrist = extracted_obs["wrist_image"]
        right = extracted_obs["right_image"]
        return np.concatenate((left, wrist, right), axis=1)

    def _compose_canvas(self, left_image: np.ndarray, wrist_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        wrist = wrist_image
        size = (self._image_h // 2, self._image_w // 2)
        left = self._resize_for_canvas(left_image, size=size, dtype=wrist.dtype)
        right = self._resize_for_canvas(right_image, size=size, dtype=wrist.dtype)
        return np.concatenate((wrist, np.concatenate((left, right), axis=1)), axis=0)

    @staticmethod
    def _resize_for_canvas(image: np.ndarray, *, size: tuple[int, int], dtype: np.dtype) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(tensor, size=size, mode="bilinear")
        return resized.squeeze(0).permute(1, 2, 0).numpy().astype(dtype, copy=False)


def droid_action_to_ur5(action: np.ndarray) -> np.ndarray:
    """Map joint-space arm/gripper actions to UR5's 7D env action chunk."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, ...]
    if action.ndim != 2 or action.shape[-1] < 7:
        raise ValueError(f"Expected joint action shape (H, D>=7) with a gripper channel, got {action.shape}")
    return np.concatenate([action[..., :6], action[..., -1:]], axis=-1)


class Cosmos3UR5Client(Cosmos3Client):
    """Cosmos3 client for UR5e joint-position environments.

    The server may return joint-space actions or single-arm 8D absolute
    EEF pose actions. Both are normalized to the UR5 7D
    ``[arm_joint_pos(6), gripper(1)]`` action expected by the IsaacLab
    action manager.
    """

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
        if server_action_format not in ("auto", "joint", "eef_pose"):
            raise ValueError(f"Unsupported server_action_format: {server_action_format!r}")
        self.server_action_format = server_action_format
        self._profile = get_ur5_eef_profile()
        self._env_action_dim = self._profile.env_action_dim + 1
        self._pinocchio_urdf_path = pinocchio_urdf_path or UR5E_PINOCCHIO_URDF_PATH
        self._ik_pos_tol = ik_pos_tol
        self._ik_rot_tol = ik_rot_tol
        self._eef_bridge: PinocchioIKBridge | None = None
        self._last_diagnostics: dict[int, list[dict[str, Any]]] = {}
        super().__init__(remote_host=remote_host, remote_port=remote_port)

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        extracted = self._extract_observation(obs, env_id=env_id)

        if self._needs_refresh(env_id):
            request = self._pack_request(extracted, instruction)
            query_start = time.perf_counter()
            try:
                response = self._query_server(request)
            except Exception:
                elapsed_ms = (time.perf_counter() - query_start) * 1000.0
                logger.warning(
                    "[%s] event=ur5_query status=error elapsed_ms=%.1f env_id=%d horizon=%d",
                    self.__class__.__name__,
                    elapsed_ms,
                    env_id,
                    self.open_loop_horizon,
                )
                raise
            elapsed_ms = (time.perf_counter() - query_start) * 1000.0
            chunk, diagnostics = self._convert_response_chunk(response, extracted, env_id=env_id)
            self._set_chunk(env_id, chunk)
            self._last_diagnostics[env_id] = diagnostics
            format_name = diagnostics[0].get("server_action_format") if diagnostics else "unknown"
            logger.info(
                "[%s] event=ur5_query status=ok elapsed_ms=%.1f env_id=%d horizon=%d action_format=%s",
                self.__class__.__name__,
                elapsed_ms,
                env_id,
                self.open_loop_horizon,
                format_name,
            )

        return {
            "action": self._next_action(env_id),
            "viz": self._build_visualization(extracted),
            "diagnostics": self._last_diagnostics.get(env_id),
        }

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if self._eef_bridge is not None:
            self._eef_bridge.reset(env_id=env_id)
        if env_id is None:
            self._last_diagnostics.clear()
        else:
            self._last_diagnostics.pop(env_id, None)

    def _compose_canvas(self, left_image: np.ndarray, wrist_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        size = (self._image_h // 2, self._image_w // 2)
        left = self._resize_for_canvas(left_image, size=size, dtype=wrist_image.dtype)
        right = np.zeros_like(left)
        return np.concatenate((wrist_image, np.concatenate((left, right), axis=1)), axis=0)

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        left_image = image_obs["over_shoulder_left_camera"][env_id].cpu().numpy()
        left_image = image_tools.resize_with_pad(left_image, self._image_h, self._image_w)

        right_tensor = image_obs.get("over_shoulder_right_camera")
        if right_tensor is None:
            right_image = np.zeros_like(left_image)
        else:
            right_image = right_tensor[env_id].cpu().numpy()
            right_image = image_tools.resize_with_pad(right_image, self._image_h, self._image_w)

        wrist_tensor = image_obs.get("wrist_cam")
        if wrist_tensor is None:
            wrist_image = np.zeros_like(left_image)
        else:
            wrist_image = wrist_tensor[env_id].cpu().numpy()
            wrist_image = image_tools.resize_with_pad(wrist_image, self._image_h, self._image_w)

        proprio_obs = raw_obs["proprio_obs"]
        arm_joint_position = _to_numpy(proprio_obs["arm_joint_pos"][env_id]).astype(np.float32, copy=False)
        joint_position = arm_joint_position
        gripper_obs = _optional_proprio(proprio_obs, "gripper_pos", env_id=env_id)
        if gripper_obs is None:
            gripper_position = np.zeros((1,), dtype=np.float32)
        else:
            gripper_position = gripper_obs.reshape(1).astype(np.float32, copy=False)
        ee_pos = _optional_proprio(proprio_obs, "ee_pos", env_id=env_id)
        ee_quat = _optional_proprio(proprio_obs, "ee_quat", env_id=env_id)
        eef_observation = None
        eef_pose_xyzw = None
        eef_pos = None
        eef_quat_xyzw = None
        if ee_pos is not None and ee_quat is not None:
            eef_pos = ee_pos.reshape(3).astype(np.float32, copy=False)
            ee_quat = ee_quat.reshape(4).astype(np.float32, copy=False)
            eef_quat_xyzw = quat_wxyz_to_xyzw(ee_quat).astype(np.float32, copy=False)
            eef_observation = EEFPolicyObservation(
                primary_image=left_image,
                wrist_image=wrist_image,
                secondary_image=right_image,
                joint_position=arm_joint_position,
                ee_pos=eef_pos,
                ee_quat_wxyz=ee_quat,
                gripper_position=gripper_position,
                metadata={"profile": self._profile.name, "env_id": env_id},
            )
            eef_pose_xyzw = np.concatenate([eef_pos, eef_quat_xyzw, gripper_position]).astype(np.float32, copy=False)

        return {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "arm_joint_position": arm_joint_position,
            "gripper_position": gripper_position,
            "eef_observation": eef_observation,
            "eef_pos": eef_pos,
            "eef_quat_xyzw": eef_quat_xyzw,
            "eef_pose_xyzw": eef_pose_xyzw,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        request["observation/arm_joint_position"] = extracted_obs["arm_joint_position"]
        requires_eef = self.server_action_format == "eef_pose"
        if requires_eef and extracted_obs["eef_pose_xyzw"] is None:
            raise ValueError(
                f"server_action_format={self.server_action_format!r} requires UR5 observation ee_pos/ee_quat"
            )
        if extracted_obs["eef_pose_xyzw"] is not None:
            request["observation/eef_pose"] = _history_row(extracted_obs["eef_pose_xyzw"])
            request["observation/eef_pos"] = _history_row(extracted_obs["eef_pos"])
            request["observation/eef_quat"] = _history_row(extracted_obs["eef_quat_xyzw"])
            request["observation/gripper_position"] = _history_row(extracted_obs["gripper_position"])
        return request

    def _convert_response_chunk(
        self, response: dict, extracted_obs: dict, *, env_id: int
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not isinstance(response, dict) or "action" not in response:
            raise ValueError(f"Expected Cosmos response dict with an 'action' key, got {type(response).__name__}")
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        action_format = self._resolve_action_format(action)
        if action_format == "joint":
            chunk = np.nan_to_num(droid_action_to_ur5(action).astype(np.float32, copy=True))
            chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
            self._validate_horizon(chunk)
            return chunk, [
                {
                    "env_id": env_id,
                    "server_action_format": "joint",
                    "source_action_shape": tuple(action.shape),
                }
            ]

        eef_observation = extracted_obs.get("eef_observation")
        if eef_observation is None:
            raise ValueError("Server returned EEF action but UR5 observation does not contain ee_pos/ee_quat")
        bridge = self._get_eef_bridge()
        if action_format != "eef_pose":
            raise ValueError(f"Unexpected EEF action format: {action_format}")
        eef_chunk = parse_eef_pose_action(action)
        if eef_chunk.horizon != self.open_loop_horizon:
            raise ValueError(f"Expected EEF horizon {self.open_loop_horizon}, got {eef_chunk.raw.shape}")
        result = bridge.convert_chunk(eef_observation, eef_chunk, env_id=env_id)
        raw_gripper = result.raw_gripper if result.raw_gripper is not None else eef_chunk.gripper
        gripper = (np.asarray(raw_gripper, dtype=np.float32).reshape(eef_chunk.horizon, 1) > 0.5).astype(np.float32)
        arm_chunk = np.nan_to_num(result.actions.astype(np.float32, copy=True))
        chunk = np.concatenate([arm_chunk, gripper], axis=-1)
        self._validate_horizon(chunk)
        for diag in result.diagnostics:
            diag["server_action_format"] = action_format
            diag["source_action_shape"] = tuple(action.shape)
            diag["ik_backend"] = "pinocchio"
        return chunk, result.diagnostics

    def _resolve_action_format(self, action: np.ndarray) -> Literal["joint", "eef_pose"]:
        if self.server_action_format != "auto":
            return self.server_action_format
        if action.ndim != 2:
            raise ValueError(f"Expected server action shape (H, D), got {action.shape}")
        if action.shape[-1] == 8:
            # 8D is ambiguous with some joint+gripper layouts; UR5 auto treats it as EEF pose.
            return "eef_pose"
        if action.shape[-1] >= 6:
            return "joint"
        raise ValueError(f"Cannot infer UR5 server action format from shape {action.shape}")

    def _get_eef_bridge(self) -> PinocchioIKBridge:
        if self._eef_bridge is None:
            self._eef_bridge = PinocchioIKBridge(
                profile=self._profile,
                urdf_path=self._pinocchio_urdf_path,
                position_threshold=self._ik_pos_tol,
                rotation_threshold=self._ik_rot_tol,
            )
        return self._eef_bridge

    def _validate_horizon(self, chunk: np.ndarray) -> None:
        if chunk.ndim != 2 or chunk.shape != (self.open_loop_horizon, self._env_action_dim):
            raise ValueError(
                f"Expected UR5 action chunk shape ({self.open_loop_horizon}, {self._env_action_dim}), "
                f"got {chunk.shape}"
            )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _optional_proprio(proprio_obs: dict, key: str, *, env_id: int) -> np.ndarray | None:
    value = proprio_obs.get(key)
    if value is None:
        return None
    return _to_numpy(value[env_id])


def _history_row(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(1, -1)
