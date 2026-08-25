# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from policies.cosmos3.openpi_compat import WebsocketClientPolicy

from policies.cosmos3.specs import (
    DEFAULT_CONTROL_FPS,
    JOINT_CURRENT_STATE_CONDITIONING,
    STATELESS_CONDITIONING,
    ClientCapability,
    ObservationCapability,
    PolicyContract,
    binarize_close_fraction,
    expand_action_chunk,
    validate_same_policy_semantics,
    validate_server_metadata,
    validate_wire_action_chunk,
)
from robolab.core.motion.eef import (
    EEFPolicyObservation,
    parse_eef_pose_action,
    quat_xyzw_to_wxyz,
    quat_to_matrix_wxyz,
    quat_wxyz_to_xyzw,
    resolve_operational_frame_pose,
)
from robolab.core.motion.pinocchio import PinocchioIKBridge
from robolab.eval.base_client import InferenceClient
from robolab.robots.ur5_profile import (
    UR5E_PINOCCHIO_URDF_PATH,
    get_ur5_berkeley_eef_profile,
    get_ur5_eef_profile,
)

logger = logging.getLogger(__name__)


class Cosmos3Client(InferenceClient):
    """Cosmos3 joint-policy client configured by a RoboLab capability."""

    IMAGE_W = 640
    IMAGE_H = 360

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        capability: ClientCapability,
        control_fps: int = DEFAULT_CONTROL_FPS,
        execute_horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.capability = capability
        self.control_fps = int(control_fps)
        self.policy_contract: PolicyContract | None = None
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._image_h, self._image_w = capability.observation.view_shape_hw

        display = f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
        self.client = self._connect()
        assert self.policy_contract is not None
        self.execute_horizon = self.policy_contract.chunk_size if execute_horizon is None else int(execute_horizon)
        if not 1 <= self.execute_horizon <= self.policy_contract.chunk_size:
            raise ValueError(
                f"execute_horizon must be in [1, {self.policy_contract.chunk_size}], got {self.execute_horizon}"
            )
        self.open_loop_horizon = self.execute_horizon
        print(f"[{self.__class__.__name__}] Connected to {display}.")

    def _connect(self) -> WebsocketClientPolicy:
        client = WebsocketClientPolicy(self._remote_host, self._remote_port)
        actual = validate_server_metadata(client.get_server_metadata(), self.capability)
        self._validate_policy_contract(actual)
        actual.hold_ratio(self.control_fps)
        if self.policy_contract is not None:
            validate_same_policy_semantics(self.policy_contract, actual)
        self.policy_contract = actual
        logger.info(
            "[%s] validated server contract profile=%s robot=%s action_space=%s fps=%d chunk=%d",
            self.__class__.__name__,
            actual.profile_id,
            actual.robot,
            actual.action_space,
            actual.policy_fps,
            actual.chunk_size,
        )
        return client

    def close(self) -> None:
        self.client.close()

    def _validate_policy_contract(self, contract: PolicyContract) -> None:
        """Subclass hook for action codecs beyond generic joint position."""

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
        canvas_views = self._extract_canvas_views(raw_obs["image_obs"], env_id=env_id)

        joint_position = raw_obs["proprio_obs"]["arm_joint_pos"][env_id].cpu().numpy()
        gripper_position = raw_obs["proprio_obs"]["gripper_pos"][env_id].cpu().numpy()

        return {
            "canvas_views": canvas_views,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        image = self._compose_canvas(extracted_obs["canvas_views"])

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
        if not isinstance(response, dict) or "action" not in response:
            raise ValueError(f"Expected Cosmos3 response dict with an 'action' key, got {type(response).__name__}")
        assert self.policy_contract is not None
        return validate_wire_action_chunk(response["action"], self.policy_contract)

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return binarize_close_fraction(chunk)

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= len(self._chunks[env_id])

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        assert self.policy_contract is not None
        expanded = expand_action_chunk(chunk, self.policy_contract, control_fps=self.control_fps)
        execute_horizon = getattr(self, "execute_horizon", self.policy_contract.chunk_size)
        buffer_horizon = execute_horizon * self.policy_contract.hold_ratio(self.control_fps)
        expanded = expanded[:buffer_horizon]
        logger.info(
            "[%s] event=action_buffer env_id=%d profile=%s source_chunk_size=%d buffer_steps=%d",
            self.__class__.__name__,
            env_id,
            self.policy_contract.profile_id,
            len(chunk),
            len(expanded),
        )
        super()._set_chunk(env_id, expanded)

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        primary, aux_left, aux_right = extracted_obs["canvas_views"]
        # Preserve the historical horizontal debug view while the model canvas
        # remains primary-on-top and auxiliaries on the bottom row.
        return np.concatenate((aux_left, primary, aux_right), axis=1)

    def _extract_canvas_views(self, image_obs: dict, *, env_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation = self.capability.observation
        if len(observation.view_roles) != 3:
            raise ValueError(
                "Cosmos3 primary_top_aux_bottom_pair canvas requires exactly three ordered view roles, "
                f"got {observation.view_roles!r}"
            )

        real_images: dict[str, np.ndarray] = {}
        for role in observation.view_roles:
            source = observation.role_sources[role]
            if source is None or source in real_images:
                continue
            if source not in image_obs:
                raise KeyError(f"Cosmos3 observation role {role!r} requires image_obs[{source!r}]")
            image = _to_numpy(image_obs[source][env_id])
            # Training resizes every source view with a plain bilinear stretch
            # (canvas_utils.resize_view), never a letterboxed fit, so non-16:9
            # cameras must reproduce the same trained distortion here.
            real_images[source] = self._resize_for_canvas(
                image, size=(self._image_h, self._image_w), dtype=image.dtype
            )
        if not real_images:
            raise ValueError("Cosmos3 observation preset has no available real camera source")

        reference = next(iter(real_images.values()))
        views = tuple(
            np.zeros_like(reference)
            if observation.role_sources[role] is None
            else real_images[observation.role_sources[role]]
            for role in observation.view_roles
        )
        return views

    def _compose_canvas(self, views: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        primary, aux_left, aux_right = views
        size = (self._image_h // 2, self._image_w // 2)
        left = self._resize_for_canvas(aux_left, size=size, dtype=primary.dtype)
        right = self._resize_for_canvas(aux_right, size=size, dtype=primary.dtype)
        return np.concatenate((primary, np.concatenate((left, right), axis=1)), axis=0)

    @staticmethod
    def _resize_for_canvas(image: np.ndarray, *, size: tuple[int, int], dtype: np.dtype) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(tensor, size=size, mode="bilinear")
        return resized.squeeze(0).permute(1, 2, 0).numpy().astype(dtype, copy=False)


class EEFControllerAdapter(Protocol):
    """One-way conversion from canonical EEF wire actions to an env layout."""

    arm_dof: int
    policy_frame: str
    controller_frame: str
    env_action_dim: int

    def convert(self, chunk: np.ndarray) -> np.ndarray:
        """Convert ``[xyz, quat_xyzw, close]`` into controller actions."""


@dataclass(frozen=True)
class IsaacLabAbsIKAdapter:
    """Compose a fixed policy-frame-to-controller transform for absolute IK."""

    arm_dof: int
    policy_frame: str
    controller_frame: str
    controller_in_policy_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller_in_policy_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    env_action_dim: int = 8

    def convert(self, chunk: np.ndarray) -> np.ndarray:
        action = parse_eef_pose_action(chunk)
        offset = np.asarray(self.controller_in_policy_xyz, dtype=np.float32)
        position = action.position + np.einsum(
            "nij,j->ni", quat_to_matrix_wxyz(action.quat_wxyz), offset
        )
        fixed = np.asarray(self.controller_in_policy_quat_wxyz, dtype=np.float32)
        quat_wxyz = _quat_multiply_wxyz(action.quat_wxyz, np.broadcast_to(fixed, action.quat_wxyz.shape))
        quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
        return np.concatenate((position, quat_xyzw, action.gripper), axis=-1).astype(np.float32, copy=False)


class Cosmos3EEFClient(Cosmos3Client):
    """Robot-agnostic canonical EEF client; robot facts live in the adapter."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        capability: ClientCapability,
        adapter: EEFControllerAdapter,
        execute_horizon: int = 8,
        control_fps: int = DEFAULT_CONTROL_FPS,
        eef_pos_key: str = "eef_pos",
        eef_quat_key: str = "eef_quat",
    ) -> None:
        if adapter.arm_dof != capability.arm_dof:
            raise ValueError("EEF adapter and client capability arm_dof must match")
        self.adapter = adapter
        self._eef_pos_key = eef_pos_key
        self._eef_quat_key = eef_quat_key
        super().__init__(
            remote_host,
            remote_port,
            capability=capability,
            control_fps=control_fps,
            execute_horizon=execute_horizon,
        )

    def _validate_policy_contract(self, contract: PolicyContract) -> None:
        expected = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
        if contract.action_space != "eef_absolute" or contract.action_layout != expected:
            raise ValueError("Cosmos3 EEF clients require canonical absolute xyz+quat_xyzw+gripper actions")
        if (contract.quaternion_order, contract.pose_mode, contract.eef_frame) != (
            "xyzw",
            "absolute",
            self.adapter.policy_frame,
        ):
            raise ValueError(
                "Cosmos3 EEF contract does not match adapter frame/pose convention: "
                f"server={(contract.eef_frame, contract.quaternion_order, contract.pose_mode)!r}, "
                f"adapter_frame={self.adapter.policy_frame!r}"
            )

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        canvas_views = self._extract_canvas_views(raw_obs["image_obs"], env_id=env_id)
        proprio = raw_obs["proprio_obs"]
        position = _to_numpy(proprio[self._eef_pos_key][env_id]).reshape(3).astype(np.float32, copy=False)
        quat_xyzw = _to_numpy(proprio[self._eef_quat_key][env_id]).reshape(4).astype(np.float32, copy=False)
        gripper = _optional_proprio(proprio, "gripper_pos", env_id=env_id)
        gripper = np.zeros(1, dtype=np.float32) if gripper is None else gripper.reshape(1).astype(np.float32)
        return {
            "canvas_views": canvas_views,
            "eef_pose": np.concatenate((position, quat_xyzw)).astype(np.float32),
            "gripper_position": gripper,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "observation/image": self._compose_canvas(extracted_obs["canvas_views"]),
            "observation/eef_pose": _history_row(extracted_obs["eef_pose"]),
            "observation/gripper_position": _history_row(extracted_obs["gripper_position"]),
            "prompt": instruction,
        }

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        converted = self.adapter.convert(chunk)
        expected = (self.policy_contract.chunk_size, self.adapter.env_action_dim)
        if converted.shape != expected or not np.isfinite(converted).all():
            raise ValueError(f"EEF adapter returned invalid controller action shape {converted.shape}; expected {expected}")
        return binarize_close_fraction(converted)


class Cosmos3UR5Client(Cosmos3Client):
    """Cosmos3 client that decodes explicit UR5 joint or EEF contracts."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        *,
        observation: ObservationCapability,
        control_fps: int = DEFAULT_CONTROL_FPS,
        pinocchio_urdf_path: str | None = None,
        ik_pos_tol: float = 0.005,
        ik_rot_tol: float = 0.05,
        execute_horizon: int = 8,
    ) -> None:
        self._profile = get_ur5_eef_profile()
        self._env_action_dim = self._profile.env_action_dim + 1
        self._pinocchio_urdf_path = pinocchio_urdf_path or UR5E_PINOCCHIO_URDF_PATH
        self._ik_pos_tol = ik_pos_tol
        self._ik_rot_tol = ik_rot_tol
        self._eef_bridge: PinocchioIKBridge | None = None
        self._last_diagnostics: dict[int, list[dict[str, Any]]] = {}
        super().__init__(
            remote_host=remote_host,
            remote_port=remote_port,
            capability=ClientCapability(
                robot="ur5",
                arm_dof=6,
                action_spaces=("joint_position", "eef_absolute"),
                joint_action_layout=(
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow",
                    "wrist_1",
                    "wrist_2",
                    "wrist_3",
                ),
                conditioning_by_action_space={
                    "joint_position": JOINT_CURRENT_STATE_CONDITIONING,
                    "eef_absolute": STATELESS_CONDITIONING,
                },
                observation=observation,
            ),
            control_fps=control_fps,
            execute_horizon=execute_horizon,
        )

    def _validate_policy_contract(self, contract: PolicyContract) -> None:
        if contract.action_space == "joint_position":
            return
        expected_layout = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
        if contract.action_layout != expected_layout:
            raise ValueError(
                f"UR5 EEF policies require xyz+quat_xyzw+gripper wire actions; got layout={contract.action_layout!r}"
            )
        if (contract.quaternion_order, contract.pose_mode) != ("xyzw", "absolute"):
            raise ValueError(
                "UR5 EEF policies require absolute xyz+quat_xyzw+gripper wire actions; "
                f"got quaternion_order={contract.quaternion_order!r}, pose_mode={contract.pose_mode!r}"
            )
        profile_by_frame = {
            "tool0": get_ur5_eef_profile,
            "berkeley_tcp": get_ur5_berkeley_eef_profile,
        }
        try:
            profile = profile_by_frame[contract.eef_frame]()
        except KeyError as exc:
            raise ValueError(
                f"UR5 EEF policies require eef_frame='tool0' or 'berkeley_tcp'; got {contract.eef_frame!r}"
            ) from exc
        self._profile = profile
        self._env_action_dim = profile.env_action_dim + 1

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
                    "[%s] event=ur5_query status=error elapsed_ms=%.1f env_id=%d profile=%s",
                    self.__class__.__name__,
                    elapsed_ms,
                    env_id,
                    self.policy_contract.profile_id,
                )
                raise
            elapsed_ms = (time.perf_counter() - query_start) * 1000.0
            chunk, diagnostics = self._convert_response_chunk(response, extracted, env_id=env_id)
            self._set_chunk(env_id, chunk)
            self._last_diagnostics[env_id] = diagnostics
            logger.info(
                "[%s] event=ur5_query status=ok elapsed_ms=%.1f env_id=%d profile=%s "
                "source_chunk_size=%d buffer_steps=%d action_space=%s",
                self.__class__.__name__,
                elapsed_ms,
                env_id,
                self.policy_contract.profile_id,
                self.policy_contract.chunk_size,
                len(self._chunks[env_id]),
                self.policy_contract.action_space,
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

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        canvas_views = self._extract_canvas_views(image_obs, env_id=env_id)
        primary_image, aux_left_image, aux_right_image = canvas_views

        proprio_obs = raw_obs["proprio_obs"]
        arm_joint_position = _to_numpy(proprio_obs["arm_joint_pos"][env_id]).astype(np.float32, copy=False)
        joint_position = arm_joint_position
        gripper_obs = _optional_proprio(proprio_obs, "gripper_pos", env_id=env_id)
        if gripper_obs is None:
            gripper_position = np.zeros((1,), dtype=np.float32)
        else:
            gripper_position = gripper_obs.reshape(1).astype(np.float32, copy=False)
        ee_pos = _optional_proprio(proprio_obs, self._profile.ee_pos_key, env_id=env_id)
        ee_quat = _optional_proprio(proprio_obs, self._profile.ee_quat_key, env_id=env_id)
        eef_observation = None
        eef_pose_xyzw = None
        eef_pos = None
        eef_quat_xyzw = None
        if ee_pos is not None and ee_quat is not None:
            ee_quat_wxyz = quat_xyzw_to_wxyz(ee_quat.reshape(4))
            eef_pos, ee_quat_wxyz = resolve_operational_frame_pose(
                ee_pos.reshape(3),
                ee_quat_wxyz,
                self._profile.fixed_operational_frame,
            )
            eef_quat_xyzw = quat_wxyz_to_xyzw(ee_quat_wxyz).astype(np.float32, copy=False)
            eef_observation = EEFPolicyObservation(
                primary_image=aux_left_image,
                wrist_image=primary_image,
                secondary_image=aux_right_image,
                joint_position=arm_joint_position,
                ee_pos=eef_pos,
                ee_quat_wxyz=ee_quat_wxyz,
                gripper_position=gripper_position,
                metadata={
                    "profile": self._profile.name,
                    "eef_frame": self._profile.ee_link,
                    "env_id": env_id,
                },
            )
            eef_pose_xyzw = np.concatenate([eef_pos, eef_quat_xyzw, gripper_position]).astype(np.float32, copy=False)

        return {
            "canvas_views": canvas_views,
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
        assert self.policy_contract is not None
        requires_eef = self.policy_contract.action_space == "eef_absolute"
        if requires_eef and extracted_obs["eef_pose_xyzw"] is None:
            raise ValueError(
                f"server profile {self.policy_contract.profile_id!r} requires UR5 observation ee_pos/ee_quat"
            )
        if requires_eef:
            request["observation/eef_pose"] = _history_row(extracted_obs["eef_pose_xyzw"])
            request["observation/eef_pos"] = _history_row(extracted_obs["eef_pos"])
            request["observation/eef_quat"] = _history_row(extracted_obs["eef_quat_xyzw"])
            request["observation/gripper_position"] = _history_row(extracted_obs["gripper_position"])
        return request

    def _convert_response_chunk(
        self, response: dict, extracted_obs: dict, *, env_id: int
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        action = self._unpack_response(response)
        assert self.policy_contract is not None
        if self.policy_contract.action_space == "joint_position":
            return self._convert_joint_chunk(action, env_id=env_id)
        return self._convert_eef_chunk(action, extracted_obs, env_id=env_id)

    def _convert_joint_chunk(self, action: np.ndarray, *, env_id: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
        chunk = self._postprocess_chunk(action.astype(np.float32, copy=False))
        self._validate_decoded_chunk(chunk)
        return chunk, [
            {
                "env_id": env_id,
                "action_space": "joint_position",
                "source_action_shape": tuple(action.shape),
            }
        ]

    def _convert_eef_chunk(
        self, action: np.ndarray, extracted_obs: dict, *, env_id: int
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        eef_observation = extracted_obs.get("eef_observation")
        if eef_observation is None:
            raise ValueError("EEF policy requires a UR5 observation containing ee_pos/ee_quat")

        eef_chunk = parse_eef_pose_action(action)
        bridge = self._get_eef_bridge()
        result = bridge.convert_chunk(eef_observation, eef_chunk, env_id=env_id)
        raw_gripper = result.raw_gripper if result.raw_gripper is not None else eef_chunk.gripper
        gripper = np.asarray(raw_gripper, dtype=np.float32).reshape(eef_chunk.horizon, 1)
        arm_chunk = np.asarray(result.actions, dtype=np.float32)
        if not np.isfinite(arm_chunk).all() or not np.isfinite(gripper).all():
            raise ValueError("UR5 EEF conversion produced non-finite actions")
        chunk = np.concatenate([arm_chunk, gripper], axis=-1)
        chunk = self._postprocess_chunk(chunk)
        self._validate_decoded_chunk(chunk)
        for diag in result.diagnostics:
            diag["action_space"] = "eef_absolute"
            diag["source_action_shape"] = tuple(action.shape)
            diag["ik_backend"] = "pinocchio"
        return chunk, result.diagnostics

    def _get_eef_bridge(self) -> PinocchioIKBridge:
        if self._eef_bridge is None:
            self._eef_bridge = PinocchioIKBridge(
                profile=self._profile,
                urdf_path=self._pinocchio_urdf_path,
                position_threshold=self._ik_pos_tol,
                rotation_threshold=self._ik_rot_tol,
            )
        return self._eef_bridge

    def _validate_decoded_chunk(self, chunk: np.ndarray) -> None:
        assert self.policy_contract is not None
        expected_shape = (self.policy_contract.chunk_size, self._env_action_dim)
        if chunk.ndim != 2 or chunk.shape != expected_shape:
            raise ValueError(f"Expected decoded UR5 action shape {expected_shape}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("Decoded UR5 action chunk contains non-finite values")


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


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Vectorized Hamilton product for normalized wxyz quaternions."""

    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    result = np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("EEF frame conversion produced a zero-length quaternion")
    return result / norm
