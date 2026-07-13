# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinocchio EEF-to-joint bridge."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from robolab.core.motion.eef import (
    EEFActionChunk,
    EEFPolicyObservation,
    MotionBridgeResult,
    RobotEEFProfile,
    matrix_to_quat_wxyz,
    quat_to_matrix_wxyz,
)

logger = logging.getLogger(__name__)


class PinocchioDependencyError(ImportError):
    """Raised when the optional Pinocchio backend is requested but unavailable."""


@dataclass(frozen=True)
class _PinocchioModules:
    pin: Any


def _load_pinocchio_modules() -> _PinocchioModules:
    if importlib.util.find_spec("pinocchio") is None:
        raise PinocchioDependencyError(
            "Pinocchio is not installed in the current Python environment. "
            "Install pinocchio before using PinocchioIKBridge."
        )
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise PinocchioDependencyError("Failed to import Pinocchio.") from exc
    return _PinocchioModules(pin=pin)


class PinocchioIKBridge:
    """Convert absolute EEF pose chunks into joint-position chunks with Pinocchio IK."""

    def __init__(
        self,
        *,
        profile: RobotEEFProfile,
        urdf_path: str,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        max_iterations: int = 80,
        damping: float = 1e-6,
        step_size: float = 0.5,
    ) -> None:
        self.profile = profile
        self.urdf_path = urdf_path
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_iterations = max_iterations
        self.damping = damping
        self.step_size = step_size
        self._last_solution: dict[int, np.ndarray] = {}
        self._last_action: dict[int, np.ndarray] = {}

        self._modules = _load_pinocchio_modules()
        self._pin = self._modules.pin
        self._model = self._pin.buildModelFromUrdf(urdf_path)
        self._add_fixed_operational_frame()
        self._data = self._model.createData()
        if not self._model.existFrame(profile.ee_link):
            raise ValueError(f"Pinocchio model has no frame {profile.ee_link!r}: {urdf_path}")
        self._frame_id = self._model.getFrameId(profile.ee_link)
        if self._model.nq != profile.env_action_dim or self._model.nv != profile.env_action_dim:
            raise ValueError(
                f"Pinocchio model dim nq={self._model.nq}, nv={self._model.nv} does not match "
                f"{profile.name} env action dim {profile.env_action_dim}"
            )

    def _add_fixed_operational_frame(self) -> None:
        fixed_frame = self.profile.fixed_operational_frame
        if fixed_frame is None or self._model.existFrame(self.profile.ee_link):
            return
        if not self._model.existFrame(fixed_frame.parent_link):
            raise ValueError(
                f"Pinocchio model has no parent frame {fixed_frame.parent_link!r} required by "
                f"operational frame {self.profile.ee_link!r}: {self.urdf_path}"
            )

        pin = self._pin
        parent_frame_id = self._model.getFrameId(fixed_frame.parent_link)
        parent_frame = self._model.frames[parent_frame_id]
        offset_rotation = np.asarray(
            quat_to_matrix_wxyz(np.asarray(fixed_frame.quat_wxyz, dtype=np.float32)), dtype=np.float64
        )
        offset = pin.SE3(offset_rotation, np.asarray(fixed_frame.xyz, dtype=np.float64))
        placement_in_parent_joint = parent_frame.placement * offset
        frame = pin.Frame(
            self.profile.ee_link,
            parent_frame.parent,
            parent_frame_id,
            placement_in_parent_joint,
            pin.FrameType.OP_FRAME,
        )
        self._model.addFrame(frame, False)

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._last_solution.clear()
            self._last_action.clear()
        else:
            self._last_solution.pop(env_id, None)
            self._last_action.pop(env_id, None)

    def forward_pose(self, joint_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        placement = self._frame_placement(np.asarray(joint_position, dtype=np.float64).reshape(self._model.nq))
        pos = np.asarray(placement.translation, dtype=np.float32).reshape(3)
        quat = matrix_to_quat_wxyz(np.asarray(placement.rotation, dtype=np.float32))
        return pos, quat

    def convert_chunk(
        self,
        observation: EEFPolicyObservation,
        action_chunk: EEFActionChunk,
        *,
        env_id: int = 0,
    ) -> MotionBridgeResult:
        joint_position = np.asarray(observation.joint_position, dtype=np.float32).reshape(-1)
        if joint_position.shape[0] != self.profile.env_action_dim:
            raise ValueError(
                f"{self.profile.name} bridge expected joint_position dim {self.profile.env_action_dim}, "
                f"got {joint_position.shape}"
            )

        hold_action = self._last_action.get(env_id, joint_position.copy())
        seed = self._last_solution.get(env_id, joint_position.copy())
        actions = np.zeros((action_chunk.horizon, self.profile.env_action_dim), dtype=np.float32)
        diagnostics: list[dict[str, Any]] = []

        for step in range(action_chunk.horizon):
            target_pos = np.asarray(action_chunk.position[step], dtype=np.float32).reshape(3)
            target_quat = np.asarray(action_chunk.quat_wxyz[step], dtype=np.float32).reshape(4)
            solution, diag = self._solve_step(target_pos, target_quat, seed)
            diag.update(
                {
                    "env_id": env_id,
                    "step": step,
                    "action_kind": "eef_pose",
                    "gripper": float(action_chunk.gripper[step, 0]),
                }
            )

            if solution is None:
                actions[step] = hold_action
                diag["fallback"] = "hold_previous"
            else:
                solution = np.nan_to_num(solution.astype(np.float32, copy=False))
                actions[step] = solution
                hold_action = solution
                seed = solution
                diag["fallback"] = None

            diagnostics.append(diag)

        self._last_solution[env_id] = seed.astype(np.float32, copy=True)
        self._last_action[env_id] = hold_action.astype(np.float32, copy=True)
        return MotionBridgeResult(actions=actions, diagnostics=diagnostics, raw_gripper=action_chunk.gripper.copy())

    def _solve_step(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        seed: np.ndarray,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        pin = self._pin
        q = np.asarray(seed, dtype=np.float64).reshape(self._model.nq).copy()
        target = pin.SE3(
            np.asarray(quat_to_matrix_wxyz(target_quat_wxyz), dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64).reshape(3),
        )
        best_q = q.copy()
        best_error = float("inf")
        best_pos_error = float("inf")
        best_rot_error = float("inf")

        for iteration in range(self.max_iterations + 1):
            current = self._frame_placement(q)
            error_se3 = current.actInv(target)
            error = np.asarray(pin.log6(error_se3).vector, dtype=np.float64)
            position_error = float(np.linalg.norm(error[:3]))
            rotation_error = float(np.linalg.norm(error[3:]))
            total_error = float(np.linalg.norm(error))
            if total_error < best_error:
                best_q = q.copy()
                best_error = total_error
                best_pos_error = position_error
                best_rot_error = rotation_error
            if position_error <= self.position_threshold and rotation_error <= self.rotation_threshold:
                return q.astype(np.float32), {
                    "success": True,
                    "iterations": iteration,
                    "position_error": position_error,
                    "rotation_error": rotation_error,
                }

            jacobian = pin.computeFrameJacobian(self._model, self._data, q, self._frame_id, pin.ReferenceFrame.LOCAL)
            lhs = jacobian @ jacobian.T + self.damping * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(lhs, error)
            q = pin.integrate(self._model, q, self.step_size * delta)
            q = self._clip_to_position_limits(q)

        logger.debug(
            "Pinocchio IK failed: profile=%s pos_err=%.4f rot_err=%.4f total_err=%.4f",
            self.profile.name,
            best_pos_error,
            best_rot_error,
            best_error,
        )
        return None, {
            "success": False,
            "iterations": self.max_iterations,
            "position_error": best_pos_error,
            "rotation_error": best_rot_error,
        }

    def _frame_placement(self, q: np.ndarray) -> Any:
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacement(self._model, self._data, self._frame_id)
        return self._data.oMf[self._frame_id]

    def _clip_to_position_limits(self, q: np.ndarray) -> np.ndarray:
        lower = np.asarray(self._model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self._model.upperPositionLimit, dtype=np.float64)
        valid_lower = np.isfinite(lower) & (lower > -1e19)
        valid_upper = np.isfinite(upper) & (upper < 1e19)
        clipped = q.copy()
        clipped[valid_lower] = np.maximum(clipped[valid_lower], lower[valid_lower])
        clipped[valid_upper] = np.minimum(clipped[valid_upper], upper[valid_upper])
        return clipped
