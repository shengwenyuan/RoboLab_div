# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuRobo EEF-to-joint bridge.

The bridge is policy-agnostic: it consumes the common EEF action chunk from
``robolab.core.motion.eef`` and emits the concrete joint-position action chunk
expected by a RoboLab environment.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import yaml

from robolab.constants import PACKAGE_DIR
from robolab.core.motion.eef import (
    EEFActionChunk,
    EEFPolicyObservation,
    MotionBridgeResult,
    RobotEEFProfile,
)

logger = logging.getLogger(__name__)


class CuRoboDependencyError(ImportError):
    """Raised when the optional cuRobo backend is requested but unavailable."""


class CuRoboWorldProvider(Protocol):
    """Build a cuRobo-compatible world representation."""

    def get_world_dict(self) -> Mapping[str, Any] | None:
        """Return a world dict compatible with ``WorldConfig.from_dict``."""


@dataclass(frozen=True)
class NullWorldProvider:
    """A minimal static world used when scene collision parsing is not enabled."""

    def get_world_dict(self) -> Mapping[str, Any]:
        # A tiny obstacle far from the workspace keeps world parsing exercised
        # without adding meaningful scene constraints.
        return {
            "cuboid": {
                "robolab_far_dummy": {
                    "dims": [0.001, 0.001, 0.001],
                    "pose": [100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
                }
            },
            "mesh": {},
        }


@dataclass(frozen=True)
class _CuRoboModules:
    IKSolver: Any
    IKSolverConfig: Any
    Pose: Any
    RobotConfig: Any
    TensorDeviceType: Any
    WorldConfig: Any | None
    torch: Any


def get_default_ur5e_robot_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "configs", "ur5e_robolab.yml")


def _load_curobo_modules() -> _CuRoboModules:
    if importlib.util.find_spec("curobo") is None:
        raise CuRoboDependencyError(
            "cuRobo is not installed in the current Python environment. "
            "Install cuRobo in the IsaacLab Python environment before using CuRoboIKBridge."
        )

    try:
        import torch
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    except ImportError as exc:
        raise CuRoboDependencyError(
            "Failed to import cuRobo IK modules. Check that cuRobo and its compiled dependencies "
            "are installed in the active IsaacLab Python environment."
        ) from exc

    return _CuRoboModules(
        IKSolver=IKSolver,
        IKSolverConfig=IKSolverConfig,
        Pose=Pose,
        RobotConfig=RobotConfig,
        TensorDeviceType=TensorDeviceType,
        WorldConfig=WorldConfig,
        torch=torch,
    )


def _expand_config_paths(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("{ROBOLAB_PACKAGE_DIR}", PACKAGE_DIR)
    if isinstance(value, list):
        return [_expand_config_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_config_paths(v) for k, v in value.items()}
    return value


def _load_robot_cfg(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "robot_cfg" not in data:
        raise ValueError(f"cuRobo robot config must contain top-level 'robot_cfg': {path}")
    return _expand_config_paths(copy.deepcopy(data["robot_cfg"]))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class CuRoboIKBridge:
    """Convert absolute EEF pose chunks into joint-position chunks with cuRobo IK."""

    def __init__(
        self,
        *,
        profile: RobotEEFProfile,
        robot_config_path: str | None = None,
        device: str = "cuda:0",
        num_seeds: int = 20,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        use_cuda_graph: bool = False,
        self_collision_check: bool = True,
        self_collision_opt: bool = True,
        world_provider: CuRoboWorldProvider | None = None,
    ) -> None:
        self.profile = profile
        self.robot_config_path = robot_config_path or get_default_ur5e_robot_config_path()
        self.device = device
        self.num_seeds = num_seeds
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.use_cuda_graph = use_cuda_graph
        self.self_collision_check = self_collision_check
        self.self_collision_opt = self_collision_opt
        self.world_provider = world_provider or NullWorldProvider()
        self._last_solution: dict[int, np.ndarray] = {}
        self._last_action: dict[int, np.ndarray] = {}

        self._modules = _load_curobo_modules()
        self._tensor_args = self._modules.TensorDeviceType(device=device)
        self._solver = self._build_solver()

    def _build_solver(self) -> Any:
        robot_cfg_dict = _load_robot_cfg(self.robot_config_path)
        try:
            robot_cfg = self._modules.RobotConfig.from_dict(robot_cfg_dict, self._tensor_args)
        except TypeError:
            robot_cfg = self._modules.RobotConfig.from_dict(robot_cfg_dict)

        world_dict = self.world_provider.get_world_dict()
        if world_dict is not None and self._modules.WorldConfig is not None:
            world_cfg = self._modules.WorldConfig.from_dict(dict(world_dict))
        else:
            world_cfg = world_dict

        ik_config = self._modules.IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=self.rotation_threshold,
            position_threshold=self.position_threshold,
            num_seeds=self.num_seeds,
            self_collision_check=self.self_collision_check,
            self_collision_opt=self.self_collision_opt,
            tensor_args=self._tensor_args,
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._modules.IKSolver(ik_config)

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._last_solution.clear()
            self._last_action.clear()
        else:
            self._last_solution.pop(env_id, None)
            self._last_action.pop(env_id, None)

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
        torch = self._modules.torch
        tensor_kwargs = self._tensor_args.as_torch_dict()
        pos_t = torch.as_tensor(target_pos, **tensor_kwargs).view(1, 3)
        quat_t = torch.as_tensor(target_quat_wxyz, **tensor_kwargs).view(1, 4)
        seed_t = torch.as_tensor(seed, **tensor_kwargs).view(1, 1, -1)
        retract_t = torch.as_tensor(seed, **tensor_kwargs).view(1, -1)

        goal = self._modules.Pose(pos_t, quat_t)
        result = self._solver.solve_single(
            goal,
            retract_config=retract_t,
            seed_config=seed_t,
            num_seeds=self.num_seeds,
            use_nn_seed=False,
        )

        success = bool(_tensor_to_numpy(result.success).reshape(-1)[0])
        diag = {
            "success": success,
            "position_error": _safe_first_float(getattr(result, "position_error", None)),
            "rotation_error": _safe_first_float(getattr(result, "rotation_error", None)),
            "solve_time": _safe_first_float(getattr(result, "solve_time", None)),
        }
        if not success:
            return None, diag

        solution = getattr(result, "solution", None)
        if solution is None and getattr(result, "js_solution", None) is not None:
            solution = getattr(result.js_solution, "position", None)
        if solution is None:
            logger.warning("cuRobo IK reported success but did not return a solution")
            diag["success"] = False
            diag["error"] = "missing_solution"
            return None, diag

        sol_np = _tensor_to_numpy(solution).astype(np.float32)
        sol_np = sol_np.reshape(-1, self.profile.env_action_dim)[0]
        return sol_np, diag


def _safe_first_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = _tensor_to_numpy(value).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])
