# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for keypoint-IK trajectory generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


Vector3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]


@dataclass(frozen=True)
class KeypointSpec:
    """A named end-effector target relative to an anchor pose."""

    name: str
    anchor: str
    offset_xyz: Vector3 = (0.0, 0.0, 0.0)
    hold_steps: int = 0
    gripper: str | float = "open"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "KeypointSpec":
        return cls(
            name=str(data["name"]),
            anchor=str(data["anchor"]),
            offset_xyz=_vector3(data.get("offset_xyz", (0.0, 0.0, 0.0))),
            hold_steps=int(data.get("hold_steps", 0)),
            gripper=data.get("gripper", "open"),
        )


@dataclass(frozen=True)
class DatasetRunConfig:
    """Serializable config for one keypoint-IK dataset/debug run."""

    robot: str = "ur5e"
    task: str = "banana_in_bowl_task.py"
    camera_preset: str = "wrist_left_right"
    ik_backend: str = "auto"
    target_object: str = "banana"
    container_object: str = "bowl"
    output_dir: str = "/mlp_vepfs/share/swy/cosmos3-framework/keypoint_ik_debug"
    open_gripper: float = 0.0
    close_gripper: float = 1.0
    max_cartesian_step_m: float = 0.015
    start_hold_steps: int = 4
    post_release_hold_steps: int = 30
    keypoints: list[KeypointSpec] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DatasetRunConfig":
        gripper_cfg = data.get("gripper", {})
        keypoints = [KeypointSpec.from_mapping(item) for item in data.get("keypoints", [])]
        return cls(
            robot=str(data.get("robot", "ur5e")),
            task=str(data.get("task", "banana_in_bowl_task.py")),
            camera_preset=str(data.get("camera_preset", "wrist_left_right")),
            ik_backend=str(data.get("ik_backend", "auto")),
            target_object=str(data.get("target_object", "banana")),
            container_object=str(data.get("container_object", "bowl")),
            output_dir=str(data.get("output_dir", cls.output_dir)),
            open_gripper=float(gripper_cfg.get("open", data.get("open_gripper", 0.0))),
            close_gripper=float(gripper_cfg.get("close", data.get("close_gripper", 1.0))),
            max_cartesian_step_m=float(data.get("max_cartesian_step_m", 0.015)),
            start_hold_steps=int(data.get("start_hold_steps", 4)),
            post_release_hold_steps=int(data.get("post_release_hold_steps", 30)),
            keypoints=keypoints,
        )


@dataclass(frozen=True)
class ResolvedKeypoint:
    """A concrete EEF target in the robot/env frame."""

    name: str
    anchor: str
    position: np.ndarray
    quat_wxyz: np.ndarray
    gripper: float
    hold_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "anchor": self.anchor,
            "position": self.position.astype(float).tolist(),
            "quat_wxyz": self.quat_wxyz.astype(float).tolist(),
            "gripper": float(self.gripper),
            "hold_steps": int(self.hold_steps),
        }


@dataclass(frozen=True)
class ResolvedTrajectory:
    """Raw EEF action sequence plus the resolved waypoint list."""

    eef_actions: np.ndarray
    keypoints: list[ResolvedKeypoint]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "eef_action_shape": list(self.eef_actions.shape),
            "keypoints": [keypoint.to_dict() for keypoint in self.keypoints],
        }


@dataclass(frozen=True)
class EnvActionPlan:
    """Final env-native action sequence and IK diagnostics."""

    env_actions: np.ndarray
    arm_actions: np.ndarray
    raw_gripper: np.ndarray
    diagnostics: list[dict[str, Any]]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "env_action_shape": list(self.env_actions.shape),
            "arm_action_shape": list(self.arm_actions.shape),
            "raw_gripper_shape": list(self.raw_gripper.shape),
            "diagnostics": self.diagnostics,
        }


def load_dataset_run_config(path: str | Path) -> DatasetRunConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected config mapping in {path}")
    return DatasetRunConfig.from_mapping(data)


def config_to_dict(config: DatasetRunConfig) -> dict[str, Any]:
    data = asdict(config)
    data["keypoints"] = [asdict(keypoint) for keypoint in config.keypoints]
    return data


def _vector3(value: Any) -> Vector3:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return float(arr[0]), float(arr[1]), float(arr[2])
