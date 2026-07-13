# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers before exporting keypoint-IK runs to LeRobot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py


REQUIRED_CAMERA_KEYS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)


def find_robolab_hdf5_files(root: str | Path) -> list[Path]:
    """Find RoboLab HDF5 files below a raw dataset directory."""

    root = Path(root)
    if not root.exists():
        return []
    files = []
    files.extend(root.rglob("data.hdf5"))
    files.extend(root.rglob("run_*.hdf5"))
    return sorted(set(files))


def validate_keypoint_hdf5(
    hdf5_path: str | Path,
    *,
    required_cameras: tuple[str, ...] = REQUIRED_CAMERA_KEYS,
    action_width: int = 7,
    require_hdf5_cameras: bool = False,
    require_videos: bool = True,
) -> dict[str, Any]:
    """Validate one RoboLab HDF5 file against the keypoint-IK data contract."""

    hdf5_path = Path(hdf5_path)
    errors: list[str] = []
    demos: list[dict[str, Any]] = []

    try:
        hdf5_file = h5py.File(hdf5_path, "r")
    except OSError as exc:
        return {
            "path": str(hdf5_path),
            "ok": False,
            "errors": [f"cannot open hdf5: {exc}"],
            "demos": [],
        }

    with hdf5_file as f:
        if "data" not in f:
            return {"path": str(hdf5_path), "ok": False, "errors": ["missing data group"], "demos": []}
        data = f["data"]
        demo_names = sorted((name for name in data if name.startswith("demo_")), key=lambda x: int(x.split("_")[1]))
        if not demo_names:
            errors.append("no demo_* groups")
        for demo_name in demo_names:
            demo = data[demo_name]
            demo_errors: list[str] = []
            actions_shape = _dataset_shape(demo, "actions")
            frame_count = int(actions_shape[0]) if actions_shape else 0
            if not actions_shape:
                demo_errors.append("missing actions")
            elif len(actions_shape) != 2 or actions_shape[1] != action_width:
                demo_errors.append(f"actions shape {actions_shape} does not match (*, {action_width})")

            joint_shape = _dataset_shape(demo, "states/articulation/robot/joint_position")
            if not joint_shape:
                demo_errors.append("missing states/articulation/robot/joint_position")
            elif frame_count and joint_shape[0] != frame_count:
                demo_errors.append(f"joint_position frames {joint_shape[0]} != actions frames {frame_count}")

            ee_pos_shape = _dataset_shape(demo, "ee_pose/position")
            ee_quat_shape = _dataset_shape(demo, "ee_pose/orientation")
            if not ee_pos_shape:
                demo_errors.append("missing ee_pose/position")
            elif frame_count and ee_pos_shape[0] != frame_count:
                demo_errors.append(f"ee_pose/position frames {ee_pos_shape[0]} != actions frames {frame_count}")
            if not ee_quat_shape:
                demo_errors.append("missing ee_pose/orientation")
            elif frame_count and ee_quat_shape[0] != frame_count:
                demo_errors.append(f"ee_pose/orientation frames {ee_quat_shape[0]} != actions frames {frame_count}")

            action_joint_shape = _dataset_shape(demo, "action_descriptions/joint_position")
            tool0_pose_shape = _dataset_shape(demo, "action_descriptions/tool0_pose_wxyz")
            if not action_joint_shape:
                demo_errors.append("missing action_descriptions/joint_position")
            elif frame_count and action_joint_shape[0] != frame_count:
                demo_errors.append(f"action_descriptions/joint_position frames {action_joint_shape[0]} != actions frames {frame_count}")
            if not tool0_pose_shape:
                demo_errors.append("missing action_descriptions/tool0_pose_wxyz")
            elif frame_count and tool0_pose_shape[0] != frame_count:
                demo_errors.append(f"action_descriptions/tool0_pose_wxyz frames {tool0_pose_shape[0]} != actions frames {frame_count}")

            obs_group = demo.get("obs")
            camera_shapes: dict[str, tuple[int, ...] | None] = {}
            if obs_group is None:
                if require_hdf5_cameras:
                    demo_errors.append("missing obs group")
            else:
                for camera in required_cameras:
                    shape = _dataset_shape(obs_group, camera)
                    camera_shapes[camera] = shape
                    if not shape and require_hdf5_cameras:
                        demo_errors.append(f"missing obs/{camera}")
                    elif frame_count and shape[0] != frame_count:
                        demo_errors.append(f"obs/{camera} frames {shape[0]} != actions frames {frame_count}")

            demo_num = int(demo_name.split("_")[1])
            video_paths = _find_demo_videos(hdf5_path.parent, demo_num, required_cameras)
            if require_videos:
                for camera, path in video_paths.items():
                    if path is None:
                        demo_errors.append(f"missing video for {camera}")

            demo_report = {
                "demo": demo_name,
                "ok": not demo_errors,
                "errors": demo_errors,
                "num_frames": frame_count,
                "actions_shape": actions_shape,
                "joint_position_shape": joint_shape,
                "ee_position_shape": ee_pos_shape,
                "ee_orientation_shape": ee_quat_shape,
                "action_joint_position_shape": action_joint_shape,
                "action_tool0_pose_shape": tool0_pose_shape,
                "camera_shapes": camera_shapes,
                "video_paths": {key: None if path is None else str(path) for key, path in video_paths.items()},
                "success": bool(demo.attrs.get("success", False)),
            }
            demos.append(demo_report)
            errors.extend(f"{demo_name}: {error}" for error in demo_errors)

    return {"path": str(hdf5_path), "ok": not errors, "errors": errors, "demos": demos}


def validate_keypoint_dataset(
    root: str | Path,
    *,
    require_hdf5_cameras: bool = False,
    require_videos: bool = True,
) -> dict[str, Any]:
    """Validate every RoboLab HDF5 file under a raw dataset directory."""

    files = find_robolab_hdf5_files(root)
    reports = [
        validate_keypoint_hdf5(
            path,
            require_hdf5_cameras=require_hdf5_cameras,
            require_videos=require_videos,
        )
        for path in files
    ]
    total_demos = sum(len(report["demos"]) for report in reports)
    ok_demos = sum(1 for report in reports for demo in report["demos"] if demo["ok"])
    successful_demos = sum(1 for report in reports for demo in report["demos"] if demo["success"])
    return {
        "root": str(Path(root)),
        "ok": bool(files) and all(report["ok"] for report in reports),
        "hdf5_count": len(files),
        "demo_count": total_demos,
        "ok_demo_count": ok_demos,
        "successful_demo_count": successful_demos,
        "reports": reports,
    }


def _dataset_shape(group: h5py.Group, key: str) -> tuple[int, ...] | None:
    value = group.get(key)
    if value is None:
        return None
    return tuple(int(dim) for dim in value.shape)


def _find_demo_videos(task_dir: Path, demo_num: int, cameras: tuple[str, ...]) -> dict[str, Path | None]:
    result: dict[str, Path | None] = {}
    for camera in cameras:
        matches = sorted(task_dir.glob(f"*_{demo_num}__{camera}.mp4"))
        result[camera] = matches[0] if matches else None
    return result
