# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Save initial UR5 camera frames in the same layouts used by Cosmos3 UR5 eval.

This is a server-free dry-run for debugging camera poses. It registers a UR5
task with the requested camera preset, resets the env, holds the reset arm
pose for a short settle window, and saves one capture:

* the Cosmos3 UR5 ``observation/image`` canvas that would be sent to the server;
* the image_obs mosaic used by eval sensor videos such as ``*_0.mp4``;
* individual registered camera frames for easier inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2  # noqa: F401  must be imported before isaaclab
import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.app import AppLauncher
from PIL import Image


CAMERA_PRESET_CHOICES = [
    "berkeley_eef",
    "robomind_single",
    "left",
    "right",
    "left_right",
    "wrist",
    "wrist_left",
    "wrist_left_right",
    "left_right_head",
    "wrist_left_right_head",
]


parser = argparse.ArgumentParser(
    description="Save initial UR5 camera mosaics for Cosmos3 server-client debugging."
)
parser.add_argument(
    "--task",
    nargs="+",
    default=["banana_in_bowl_task.py"],
    help="Task class, task filename, or task path to register. Default: banana_in_bowl_task.py",
)
parser.add_argument(
    "--task-dirs",
    "--task_dirs",
    nargs="+",
    default=None,
    help="Task subdirectories to search. Default: robolab.constants.DEFAULT_TASK_SUBFOLDERS.",
)
parser.add_argument("--env", default=None, help="Registered env name to capture. Default: first registered env.")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environment copies.")
parser.add_argument("--env-id", "--env_id", type=int, default=0, help="Environment index to save.")
parser.add_argument(
    "--camera-preset",
    choices=CAMERA_PRESET_CHOICES,
    default="berkeley_eef",
    help=(
        "UR5 camera preset used during registration. Use robomind_single with "
        "--initial-pose-preset robomind_ur5. Default: berkeley_eef."
    ),
)
parser.add_argument(
    "--initial-pose-preset",
    default="default",
    help=(
        "UR5 reset geometry: default preserves standard RoboLab home; "
        "rh20t_ur5 applies a RoboLab-safe cfg4-aligned arm and closed-gripper reset; "
        "robomind_ur5 applies the verified RoboMIND joint pose and base yaw."
    ),
)
parser.add_argument(
    "--primary-image-key",
    default="over_shoulder_left_camera",
    help="Primary image_obs key used as the Cosmos exterior camera.",
)
parser.add_argument(
    "--wrist-image-key",
    default="wrist_cam",
    help="Wrist image_obs key. Missing wrist images are represented as zeros, matching Cosmos3UR5Client.",
)
parser.add_argument(
    "--secondary-image-key",
    default="over_shoulder_right_camera",
    help="Secondary image_obs key used for debug mosaics.",
)
parser.add_argument(
    "--server-right-tile",
    choices=["zero", "observation"],
    default="zero",
    help=(
        "Right-bottom tile in the saved Cosmos server canvas. "
        "'zero' matches Cosmos3UR5Client; 'observation' is useful when tuning a right camera."
    ),
)
parser.add_argument(
    "--server-left-tile",
    choices=["zero", "observation"],
    default="observation",
    help=(
        "Left-bottom tile in the saved Cosmos server canvas. 'observation' matches the "
        "Berkeley/DROID contracts; 'zero' matches single-camera contracts such as robomind_single, "
        "whose canvas is the top camera over two black tiles."
    ),
)
parser.add_argument("--cosmos-image-height", type=int, default=360, help="Cosmos policy image height.")
parser.add_argument("--cosmos-image-width", type=int, default=640, help="Cosmos policy image width.")
parser.add_argument(
    "--video-scale",
    type=float,
    default=0.5,
    help="Scale used for the *_0.mp4-style image_obs mosaic. Default matches eval episode sensor videos.",
)
parser.add_argument(
    "--settle-seconds",
    "--settle_seconds",
    type=float,
    default=10.0,
    help="Hold the reset joint pose for this many seconds before saving the single capture.",
)
parser.add_argument(
    "--gripper",
    type=float,
    default=None,
    help="Held gripper command during optional settle steps. Default preserves reset gripper observation.",
)
parser.add_argument(
    "--output-dir",
    "--output_dir",
    default=None,
    help="Directory for saved images. Default: output/ur5_cosmos_initial_cameras/<timestamp>_<env>.",
)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, PACKAGE_DIR, get_timestamp, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_viewport_cams  # noqa: E402
from robolab.registrations.ur5.auto_env_registrations_jointpos import auto_register_ur5_envs  # noqa: E402
from robolab.registrations.ur5.camera_presets import get_camera_preset  # noqa: E402
from robolab.registrations.ur5.initial_pose_presets import (  # noqa: E402
    get_ur5_initial_pose_preset,
)
from robolab.robots.ur5 import ARM_JOINT_NAMES  # noqa: E402


_RESIZE_BACKEND = "torch_bilinear_stretch (matches training canvas_utils.resize_view and Cosmos3Client)"


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value).strip("_") or "capture"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _resize_full_view(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Plain bilinear stretch to the contract view size, matching Cosmos3Client.

    Training resizes every source view with canvas_utils.resize_view (a straight
    F.interpolate with no letterboxing), so non-16:9 cameras such as the RoboMIND
    4:3 top camera must be stretched, not padded, to reproduce the trained frames.
    """
    image = _as_uint8_rgb(image)
    if image.shape[:2] == (height, width):
        return image
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear")
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8, copy=False)


def _resize_for_canvas(image: np.ndarray, *, size: tuple[int, int], dtype: np.dtype) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(_as_uint8_rgb(image))).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=size, mode="bilinear")
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(dtype, copy=False)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_uint8_rgb(image)).save(path)


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    image = _as_uint8_rgb(image)
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    resized = Image.fromarray(image).resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _compose_image_obs_mosaic(image_obs: dict[str, np.ndarray], *, scale: float) -> np.ndarray | None:
    if not image_obs:
        return None
    target_height = max(1, int(round(max(image.shape[0] for image in image_obs.values()) * scale)))
    tiles = [_resize_to_height(image, target_height) for image in image_obs.values()]
    return np.concatenate(tiles, axis=1)


def _extract_images(obs: dict, group_name: str, *, env_id: int) -> dict[str, np.ndarray]:
    group = obs.get(group_name, {})
    return {key: _as_uint8_rgb(value[env_id]) for key, value in group.items()}


def _zeros_like_reference(reference: np.ndarray) -> np.ndarray:
    return np.zeros_like(_as_uint8_rgb(reference))


def _get_optional_image(images: dict[str, np.ndarray], key: str, reference: np.ndarray) -> np.ndarray:
    image = images.get(key)
    if image is None:
        return _zeros_like_reference(reference)
    return image


def _compose_cosmos_ur5_server_canvas(
    *,
    left_image: np.ndarray,
    wrist_image: np.ndarray,
    right_image: np.ndarray,
    left_tile: str,
    right_tile: str,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    left = _resize_full_view(left_image, image_h, image_w)
    wrist = _resize_full_view(wrist_image, image_h, image_w)
    right = _resize_full_view(right_image, image_h, image_w)

    tile_size = (image_h // 2, image_w // 2)
    if left_tile == "zero":
        left_tile_image = np.zeros((*tile_size, 3), dtype=wrist.dtype)
    else:
        left_tile_image = _resize_for_canvas(left, size=tile_size, dtype=wrist.dtype)
    if right_tile == "zero":
        right_tile_image = np.zeros_like(left_tile_image)
    else:
        right_tile_image = _resize_for_canvas(right, size=tile_size, dtype=wrist.dtype)
    return np.concatenate((wrist, np.concatenate((left_tile_image, right_tile_image), axis=1)), axis=0)


def _compose_client_viz(
    *,
    left_image: np.ndarray,
    wrist_image: np.ndarray,
    right_image: np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    left = _resize_full_view(left_image, image_h, image_w)
    wrist = _resize_full_view(wrist_image, image_h, image_w)
    right = _resize_full_view(right_image, image_h, image_w)
    return np.concatenate((left, wrist, right), axis=1)


def _hold_action(
    env,
    *,
    arm_joint_pos: torch.Tensor,
    gripper_pos: torch.Tensor,
    gripper_override: float | None,
) -> torch.Tensor:
    arm_action = arm_joint_pos.to(device=env.device).clone()
    if arm_action.ndim == 1:
        arm_action = arm_action.unsqueeze(0).repeat(env.num_envs, 1)
    gripper_action = gripper_pos.to(device=env.device).clone().reshape(env.num_envs, 1)
    if gripper_override is not None:
        gripper_action = torch.full((env.num_envs, 1), gripper_override, device=env.device)
    return torch.cat([arm_action, gripper_action], dim=1)


def _make_output_dir(env_name: str) -> Path:
    if args_cli.output_dir:
        return Path(args_cli.output_dir).expanduser().resolve()
    name = f"{get_timestamp()}_{_safe_name(env_name)}"
    return Path(PACKAGE_DIR, "output", "ur5_cosmos_initial_cameras", name)


def _select_env() -> str:
    envs = get_envs()
    if not envs:
        raise RuntimeError(f"No UR5 environments were registered for task(s): {args_cli.task}")
    if args_cli.env is None:
        return envs[0]
    if args_cli.env not in envs:
        raise ValueError(f"Requested env {args_cli.env!r} is not registered. Available: {envs}")
    return args_cli.env


def _save_capture(obs: dict, env_cfg, output_dir: Path, *, capture_stage: str, effective_settle_steps: int) -> dict:
    env_id = args_cli.env_id
    image_obs = _extract_images(obs, "image_obs", env_id=env_id)
    viewport_obs = _extract_images(obs, "viewport_cam", env_id=env_id)
    proprio_obs = {
        key: _to_numpy(value[env_id]).tolist() for key, value in obs.get("proprio_obs", {}).items()
    }
    if args_cli.primary_image_key not in image_obs:
        raise KeyError(
            f"Primary image key {args_cli.primary_image_key!r} is missing from image_obs. "
            f"Registered image keys: {sorted(image_obs)}. Choose a camera preset containing the primary camera."
        )

    left_raw = image_obs[args_cli.primary_image_key]
    wrist_raw = _get_optional_image(image_obs, args_cli.wrist_image_key, left_raw)
    right_raw = _get_optional_image(image_obs, args_cli.secondary_image_key, left_raw)

    paths: dict[str, str] = {}
    server_canvas = _compose_cosmos_ur5_server_canvas(
        left_image=left_raw,
        wrist_image=wrist_raw,
        right_image=right_raw,
        left_tile=args_cli.server_left_tile,
        right_tile=args_cli.server_right_tile,
        image_h=args_cli.cosmos_image_height,
        image_w=args_cli.cosmos_image_width,
    )
    server_path = output_dir / "cosmos3_ur5_server_observation_image.png"
    _save_png(server_path, server_canvas)
    paths["cosmos3_ur5_server_observation_image"] = str(server_path)

    client_viz = _compose_client_viz(
        left_image=left_raw,
        wrist_image=wrist_raw,
        right_image=right_raw,
        image_h=args_cli.cosmos_image_height,
        image_w=args_cli.cosmos_image_width,
    )
    client_viz_path = output_dir / "cosmos3_ur5_client_viz_left_wrist_right.png"
    _save_png(client_viz_path, client_viz)
    paths["cosmos3_ur5_client_viz_left_wrist_right"] = str(client_viz_path)

    sensor_video_frame = _compose_image_obs_mosaic(image_obs, scale=args_cli.video_scale)
    if sensor_video_frame is not None:
        sensor_path = output_dir / "sensor_video_like_0_mp4_image_obs_combined.png"
        _save_png(sensor_path, sensor_video_frame)
        paths["sensor_video_like_0_mp4_image_obs_combined"] = str(sensor_path)

    viewport_frame = unpack_viewport_cams(obs, env_id=env_id).get("combined_image") if viewport_obs else None
    if viewport_frame is not None:
        viewport_path = output_dir / "viewport_video_like_0_viewport_mp4.png"
        _save_png(viewport_path, viewport_frame)
        paths["viewport_video_like_0_viewport_mp4"] = str(viewport_path)

    individual_dir = output_dir / "individual_cameras"
    individual_paths: dict[str, str] = {}
    for key, image in image_obs.items():
        path = individual_dir / f"{_safe_name(key)}.png"
        _save_png(path, image)
        individual_paths[key] = str(path)

    viewport_paths: dict[str, str] = {}
    for key, image in viewport_obs.items():
        path = individual_dir / f"viewport_{_safe_name(key)}.png"
        _save_png(path, image)
        viewport_paths[key] = str(path)

    metadata = {
        "env_name": args_cli.env,
        "instruction": getattr(env_cfg, "instruction", None),
        "camera_preset": args_cli.camera_preset,
        "initial_pose_preset": args_cli.initial_pose_preset,
        "task": args_cli.task,
        "task_dirs": args_cli.task_dirs or DEFAULT_TASK_SUBFOLDERS,
        "env_id": env_id,
        "num_envs": args_cli.num_envs,
        "settle_seconds": args_cli.settle_seconds,
        "effective_settle_steps": effective_settle_steps,
        "capture_stage": capture_stage,
        "cosmos_image_height": args_cli.cosmos_image_height,
        "cosmos_image_width": args_cli.cosmos_image_width,
        "video_scale": args_cli.video_scale,
        "resize_backend": _RESIZE_BACKEND,
        "server_canvas": {
            "path": str(server_path),
            "shape": list(server_canvas.shape),
            "primary_image_key": args_cli.primary_image_key,
            "wrist_image_key": args_cli.wrist_image_key,
            "secondary_image_key": args_cli.secondary_image_key,
            "left_tile": args_cli.server_left_tile,
            "right_tile": args_cli.server_right_tile,
            "matches_cosmos3_ur5_client": args_cli.server_right_tile == "zero",
        },
        "client_viz": {"path": str(client_viz_path), "shape": list(client_viz.shape)},
        "sensor_video_like_0_mp4": {
            "path": paths.get("sensor_video_like_0_mp4_image_obs_combined"),
            "shape": list(sensor_video_frame.shape) if sensor_video_frame is not None else None,
            "image_obs_key_order": list(image_obs.keys()),
        },
        "viewport_video_like_0_viewport_mp4": {
            "path": paths.get("viewport_video_like_0_viewport_mp4"),
            "shape": list(viewport_frame.shape) if viewport_frame is not None else None,
            "viewport_key_order": list(viewport_obs.keys()),
        },
        "individual_image_obs": individual_paths,
        "individual_viewport_cam": viewport_paths,
        "proprio_obs": proprio_obs,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    paths["metadata"] = str(metadata_path)
    return {"paths": paths, "metadata": metadata}


def main() -> None:
    if args_cli.env_id < 0 or args_cli.env_id >= args_cli.num_envs:
        raise ValueError(f"--env-id must be in [0, {args_cli.num_envs - 1}], got {args_cli.env_id}")
    if args_cli.video_scale <= 0.0:
        raise ValueError(f"--video-scale must be > 0, got {args_cli.video_scale}")
    if args_cli.settle_seconds < 0.0:
        raise ValueError(f"--settle-seconds must be >= 0, got {args_cli.settle_seconds}")

    task_dirs = args_cli.task_dirs or DEFAULT_TASK_SUBFOLDERS
    initial_pose_preset = get_ur5_initial_pose_preset(args_cli.initial_pose_preset)
    auto_register_ur5_envs(
        task_dirs=task_dirs,
        task=args_cli.task,
        cameras=get_camera_preset(args_cli.camera_preset),
        initial_arm_joint_positions=initial_pose_preset.arm_joint_positions,
        initial_gripper_close_fraction=initial_pose_preset.gripper_close_fraction,
        initial_root_rot_wxyz=initial_pose_preset.root_rot_wxyz,
    )
    env_name = _select_env()
    args_cli.env = env_name

    output_dir = _make_output_dir(env_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(output_dir))

    print(f"[RoboLab] Creating env: {env_name}", flush=True)
    print(f"[RoboLab] Saving camera debug images to: {output_dir}", flush=True)
    env, env_cfg = create_env(env_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    try:
        obs, _ = env.reset()
        env_step_dt = float(env_cfg.sim.render_interval * env_cfg.sim.dt)
        effective_settle_steps = int(math.ceil(args_cli.settle_seconds / env_step_dt))

        if effective_settle_steps > 0:
            arm_joint_pos = obs["proprio_obs"]["arm_joint_pos"].clone()
            gripper_pos = obs["proprio_obs"]["gripper_pos"].clone()
            print(
                f"[RoboLab] Holding reset pose for {args_cli.settle_seconds:g}s "
                f"({effective_settle_steps} env steps) before capture.",
                flush=True,
            )
            for _ in range(effective_settle_steps):
                obs, _, _, _, _ = env.step(
                    _hold_action(
                        env,
                        arm_joint_pos=arm_joint_pos,
                        gripper_pos=gripper_pos,
                        gripper_override=args_cli.gripper,
                    )
                )

        result = _save_capture(
            obs,
            env_cfg,
            output_dir,
            capture_stage="after_hold_settle",
            effective_settle_steps=effective_settle_steps,
        )
        print(f"[RoboLab] Instruction: {env_cfg.instruction}", flush=True)
        print(f"[RoboLab] Saved after-hold capture ({effective_settle_steps} steps):", flush=True)
        for label, path in result["paths"].items():
            print(f"  - {label}: {path}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[RoboLab] Terminated with error: {exc}", flush=True)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
