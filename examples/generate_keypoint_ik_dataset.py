# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Generate or debug a UR5 keypoint-IK pick-and-place trajectory.

This first-stage tool focuses on finding one successful banana path before
scaling to object/background/lighting variants.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401  must be imported before isaaclab
import numpy as np
import torch
from isaaclab.app import AppLauncher

DEFAULT_OUTPUT_ROOT = "/mlp_vepfs/share/swy/cosmos3-framework/keypoint_ik_debug"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="banana_in_bowl_task.py", help="Task class, filename, or path to register.")
parser.add_argument("--task-dirs", nargs="+", default=None, help="Task subdirs to search during registration.")
parser.add_argument("--env", default=None, help="Registered env name. Default: first registered UR5 env.")
parser.add_argument("--target-object", default="banana", help="Object to grasp.")
parser.add_argument("--container-object", default="bowl", help="Object/container xy target for release.")
parser.add_argument("--object-a-label", default=None, help="Human-readable object A name for prompts and metadata.")
parser.add_argument("--object-b-label", default=None, help="Human-readable object B name for prompts and metadata.")
parser.add_argument("--camera-preset", default="wrist_left_right", help="UR5 camera preset.")
parser.add_argument("--ik-backend", choices=["auto", "curobo", "pinocchio"], default="auto")
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument("--output-dir", default=None)
parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--record-video",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Write one MP4 per policy camera. Defaults to --record.",
)
parser.add_argument(
    "--record-images-to-hdf5",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also store image observations inside HDF5. This is large; MP4 videos are preferred for LeRobot.",
)
parser.add_argument("--dry-run", action="store_true", help="Resolve and convert actions but do not step the env.")
parser.add_argument("--allow-ik-fallback", action="store_true")
parser.add_argument("--max-cartesian-step-m", type=float, default=0.015)
parser.add_argument("--action-repeat", type=int, default=1, help="Number of env steps to hold each generated action.")
parser.add_argument("--settle-steps", type=int, default=0, help="Open-gripper hold steps after reset before reading object poses.")
parser.add_argument("--start-hold-steps", type=int, default=4)
parser.add_argument("--post-release-hold-steps", type=int, default=30)
parser.add_argument(
    "--grasp-mode",
    choices=["explicit", "table_fixed"],
    default="explicit",
    help="Use explicit z offsets or table-relative dynamic object-height grasp points.",
)
parser.add_argument(
    "--grasp-clearance-policy",
    choices=["half_height_clamped"],
    default="half_height_clamped",
    help="Policy used by --grasp-mode table_fixed.",
)
parser.add_argument("--min-grasp-clearance-m", type=float, default=0.018)
parser.add_argument("--max-grasp-clearance-m", type=float, default=0.055)
parser.add_argument("--table-pre-grasp-clearance-m", type=float, default=0.12)
parser.add_argument("--table-lift-clearance-m", type=float, default=0.24)
parser.add_argument("--table-release-clearance-m", type=float, default=0.12)
parser.add_argument(
    "--table-tool0-tcp-offset-mode",
    choices=["none", "robotiq_2f85_open_pad_center"],
    default="robotiq_2f85_open_pad_center",
    help="For table_fixed, interpret table clearances as TCP/pad-center z and convert them back to tool0 z.",
)
parser.add_argument(
    "--table-tcp-collision-margin-m",
    type=float,
    default=0.005,
    help="Extra z margin above the table for the Robotiq pad collision box in table_fixed mode.",
)
parser.add_argument("--pre-grasp-z", type=float, default=0.18)
parser.add_argument("--grasp-z", type=float, default=0.045)
parser.add_argument("--lift-z", type=float, default=0.24)
parser.add_argument("--release-z", type=float, default=0.18)
parser.add_argument(
    "--success-mode",
    choices=["env_termination", "near_target_xy"],
    default="env_termination",
    help="Use env termination or post-hoc final object XY distance as success.",
)
parser.add_argument("--place-xy-tolerance-m", type=float, default=0.10)
parser.add_argument("--min-final-z-clearance-m", type=float, default=0.001)
parser.add_argument(
    "--prompt-template",
    default="pick up the {target_object} and place it near the {container_object}",
)
parser.add_argument(
    "--target-xy-offset",
    nargs=2,
    type=float,
    default=(0.0, 0.0),
    metavar=("DX", "DY"),
    help="XY offset from target object centroid for grasp keypoints.",
)
parser.add_argument(
    "--release-xy-offset",
    nargs=2,
    type=float,
    default=(0.0, 0.0),
    metavar=("DX", "DY"),
    help="XY offset from container centroid for release keypoints.",
)
parser.add_argument(
    "--grasp-quat-wxyz",
    nargs=4,
    type=float,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Fixed EEF orientation. Default: reset ee_quat.",
)
parser.add_argument(
    "--grasp-yaw-deg",
    type=float,
    default=0.0,
    help="World-Z yaw offset, in degrees, applied after selecting the base grasp quaternion.",
)
parser.add_argument("--background-random", action="store_true", help="Use UR5 background randomization registration.")
parser.add_argument("--background-seed", type=int, default=0)
parser.add_argument(
    "--table-material",
    default=None,
    help="Optional material name under /world/Looks to bind to the tabletop, e.g. Walnut_Planks or Bamboo.",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, get_timestamp, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.motion.eef import normalize_quat_wxyz  # noqa: E402
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.data_gen.keypoint_ik.executor import (  # noqa: E402
    build_eef_observation_from_env_obs,
    convert_trajectory_with_bridge,
    create_ur5_ik_bridge,
    tensor_to_numpy,
)
from robolab.data_gen.keypoint_ik.recording import write_json, write_npz  # noqa: E402
from robolab.data_gen.keypoint_ik.schemas import KeypointSpec  # noqa: E402
from robolab.data_gen.keypoint_ik.tcp_offsets import (  # noqa: E402
    ROBOTIQ_2F85_OPEN_PAD_CENTER_FROM_TOOL0_XYZ_M,
    robotiq_2f85_open_pad_half_extent_world_z,
    table_tool0_z_from_tcp_center_z,
    tool0_offset_world_z,
)
from robolab.data_gen.keypoint_ik.trajectory import build_eef_action_trajectory, resolve_keypoints  # noqa: E402
from robolab.registrations.ur5.auto_env_registrations_jointpos import auto_register_ur5_envs  # noqa: E402
from robolab.registrations.ur5.camera_presets import get_camera_preset  # noqa: E402
from robolab.robots.ur5_profile import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES, get_ur5_eef_profile  # noqa: E402

VIDEO_CAMERA_KEYS = ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
TABLE_TCP_OFFSET_NONE = "none"
TABLE_TCP_OFFSET_ROBOTIQ_OPEN_PAD_CENTER = "robotiq_2f85_open_pad_center"


def _output_dir() -> Path:
    if args_cli.output_dir:
        return Path(args_cli.output_dir)
    return Path(DEFAULT_OUTPUT_ROOT) / f"{get_timestamp()}_ur5_keypoint_{args_cli.target_object}_to_{args_cli.container_object}"


def _record_video_enabled() -> bool:
    return bool(args_cli.record if args_cli.record_video is None else args_cli.record_video)


def _keypoints(
    *,
    pre_grasp_z: float,
    grasp_z: float,
    target_lift_z: float,
    container_lift_z: float,
    release_z: float,
) -> list[KeypointSpec]:
    tx, ty = args_cli.target_xy_offset
    rx, ry = args_cli.release_xy_offset
    return [
        KeypointSpec("pre_grasp", "target_object", (tx, ty, pre_grasp_z), 8, "open"),
        KeypointSpec("grasp", "target_object", (tx, ty, grasp_z), 8, "open"),
        KeypointSpec("close", "target_object", (tx, ty, grasp_z), 12, "close"),
        KeypointSpec("lift", "target_object", (tx, ty, target_lift_z), 8, "close"),
        KeypointSpec("pre_release", "container_object", (rx, ry, container_lift_z), 8, "close"),
        KeypointSpec("release", "container_object", (rx, ry, release_z), 12, "open"),
    ]


def _select_env() -> str:
    if args_cli.env:
        return get_envs(env=args_cli.env)[0]
    envs = get_envs()
    if not envs:
        raise RuntimeError("No UR5 environments registered.")
    return envs[0]


def _pose_xyz(world, name: str) -> np.ndarray:
    _, centroid = world.get_bbox(name, env_id=0)
    return np.asarray(centroid, dtype=np.float32).reshape(3)


def _bbox_corners_xyz(world, name: str) -> np.ndarray:
    corners, _ = world.get_bbox(name, env_id=0)
    return np.asarray([[float(corner[0]), float(corner[1]), float(corner[2])] for corner in corners], dtype=np.float32)


def _bbox_height(world, name: str) -> float:
    corners = _bbox_corners_xyz(world, name)
    return float(corners[:, 2].max() - corners[:, 2].min())


def _table_top_z(world) -> float:
    corners = _bbox_corners_xyz(world, "table")
    return float(corners[:, 2].max())


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _table_tool0_to_tcp_offset_xyz() -> np.ndarray:
    mode = args_cli.table_tool0_tcp_offset_mode
    if mode == TABLE_TCP_OFFSET_NONE:
        return np.zeros(3, dtype=np.float32)
    if mode == TABLE_TCP_OFFSET_ROBOTIQ_OPEN_PAD_CENTER:
        return ROBOTIQ_2F85_OPEN_PAD_CENTER_FROM_TOOL0_XYZ_M.copy()
    raise ValueError(f"Unsupported table tool0 TCP offset mode {mode!r}")


def _table_tcp_collision_half_extent_world_z(tool0_quat_wxyz: np.ndarray) -> float:
    if args_cli.table_tool0_tcp_offset_mode == TABLE_TCP_OFFSET_NONE:
        return 0.0
    return robotiq_2f85_open_pad_half_extent_world_z(tool0_quat_wxyz)


def _resolve_keypoint_heights(world, *, tool0_quat_wxyz: np.ndarray | None = None) -> dict[str, object]:
    table_top_z = _table_top_z(world)
    target_height = _bbox_height(world, args_cli.target_object)
    if args_cli.grasp_mode == "explicit":
        return {
            "grasp_mode": "explicit",
            "grasp_clearance_policy": "none",
            "table_top_z": table_top_z,
            "target_height": target_height,
            "grasp_clearance_m": None,
            "object_grasp_clearance_m": None,
            "tcp_safe_min_grasp_clearance_m": None,
            "tool0_tcp_offset_policy": TABLE_TCP_OFFSET_NONE,
            "tool0_to_tcp_offset_xyz_m": [0.0, 0.0, 0.0],
            "tool0_to_tcp_world_z_m": 0.0,
            "tcp_collision_half_extent_world_z_m": 0.0,
            "tcp_collision_margin_m": 0.0,
            "pre_grasp_z": float(args_cli.pre_grasp_z),
            "grasp_z": float(args_cli.grasp_z),
            "target_lift_z": float(args_cli.lift_z),
            "container_lift_z": float(args_cli.lift_z),
            "release_z": float(args_cli.release_z),
            "absolute_keypoint_z": None,
        }

    if tool0_quat_wxyz is None:
        raise ValueError("table_fixed height resolution requires the target tool0 quaternion")

    tool0_quat = normalize_quat_wxyz(np.asarray(tool0_quat_wxyz, dtype=np.float32).reshape(4))
    target_z = float(_pose_xyz(world, args_cli.target_object)[2])
    container_z = float(_pose_xyz(world, args_cli.container_object)[2])
    object_grasp_clearance = _clamp(
        target_height * 0.5,
        args_cli.min_grasp_clearance_m,
        args_cli.max_grasp_clearance_m,
    )
    tool0_to_tcp = _table_tool0_to_tcp_offset_xyz()
    tool0_to_tcp_world_z = tool0_offset_world_z(tool0_quat, tool0_to_tcp)
    tcp_collision_half_extent_world_z = _table_tcp_collision_half_extent_world_z(tool0_quat)
    tcp_collision_margin = 0.0
    if args_cli.table_tool0_tcp_offset_mode != TABLE_TCP_OFFSET_NONE:
        tcp_collision_margin = max(0.0, float(args_cli.table_tcp_collision_margin_m))
    tcp_safe_min_grasp_clearance = tcp_collision_half_extent_world_z + tcp_collision_margin
    grasp_clearance = max(object_grasp_clearance, tcp_safe_min_grasp_clearance)

    pre_tcp_abs = table_top_z + float(args_cli.table_pre_grasp_clearance_m)
    grasp_tcp_abs = table_top_z + grasp_clearance
    lift_tcp_abs = table_top_z + float(args_cli.table_lift_clearance_m)
    release_tcp_abs = table_top_z + float(args_cli.table_release_clearance_m)
    pre_tool0_abs = table_tool0_z_from_tcp_center_z(
        pre_tcp_abs,
        tool0_quat_wxyz=tool0_quat,
        tool0_to_tcp_xyz=tool0_to_tcp,
    )
    grasp_tool0_abs = table_tool0_z_from_tcp_center_z(
        grasp_tcp_abs,
        tool0_quat_wxyz=tool0_quat,
        tool0_to_tcp_xyz=tool0_to_tcp,
    )
    lift_tool0_abs = table_tool0_z_from_tcp_center_z(
        lift_tcp_abs,
        tool0_quat_wxyz=tool0_quat,
        tool0_to_tcp_xyz=tool0_to_tcp,
    )
    release_tool0_abs = table_tool0_z_from_tcp_center_z(
        release_tcp_abs,
        tool0_quat_wxyz=tool0_quat,
        tool0_to_tcp_xyz=tool0_to_tcp,
    )
    return {
        "grasp_mode": "table_fixed",
        "grasp_clearance_policy": args_cli.grasp_clearance_policy,
        "table_top_z": table_top_z,
        "target_height": target_height,
        "grasp_clearance_m": grasp_clearance,
        "object_grasp_clearance_m": object_grasp_clearance,
        "tcp_safe_min_grasp_clearance_m": tcp_safe_min_grasp_clearance,
        "tool0_tcp_offset_policy": args_cli.table_tool0_tcp_offset_mode,
        "tool0_to_tcp_offset_xyz_m": tool0_to_tcp.astype(float).tolist(),
        "tool0_to_tcp_world_z_m": tool0_to_tcp_world_z,
        "tcp_collision_half_extent_world_z_m": tcp_collision_half_extent_world_z,
        "tcp_collision_margin_m": tcp_collision_margin,
        "pre_grasp_z": pre_tool0_abs - target_z,
        "grasp_z": grasp_tool0_abs - target_z,
        "target_lift_z": lift_tool0_abs - target_z,
        "container_lift_z": lift_tool0_abs - container_z,
        "release_z": release_tool0_abs - container_z,
        "absolute_keypoint_z": {
            "pre_grasp_z": pre_tool0_abs,
            "grasp_z": grasp_tool0_abs,
            "lift_z": lift_tool0_abs,
            "release_z": release_tool0_abs,
            "pre_grasp_tool0_z": pre_tool0_abs,
            "grasp_tool0_z": grasp_tool0_abs,
            "lift_tool0_z": lift_tool0_abs,
            "release_tool0_z": release_tool0_abs,
            "pre_grasp_tcp_center_z": pre_tcp_abs,
            "grasp_tcp_center_z": grasp_tcp_abs,
            "lift_tcp_center_z": lift_tcp_abs,
            "release_tcp_center_z": release_tcp_abs,
            "target_centroid_z": target_z,
            "container_centroid_z": container_z,
        },
    }


def _format_object_name(name: str) -> str:
    return name.replace("_", " ")


def _format_prompt() -> str:
    object_a = args_cli.object_a_label or args_cli.target_object
    object_b = args_cli.object_b_label or args_cli.container_object
    return args_cli.prompt_template.format(
        target_object=_format_object_name(object_a),
        container_object=_format_object_name(object_b),
        object_a=_format_object_name(object_a),
        object_b=_format_object_name(object_b),
    )


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float32).reshape(4)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float32).reshape(4)
    return normalize_quat_wxyz(
        np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float32,
        )
    )


def _apply_world_yaw(quat_wxyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    base = normalize_quat_wxyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))
    if abs(float(yaw_deg)) < 1e-6:
        return base
    half = math.radians(float(yaw_deg)) * 0.5
    yaw = np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)
    return _quat_multiply_wxyz(yaw, base)


def _change_table_material(material_name: str | None) -> bool:
    if not material_name:
        return False
    try:
        import omni.usd
        from pxr import UsdShade
    except Exception as exc:
        print(f"[keypoint-ik] table material skipped; USD APIs unavailable: {exc}", flush=True)
        return False

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[keypoint-ik] table material skipped; no USD stage", flush=True)
        return False

    table_top_prim = None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath()).lower()
        prim_name = prim.GetName().lower()
        if prim_name == "top" and "table" in prim_path and "franka" not in prim_path:
            table_top_prim = prim
            break
    if table_top_prim is None:
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath()).lower()
            if "table" in prim_path and "franka" not in prim_path and prim.GetTypeName() in {"Cube", "Mesh"}:
                table_top_prim = prim
                break
    if table_top_prim is None:
        print("[keypoint-ik] table material skipped; no table mesh found", flush=True)
        return False

    material_prim = None
    for path in (f"/Root/Looks/{material_name}", f"/World/Looks/{material_name}", f"/world/Looks/{material_name}"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            material_prim = prim
            break
    if material_prim is None:
        for prim in stage.Traverse():
            if prim.GetName() == material_name and prim.GetTypeName() == "Material":
                material_prim = prim
                break
    if material_prim is None:
        print(f"[keypoint-ik] table material skipped; material {material_name!r} not found", flush=True)
        return False

    UsdShade.MaterialBindingAPI(table_top_prim).Bind(
        UsdShade.Material(material_prim),
        UsdShade.Tokens.weakerThanDescendants,
    )
    print(f"[keypoint-ik] table_material={material_name}", flush=True)
    return True


def _finger_pad_positions(env) -> dict[str, list[float]]:
    robot = env.scene["robot"]
    result = {}
    for body_name in ("left_inner_finger_pad", "right_inner_finger_pad"):
        if body_name not in robot.data.body_names:
            continue
        body_idx = robot.data.body_names.index(body_name)
        pos = robot.data.body_pos_w[0, body_idx, :] - env.scene.env_origins[0]
        result[body_name] = tensor_to_numpy(pos).astype(float).tolist()
    return result


def _trace_row(obs: dict, env, world, *, step: int, phase: str, target_object: str, container_object: str) -> dict:
    proprio = obs.get("proprio_obs", {})
    gripper = proprio.get("gripper_pos")
    arm_joint_pos = proprio.get("arm_joint_pos")
    return {
        "step": int(step),
        "phase": phase,
        "ee_pos": tensor_to_numpy(proprio["ee_pos"][0]).astype(float).tolist(),
        "ee_quat_wxyz": tensor_to_numpy(proprio["ee_quat"][0]).astype(float).tolist(),
        "arm_joint_pos": None if arm_joint_pos is None else tensor_to_numpy(arm_joint_pos[0]).astype(float).tolist(),
        "gripper_pos": None if gripper is None else tensor_to_numpy(gripper[0]).astype(float).tolist(),
        "finger_pads": _finger_pad_positions(env),
        "target_xyz": _pose_xyz(world, target_object).astype(float).tolist(),
        "container_xyz": _pose_xyz(world, container_object).astype(float).tolist(),
    }


def _step_actions(
    env,
    env_actions: np.ndarray,
    *,
    action_repeat: int,
    post_release_hold_steps: int,
    world,
    target_object: str,
    container_object: str,
    stop_on_env_success: bool = True,
    video_writers: dict[str, cv2.VideoWriter] | None = None,
) -> tuple[dict, bool, bool, list[dict]]:
    obs = {}
    env_success = False
    truncated = False
    trace: list[dict] = []
    sim_step = 0
    for action_index, action_np in enumerate(env_actions):
        action = torch.as_tensor(action_np, dtype=torch.float32, device=env.device).view(1, -1)
        action = action.repeat(env.num_envs, 1)
        for repeat_index in range(max(1, int(action_repeat))):
            obs, _, term, trunc, _ = env.step(action)
            _write_video_frames(video_writers, obs, env_id=0)
            row = _trace_row(
                obs,
                env,
                world,
                step=sim_step,
                phase="action",
                target_object=target_object,
                container_object=container_object,
            )
            row["action_index"] = int(action_index)
            row["repeat_index"] = int(repeat_index)
            row["joint_action"] = np.asarray(action_np, dtype=np.float32).astype(float).tolist()
            trace.append(row)
            if bool(term[0].item()):
                env_success = True
                print(f"[keypoint-ik] episode terminated successfully at sim step {sim_step}", flush=True)
                if stop_on_env_success:
                    return obs, env_success, truncated, trace
            if bool(trunc[0].item()):
                truncated = True
                print(f"[keypoint-ik] episode truncated at sim step {sim_step}", flush=True)
                return obs, env_success, truncated, trace
            sim_step += 1

    if post_release_hold_steps > 0:
        last = torch.as_tensor(env_actions[-1], dtype=torch.float32, device=env.device).view(1, -1)
        last = last.repeat(env.num_envs, 1)
        for step in range(post_release_hold_steps):
            obs, _, term, trunc, _ = env.step(last)
            _write_video_frames(video_writers, obs, env_id=0)
            row = _trace_row(
                obs,
                env,
                world,
                step=sim_step,
                phase="post_hold",
                target_object=target_object,
                container_object=container_object,
            )
            row["joint_action"] = np.asarray(env_actions[-1], dtype=np.float32).astype(float).tolist()
            trace.append(row)
            if bool(term[0].item()):
                env_success = True
                print(f"[keypoint-ik] episode terminated successfully during post-release hold {step}", flush=True)
                if stop_on_env_success:
                    break
            if bool(trunc[0].item()):
                truncated = True
                print(f"[keypoint-ik] episode truncated during post-release hold {step}", flush=True)
                break
            sim_step += 1
    return obs, env_success, truncated, trace


def _near_target_xy_success(
    *,
    final_target_xyz: np.ndarray,
    target_container_xyz: np.ndarray,
    table_top_z: float,
    trace: list[dict],
    truncated: bool,
) -> dict:
    final_target_xyz = np.asarray(final_target_xyz, dtype=np.float32).reshape(3)
    target_container_xyz = np.asarray(target_container_xyz, dtype=np.float32).reshape(3)
    final_xy_distance = float(np.linalg.norm(final_target_xyz[:2] - target_container_xyz[:2]))
    min_final_z = float(table_top_z + args_cli.min_final_z_clearance_m)
    final_z_ok = bool(float(final_target_xyz[2]) >= min_final_z)
    gripper_pos = None
    if trace:
        gripper_pos = trace[-1].get("gripper_pos")
    gripper_open = True
    if gripper_pos is not None:
        gripper_open = bool(float(np.asarray(gripper_pos).reshape(-1)[0]) <= 0.35)
    success = (
        final_xy_distance <= float(args_cli.place_xy_tolerance_m)
        and final_z_ok
        and gripper_open
        and not bool(truncated)
    )
    return {
        "mode": "near_target_xy",
        "success": bool(success),
        "final_xy_distance": final_xy_distance,
        "place_xy_tolerance_m": float(args_cli.place_xy_tolerance_m),
        "final_z": float(final_target_xyz[2]),
        "min_final_z": min_final_z,
        "final_z_ok": final_z_ok,
        "gripper_open": gripper_open,
        "truncated": bool(truncated),
    }


def _settle_after_reset(
    env,
    obs: dict,
    settle_steps: int,
    *,
    video_writers: dict[str, cv2.VideoWriter] | None = None,
) -> dict:
    if settle_steps <= 0:
        return obs
    arm = obs["proprio_obs"]["arm_joint_pos"]
    open_gripper = torch.zeros((env.num_envs, 1), dtype=arm.dtype, device=arm.device)
    action = torch.cat([arm, open_gripper], dim=-1)
    for _ in range(int(settle_steps)):
        obs, _, term, trunc, _ = env.step(action)
        _write_video_frames(video_writers, obs, env_id=0)
        if bool(term[0].item()) or bool(trunc[0].item()):
            break
    return obs


def _open_video_writers(out_dir: Path, episode_index: int, obs: dict, *, fps: float = 15.0) -> dict[str, cv2.VideoWriter]:
    writers: dict[str, cv2.VideoWriter] = {}
    image_obs = obs.get("image_obs", {})
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for camera in VIDEO_CAMERA_KEYS:
        if camera not in image_obs:
            continue
        frame = _camera_frame_bgr(obs, camera, env_id=0)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        video_path = out_dir / f"keypoint_ik_{episode_index}__{camera}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[keypoint-ik] failed to open video writer for {video_path}", flush=True)
            continue
        writers[camera] = writer
    return writers


def _write_video_frames(writers: dict[str, cv2.VideoWriter] | None, obs: dict, *, env_id: int) -> None:
    if not writers:
        return
    for camera, writer in writers.items():
        frame = _camera_frame_bgr(obs, camera, env_id=env_id)
        if frame is not None:
            writer.write(frame)


def _close_video_writers(writers: dict[str, cv2.VideoWriter] | None) -> None:
    if not writers:
        return
    for writer in writers.values():
        writer.release()


def _camera_frame_bgr(obs: dict, camera: str, *, env_id: int) -> np.ndarray | None:
    image_obs = obs.get("image_obs", {})
    if camera not in image_obs:
        return None
    frame = tensor_to_numpy(image_obs[camera][env_id])
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3][..., ::-1])


def _camera_calibration_from_env_cfg(out_dir: Path) -> dict:
    env_cfg_path = out_dir / "env_cfg.json"
    if not env_cfg_path.exists():
        return {}
    try:
        cfg = json.loads(env_cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    result = {}
    scene = cfg.get("scene", {})
    for camera in VIDEO_CAMERA_KEYS:
        value = scene.get(camera)
        if not isinstance(value, dict):
            continue
        offset = value.get("offset", {}) if isinstance(value.get("offset"), dict) else {}
        spawn = value.get("spawn", {}) if isinstance(value.get("spawn"), dict) else {}
        prim_path = str(value.get("prim_path", ""))
        entry = {
            "width": value.get("width"),
            "height": value.get("height"),
            "data_types": value.get("data_types", ["rgb"]),
            "intrinsics": {
                "projection_type": spawn.get("projection_type"),
                "focal_length": spawn.get("focal_length"),
                "focus_distance": spawn.get("focus_distance"),
                "horizontal_aperture": spawn.get("horizontal_aperture"),
                "vertical_aperture": spawn.get("vertical_aperture"),
                "clipping_range": spawn.get("clipping_range"),
            },
        }
        transform = {
            "pos": offset.get("pos"),
            "quat_wxyz": offset.get("rot"),
            "convention": offset.get("convention"),
        }
        if camera == "wrist_cam":
            entry["parent_link"] = "tool0" if "/tool0/" in prim_path else "robot"
            entry["extrinsics_link_from_camera"] = transform
        else:
            entry["extrinsics_world_from_camera"] = transform
        result[camera] = entry
    return result


def _patch_hdf5_episode_metadata(
    out_dir: Path,
    episode_index: int,
    *,
    attrs: dict,
    execution_trace: list[dict],
) -> None:
    if not args_cli.record:
        return
    hdf5_path = out_dir / "data.hdf5"
    if not hdf5_path.exists():
        return
    try:
        import h5py

        with h5py.File(hdf5_path, "a") as f:
            demo = f.get(f"data/demo_{episode_index}")
            if demo is None:
                return
            for key, value in attrs.items():
                if value is not None:
                    demo.attrs[key] = value
            _ensure_hdf5_action_descriptions(demo, execution_trace)
    except OSError as exc:
        print(f"[keypoint-ik] failed to patch hdf5 attrs: {exc}", flush=True)


def _ensure_hdf5_action_descriptions(demo, execution_trace: list[dict]) -> None:
    import h5py

    actions = demo.get("actions")
    if actions is None:
        return
    frame_count = int(actions.shape[0])
    joint_actions = np.asarray(actions, dtype=np.float32)

    ee_group = demo.get("ee_pose")
    if ee_group is None or "position" not in ee_group or "orientation" not in ee_group:
        position, orientation = _trace_pose_arrays(execution_trace, frame_count)
        ee_group = demo.require_group("ee_pose")
        _write_or_replace_dataset(ee_group, "position", position)
        _write_or_replace_dataset(ee_group, "orientation", orientation)

    ee_position = _frame_matrix(np.asarray(ee_group["position"], dtype=np.float32), frame_count=frame_count, width=3)
    ee_orientation = _frame_matrix(np.asarray(ee_group["orientation"], dtype=np.float32), frame_count=frame_count, width=4)
    tool0_pose = None
    if ee_position is not None and ee_orientation is not None:
        common = min(len(ee_position), len(ee_orientation), frame_count)
        tool0_pose = np.concatenate([ee_position[:common], ee_orientation[:common]], axis=1).astype(np.float32)

    joint_positions = None
    joint_dataset = demo.get("states/articulation/robot/joint_position")
    if joint_dataset is not None:
        joint_positions = _frame_matrix(np.asarray(joint_dataset, dtype=np.float32), frame_count=frame_count)

    desc_group = demo.require_group("action_descriptions")
    desc_group.attrs["frame_convention"] = "env-local for position, wxyz quaternion for tool0"
    desc_group.attrs["joint_action_names"] = json.dumps([*ARM_JOINT_NAMES, "gripper_close"])
    desc_group.attrs["arm_joint_names"] = json.dumps(ARM_JOINT_NAMES)
    desc_group.attrs["gripper_joint_names"] = json.dumps(GRIPPER_JOINT_NAMES)
    _write_or_replace_dataset(desc_group, "joint_action", joint_actions)
    if joint_positions is not None:
        _write_or_replace_dataset(desc_group, "joint_position", joint_positions)
    if ee_position is not None:
        _write_or_replace_dataset(desc_group, "tool0_position", ee_position)
    if ee_orientation is not None:
        _write_or_replace_dataset(desc_group, "tool0_quat_wxyz", ee_orientation)
    if tool0_pose is not None:
        _write_or_replace_dataset(desc_group, "tool0_pose_wxyz", tool0_pose)
    demo.attrs["action_description_fields"] = json.dumps(sorted(desc_group.keys()))


def _trace_pose_arrays(execution_trace: list[dict], frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([row.get("ee_pos", [0.0, 0.0, 0.0]) for row in execution_trace], dtype=np.float32)
    orientations = np.asarray([row.get("ee_quat_wxyz", [1.0, 0.0, 0.0, 0.0]) for row in execution_trace], dtype=np.float32)
    if positions.size == 0:
        positions = np.zeros((0, 3), dtype=np.float32)
        orientations = np.zeros((0, 4), dtype=np.float32)
    if len(positions) < frame_count:
        pad_count = frame_count - len(positions)
        pad_pos = np.repeat((positions[:1] if len(positions) else np.zeros((1, 3), dtype=np.float32)), pad_count, axis=0)
        pad_quat = np.repeat(
            (orientations[:1] if len(orientations) else np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)),
            pad_count,
            axis=0,
        )
        positions = np.concatenate([pad_pos, positions], axis=0)
        orientations = np.concatenate([pad_quat, orientations], axis=0)
    elif len(positions) > frame_count:
        positions = positions[-frame_count:]
        orientations = orientations[-frame_count:]
    return positions.astype(np.float32), orientations.astype(np.float32)


def _frame_matrix(value: np.ndarray, *, frame_count: int, width: int | None = None) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim == 1 and width is not None and arr.size % width == 0:
        arr = arr.reshape((-1, width))
    if arr.ndim != 2:
        return None
    if width is not None and arr.shape[1] != width:
        return None
    if len(arr) > frame_count:
        arr = arr[:frame_count]
    return arr.astype(np.float32, copy=False)


def _write_or_replace_dataset(group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.asarray(data, dtype=np.float32))


def main() -> None:
    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(out_dir))
    robolab.constants.RECORD_IMAGE_DATA = bool(args_cli.record and args_cli.record_images_to_hdf5)
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True

    task_dirs = args_cli.task_dirs or DEFAULT_TASK_SUBFOLDERS
    auto_register_ur5_envs(
        task_dirs=task_dirs,
        task=args_cli.task,
        cameras=get_camera_preset(args_cli.camera_preset),
        randomize_background=args_cli.background_random,
        background_seed=args_cli.background_seed,
    )
    env_name = _select_env()
    print(f"[keypoint-ik] output_dir={out_dir}", flush=True)
    print(f"[keypoint-ik] env={env_name}", flush=True)

    profile = get_ur5_eef_profile()
    bridge, resolved_backend = create_ur5_ik_bridge(backend=args_cli.ik_backend, device=args_cli.device)
    print(f"[keypoint-ik] ik_backend={resolved_backend}", flush=True)

    env, env_cfg = create_env(env_name, device=args_cli.device, num_envs=1, use_fabric=True)
    table_material_applied = _change_table_material(args_cli.table_material)
    try:
        for episode_index in range(args_cli.num_episodes):
            if args_cli.record and env.recorder_manager is not None:
                if hasattr(env.recorder_manager, "set_hdf5_file"):
                    env.recorder_manager.set_hdf5_file("data.hdf5")
                if hasattr(env.recorder_manager, "set_episode_index"):
                    env.recorder_manager.set_episode_index(episode_index, env_ids=[0])

            obs, _ = env.reset()
            video_writers = _open_video_writers(out_dir, episode_index, obs) if _record_video_enabled() else None
            obs = _settle_after_reset(env, obs, args_cli.settle_steps, video_writers=video_writers)
            world = get_world(env)
            target_xyz = _pose_xyz(world, args_cli.target_object)
            container_xyz = _pose_xyz(world, args_cli.container_object)
            eef_observation = build_eef_observation_from_env_obs(obs, profile, env_id=0)
            grasp_quat = (
                np.asarray(args_cli.grasp_quat_wxyz, dtype=np.float32)
                if args_cli.grasp_quat_wxyz is not None
                else eef_observation.ee_quat_wxyz
            )
            grasp_quat = _apply_world_yaw(grasp_quat, args_cli.grasp_yaw_deg)
            height_cfg = _resolve_keypoint_heights(world, tool0_quat_wxyz=grasp_quat)
            anchors = {
                "target_object": target_xyz,
                "container_object": container_xyz,
            }
            resolved_keypoints = resolve_keypoints(
                _keypoints(
                    pre_grasp_z=float(height_cfg["pre_grasp_z"]),
                    grasp_z=float(height_cfg["grasp_z"]),
                    target_lift_z=float(height_cfg["target_lift_z"]),
                    container_lift_z=float(height_cfg["container_lift_z"]),
                    release_z=float(height_cfg["release_z"]),
                ),
                anchors_xyz=anchors,
                quat_wxyz=grasp_quat,
                open_gripper=0.0,
                close_gripper=1.0,
            )
            trajectory = build_eef_action_trajectory(
                start_pos=eef_observation.ee_pos,
                start_quat_wxyz=eef_observation.ee_quat_wxyz,
                start_gripper=float(
                    0.0
                    if eef_observation.gripper_position is None
                    else np.asarray(eef_observation.gripper_position).reshape(-1)[0]
                ),
                keypoints=resolved_keypoints,
                max_cartesian_step_m=args_cli.max_cartesian_step_m,
                start_hold_steps=args_cli.start_hold_steps,
            )
            action_plan = convert_trajectory_with_bridge(
                bridge=bridge,
                observation=eef_observation,
                eef_actions=trajectory.eef_actions,
                allow_ik_fallback=args_cli.allow_ik_fallback,
            )
            episode_dir = out_dir / f"episode_{episode_index:06d}"
            write_npz(
                episode_dir / "actions_debug.npz",
                eef_actions=trajectory.eef_actions,
                arm_actions=action_plan.arm_actions,
                env_actions=action_plan.env_actions,
                raw_gripper=action_plan.raw_gripper,
            )
            metadata = {
                "env_name": env_name,
                "instruction": env_cfg.instruction,
                "robot_body_names": list(env.scene["robot"].data.body_names),
                "camera_preset": args_cli.camera_preset,
                "ik_backend": resolved_backend,
                "background_random": bool(args_cli.background_random),
                "background_seed": int(args_cli.background_seed),
                "table_material": args_cli.table_material,
                "table_material_applied": bool(table_material_applied),
                "record": bool(args_cli.record),
                "record_video": bool(_record_video_enabled()),
                "record_images_to_hdf5": bool(args_cli.record and args_cli.record_images_to_hdf5),
                "video_camera_keys": list(VIDEO_CAMERA_KEYS),
                "prompt": _format_prompt(),
                "object_a": args_cli.target_object,
                "object_b": args_cli.container_object,
                "object_a_label": args_cli.object_a_label or _format_object_name(args_cli.target_object),
                "object_b_label": args_cli.object_b_label or _format_object_name(args_cli.container_object),
                "target_object": args_cli.target_object,
                "container_object": args_cli.container_object,
                "target_xyz": target_xyz.tolist(),
                "container_xyz": container_xyz.tolist(),
                "grasp_mode": height_cfg["grasp_mode"],
                "grasp_clearance_policy": height_cfg["grasp_clearance_policy"],
                "table_top_z": height_cfg["table_top_z"],
                "target_height": height_cfg["target_height"],
                "computed_clearances": {
                    "grasp_clearance_m": height_cfg["grasp_clearance_m"],
                    "pre_grasp_z": height_cfg["pre_grasp_z"],
                    "grasp_z": height_cfg["grasp_z"],
                    "target_lift_z": height_cfg["target_lift_z"],
                    "container_lift_z": height_cfg["container_lift_z"],
                    "release_z": height_cfg["release_z"],
                    "absolute_keypoint_z": height_cfg["absolute_keypoint_z"],
                    "object_grasp_clearance_m": height_cfg.get("object_grasp_clearance_m"),
                    "tcp_safe_min_grasp_clearance_m": height_cfg.get("tcp_safe_min_grasp_clearance_m"),
                    "tool0_tcp_offset_policy": height_cfg.get("tool0_tcp_offset_policy"),
                    "tool0_to_tcp_offset_xyz_m": height_cfg.get("tool0_to_tcp_offset_xyz_m"),
                    "tool0_to_tcp_world_z_m": height_cfg.get("tool0_to_tcp_world_z_m"),
                    "tcp_collision_half_extent_world_z_m": height_cfg.get("tcp_collision_half_extent_world_z_m"),
                    "tcp_collision_margin_m": height_cfg.get("tcp_collision_margin_m"),
                },
                "success_mode": args_cli.success_mode,
                "place_xy_tolerance_m": float(args_cli.place_xy_tolerance_m),
                "start_ee_pos": eef_observation.ee_pos.tolist(),
                "start_ee_quat_wxyz": eef_observation.ee_quat_wxyz.tolist(),
                "grasp_quat_wxyz": np.asarray(grasp_quat, dtype=np.float32).tolist(),
                "grasp_yaw_deg": float(args_cli.grasp_yaw_deg),
                "settle_steps": int(args_cli.settle_steps),
                "trajectory": trajectory.to_metadata(),
                "action_plan": action_plan.to_metadata(),
                "action_repeat": args_cli.action_repeat,
                "action_layout": {
                    "type": "ur5_joint_position_with_binary_gripper",
                    "arm_joint_names": ARM_JOINT_NAMES,
                    "gripper_joint_names": GRIPPER_JOINT_NAMES,
                    "columns": [*ARM_JOINT_NAMES, "gripper_close"],
                },
            }
            if not args_cli.dry_run:
                obs, env_success, truncated, execution_trace = _step_actions(
                    env,
                    action_plan.env_actions,
                    action_repeat=args_cli.action_repeat,
                    post_release_hold_steps=args_cli.post_release_hold_steps,
                    world=world,
                    target_object=args_cli.target_object,
                    container_object=args_cli.container_object,
                    stop_on_env_success=args_cli.success_mode == "env_termination",
                    video_writers=video_writers,
                )
                final_target_xyz = _pose_xyz(world, args_cli.target_object)
                final_container_xyz = _pose_xyz(world, args_cli.container_object)
                if args_cli.success_mode == "near_target_xy":
                    success_checks = _near_target_xy_success(
                        final_target_xyz=final_target_xyz,
                        target_container_xyz=container_xyz,
                        table_top_z=float(height_cfg["table_top_z"]),
                        trace=execution_trace,
                        truncated=truncated,
                    )
                    success = bool(success_checks["success"])
                else:
                    success_checks = {"mode": "env_termination", "success": bool(env_success), "truncated": bool(truncated)}
                    success = bool(env_success)
                metadata["success"] = bool(success)
                metadata["env_termination_success"] = bool(env_success)
                metadata["truncated"] = bool(truncated)
                metadata["success_checks"] = success_checks
                metadata["execution_trace"] = execution_trace
                metadata["final_target_xyz"] = final_target_xyz.tolist()
                metadata["final_container_xyz"] = final_container_xyz.tolist()
                if args_cli.record:
                    end_episode(env)
                    _patch_hdf5_episode_metadata(
                        out_dir,
                        episode_index,
                        attrs={
                            "success": bool(success),
                            "prompt": metadata["prompt"],
                            "object_a": args_cli.target_object,
                            "object_b": args_cli.container_object,
                            "object_a_label": metadata["object_a_label"],
                            "object_b_label": metadata["object_b_label"],
                            "success_mode": args_cli.success_mode,
                        },
                        execution_trace=execution_trace,
                    )
            else:
                metadata["success"] = None
                metadata["dry_run"] = True
            metadata["camera_calibration"] = _camera_calibration_from_env_cfg(out_dir)
            write_json(episode_dir / "metadata.json", metadata)
            _close_video_writers(video_writers)
            print(
                f"[keypoint-ik] episode={episode_index} "
                f"eef={trajectory.eef_actions.shape} env_actions={action_plan.env_actions.shape} "
                f"success={metadata['success']}",
                flush=True,
            )
            del obs
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[keypoint-ik] terminated with error: {exc}", flush=True)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
