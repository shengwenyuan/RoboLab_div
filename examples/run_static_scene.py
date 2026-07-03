# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Launch one RoboLab task scene and hold the robot still for live viewing."""

import argparse
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch a single RoboLab scene and hold it still.")
parser.add_argument(
    "--task",
    default="banana_in_bowl_task.py",
    help="Task class, task filename, or task path to register. Default: banana_in_bowl_task.py",
)
parser.add_argument(
    "--env",
    default=None,
    help="Registered env name to run. Default: first environment registered from --task.",
)
parser.add_argument(
    "--robot",
    choices=["droid", "ur5e"],
    default="droid",
    help="Robot to launch in the task scene. Default: droid.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environment copies.")
parser.add_argument(
    "--num-steps",
    type=int,
    default=-1,
    help="Number of simulation steps to run. Use -1 to run until the app exits.",
)
parser.add_argument(
    "--gripper",
    type=float,
    default=0.0,
    help="Held gripper command: 0.0=open, >0.5=closed. Default: 0.0.",
)
parser.add_argument(
    "--background",
    choices=["none", "home_office"],
    default="none",
    help="Dome background. 'none' avoids downloading the large home_office.exr LFS asset.",
)
parser.add_argument("--print-every", type=int, default=120, help="Print a heartbeat every N steps. Use 0 to disable.")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR  # noqa: E402
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa: E402
from robolab.robots.droid import (  # noqa: E402
    DroidCfg,
    DroidJointPositionActionCfg,
    ProprioceptionObservationCfg,
    WristCameraCfg,
    contact_gripper,
)
from robolab.robots.ur5 import (  # noqa: E402
    ARM_JOINT_NAMES as UR5_ARM_JOINT_NAMES,
)
from robolab.robots.ur5 import (  # noqa: E402
    ProprioceptionObservationCfg as UR5ProprioceptionObservationCfg,
)
from robolab.robots.ur5 import (  # noqa: E402
    UR5eCfg,
    UR5eJointPositionActionCfg,
    contact_gripper as ur5e_contact_gripper,
)
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg  # noqa: E402
from robolab.variations.camera import EgocentricMirroredCameraCfg, OverShoulderLeftCameraCfg  # noqa: E402
from robolab.variations.lighting import SphereLightCfg  # noqa: E402


@configclass
class PlainBackgroundCfg:
    dome_light = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            visible_in_primary_ray=True,
        ),
    )


def register_single_task(task: str) -> None:
    if args_cli.robot == "droid":
        image_cameras = WRIST_LEFT
        proprio_obs_cfg = ProprioceptionObservationCfg
        scene_cameras = [camera for camera in WRIST_LEFT if camera is not WristCameraCfg]
        actions_cfg = DroidJointPositionActionCfg
        robot_cfg = DroidCfg
        task_contact_gripper = contact_gripper
        env_postfix = ""
    else:
        image_cameras = [OverShoulderLeftCameraCfg]
        proprio_obs_cfg = UR5ProprioceptionObservationCfg
        scene_cameras = image_cameras
        actions_cfg = UR5eJointPositionActionCfg
        robot_cfg = UR5eCfg
        task_contact_gripper = ur5e_contact_gripper
        env_postfix = "UR5eJointPosition"

    image_obs_cfg = generate_image_obs_from_cameras(image_cameras)
    viewport_obs_cfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    observation_cfg = generate_obs_cfg(
        {
            "image_obs": image_obs_cfg(),
            "proprio_obs": proprio_obs_cfg(),
            "viewport_cam": viewport_obs_cfg(),
        }
    )
    background_cfg = HomeOfficeBackgroundCfg if args_cli.background == "home_office" else PlainBackgroundCfg

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=DEFAULT_TASK_SUBFOLDERS,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix=env_postfix,
        observations_cfg=observation_cfg(),
        actions_cfg=actions_cfg(),
        robot_cfg=robot_cfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=background_cfg,
        contact_gripper=task_contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )


def main() -> None:
    register_single_task(args_cli.task)
    envs = get_envs()
    if not envs:
        raise RuntimeError(f"No environments were registered for task '{args_cli.task}'.")

    env_name = args_cli.env or envs[0]
    if env_name not in envs:
        raise ValueError(f"Requested env '{env_name}' is not registered. Available: {envs}")

    print(f"[RoboLab] Registered envs: {envs}", flush=True)
    print(f"[RoboLab] Creating env: {env_name}", flush=True)
    env, env_cfg = create_env(env_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    try:
        obs, _ = env.reset()
        del obs
        robot = env.scene["robot"]
        print(f"[RoboLab] Static scene ready: {env_name}", flush=True)
        print(f"[RoboLab] Instruction: {env_cfg.instruction}", flush=True)
        print("[RoboLab] Holding current joint positions. Stop with Ctrl+C or close the viewer.", flush=True)

        step = 0
        while simulation_app.is_running() and (args_cli.num_steps < 0 or step < args_cli.num_steps):
            if args_cli.robot == "droid":
                arm_action = robot.data.joint_pos[:, :7].clone()
                gripper_action = torch.full((env.num_envs, 1), args_cli.gripper, device=env.device)
                action = torch.cat([arm_action, gripper_action], dim=1)
            else:
                joint_ids = [robot.data.joint_names.index(name) for name in UR5_ARM_JOINT_NAMES]
                action = robot.data.joint_pos[:, joint_ids].clone()
            env.step(action)
            step += 1
            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                print(f"[RoboLab] Static scene heartbeat: step={step}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        simulation_app.close()
    except Exception as exc:
        print(f"[RoboLab] Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
