# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Cosmos3 UR5 EEF policy with keypoint-IK camera geometry."""

import argparse
import logging
import sys
import traceback

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "cosmos3_ur5_keypoint_ik"
KEYPOINT_IK_CAMERA_PRESET = "wrist_left_right"
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--remote-host", default="localhost", help="Remote host for policy server.")
parser.add_argument("--remote-port", default=8000, type=int, help="Remote port for policy server.")
parser.add_argument(
    "--server-action-format",
    choices=["auto", "joint", "eef_pose"],
    default="eef_pose",
    help="Server action description. The keypoint-IK UR5 EEF model should normally use eef_pose.",
)
parser.add_argument(
    "--camera-preset",
    choices=[KEYPOINT_IK_CAMERA_PRESET],
    default=KEYPOINT_IK_CAMERA_PRESET,
    help="Fixed keypoint-IK camera preset. This runner intentionally only supports wrist_left_right.",
)

from robolab.eval.runner import add_common_eval_args, clear_task_filter_for_explicit_paths, run_evaluation  # noqa: E402

add_common_eval_args(parser)
parser.set_defaults(video_mode="all")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.keypoint_ik_client import Cosmos3UR5KeypointIKClient  # noqa: E402
from robolab.registrations.ur5.auto_env_registrations_jointpos import auto_register_ur5_envs  # noqa: E402
from robolab.registrations.ur5.camera_presets import get_camera_preset  # noqa: E402

auto_register_ur5_envs(task=args_cli.task, cameras=get_camera_preset(args_cli.camera_preset))
if clear_task_filter_for_explicit_paths(args_cli):
    logger.debug("Registered explicit task path(s); cleared eval task filter.")


def make_client(args: argparse.Namespace) -> Cosmos3UR5KeypointIKClient:
    return Cosmos3UR5KeypointIKClient(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        server_action_format=args.server_action_format,
    )


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
