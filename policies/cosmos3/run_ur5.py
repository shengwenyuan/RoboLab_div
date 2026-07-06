# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Cosmos3 backend on UR5e registered tasks."""

import argparse
import logging
import sys
import traceback

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "cosmos3_ur5"
CAMERA_PRESET_CHOICES = [
    "berkeley_eef",
    "left",
    "right",
    "left_right",
    "wrist",
    "wrist_left",
    "wrist_left_right",
    "left_right_head",
    "wrist_left_right_head",
]
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate the Cosmos3 policy backend on UR5e.")
parser.add_argument(
    "--remote-host", default="localhost", help="Remote host for policy server (default: localhost)."
)
parser.add_argument(
    "--remote-port", default=8000, type=int, help="Remote port for policy server (default: 8000)."
)
parser.add_argument(
    "--server-action-format",
    choices=["auto", "joint", "eef_pose"],
    default="eef_pose",
    help=(
        "Server action description: joint, single-arm 8D EEF pose, "
        "or shape-based auto detection (default: eef_pose)."
    ),
)
parser.add_argument(
    "--camera-preset",
    choices=CAMERA_PRESET_CHOICES,
    default="berkeley_eef",
    help="UR5 camera preset for policy observations (default: berkeley_eef).",
)
from robolab.eval.runner import add_common_eval_args, clear_task_filter_for_explicit_paths, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.client import Cosmos3UR5Client
from robolab.registrations.ur5.auto_env_registrations_jointpos import auto_register_ur5_envs
from robolab.registrations.ur5.camera_presets import get_camera_preset

auto_register_ur5_envs(task=args_cli.task, cameras=get_camera_preset(args_cli.camera_preset))
if clear_task_filter_for_explicit_paths(args_cli):
    logger.debug("Registered explicit task path(s); cleared eval task filter.")


def make_client(args: argparse.Namespace) -> Cosmos3UR5Client:
    """Create a Cosmos3 UR5 adapter client."""
    return Cosmos3UR5Client(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        server_action_format=args.server_action_format,
    )


def main() -> None:
    """Run Cosmos3 UR5 evaluation."""
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
