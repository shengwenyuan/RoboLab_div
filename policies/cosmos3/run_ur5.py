# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Cosmos3 backend on UR5e registered tasks."""

import argparse
import logging
import sys
import traceback

import cv2  # noqa: F401  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "cosmos3_ur5"
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate the Cosmos3 policy backend on UR5e.")
parser.add_argument("--remote-host", default="localhost", help="Remote host for policy server (default: localhost).")
parser.add_argument("--remote-port", default=8000, type=int, help="Remote port for policy server (default: 8000).")
parser.add_argument("--execute-horizon", default=8, type=int, help="Actions to execute before replanning.")
parser.add_argument(
    "--camera-preset",
    default="wrist_left_right",
    help=(
        "UR5 camera preset for policy observations (default: wrist_left_right). "
        "Use robomind_single together with --initial-pose-preset robomind_ur5; "
        "standard and Berkeley evaluations normally use the default initial pose. "
        "The camera canvas contract is checked against the server manifest during handshake."
    ),
)
parser.add_argument(
    "--initial-pose-preset",
    default="default",
    help=(
        "UR5 simulator reset geometry: default preserves the standard RoboLab home; "
        "robomind_ur5 applies the verified RoboMIND joint pose and base yaw. "
        "This setting does not override the model action schema."
    ),
)
from robolab.eval.runner import add_common_eval_args, clear_task_filter_for_explicit_paths, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.client import Cosmos3UR5Client
from policies.cosmos3.specs import ObservationCapability
from robolab.registrations.ur5.auto_env_registrations_jointpos import auto_register_ur5_envs
from robolab.registrations.ur5.camera_presets import get_cosmos3_camera_preset
from robolab.registrations.ur5.initial_pose_presets import get_ur5_initial_pose_preset

try:
    camera_preset = get_cosmos3_camera_preset(args_cli.camera_preset)
    initial_pose_preset = get_ur5_initial_pose_preset(args_cli.initial_pose_preset)
except ValueError as exc:
    simulation_app.close()
    parser.error(str(exc))

auto_register_ur5_envs(
    task=args_cli.task,
    cameras=list(camera_preset.cameras),
    initial_arm_joint_positions=initial_pose_preset.arm_joint_positions,
    initial_root_rot_wxyz=initial_pose_preset.root_rot_wxyz,
)
if clear_task_filter_for_explicit_paths(args_cli):
    logger.debug("Registered explicit task path(s); cleared eval task filter.")


def make_client(args: argparse.Namespace) -> Cosmos3UR5Client:
    """Create a Cosmos3 UR5 adapter client."""
    return Cosmos3UR5Client(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        observation=ObservationCapability.from_mapping(camera_preset.observation_metadata()),
        execute_horizon=args.execute_horizon,
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
