# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Cosmos3 DROID-model backend on UR5e registered tasks."""

import argparse
import logging
import sys
import traceback

import cv2  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "cosmos3_ur5"
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate the Cosmos3 policy backend on UR5e.")
parser.add_argument(
    "--remote-host", default="localhost", help="Remote host for policy server (default: localhost)."
)
parser.add_argument(
    "--remote-port", default=8000, type=int, help="Remote port for policy server (default: 8000)."
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

auto_register_ur5_envs(task=args_cli.task)
if clear_task_filter_for_explicit_paths(args_cli):
    logger.debug("Registered explicit task path(s); cleared eval task filter.")


def make_client(args: argparse.Namespace) -> Cosmos3UR5Client:
    """Create a Cosmos3 UR5 adapter client."""
    return Cosmos3UR5Client(remote_host=args.remote_host, remote_port=args.remote_port)


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
