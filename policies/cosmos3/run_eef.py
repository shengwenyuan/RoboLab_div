# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Cosmos3 DROID EEF policy through IsaacLab absolute IK."""

import argparse
import sys
import traceback

import cv2  # noqa: F401  # Must import before IsaacLab.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", default=8000, type=int)
parser.add_argument("--execute-horizon", default=8, type=int)

from robolab.eval.runner import add_common_eval_args, clear_task_filter_for_explicit_paths, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

from policies.cosmos3.client import Cosmos3EEFClient, IsaacLabAbsIKAdapter
from policies.cosmos3.specs import STATELESS_CONDITIONING, ClientCapability, STANDARD_THREE_VIEW_OBSERVATION
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
from robolab.robots.droid import PANDA_LINK8_TO_BASE_POS, PANDA_LINK8_TO_BASE_ROT

CAPABILITY = ClientCapability(
    robot="droid",
    arm_dof=7,
    action_spaces=("eef_absolute",),
    joint_action_layout=(),
    conditioning_by_action_space={"eef_absolute": STATELESS_CONDITIONING},
    observation=STANDARD_THREE_VIEW_OBSERVATION,
)
ADAPTER = IsaacLabAbsIKAdapter(
    arm_dof=7,
    policy_frame="panda_link8",
    controller_frame="base_link",
    controller_in_policy_xyz=PANDA_LINK8_TO_BASE_POS,
    controller_in_policy_quat_wxyz=PANDA_LINK8_TO_BASE_ROT,
)

auto_register_droid_abs_ik_envs(task=args_cli.task, cameras=WRIST_LEFT_RIGHT_HEAD)
clear_task_filter_for_explicit_paths(args_cli)


def make_client(args: argparse.Namespace) -> Cosmos3EEFClient:
    return Cosmos3EEFClient(
        args.remote_host,
        args.remote_port,
        capability=CAPABILITY,
        adapter=ADAPTER,
        execute_horizon=args.execute_horizon,
        eef_pos_key="panda_link8_pos",
        eef_quat_key="panda_link8_quat",
    )


def main() -> None:
    run_evaluation(args_cli, policy="cosmos3_droid_eef", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
