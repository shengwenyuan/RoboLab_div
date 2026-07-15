# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named UR5 reset poses for policy evaluation coordinate conventions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UR5InitialPosePreset:
    """A canonical UR5 reset pose in one evaluation coordinate convention."""

    arm_joint_positions: tuple[float, ...] | None = None
    root_rot_wxyz: tuple[float, float, float, float] | None = None


UR5_INITIAL_POSE_PRESETS = {
    "default": UR5InitialPosePreset(),
    # Representative real RoboMIND pose matched to episode 2836. The +52.401°
    # root yaw rotates this real pose toward the RoboLab work area.
    "robomind_ur5": UR5InitialPosePreset(
        arm_joint_positions=(
            2.216296672821045,
            -1.8964468240737915,
            -1.6191987991333008,
            -1.1574769020080566,
            1.5983433723449707,
            1.0311790704727173,
        ),
        root_rot_wxyz=(0.8972531204891708, 0.0, 0.0, 0.44151652038450995),
    ),
}


def get_ur5_initial_pose_preset(name: str) -> UR5InitialPosePreset:
    """Resolve one explicit UR5 initial-pose preset by name."""

    try:
        return UR5_INITIAL_POSE_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(UR5_INITIAL_POSE_PRESETS))
        raise ValueError(f"Unsupported UR5 initial-pose preset {name!r}; choose one of: {choices}") from exc
