# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UR5-specific camera preset bundles for policy image observations."""

from robolab.robots.ur5 import WristCameraCfg
from robolab.variations.camera import (
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)


class ZeroOverShoulderRightCameraCfg:
    """Sentinel for the Berkeley missing right-camera slot.

    UR5 registration converts this into a zero image observation named
    ``over_shoulder_right_camera`` without spawning a physical camera.
    """


WRIST = [WristCameraCfg]
LEFT = [OverShoulderLeftCameraCfg]
RIGHT = [OverShoulderRightCameraCfg]
LEFT_RIGHT = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg]
WRIST_LEFT = [OverShoulderLeftCameraCfg, WristCameraCfg]
WRIST_LEFT_RIGHT = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg, WristCameraCfg]
LEFT_RIGHT_HEAD = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg, HeadCameraCfg]
WRIST_LEFT_RIGHT_HEAD = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg, HeadCameraCfg, WristCameraCfg]

# Berkeley policy input is external+wrist+zero. The head camera is kept in image_obs
# so saved sensor videos retain a DROID-style global panel.
BERKELEY_EEF = [OverShoulderLeftCameraCfg, ZeroOverShoulderRightCameraCfg, HeadCameraCfg, WristCameraCfg]

CAMERA_PRESETS = {
    "berkeley_eef": BERKELEY_EEF,
    "left": LEFT,
    "right": RIGHT,
    "left_right": LEFT_RIGHT,
    "wrist": WRIST,
    "wrist_left": WRIST_LEFT,
    "wrist_left_right": WRIST_LEFT_RIGHT,
    "left_right_head": LEFT_RIGHT_HEAD,
    "wrist_left_right_head": WRIST_LEFT_RIGHT_HEAD,
}


def get_camera_preset(name: str):
    try:
        return CAMERA_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported UR5 camera preset: {name!r}") from exc
