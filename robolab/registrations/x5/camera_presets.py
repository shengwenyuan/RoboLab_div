# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
X5-specific camera preset bundles for the policy's image observations.

Each preset is a list of camera config classes that feed both the scene
(``camera_cfg=``) and the image observation group (via
``generate_image_obs_from_cameras``). Viewport-only cameras are attached
separately inside the registration function and are not listed here.
"""

from robolab.robots.x5 import WristCameraCfg
from robolab.variations.camera import (
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)

WRIST = [WristCameraCfg]

WRIST_LEFT = [
    OverShoulderLeftCameraCfg,
    WristCameraCfg,
]

WRIST_RIGHT = [
    OverShoulderRightCameraCfg,
    WristCameraCfg,
]

WRIST_LEFT_RIGHT = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    WristCameraCfg,
]

WRIST_LEFT_RIGHT_HEAD = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    HeadCameraCfg,
    WristCameraCfg,
]

LEFT_RIGHT = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
]
