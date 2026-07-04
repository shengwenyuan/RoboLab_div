# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UR5-specific camera preset bundles for policy image observations."""

from robolab.variations.camera import (
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)

LEFT = [OverShoulderLeftCameraCfg]
RIGHT = [OverShoulderRightCameraCfg]
LEFT_RIGHT = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg]
LEFT_RIGHT_HEAD = [OverShoulderLeftCameraCfg, OverShoulderRightCameraCfg, HeadCameraCfg]

# Default policy observation bundle. UR5 currently has no robot-mounted wrist camera;
# Cosmos3UR5Client mirrors the exterior image into the DROID wrist-image slot.
WRIST_LEFT = LEFT
