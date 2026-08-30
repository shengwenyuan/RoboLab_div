# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UR5-specific camera preset bundles for policy image observations."""

from dataclasses import dataclass

from robolab.robots.ur5 import WristCameraCfg
from robolab.variations.camera import (
    BerkeleyUR5LeftCameraCfg,
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    RoboMindGlobalCameraCfg,
    RoboMindTopCameraCfg,
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
BERKELEY_EEF = [BerkeleyUR5LeftCameraCfg, ZeroOverShoulderRightCameraCfg, HeadCameraCfg, WristCameraCfg]
# The wide global camera is first so *_0.mp4 sensor mosaics show it on the left.
# Cosmos3 still selects only robomind_top_camera by role_sources below; the
# global camera is diagnostic-only and both policy auxiliary slots stay black.
ROBOMIND_SINGLE = [RoboMindGlobalCameraCfg, RoboMindTopCameraCfg]

CAMERA_PRESETS = {
    "berkeley_eef": BERKELEY_EEF,
    "robomind_single": ROBOMIND_SINGLE,
    "left": LEFT,
    "right": RIGHT,
    "left_right": LEFT_RIGHT,
    "wrist": WRIST,
    "wrist_left": WRIST_LEFT,
    "wrist_left_right": WRIST_LEFT_RIGHT,
    "left_right_head": LEFT_RIGHT_HEAD,
    "wrist_left_right_head": WRIST_LEFT_RIGHT_HEAD,
}


@dataclass(frozen=True)
class Cosmos3CameraPreset:
    """Physical cameras plus the canvas semantics they can satisfy."""

    cameras: tuple[type, ...]
    layout_id: str
    view_roles: tuple[str, ...]
    role_sources: dict[str, str | None]
    missing_view_policies: tuple[str, ...]
    view_shape_hw: tuple[int, int] = (360, 640)
    canvas_shape_hw: tuple[int, int] = (540, 640)
    viewpoint: str = "concat_view"

    def observation_metadata(self) -> dict[str, object]:
        return {
            "layout_id": self.layout_id,
            "view_shape_hw": self.view_shape_hw,
            "canvas_shape_hw": self.canvas_shape_hw,
            "view_roles": self.view_roles,
            "role_sources": self.role_sources,
            "missing_view_policies": self.missing_view_policies,
            "viewpoint": self.viewpoint,
        }


_THREE_REAL_VIEWS = {
    "layout_id": "primary_top_aux_bottom_pair",
    "view_roles": ("primary", "aux_left", "aux_right"),
    "role_sources": {
        "primary": "wrist_cam",
        "aux_left": "over_shoulder_left_camera",
        "aux_right": "over_shoulder_right_camera",
    },
    "missing_view_policies": ("error", "black"),
}

COSMOS3_CAMERA_PRESETS = {
    "berkeley_eef": Cosmos3CameraPreset(
        cameras=tuple(BERKELEY_EEF),
        layout_id="primary_top_aux_bottom_pair_right_missing",
        view_roles=("primary", "aux_left", "aux_right"),
        role_sources={
            "primary": "wrist_cam",
            "aux_left": "over_shoulder_left_camera",
            "aux_right": None,
        },
        missing_view_policies=("black",),
    ),
    "robomind_single": Cosmos3CameraPreset(
        cameras=tuple(ROBOMIND_SINGLE),
        layout_id="primary_top_aux_bottom_pair",
        view_roles=("primary", "aux_left", "aux_right"),
        role_sources={
            "primary": "robomind_top_camera",
            "aux_left": None,
            "aux_right": None,
        },
        missing_view_policies=("black",),
    ),
    "rh20t_vertical_pair": Cosmos3CameraPreset(
        cameras=tuple(LEFT_RIGHT),
        layout_id="vertical_pair",
        view_roles=("primary", "aux_left"),
        role_sources={
            "primary": "over_shoulder_left_camera",
            "aux_left": "over_shoulder_right_camera",
        },
        missing_view_policies=("error",),
        canvas_shape_hw=(720, 640),
    ),
    "wrist_left_right": Cosmos3CameraPreset(cameras=tuple(WRIST_LEFT_RIGHT), **_THREE_REAL_VIEWS),
    "wrist_left_right_head": Cosmos3CameraPreset(cameras=tuple(WRIST_LEFT_RIGHT_HEAD), **_THREE_REAL_VIEWS),
}


def get_camera_preset(name: str):
    try:
        return CAMERA_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported UR5 camera preset: {name!r}") from exc


def get_cosmos3_camera_preset(name: str) -> Cosmos3CameraPreset:
    try:
        return COSMOS3_CAMERA_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(COSMOS3_CAMERA_PRESETS))
        raise ValueError(f"Unsupported Cosmos3 UR5 camera preset {name!r}; choose one of: {choices}") from exc
