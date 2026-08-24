# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class OverShoulderLeftCameraCfg:
    """Left over-shoulder camera, matching DROID exterior_image_1_left placement.

    Mounted to the left of the robot workspace at (0.05, 0.57, 0.66).
    Look direction: (0.63, -0.48, -0.62) — looking right-and-down toward workspace.
    """
    over_shoulder_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.195, 0.399, 0.805, -0.393), convention="opengl"
        ),
    )


@configclass
class OverShoulderRightCameraCfg:
    """Right over-shoulder camera, matching DROID exterior_image_2_left placement.

    Mirror of OverShoulderLeftCameraCfg across the XZ plane (Y → -Y).
    Mounted to the right of the robot workspace at (0.05, -0.57, 0.66).
    Look direction: (0.628, +0.490, -0.606) — looking left-and-down toward workspace.
    Up direction in world: (0.477, +0.372, +0.796) — Z component positive (upright image).

    Derivation: the correct XZ mirror requires det(R) = +1. The rotation matrix columns are
    the XZ-mirrored left-camera basis vectors with the right-vector sign corrected for
    right-handedness. In XYZW order the quaternion is (0.399, -0.195, -0.393, 0.805).
    """
    over_shoulder_right_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/over_shoulder_right_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66), rot=(0.399, -0.195, -0.393, 0.805), convention="opengl"
        ),
    )


@configclass
class HeadCameraCfg:
    """Front-facing overhead camera, simulating an operator's head/eye view.

    Positioned 1.5 m in front of and 1.0 m above the robot, looking back toward
    the workspace.  Look direction: (-0.83, 0, -0.55) — straight-on frontal view.
    Rotation computed as pure Y-axis rotation mapping camera -Z to look direction:
    q = (0, sin(28.3°), 0, cos(28.3°)) = (0, 0.474, 0, 0.881).
    """
    head_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/head_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.0), rot=(0.0, 0.474, 0.0, 0.881), convention="opengl"
        ),
    )



################################################################################
# Egocentric cameras
################################################################################
@configclass
class EgocentricWideAngleCameraCfg:
    egocentric_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.5), rot=(0.271, -0.271, -0.653, 0.653), convention="opengl"
        ),
    )


################################################################################
# Egocentric mirrored, means the camera is looking at the robot from the front,
# Assuming the robot is at origin.
################################################################################
@configclass
class EgocentricMirroredWideAngleHighCameraCfg:
   egocentric_mirrored_wide_angle_high_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_high_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 1), rot=(0.271, 0.271, 0.653, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredWideAngleCameraCfg:
   egocentric_mirrored_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 0.5), rot=(0.271, 0.271, 0.653, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredCameraCfg:
   egocentric_mirrored_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_camera",
    # height=720,
    # width=1280,
    height = 480,
    width = 864,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.0),
        rot=(0.271, 0.271, 0.653, 0.653),
        convention="opengl"
    ),
)


@configclass
class RoboMindGlobalCameraCfg:
    """Wide global view used only to make RoboMIND evaluation videos readable.

    The camera looks from the side opposite the UR5 base toward the workspace
    center. Its 16:9 frame has the same 480 px height as ``RoboMindTopCameraCfg``
    so ``unpack_image_obs`` can concatenate it to the left of the local top view
    without padding or rescaling one camera relative to the other.

    Pose derivation (env frame, physical quaternion shown as w, x, y, z):
    pos (1.5, 0, 1.0), look-at (0.5, 0, 0) gives view direction
    (-0.707, 0, -0.707) and quaternion (0.653, 0.271, 0.271, 0.653).
    """

    robomind_global_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robomind_global_camera",
        height=480,
        width=864,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.0), rot=(0.271, 0.271, 0.653, 0.653), convention="opengl"
        ),
    )


@configclass
class RoboMindTopCameraCfg:
    """RoboMIND1-UR ``camera_top`` style view: across from the arm base, looking down-back.

    Approximates the real RoboMIND UR rig: a 4:3 RGB sensor (640x480, configured
    for ~56x44 deg FOV) mounted above the side of the workspace opposite the
    UR5e base, pitched down toward the robot so the tabletop workspace fills the
    frame. The source dataset stores 640x480 JPEG frames without camera intrinsics;
    training stretched them to 360x640, and the client canvas resize replicates
    that same stretch.

    Pose derivation (env frame: base at origin, workspace toward +X):
    mirror the old base-side position (0.0, 0, 1.05) about the workspace center
    x=0.62 to get (1.24, 0, 1.05), then move 0.19 m toward the table along -X
    so its near edge meets the bottom of the image. The final pos is
    (1.05, 0, 1.05). Height and the ~59 deg depression stay unchanged. The view
    direction is (-0.509, 0, -0.861), and the world-up look-at basis gives camera
    up (-0.861, 0, 0.509). The physical wxyz quaternion is
    (0.682, 0.186, 0.186, 0.682); ``OffsetCfg.rot`` below is xyzw.

    Tune pose/FOV against real dataset frames with
    ``examples/save_ur5_cosmos_initial_cameras.py --camera-preset robomind_single``.
    """
    robomind_top_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robomind_top_camera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=4.032,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.05, 0.0, 1.05), rot=(0.186, 0.186, 0.682, 0.682), convention="opengl"
        ),
    )


@configclass
class BerkeleyUR5LeftCameraCfg:
    """UR5 Berkeley-style left over-shoulder camera with tighter framing."""

    over_shoulder_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.7,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.52), rot=(-0.195, 0.399, 0.805, -0.393), convention="opengl"
        ),
    )
