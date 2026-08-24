# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import random

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def zero_image_like_camera(env, sensor_cfg):
    import torch

    camera = env.scene.sensors[sensor_cfg.name]
    return torch.zeros_like(camera.data.output["rgb"])


def auto_register_ur5_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, lighting_intensity=None, task=None, cameras=None,
                           randomize_background=False, background_seed=None,
                           env_postfix="UR5eJointPosition", initial_arm_joint_positions=None,
                           initial_root_rot_wxyz=None):
    """Automatically discover and register tasks with the UR5e robot."""
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        _image_observation_func,
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.registrations.ur5.camera_presets import BERKELEY_EEF, ZeroOverShoulderRightCameraCfg
    from robolab.robots.ur5 import (
        ProprioceptionObservationCfg,
        UR5eCfg,
        UR5eJointPositionActionCfg,
        UR5eWithWristCameraCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.robots.ur5_profile import ARM_JOINT_NAMES
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    if cameras is None:
        cameras = BERKELEY_EEF

    def _make_image_obs_cfg(camera_cfgs):
        @configclass
        class ImageObsCfg(ObsGroup):
            """UR5 image observations, including optional synthetic missing-view slots."""

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = False

        for camera_cfg in camera_cfgs:
            if camera_cfg is ZeroOverShoulderRightCameraCfg:
                camera_name = "over_shoulder_right_camera"
                if hasattr(ImageObsCfg, camera_name):
                    raise ValueError("Zero right camera cannot be combined with a physical right camera")
                setattr(
                    ImageObsCfg,
                    camera_name,
                    ObsTerm(
                        func=zero_image_like_camera,
                        params={"sensor_cfg": SceneEntityCfg("over_shoulder_left_camera")},
                    ),
                )
                continue

            camera_cfg_instance = camera_cfg()
            for attr_name in dir(camera_cfg_instance):
                if attr_name.startswith("_"):
                    continue
                attr_value = getattr(camera_cfg_instance, attr_name)
                if not isinstance(attr_value, CameraCfg):
                    continue
                if hasattr(ImageObsCfg, attr_name):
                    raise ValueError(f"Duplicate image observation camera: {attr_name}")
                setattr(
                    ImageObsCfg,
                    attr_name,
                    ObsTerm(
                        func=_image_observation_func(),
                        params={
                            "sensor_cfg": SceneEntityCfg(attr_name),
                            "data_type": "rgb",
                            "normalize": False,
                        },
                    ),
                )
        return ImageObsCfg

    real_cameras = [camera for camera in cameras if camera is not ZeroOverShoulderRightCameraCfg]
    has_wrist_camera = any(camera is WristCameraCfg for camera in real_cameras)
    robot_cfg = UR5eWithWristCameraCfg if has_wrist_camera else UR5eCfg
    if initial_arm_joint_positions is not None or initial_root_rot_wxyz is not None:
        configured_robot = copy.deepcopy(robot_cfg().robot)
        if initial_arm_joint_positions is not None:
            if len(initial_arm_joint_positions) != len(ARM_JOINT_NAMES):
                raise ValueError(
                    f"Expected {len(ARM_JOINT_NAMES)} UR5 arm joints, got {len(initial_arm_joint_positions)}"
                )
            configured_robot.init_state.joint_pos.update(
                dict(zip(ARM_JOINT_NAMES, map(float, initial_arm_joint_positions)))
            )
        if initial_root_rot_wxyz is not None:
            if len(initial_root_rot_wxyz) != 4:
                raise ValueError(f"Expected a 4-value root quaternion, got {len(initial_root_rot_wxyz)}")
            root_wxyz = tuple(map(float, initial_root_rot_wxyz))
            configured_robot.init_state.rot = (*root_wxyz[1:], root_wxyz[0])
        robot_cfg = type(f"{robot_cfg.__name__}InitialPoseCfg", (robot_cfg,), {"robot": configured_robot})
    scene_cameras = [camera for camera in real_cameras if camera is not WristCameraCfg]

    ImageObsCfg = _make_image_obs_cfg(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg(),
    })

    if randomize_background:
        from robolab.variations.backgrounds import find_background_files, generate_background_config

        rng = random.Random(background_seed)
        all_bgs = find_background_files()
        default_bg_path = getattr(HomeOfficeBackgroundCfg().dome_light.spawn, "texture_file", None)
        all_bgs = [p for p in all_bgs if p != default_bg_path]
        if not all_bgs:
            raise FileNotFoundError(
                "No backgrounds available for randomization after excluding the default."
            )

        def _bg_factory():
            return generate_background_config(rng.choice(all_bgs))

        background_cfg = _bg_factory
    else:
        background_cfg = HomeOfficeBackgroundCfg

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix=env_postfix,
        observations_cfg=ObservationCfg(),
        actions_cfg=UR5eJointPositionActionCfg(),
        robot_cfg=robot_cfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper,
        ee_body_name="tool0",
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
