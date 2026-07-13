# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URDF spawning helpers for the UR5e + Robotiq 2F-85 assembly."""

from __future__ import annotations

from isaaclab.sim.spawners.from_files import spawn_from_urdf
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths
from pxr import Sdf, Usd, UsdPhysics

# Menagerie v4 explicitly excludes these mechanism-internal pairs.
ROBOTIQ_INTERNAL_COLLISION_FILTER_PAIRS = (
    ("robotiq_arg2f_base_link", "left_outer_knuckle"),
    ("robotiq_arg2f_base_link", "right_outer_knuckle"),
    ("robotiq_arg2f_base_link", "left_inner_knuckle"),
    ("robotiq_arg2f_base_link", "right_inner_knuckle"),
    ("left_outer_finger", "left_inner_finger"),
    ("right_outer_finger", "right_inner_finger"),
)

# The UR5 description uses coarse flange/wrist collision proxies that overlap
# the bolted coupling. These are fixed mounting neighbors, not independently
# colliding mechanism parts. Tool0 is the direct fixed-joint parent and is
# already excluded by PhysX.
ROBOTIQ_MOUNT_COLLISION_FILTER_PAIRS = (
    ("robotiq_arg2f_base_link", "flange"),
    ("robotiq_arg2f_base_link", "wrist_3_link"),
)

ROBOTIQ_COLLISION_FILTER_PAIRS = (
    *ROBOTIQ_INTERNAL_COLLISION_FILTER_PAIRS,
    *ROBOTIQ_MOUNT_COLLISION_FILTER_PAIRS,
)

ROBOTIQ_PAD_FRICTION_BY_SECTION = {"pad_lower": 0.7, "pad_upper": 0.6}
ROBOTIQ_PAD_COLLISION_ROOT_PATHS = tuple(f"{side}_inner_finger_pad/collisions" for side in ("left", "right"))
ROBOTIQ_PAD_COLLIDER_PATHS = tuple(
    f"{side}_inner_finger_pad/collisions/{section}/box"
    for side in ("left", "right")
    for section in ROBOTIQ_PAD_FRICTION_BY_SECTION
)


def spawn_ur5e_robotiq_2f85(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the assembly, filter mounting contacts, and bind pad friction."""

    prim = spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    robot_paths = find_matching_prim_paths(str(prim_path), stage)
    if not robot_paths:
        raise RuntimeError(f"Spawned UR5e prim path did not resolve: {prim_path!r}")

    for robot_path in robot_paths:
        for first_link, second_link in ROBOTIQ_COLLISION_FILTER_PAIRS:
            first_path = Sdf.Path(f"{robot_path}/{first_link}")
            second_path = Sdf.Path(f"{robot_path}/{second_link}")
            first_prim = stage.GetPrimAtPath(first_path)
            second_prim = stage.GetPrimAtPath(second_path)
            if not first_prim.IsValid() or not second_prim.IsValid():
                raise RuntimeError(
                    f"Cannot apply Robotiq collision filter to missing link pair {first_path} <-> {second_path}"
                )
            relation = UsdPhysics.FilteredPairsAPI.Apply(first_prim).CreateFilteredPairsRel()
            relation.AddTarget(second_path)

        # Only the tiny collision subtrees are instanced. Expand those two
        # roots so lower and upper pad boxes can retain distinct source
        # friction instead of disabling instancing for the complete robot.
        for relative_path in ROBOTIQ_PAD_COLLISION_ROOT_PATHS:
            collision_root_path = f"{robot_path}/{relative_path}"
            collision_root = stage.GetPrimAtPath(collision_root_path)
            if not collision_root.IsValid() or not collision_root.IsInstance():
                raise RuntimeError(f"Expected instanced Robotiq collision root at {collision_root_path}")
            collision_root.SetInstanceable(False)

        for section, friction in ROBOTIQ_PAD_FRICTION_BY_SECTION.items():
            pad_material_cfg = RigidBodyMaterialCfg(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            )
            material_path = f"{robot_path}/robotiq_{section}_physics_material"
            pad_material_cfg.func(material_path, pad_material_cfg)
            for side in ("left", "right"):
                collider_path = f"{robot_path}/{side}_inner_finger_pad/collisions/{section}/box"
                collider_prim = stage.GetPrimAtPath(collider_path)
                if (
                    not collider_prim.IsValid()
                    or collider_prim.IsInstanceProxy()
                    or not collider_prim.HasAPI(UsdPhysics.CollisionAPI)
                ):
                    raise RuntimeError(f"Cannot bind Robotiq pad material to collider {collider_path}")
                bind_physics_material(collider_path, material_path, stage)

    return prim
