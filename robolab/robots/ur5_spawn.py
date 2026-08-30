# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URDF spawning helpers for the UR5e + Robotiq 2F-85 assembly."""

from __future__ import annotations

from isaaclab.sim.spawners.from_files import spawn_from_urdf
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.schemas import activate_contact_sensors
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


def _resolve_link_prim(stage: Usd.Stage, robot_path: str, link_name: str) -> Usd.Prim:
    """Resolve flat Isaac 5 links and nested Isaac 6 URDF link trees."""

    direct = stage.GetPrimAtPath(Sdf.Path(f"{robot_path}/{link_name}"))
    if direct.IsValid():
        return direct
    root = stage.GetPrimAtPath(robot_path)
    matches = [prim for prim in Usd.PrimRange(root) if prim.GetName() == link_name]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Expected one UR5e link {link_name!r} below {robot_path}, found {paths}")
    return matches[0]


def _resolve_pad_collider(stage: Usd.Stage, pad: Usd.Prim, section: str) -> Usd.Prim:
    """Resolve Isaac 6 direct pad colliders and Isaac 5 nested colliders."""

    for relative_path in (section, f"collisions/{section}/box"):
        collider = stage.GetPrimAtPath(pad.GetPath().AppendPath(relative_path))
        if collider.IsValid():
            return collider
    raise RuntimeError(f"Missing Robotiq {section} collider below {pad.GetPath()}")


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
            first_prim = _resolve_link_prim(stage, robot_path, first_link)
            second_prim = _resolve_link_prim(stage, robot_path, second_link)
            relation = UsdPhysics.FilteredPairsAPI.Apply(first_prim).CreateFilteredPairsRel()
            relation.AddTarget(second_prim.GetPath())

        # Isaac 6 nests rigid links, while the generic activation traversal
        # stops at the first rigid body. Explicitly activate the two pad links.
        for side in ("left", "right"):
            pad = _resolve_link_prim(stage, robot_path, f"{side}_inner_finger_pad")
            activate_contact_sensors(str(pad.GetPath()), stage=stage)

        # Isaac 5 may instance these tiny collision subtrees. Expand only
        # those roots; Isaac 6 emits editable colliders directly on the pad.
        for relative_path in ROBOTIQ_PAD_COLLISION_ROOT_PATHS:
            link_name, child_path = relative_path.split("/", 1)
            collision_root_path = _resolve_link_prim(stage, robot_path, link_name).GetPath().AppendPath(child_path)
            collision_root = stage.GetPrimAtPath(collision_root_path)
            if collision_root.IsValid() and collision_root.IsInstance():
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
                pad = _resolve_link_prim(stage, robot_path, f"{side}_inner_finger_pad")
                collider = _resolve_pad_collider(stage, pad, section)
                if collider.IsInstanceProxy() or not collider.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError(f"Cannot bind Robotiq pad material to collider {collider.GetPath()}")
                bind_physics_material(str(collider.GetPath()), material_path, stage)

    return prim
