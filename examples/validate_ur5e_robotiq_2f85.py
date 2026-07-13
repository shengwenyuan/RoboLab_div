# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offscreen physics/video validation for the RoboLab UR5e + Robotiq 2F-85.

The validation deliberately does not use a registered RoboLab task.  It clones
``UR5eCfg().robot`` in memory, enables URDF mimic parsing for that clone, and
narrows the gripper actuator to ``finger_joint``.  The five follower joints are
therefore constrained by PhysX rather than driven independently.

The recorded sequence is:

1. settle open;
2. slowly close, hold, slowly reopen, and hold;
3. move above a simple static table;
4. descend slowly until the first gripper/table contact, hold, and retract.

Artifacts are an annotated H.264 MP4, full-rate trace JSON and CSV files, and a
machine-readable summary JSON.

Example:

.. code-block:: bash

    /workspace/isaaclab/isaaclab.sh -p \
        examples/validate_ur5e_robotiq_2f85.py \
        --headless --device cuda:0 \
        --output-dir /tmp/ur5e_2f85_validation
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2  # must be imported before Isaac Lab/Isaac Sim
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate UR5e + Robotiq 2F-85 mimic motion and table contact offscreen.")
parser.add_argument(
    "--output-dir",
    type=str,
    default="/tmp/robolab_ur5e_2f85_validation",
    help="Directory for MP4, trace JSON/CSV, and summary JSON.",
)
parser.add_argument(
    "--usd-cache-dir",
    type=str,
    default="/tmp/robolab_validate_ur5e_robotiq_2f85_v4_usd",
    help="Isolated USD conversion cache used by this validation.",
)
parser.add_argument(
    "--reuse-usd-cache",
    action="store_true",
    help="Reuse the validation USD cache. By default conversion is forced so mesh-only edits are tested.",
)
parser.add_argument("--physics-hz", type=int, default=120, help="Physics frequency. Default: 120 Hz.")
parser.add_argument("--video-fps", type=int, default=30, help="Output video frame rate. Default: 30 fps.")
parser.add_argument("--width", type=int, default=1280, help="Inspection camera width.")
parser.add_argument("--height", type=int, default=720, help="Inspection camera height.")
parser.add_argument("--table-top", type=float, default=0.15, help="World Z of the table top in metres.")
parser.add_argument("--table-thickness", type=float, default=0.10, help="Table slab thickness in metres.")
parser.add_argument(
    "--approach-clearance",
    type=float,
    default=0.06,
    help="Desired pad-centre height above the table before descent, in metres.",
)
parser.add_argument(
    "--descent-overtravel",
    type=float,
    default=0.03,
    help="Pad-centre target below the table top if no contact is detected, in metres.",
)
parser.add_argument(
    "--contact-threshold",
    type=float,
    default=1.0,
    help="Force threshold used to declare first contact, in newtons.",
)
parser.add_argument(
    "--sustained-contact-threshold",
    type=float,
    default=0.1,
    help="Force threshold used to verify sustained hold contact, in newtons.",
)
parser.add_argument(
    "--abort-force",
    type=float,
    default=150.0,
    help="Validation fails and contact hold is shortened above this force, in newtons.",
)
parser.add_argument(
    "--contact-offset",
    type=float,
    default=0.001,
    help="Explicit table collision contact offset in metres.",
)
parser.add_argument(
    "--max-arm-speed",
    type=float,
    default=1.0,
    help="Maximum differential-IK command change, in rad/s per joint.",
)
parser.add_argument(
    "--quick",
    action="store_true",
    help="Run a shorter smoke sequence while retaining all validation phases.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.physics_hz <= 0 or args_cli.video_fps <= 0:
    parser.error("--physics-hz and --video-fps must be positive")
if args_cli.physics_hz < args_cli.video_fps:
    parser.error("--physics-hz must be greater than or equal to --video-fps")
if args_cli.width <= 0 or args_cli.height <= 0:
    parser.error("--width and --height must be positive")
if args_cli.contact_offset < 0.0:
    parser.error("--contact-offset must be non-negative")
if args_cli.contact_threshold <= 0.0 or args_cli.sustained_contact_threshold <= 0.0:
    parser.error("contact thresholds must be positive")

# RTX sensors must remain enabled even in fully headless mode.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_apply, subtract_frame_transforms  # noqa: E402
from pxr import PhysxSchema, UsdPhysics, UsdShade  # noqa: E402

from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.robots.ur5 import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, UR5eCfg  # noqa: E402
from robolab.robots.ur5_profile import ARM_JOINT_NAMES  # noqa: E402
from robolab.robots.ur5_spawn import (  # noqa: E402
    ROBOTIQ_COLLISION_FILTER_PAIRS,
    ROBOTIQ_PAD_COLLIDER_PATHS,
    ROBOTIQ_PAD_COLLISION_ROOT_PATHS,
    ROBOTIQ_PAD_FRICTION_BY_SECTION,
)

MASTER_JOINT = "finger_joint"
MIMIC_JOINT_NAMES = [
    "right_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
]
GRIPPER_JOINT_NAMES = [MASTER_JOINT, *MIMIC_JOINT_NAMES]
PAD_BODY_NAMES = ["left_inner_finger_pad", "right_inner_finger_pad"]
VALID_CLOSED_SELF_CONTACT_BODY_NAMES = {
    "left_inner_finger",
    "right_inner_finger",
    *PAD_BODY_NAMES,
}
GRIPPER_BODY_EXPR = "{ENV_REGEX_NS}/robot/(robotiq_arg2f_base_link|left_.*|right_.*)"
MOUNT_NEIGHBOR_BODY_NAMES = ("wrist_3_link", "flange", "tool0")
# Exact Menagerie-v4 box geometry in each pad-link frame.  Pad link origins are
# about 130 mm away from the contact material, so they must not be used as TCPs,
# gap measurements, camera targets, or table-approach references.
PAD_BOX_CENTERS_LOCAL_M = ((0.043258, 0.0, 0.120000), (0.043258, 0.0, 0.138750))
PAD_BOX_SIZE_M = (0.004, 0.022, 0.01875)
EXPECTED_OPEN_PAD_SURFACE_GAP_M = 0.0868782314
EXPECTED_PHYSICAL_CLOSED_PAD_SURFACE_GAP_M = 0.0
OPEN_PAD_GAP_TOLERANCE_M = 0.0005
CLOSED_PAD_GAP_TOLERANCE_M = 0.001
REOPEN_PAD_GAP_TOLERANCE_M = 0.0005
MIMIC_TRACKING_TOLERANCE_RAD = 0.005


def _json_safe(value: Any) -> Any:
    """Convert tensors/numpy values and non-finite floats to strict JSON values."""

    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(value), file, indent=2, sort_keys=False, allow_nan=False)
        file.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = ["step", "time_s", "phase", "phase_time_s", "finger_command", "contact_force_n"]
    all_fields = set().union(*(row.keys() for row in rows))
    fieldnames = [name for name in preferred if name in all_fields]
    fieldnames.extend(sorted(all_fields.difference(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows((_json_safe(row) for row in rows))


def _configure_robot() -> ArticulationCfg:
    """Clone production configuration and make validation-only overrides."""

    robot_cfg = copy.deepcopy(UR5eCfg().robot)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/robot"

    # IsaacLab 2.2 passes this unfortunately named field directly to
    # UrdfImportConfig.set_parse_mimic().  True therefore creates
    # PhysxMimicJointAPI followers; False turns them into regular joints.
    robot_cfg.spawn.convert_mimic_joints_to_normal_joints = True
    robot_cfg.spawn.force_usd_conversion = not args_cli.reuse_usd_cache
    robot_cfg.spawn.usd_dir = str(Path(args_cli.usd_cache_dir).expanduser().resolve())
    robot_cfg.spawn.merge_fixed_joints = False
    robot_cfg.spawn.self_collision = True
    robot_cfg.spawn.activate_contact_sensors = True
    # Production keeps the historical gravity-disabled UR5 behavior. This
    # dedicated physical validation opts in to gravity so the calibrated
    # 0.9 kg gripper payload participates in the table-contact sequence.
    if robot_cfg.spawn.rigid_props is not None:
        robot_cfg.spawn.rigid_props.disable_gravity = False
    if robot_cfg.spawn.articulation_props is not None:
        robot_cfg.spawn.articulation_props.enabled_self_collisions = True

    if "gripper" not in robot_cfg.actuators:
        raise KeyError(f"UR5eCfg.robot has no 'gripper' actuator: {list(robot_cfg.actuators)}")
    robot_cfg.actuators["gripper"].joint_names_expr = [MASTER_JOINT]
    return robot_cfg


def _make_scene_cfg(robot_cfg: ArticulationCfg) -> InteractiveSceneCfg:
    table_size = (0.66, 0.66, args_cli.table_thickness)
    table_center = (0.50, 0.13, args_cli.table_top - 0.5 * args_cli.table_thickness)
    configured_robot = robot_cfg

    @configclass
    class ValidationSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                    friction_combine_mode="max",
                )
            ),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/domeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.86, 1.0)),
        )
        key_light = AssetBaseCfg(
            prim_path="/World/keyLight",
            spawn=sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.92, 0.82), angle=4.0),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),
        )

        robot: ArticulationCfg = configured_robot
        table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.CuboidCfg(
                size=table_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=args_cli.contact_offset,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.9,
                    restitution=0.0,
                    friction_combine_mode="max",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.38, 0.20, 0.09),
                    roughness=0.72,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=table_center),
        )

        inspection_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/inspection_camera",
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=32.0,
                focus_distance=0.6,
                horizontal_aperture=36.0,
                clipping_range=(0.01, 5.0),
            ),
        )

        # The unfiltered multi-body sensor also sees normal opposing-jaw
        # contact. Filtered contact reporting is one-to-many only, so each
        # likely table-contact body gets its own table-filtered sensor.
        gripper_contacts = ContactSensorCfg(
            prim_path=GRIPPER_BODY_EXPR,
            update_period=0.0,
            history_length=4,
            debug_vis=False,
        )
        left_pad_table = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/left_inner_finger_pad",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
        )
        right_pad_table = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/right_inner_finger_pad",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
        )
        left_inner_finger_table = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/left_inner_finger",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
        )
        right_inner_finger_table = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/right_inner_finger",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
        )
        mount_base_arm = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/robotiq_arg2f_base_link",
            update_period=0.0,
            history_length=4,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/robot/{body_name}" for body_name in MOUNT_NEIGHBOR_BODY_NAMES],
        )

    return ValidationSceneCfg(num_envs=1, env_spacing=2.0)


def _inspect_mimic_schemas() -> dict[str, dict[str, Any]]:
    """Inspect composed joint prims instead of trusting the converter config."""

    expected = [MASTER_JOINT, *MIMIC_JOINT_NAMES]
    result: dict[str, dict[str, Any]] = {
        name: {"found": False, "path": None, "mimic_schemas": [], "drive_schemas": []} for name in expected
    }
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if name not in result or "/robot/joints/" not in path:
            continue
        schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        mimic_schemas = [schema for schema in schemas if schema.startswith("PhysxMimicJointAPI")]
        drive_schemas = [schema for schema in schemas if schema.startswith("PhysicsDriveAPI")]
        record: dict[str, Any] = {
            "found": True,
            "path": path,
            "mimic_schemas": mimic_schemas,
            "drive_schemas": drive_schemas,
        }
        if mimic_schemas:
            axis = mimic_schemas[0].split(":", 1)[1]
            api = PhysxSchema.PhysxMimicJointAPI(prim, axis)
            record.update(
                {
                    "reference_joint": [str(target) for target in api.GetReferenceJointRel().GetTargets()],
                    "reference_axis": str(api.GetReferenceJointAxisAttr().Get()),
                    "gearing": float(api.GetGearingAttr().Get()),
                    "offset": float(api.GetOffsetAttr().Get()),
                    "natural_frequency": float(prim.GetAttribute(f"physxMimicJoint:{axis}:naturalFrequency").Get()),
                    "damping_ratio": float(prim.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Get()),
                }
            )
        result[name] = record
    return result


def _inspect_collision_filters() -> dict[str, Any]:
    """Read back composed collision filters and pad physics materials."""

    stage = omni.usd.get_context().get_stage()
    robot_path = "/World/envs/env_0/robot"
    pairs: dict[str, dict[str, Any]] = {}
    for first_link, second_link in ROBOTIQ_COLLISION_FILTER_PAIRS:
        first_path = f"{robot_path}/{first_link}"
        second_path = f"{robot_path}/{second_link}"
        first_prim = stage.GetPrimAtPath(first_path)
        targets: list[str] = []
        if first_prim.IsValid() and first_prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            targets = [
                str(target) for target in UsdPhysics.FilteredPairsAPI(first_prim).GetFilteredPairsRel().GetTargets()
            ]
        key = f"{first_link}<->{second_link}"
        pairs[key] = {
            "first_path": first_path,
            "second_path": second_path,
            "targets": targets,
            "present": second_path in targets,
        }

    articulation_self_collision: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        attribute = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if attribute.IsValid():
            articulation_self_collision.append({"path": str(prim.GetPath()), "enabled": bool(attribute.Get())})

    pad_materials: dict[str, dict[str, Any]] = {}
    for relative_path in ROBOTIQ_PAD_COLLIDER_PATHS:
        section = relative_path.split("/")[-2]
        expected_friction = ROBOTIQ_PAD_FRICTION_BY_SECTION[section]
        collider_path = f"{robot_path}/{relative_path}"
        collider = stage.GetPrimAtPath(collider_path)
        bound_material = None
        relationship = None
        if collider.IsValid():
            bound_material, relationship = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        material = bound_material.GetPrim() if bound_material and bound_material.GetPrim().IsValid() else None
        material_path = str(material.GetPath()) if material is not None else None
        static_friction = None
        dynamic_friction = None
        if material is not None:
            static_value = material.GetAttribute("physics:staticFriction").Get()
            dynamic_value = material.GetAttribute("physics:dynamicFriction").Get()
            static_friction = None if static_value is None else float(static_value)
            dynamic_friction = None if dynamic_value is None else float(dynamic_value)
        pad_materials[relative_path] = {
            "collider_path": collider_path,
            "material_path": material_path,
            "binding_relationship": str(relationship.GetPath()) if relationship and relationship.IsValid() else None,
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "expected_friction": expected_friction,
            "correct": material_path == f"{robot_path}/robotiq_{section}_physics_material"
            and math.isclose(static_friction or math.nan, expected_friction, abs_tol=1.0e-6)
            and math.isclose(dynamic_friction or math.nan, expected_friction, abs_tol=1.0e-6),
        }
    pad_collision_roots = {
        relative_path: {
            "path": f"{robot_path}/{relative_path}",
            "is_instance": stage.GetPrimAtPath(f"{robot_path}/{relative_path}").IsInstance(),
            "is_instanceable": stage.GetPrimAtPath(f"{robot_path}/{relative_path}").IsInstanceable(),
            "is_instance_proxy": stage.GetPrimAtPath(f"{robot_path}/{relative_path}").IsInstanceProxy(),
        }
        for relative_path in ROBOTIQ_PAD_COLLISION_ROOT_PATHS
    }
    return {
        "pairs": pairs,
        "all_pairs_present": all(record["present"] for record in pairs.values()),
        "articulation_self_collision": articulation_self_collision,
        "articulation_self_collision_enabled": bool(articulation_self_collision)
        and all(record["enabled"] for record in articulation_self_collision),
        "pad_materials": pad_materials,
        "pad_collision_roots": pad_collision_roots,
        "all_pad_materials_correct": all(record["correct"] for record in pad_materials.values())
        and all(
            not record["is_instance"] and not record["is_instanceable"] and not record["is_instance_proxy"]
            for record in pad_collision_roots.values()
        ),
    }


class ValidationRunner:
    """Stateful single-environment validation sequence."""

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
        writer: VideoWriter,
        render_interval: int,
    ) -> None:
        self.sim = sim
        self.scene = scene
        self.robot = scene["robot"]
        self.table = scene["table"]
        self.camera = scene["inspection_camera"]
        self.gripper_sensor = scene["gripper_contacts"]
        self.left_pad_sensor = scene["left_pad_table"]
        self.right_pad_sensor = scene["right_pad_table"]
        self.left_inner_finger_sensor = scene["left_inner_finger_table"]
        self.right_inner_finger_sensor = scene["right_inner_finger_table"]
        self.mount_sensor = scene["mount_base_arm"]
        self.writer = writer
        self.render_interval = render_interval
        self.dt = sim.get_physics_dt()
        self.device = self.robot.device

        self.arm_ids = [self.robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES]
        self.master_id = self.robot.data.joint_names.index(MASTER_JOINT)
        self.trace_joint_names = [
            name for name in [*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES] if name in self.robot.data.joint_names
        ]
        self.trace_joint_ids = [self.robot.data.joint_names.index(name) for name in self.trace_joint_names]
        self.tool_body_id = self.robot.data.body_names.index("tool0")
        self.gripper_base_body_id = self.robot.data.body_names.index("robotiq_arg2f_base_link")
        self.pad_body_ids = [self.robot.data.body_names.index(name) for name in PAD_BODY_NAMES]
        self.gripper_body_ids = [
            index
            for index, name in enumerate(self.robot.data.body_names)
            if name == "robotiq_arg2f_base_link" or name.startswith("left_") or name.startswith("right_")
        ]
        if not self.robot.is_fixed_base:
            raise RuntimeError("This validation expects the UR5e base to remain fixed")
        self.ee_jacobian_id = self.tool_body_id - 1

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=self.device,
        )
        self.global_step = 0
        self.trace: list[dict[str, Any]] = []
        self.phase_events: list[dict[str, Any]] = []
        self.first_contact: dict[str, Any] | None = None
        self.max_contact_force = 0.0
        self.max_force_exceeded = False
        self.max_arm_gravity_compensation = 0.0
        self.camera_eye_offset = torch.tensor((0.58, -0.12, 0.27), device=self.device)
        self.camera_target_z_offset = -0.025

    def _steps(self, duration_s: float) -> int:
        return max(1, int(round(duration_s / self.dt)))

    def apply_gravity_compensation(self) -> None:
        """Feed forward the articulated payload gravity torque to the arm."""

        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[:, self.arm_ids]
        self.max_arm_gravity_compensation = max(
            self.max_arm_gravity_compensation,
            float(torch.max(torch.abs(gravity)).item()),
        )
        self.robot.set_joint_effort_target(gravity, joint_ids=self.arm_ids)

    def _tool_pose_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pose_w = self.robot.data.body_pose_w[:, self.tool_body_id]
        root_pose_w = self.robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

    def _pad_geometry_world(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return lower-box centres, opening-axis separation/gap, and lowest surface."""

        body_pos = self.robot.data.body_pos_w[:, self.pad_body_ids, :]
        body_quat = self.robot.data.body_quat_w[:, self.pad_body_ids, :]
        local_centres = torch.tensor(PAD_BOX_CENTERS_LOCAL_M, dtype=torch.float32, device=self.device)
        local_centres = local_centres.view(1, 1, 2, 3).expand(body_pos.shape[0], 2, -1, -1)
        expanded_quat = body_quat.unsqueeze(2).expand(-1, -1, 2, -1)
        box_centres = body_pos.unsqueeze(2) + quat_apply(
            expanded_quat.reshape(-1, 4), local_centres.reshape(-1, 3)
        ).reshape_as(local_centres)

        lower_centres = box_centres[:, :, 0, :]
        root_x = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        root_x = root_x.expand(body_pos.shape[0], -1)
        opening_axis = quat_apply(self.robot.data.body_quat_w[:, self.gripper_base_body_id, :], root_x)
        projected_centres = torch.einsum("epj,ej->ep", lower_centres, opening_axis)
        centre_separation = torch.abs(projected_centres[:, 1] - projected_centres[:, 0])
        rotations = matrix_from_quat(body_quat)
        opening_axis_local = torch.einsum("epij,ei->epj", rotations, opening_axis)
        half_size = 0.5 * torch.tensor(PAD_BOX_SIZE_M, dtype=torch.float32, device=self.device)
        projected_half_extent = (torch.abs(opening_axis_local) * half_size).sum(dim=-1)
        interval_lower = projected_centres - projected_half_extent
        interval_upper = projected_centres + projected_half_extent
        surface_gap = interval_lower.max(dim=1).values - interval_upper.min(dim=1).values

        world_z_half_extent = (torch.abs(rotations[:, :, 2, :]) * half_size).sum(dim=-1)
        lowest_surface_z = (box_centres[..., 2] - world_z_half_extent.unsqueeze(-1)).amin(dim=(1, 2))
        return lower_centres, centre_separation, surface_gap, lowest_surface_z

    def _pad_midpoint_world(self) -> torch.Tensor:
        lower_centres, _, _, _ = self._pad_geometry_world()
        return lower_centres.mean(dim=1)

    @staticmethod
    def _filtered_force(sensor: Any) -> float:
        matrix = sensor.data.force_matrix_w
        if matrix is None or matrix.numel() == 0:
            return 0.0
        return float(torch.linalg.vector_norm(matrix[0], dim=-1).max().item())

    def _contact_forces(self) -> dict[str, Any]:
        forces = self.gripper_sensor.data.net_forces_w[0]
        norms = torch.linalg.vector_norm(forces, dim=-1)
        broad_force = float(norms.max().item()) if norms.numel() else 0.0
        body_index = int(torch.argmax(norms).item()) if norms.numel() else -1
        body_name = self.gripper_sensor.body_names[body_index] if body_index >= 0 else None
        left_force = self._filtered_force(self.left_pad_sensor)
        right_force = self._filtered_force(self.right_pad_sensor)
        left_inner_force = self._filtered_force(self.left_inner_finger_sensor)
        right_inner_force = self._filtered_force(self.right_inner_finger_sensor)
        table_forces = {
            "left_inner_finger_pad": left_force,
            "right_inner_finger_pad": right_force,
            "left_inner_finger": left_inner_force,
            "right_inner_finger": right_inner_force,
        }
        table_body = max(table_forces, key=table_forces.get)
        table_force = table_forces[table_body]
        if table_force <= 0.0:
            table_body = None
        result = {
            # Phase control and validation use external table contact only.
            "contact_force_n": table_force,
            "gripper_contact_force_n": broad_force,
            "left_pad_table_force_n": left_force,
            "right_pad_table_force_n": right_force,
            "left_inner_finger_table_force_n": left_inner_force,
            "right_inner_finger_table_force_n": right_inner_force,
            "max_gripper_contact_body": body_name,
            "max_contact_body": table_body,
        }
        mount_matrix = self.mount_sensor.data.force_matrix_w
        for filter_index, neighbor_name in enumerate(MOUNT_NEIGHBOR_BODY_NAMES):
            force = 0.0
            if mount_matrix is not None and mount_matrix.numel() and filter_index < mount_matrix.shape[-2]:
                force = float(torch.linalg.vector_norm(mount_matrix[0, 0, filter_index]).item())
            result[f"mount_base_{neighbor_name}_force_n"] = force
        return result

    def _collect_row(
        self,
        phase: str,
        phase_time_s: float,
        finger_command: float,
        arm_target: torch.Tensor,
    ) -> dict[str, Any]:
        joint_pos = self.robot.data.joint_pos[0, self.trace_joint_ids].detach().cpu().tolist()
        joint_vel = self.robot.data.joint_vel[0, self.trace_joint_ids].detach().cpu().tolist()
        tool_pos = self.robot.data.body_pos_w[0, self.tool_body_id].detach().cpu().tolist()
        tool_quat = self.robot.data.body_quat_w[0, self.tool_body_id].detach().cpu().tolist()
        pad_centres, pad_center_gap, pad_surface_gap, pad_lowest_surface_z = self._pad_geometry_world()
        pad_pos = pad_centres[0].detach().cpu()
        table_pos = self.table.data.root_pos_w[0].detach().cpu().tolist()
        gripper_body_pos = self.robot.data.body_pos_w[0, self.gripper_body_ids].detach().cpu()
        forces = self._contact_forces()

        row: dict[str, Any] = {
            "step": self.global_step,
            "time_s": self.global_step * self.dt,
            "phase": phase,
            "phase_time_s": phase_time_s,
            "finger_command": float(finger_command),
            "pad_center_gap_m": float(pad_center_gap[0].item()),
            "pad_surface_gap_m": float(pad_surface_gap[0].item()),
            "pad_lowest_surface_z_m": float(pad_lowest_surface_z[0].item()),
            "left_pad_x_m": float(pad_pos[0, 0].item()),
            "left_pad_y_m": float(pad_pos[0, 1].item()),
            "left_pad_z_m": float(pad_pos[0, 2].item()),
            "right_pad_x_m": float(pad_pos[1, 0].item()),
            "right_pad_y_m": float(pad_pos[1, 1].item()),
            "right_pad_z_m": float(pad_pos[1, 2].item()),
            "min_gripper_body_center_z_m": float(gripper_body_pos[:, 2].min().item()),
            "tool0_x_m": float(tool_pos[0]),
            "tool0_y_m": float(tool_pos[1]),
            "tool0_z_m": float(tool_pos[2]),
            "tool0_qw": float(tool_quat[0]),
            "tool0_qx": float(tool_quat[1]),
            "tool0_qy": float(tool_quat[2]),
            "tool0_qz": float(tool_quat[3]),
            "table_x_m": float(table_pos[0]),
            "table_y_m": float(table_pos[1]),
            "table_z_m": float(table_pos[2]),
            **forces,
        }
        arm_target_cpu = arm_target[0].detach().cpu().tolist()
        for index, name in enumerate(self.trace_joint_names):
            row[f"joint_pos__{name}"] = float(joint_pos[index])
            row[f"joint_vel__{name}"] = float(joint_vel[index])
        for index, name in enumerate(ARM_JOINT_NAMES):
            row[f"arm_target__{name}"] = float(arm_target_cpu[index])
        return row

    def _write_video_frame(self, row: dict[str, Any]) -> None:
        rgb = self.camera.data.output["rgb"][0].detach().cpu().numpy()
        frame = np.asarray(rgb[..., :3]).copy()
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=False)
        frame = np.ascontiguousarray(frame)

        overlay_height = min(112, max(72, frame.shape[0] // 6))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.75, frame.shape[1] / 1600.0))
        line_1 = f"{row['phase']}  t={row['time_s']:.2f}s  command={row['finger_command']:.3f} rad"
        line_2 = (
            f"pad surface={1000.0 * row['pad_surface_gap_m']:.1f} mm "
            f"(centres={1000.0 * row['pad_center_gap_m']:.1f})  "
            f"table={row['contact_force_n']:.2f} N  jaw/net={row['gripper_contact_force_n']:.2f} N  "
            f"tool0 z={row['tool0_z_m']:.3f} m"
        )
        cv2.putText(frame, line_1, (18, 34), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        color = (255, 90, 90) if row["contact_force_n"] >= args_cli.contact_threshold else (130, 255, 170)
        cv2.putText(frame, line_2, (18, 70), font, scale, color, 1, cv2.LINE_AA)
        if self.first_contact is not None:
            text = f"FIRST CONTACT: {self.first_contact['phase']} / {self.first_contact['body']}"
            cv2.putText(frame, text, (18, 103), font, scale, (255, 210, 80), 1, cv2.LINE_AA)
        self.writer.write(frame)

    def step(
        self,
        *,
        phase: str,
        phase_time_s: float,
        arm_target: torch.Tensor,
        finger_command: float,
    ) -> dict[str, Any]:
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_ids)
        self.apply_gravity_compensation()
        finger = torch.tensor([[finger_command]], dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(finger, joint_ids=[self.master_id])
        # Pose the camera before the render associated with this physics step.
        # Tracking the pad midpoint keeps both the free-space linkage motion and
        # the later millimetre-scale contact readable in one close-up video.
        if self.global_step % self.render_interval == 0:
            self._track_camera()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

        row = self._collect_row(phase, phase_time_s, finger_command, arm_target)
        contact_force = float(row["contact_force_n"])
        self.max_contact_force = max(self.max_contact_force, contact_force)
        self.max_force_exceeded = self.max_force_exceeded or contact_force > args_cli.abort_force
        if self.first_contact is None and contact_force >= args_cli.contact_threshold:
            self.first_contact = {
                "step": self.global_step,
                "time_s": row["time_s"],
                "phase": phase,
                "phase_time_s": phase_time_s,
                "force_n": contact_force,
                "body": row["max_contact_body"],
                "tool0_position_m": [row["tool0_x_m"], row["tool0_y_m"], row["tool0_z_m"]],
                "pad_midpoint_z_m": 0.5 * (row["left_pad_z_m"] + row["right_pad_z_m"]),
                "pad_lowest_surface_z_m": row["pad_lowest_surface_z_m"],
            }
        self.trace.append(row)
        if self.global_step % self.render_interval == 0:
            self._write_video_frame(row)
        self.global_step += 1
        return row

    def _track_camera(self) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mid = self._pad_midpoint_world()[0]
        target = pad_mid.clone()
        target[2] += self.camera_target_z_offset
        # View mostly along -X so left/right finger separation remains visible.
        eye = target + self.camera_eye_offset
        self.camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        return eye, target

    def setup_camera(self) -> dict[str, Any]:
        eye, target = self._track_camera()
        self.sim.render()
        self.sim.render()
        return {
            "initial_eye": eye.detach().cpu().tolist(),
            "initial_target": target.detach().cpu().tolist(),
            "tracking": "pad_midpoint",
            "eye_offset_m": self.camera_eye_offset.detach().cpu().tolist(),
            "target_z_offset_m": self.camera_target_z_offset,
        }

    def cleanup_camera(self) -> None:
        """Detach annotators and destroy Hydra textures before Kit shutdown."""

        # IsaacLab Camera.__del__ performs the same detachment, but scene and
        # callback references can keep the object alive past SimulationApp.close.
        for annotators in self.camera._rep_registry.values():
            for annotator, render_product_path in zip(annotators, self.camera._render_product_paths):
                annotator.detach([render_product_path])
        self.camera._rep_registry = {name: [] for name in self.camera._rep_registry}
        self.camera._clear_callbacks()
        from omni.replicator.core.scripts.utils.viewport_manager import destroy_hydra_textures

        destroy_hydra_textures()

    def _ik_joint_target(self, desired_pos_b: torch.Tensor, desired_quat_b: torch.Tensor) -> torch.Tensor:
        current_pos_b, current_quat_b = self._tool_pose_base()
        command = torch.cat((desired_pos_b, desired_quat_b), dim=-1)
        self.ik.set_command(command)
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobian_id, :, self.arm_ids]
        current_joint_pos = self.robot.data.joint_pos[:, self.arm_ids]
        target = self.ik.compute(current_pos_b, current_quat_b, jacobian, current_joint_pos)
        max_delta = args_cli.max_arm_speed * self.dt
        target = current_joint_pos + torch.clamp(target - current_joint_pos, -max_delta, max_delta)
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        return torch.clamp(target, min=limits[..., 0], max=limits[..., 1])

    def _begin_phase(self, name: str) -> None:
        self.phase_events.append(
            {"phase": name, "start_step": self.global_step, "start_time_s": self.global_step * self.dt}
        )
        print(f"[validation] phase={name}", flush=True)

    def _end_phase(self) -> None:
        self.phase_events[-1].update({"end_step": self.global_step, "end_time_s": self.global_step * self.dt})

    def run_joint_phase(
        self,
        name: str,
        duration_s: float,
        arm_start: torch.Tensor,
        arm_end: torch.Tensor,
        finger_start: float,
        finger_end: float,
    ) -> torch.Tensor:
        self._begin_phase(name)
        count = self._steps(duration_s)
        last_target = arm_end
        for index in range(count):
            alpha = float(index + 1) / count
            arm_target = torch.lerp(arm_start, arm_end, alpha)
            finger = (1.0 - alpha) * finger_start + alpha * finger_end
            self.step(
                phase=name,
                phase_time_s=(index + 1) * self.dt,
                arm_target=arm_target,
                finger_command=finger,
            )
            last_target = arm_target
        self._end_phase()
        return last_target

    def run_ik_phase(
        self,
        name: str,
        duration_s: float,
        start_pos_b: torch.Tensor,
        end_pos_b: torch.Tensor,
        quat_b: torch.Tensor,
        *,
        stop_on_contact: bool,
    ) -> tuple[bool, torch.Tensor]:
        self._begin_phase(name)
        count = self._steps(duration_s)
        last_target = self.robot.data.joint_pos[:, self.arm_ids].clone()
        contacted = False
        for index in range(count):
            alpha = float(index + 1) / count
            desired_pos = torch.lerp(start_pos_b, end_pos_b, alpha)
            last_target = self._ik_joint_target(desired_pos, quat_b)
            row = self.step(
                phase=name,
                phase_time_s=(index + 1) * self.dt,
                arm_target=last_target,
                finger_command=GRIPPER_OPEN_POS,
            )
            contacted = float(row["contact_force_n"]) >= args_cli.contact_threshold
            if stop_on_contact and contacted:
                break
        self._end_phase()
        return contacted, last_target

    def run(self) -> dict[str, Any]:
        if args_cli.quick:
            durations = {
                "settle": 0.25,
                "gripper_ramp": 0.55,
                "gripper_hold": 0.25,
                "approach": 1.0,
                "approach_hold": 0.25,
                "descent": 1.75,
                "contact_hold": 0.45,
                "retract": 1.0,
                "final_hold": 0.25,
            }
        else:
            durations = {
                "settle": 0.75,
                "gripper_ramp": 2.0,
                "gripper_hold": 0.75,
                "approach": 2.5,
                "approach_hold": 0.75,
                "descent": 5.0,
                "contact_hold": 1.5,
                "retract": 2.5,
                "final_hold": 0.75,
            }

        home_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.run_joint_phase(
            "settle_open",
            durations["settle"],
            home_arm,
            home_arm,
            GRIPPER_OPEN_POS,
            GRIPPER_OPEN_POS,
        )
        cycle_start_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()
        cycle_start_tool = self.robot.data.body_pos_w[:, self.tool_body_id].clone()
        self.run_joint_phase(
            "gripper_close",
            durations["gripper_ramp"],
            home_arm,
            home_arm,
            GRIPPER_OPEN_POS,
            GRIPPER_CLOSED_POS,
        )
        self.run_joint_phase(
            "closed_hold",
            durations["gripper_hold"],
            home_arm,
            home_arm,
            GRIPPER_CLOSED_POS,
            GRIPPER_CLOSED_POS,
        )
        self.run_joint_phase(
            "gripper_open",
            durations["gripper_ramp"],
            home_arm,
            home_arm,
            GRIPPER_CLOSED_POS,
            GRIPPER_OPEN_POS,
        )
        self.run_joint_phase(
            "reopened_hold",
            durations["gripper_hold"],
            home_arm,
            home_arm,
            GRIPPER_OPEN_POS,
            GRIPPER_OPEN_POS,
        )
        cycle_end_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()
        cycle_end_tool = self.robot.data.body_pos_w[:, self.tool_body_id].clone()

        current_pos_b, current_quat_b = self._tool_pose_base()
        _, _, _, pad_lowest_surface_z = self._pad_geometry_world()
        approach_delta_z = args_cli.table_top + args_cli.approach_clearance - float(pad_lowest_surface_z[0])
        approach_pos_b = current_pos_b.clone()
        approach_pos_b[:, 2] += approach_delta_z
        early_contact, _ = self.run_ik_phase(
            "approach_table",
            durations["approach"],
            current_pos_b,
            approach_pos_b,
            current_quat_b,
            stop_on_contact=True,
        )
        if not early_contact:
            actual_pos_b, actual_quat_b = self._tool_pose_base()
            early_contact, _ = self.run_ik_phase(
                "approach_hold",
                durations["approach_hold"],
                actual_pos_b,
                approach_pos_b,
                actual_quat_b,
                stop_on_contact=True,
            )
        approach_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()

        contact_command = approach_arm.clone()
        if not early_contact:
            descent_start_b, descent_quat_b = self._tool_pose_base()
            _, _, _, current_pad_lowest_surface_z = self._pad_geometry_world()
            descent_delta_z = args_cli.table_top - args_cli.descent_overtravel - float(current_pad_lowest_surface_z[0])
            descent_end_b = descent_start_b.clone()
            descent_end_b[:, 2] += descent_delta_z
            _, contact_command = self.run_ik_phase(
                "slow_descent",
                durations["descent"],
                descent_start_b,
                descent_end_b,
                descent_quat_b,
                stop_on_contact=True,
            )

        contact_arm = self.robot.data.joint_pos[:, self.arm_ids].clone()
        hold_target = contact_arm if self.max_force_exceeded else contact_command
        hold_duration = min(0.20, durations["contact_hold"]) if self.max_force_exceeded else durations["contact_hold"]
        self.run_joint_phase(
            "first_contact_hold",
            hold_duration,
            hold_target,
            hold_target,
            GRIPPER_OPEN_POS,
            GRIPPER_OPEN_POS,
        )

        retract_start = self.robot.data.joint_pos[:, self.arm_ids].clone()
        self.run_joint_phase(
            "retract",
            durations["retract"],
            retract_start,
            approach_arm,
            GRIPPER_OPEN_POS,
            GRIPPER_OPEN_POS,
        )
        self.run_joint_phase(
            "final_hold",
            durations["final_hold"],
            approach_arm,
            approach_arm,
            GRIPPER_OPEN_POS,
            GRIPPER_OPEN_POS,
        )

        return {
            "durations_s": durations,
            "early_contact": early_contact,
            "cycle_arm_drift_max_rad": float(torch.max(torch.abs(cycle_end_arm - cycle_start_arm)).item()),
            "cycle_tool_drift_m": float(
                torch.linalg.vector_norm(cycle_end_tool - cycle_start_tool, dim=-1).max().item()
            ),
        }

    def summarize(
        self,
        mimic_schemas: dict[str, dict[str, Any]],
        collision_filters: dict[str, Any],
        camera_pose: dict[str, Any],
        run_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        def phase_rows(name: str) -> list[dict[str, Any]]:
            return [row for row in self.trace if row["phase"] == name]

        def tail_median(rows: list[dict[str, Any]], key: str) -> float | None:
            if not rows:
                return None
            tail = rows[max(0, len(rows) // 2) :]
            return float(np.median([float(row[key]) for row in tail]))

        open_center_gap = tail_median(phase_rows("settle_open"), "pad_center_gap_m")
        closed_center_gap = tail_median(phase_rows("closed_hold"), "pad_center_gap_m")
        reopened_center_gap = tail_median(phase_rows("reopened_hold"), "pad_center_gap_m")
        open_surface_gap = tail_median(phase_rows("settle_open"), "pad_surface_gap_m")
        closed_surface_gap = tail_median(phase_rows("closed_hold"), "pad_surface_gap_m")
        reopened_surface_gap = tail_median(phase_rows("reopened_hold"), "pad_surface_gap_m")
        gap_reduction = (
            None if open_surface_gap is None or closed_surface_gap is None else open_surface_gap - closed_surface_gap
        )
        reopen_error = (
            None
            if open_surface_gap is None or reopened_surface_gap is None
            else abs(open_surface_gap - reopened_surface_gap)
        )

        follower_motion: dict[str, float | None] = {}
        follower_tracking_error: dict[str, float | None] = {}
        loaded_follower_tracking_error: dict[str, float | None] = {}
        close_rows = phase_rows("gripper_close") + phase_rows("closed_hold")
        closed_rows = phase_rows("closed_hold")
        loaded_rows = phase_rows("first_contact_hold")
        for name in MIMIC_JOINT_NAMES:
            key = f"joint_pos__{name}"
            values = [float(row[key]) for row in close_rows if key in row]
            follower_motion[name] = max(values) - min(values) if values else None
            tracking_errors = [
                abs(float(row[key]) - float(row[f"joint_pos__{MASTER_JOINT}"]))
                for row in closed_rows
                if key in row and f"joint_pos__{MASTER_JOINT}" in row
            ]
            follower_tracking_error[name] = max(tracking_errors) if tracking_errors else None
            loaded_tracking_errors = [
                abs(float(row[key]) - float(row[f"joint_pos__{MASTER_JOINT}"]))
                for row in loaded_rows
                if key in row and f"joint_pos__{MASTER_JOINT}" in row
            ]
            loaded_follower_tracking_error[name] = max(loaded_tracking_errors) if loaded_tracking_errors else None

        numeric_values = [
            float(value)
            for row in self.trace
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        finite_trace = bool(numeric_values) and all(math.isfinite(value) for value in numeric_values)

        table_positions = np.asarray(
            [[row["table_x_m"], row["table_y_m"], row["table_z_m"]] for row in self.trace], dtype=np.float64
        )
        table_drift = float(np.max(np.linalg.norm(table_positions - table_positions[0], axis=1)))
        final_rows = phase_rows("final_hold")
        final_force = tail_median(final_rows, "contact_force_n")
        hold_rows = phase_rows("first_contact_hold")
        hold_contact_peak = max((float(row["contact_force_n"]) for row in hold_rows), default=0.0)
        hold_tail = hold_rows[max(0, len(hold_rows) // 2) :]
        hold_contact_tail_median = (
            float(np.median([float(row["contact_force_n"]) for row in hold_tail])) if hold_tail else 0.0
        )
        hold_contact_fraction = (
            sum(float(row["contact_force_n"]) >= args_cli.sustained_contact_threshold for row in hold_rows)
            / len(hold_rows)
            if hold_rows
            else 0.0
        )
        closed_self_contact_tail_median = tail_median(closed_rows, "gripper_contact_force_n")
        closed_self_contact_peak_row = max(
            closed_rows,
            key=lambda row: float(row["gripper_contact_force_n"]),
            default=None,
        )
        closed_self_contact_body = (
            None if closed_self_contact_peak_row is None else closed_self_contact_peak_row["max_gripper_contact_body"]
        )
        mount_contact_peak = {
            neighbor_name: max(
                (float(row[f"mount_base_{neighbor_name}_force_n"]) for row in self.trace),
                default=0.0,
            )
            for neighbor_name in MOUNT_NEIGHBOR_BODY_NAMES
        }

        runtime_gripper_joints = list(self.robot.actuators["gripper"].joint_names)
        master_drive_ok = bool(mimic_schemas[MASTER_JOINT]["drive_schemas"])
        followers_mimic_ok = all(bool(mimic_schemas[name]["mimic_schemas"]) for name in MIMIC_JOINT_NAMES)
        followers_drive_free = all(not mimic_schemas[name]["drive_schemas"] for name in MIMIC_JOINT_NAMES)
        follower_schema_mapping_ok = all(
            len(mimic_schemas[name].get("reference_joint", [])) == 1
            and mimic_schemas[name]["reference_joint"][0].endswith(f"/{MASTER_JOINT}")
            and mimic_schemas[name].get("reference_axis") == "rotX"
            and math.isclose(mimic_schemas[name].get("gearing", math.nan), -1.0, abs_tol=1.0e-9)
            and math.isclose(mimic_schemas[name].get("offset", math.nan), 0.0, abs_tol=1.0e-9)
            and math.isclose(mimic_schemas[name].get("natural_frequency", math.nan), 0.0, abs_tol=1.0e-9)
            and math.isclose(mimic_schemas[name].get("damping_ratio", math.nan), 0.0, abs_tol=1.0e-9)
            for name in MIMIC_JOINT_NAMES
        )
        minimum_gap_change = max(0.005, 0.10 * open_surface_gap) if open_surface_gap is not None else math.inf

        checks = {
            "arm_self_collision_remains_enabled": collision_filters["articulation_self_collision_enabled"],
            "gripper_internal_collision_filters_present": collision_filters["all_pairs_present"],
            "pad_physics_materials_match_source_coefficients": collision_filters["all_pad_materials_correct"],
            "mount_interface_has_no_contact": all(force < 1.0e-4 for force in mount_contact_peak.values()),
            "master_is_only_gripper_actuator_joint": runtime_gripper_joints == [MASTER_JOINT],
            "master_has_usd_drive": master_drive_ok,
            "all_followers_have_physx_mimic_api": followers_mimic_ok,
            "followers_have_no_usd_drive": followers_drive_free,
            "follower_mimic_schema_mapping_is_correct": follower_schema_mapping_ok,
            "all_followers_move": all(value is not None and value > 0.03 for value in follower_motion.values()),
            "all_followers_track_master": all(
                value is not None and value < MIMIC_TRACKING_TOLERANCE_RAD for value in follower_tracking_error.values()
            ),
            "all_followers_track_master_under_contact": all(
                value is not None and value < MIMIC_TRACKING_TOLERANCE_RAD
                for value in loaded_follower_tracking_error.values()
            ),
            "open_pad_gap_matches_cad": (
                open_surface_gap is not None
                and abs(open_surface_gap - EXPECTED_OPEN_PAD_SURFACE_GAP_M) < OPEN_PAD_GAP_TOLERANCE_M
            ),
            "closed_gap_is_nonpenetrating_and_submillimetre": (
                closed_surface_gap is not None
                and abs(closed_surface_gap - EXPECTED_PHYSICAL_CLOSED_PAD_SURFACE_GAP_M) < CLOSED_PAD_GAP_TOLERANCE_M
            ),
            "closed_self_contact_is_bounded": closed_self_contact_tail_median is not None
            and 0.05 <= closed_self_contact_tail_median < 50.0
            and closed_self_contact_body in VALID_CLOSED_SELF_CONTACT_BODY_NAMES,
            "pad_surface_gap_closes": gap_reduction is not None and gap_reduction > minimum_gap_change,
            "pad_surface_gap_reopens": reopen_error is not None and reopen_error < REOPEN_PAD_GAP_TOLERANCE_M,
            "arm_stable_during_gripper_cycle": (
                run_metadata["cycle_arm_drift_max_rad"] < 0.02 and run_metadata["cycle_tool_drift_m"] < 0.005
            ),
            "first_contact_detected_during_slow_descent": (
                self.first_contact is not None and self.first_contact["phase"] == "slow_descent"
            ),
            "contact_persists_during_hold": (
                hold_contact_tail_median >= args_cli.sustained_contact_threshold and hold_contact_fraction >= 0.8
            ),
            "contact_force_below_abort_limit": self.max_contact_force <= args_cli.abort_force,
            "contact_cleared_after_retract": final_force is not None and final_force < args_cli.contact_threshold,
            "table_remains_static": table_drift < 1.0e-5,
            "trace_is_finite": finite_trace,
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": {
                "pad_box_centres_local_m": PAD_BOX_CENTERS_LOCAL_M,
                "pad_box_size_m": PAD_BOX_SIZE_M,
                "open_pad_center_gap_m": open_center_gap,
                "closed_pad_center_gap_m": closed_center_gap,
                "reopened_pad_center_gap_m": reopened_center_gap,
                "open_pad_surface_gap_m": open_surface_gap,
                "closed_pad_surface_gap_m": closed_surface_gap,
                "reopened_pad_surface_gap_m": reopened_surface_gap,
                "pad_surface_gap_reduction_m": gap_reduction,
                "reopen_surface_gap_error_m": reopen_error,
                "follower_joint_motion_range_rad": follower_motion,
                "follower_master_tracking_error_rad": follower_tracking_error,
                "loaded_follower_master_tracking_error_rad": loaded_follower_tracking_error,
                "closed_self_contact_tail_median_force_n": closed_self_contact_tail_median,
                "closed_self_contact_body": closed_self_contact_body,
                "mount_contact_peak_force_n": mount_contact_peak,
                "max_arm_gravity_compensation_nm": self.max_arm_gravity_compensation,
                "max_contact_force_n": self.max_contact_force,
                "contact_hold_peak_force_n": hold_contact_peak,
                "contact_hold_tail_median_force_n": hold_contact_tail_median,
                "contact_hold_fraction_above_threshold": hold_contact_fraction,
                "final_contact_force_n": final_force,
                "table_translation_drift_m": table_drift,
                **run_metadata,
            },
            "first_contact": self.first_contact,
            "camera": camera_pose,
            "mimic_joint_schemas": mimic_schemas,
            "collision_filters": collision_filters,
            "runtime_gripper_actuator_joints": runtime_gripper_joints,
            "gripper_contact_sensor_bodies": list(self.gripper_sensor.body_names),
            "phase_events": self.phase_events,
            "num_physics_samples": len(self.trace),
        }


def main() -> int:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "ur5e_robotiq_2f85_validation.mp4"
    trace_json_path = output_dir / "ur5e_robotiq_2f85_trace.json"
    trace_csv_path = output_dir / "ur5e_robotiq_2f85_trace.csv"
    summary_path = output_dir / "ur5e_robotiq_2f85_summary.json"

    render_interval = max(1, int(round(args_cli.physics_hz / args_cli.video_fps)))
    actual_video_fps = args_cli.physics_hz / render_interval
    if not math.isclose(actual_video_fps, args_cli.video_fps, rel_tol=0.0, abs_tol=1.0e-6):
        print(
            f"[validation] requested video fps {args_cli.video_fps} is not an integer divisor of "
            f"physics Hz {args_cli.physics_hz}; using {actual_video_fps:.6g} fps",
            flush=True,
        )

    writer = VideoWriter(str(video_path), fps=int(round(actual_video_fps)))
    runner: ValidationRunner | None = None
    summary: dict[str, Any] = {
        "status": "running",
        "passed": False,
        "artifacts": {
            "video": str(video_path),
            "trace_json": str(trace_json_path),
            "trace_csv": str(trace_csv_path),
            "summary": str(summary_path),
        },
        "configuration": {
            "isaaclab_runtime_version": getattr(isaaclab, "__version__", "unknown"),
            "device": args_cli.device,
            "physics_hz": args_cli.physics_hz,
            "video_fps": actual_video_fps,
            "resolution": [args_cli.width, args_cli.height],
            "table_top_m": args_cli.table_top,
            "approach_clearance_m": args_cli.approach_clearance,
            "descent_overtravel_m": args_cli.descent_overtravel,
            "contact_threshold_n": args_cli.contact_threshold,
            "sustained_contact_threshold_n": args_cli.sustained_contact_threshold,
            "abort_force_n": args_cli.abort_force,
            "table_contact_offset_m": args_cli.contact_offset,
            "pad_box_centres_local_m": PAD_BOX_CENTERS_LOCAL_M,
            "pad_box_size_m": PAD_BOX_SIZE_M,
            "force_usd_conversion": not args_cli.reuse_usd_cache,
            "robot_gravity_enabled": True,
            "arm_gravity_compensation_enabled": True,
            "usd_cache_dir": str(Path(args_cli.usd_cache_dir).expanduser().resolve()),
            "quick": args_cli.quick,
        },
    }
    exit_code = 1

    try:
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / args_cli.physics_hz,
            render_interval=render_interval,
            device=args_cli.device,
            use_fabric=True,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                enable_ccd=False,
                bounce_threshold_velocity=0.2,
                min_position_iteration_count=1,
                max_position_iteration_count=64,
                min_velocity_iteration_count=0,
                max_velocity_iteration_count=8,
            ),
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(_make_scene_cfg(_configure_robot()))
        sim.reset()

        robot = scene["robot"]
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        scene.reset()

        runner = ValidationRunner(sim, scene, writer, render_interval)
        initial_arm = robot.data.joint_pos[:, runner.arm_ids].clone()
        # A few unrecorded steps initialize mimic constraints, contacts, and RTX buffers.
        for _ in range(4):
            robot.set_joint_position_target(initial_arm, joint_ids=runner.arm_ids)
            runner.apply_gravity_compensation()
            robot.set_joint_position_target(
                torch.tensor([[GRIPPER_OPEN_POS]], device=robot.device),
                joint_ids=[runner.master_id],
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        camera_pose = runner.setup_camera()
        mimic_schemas = _inspect_mimic_schemas()
        collision_filters = _inspect_collision_filters()
        runtime_gripper_joints = list(robot.actuators["gripper"].joint_names)
        if runtime_gripper_joints != [MASTER_JOINT]:
            raise RuntimeError(
                f"Validation must drive only {MASTER_JOINT!r}, got gripper actuator joints {runtime_gripper_joints}"
            )
        missing_mimic = [name for name in MIMIC_JOINT_NAMES if not mimic_schemas[name]["mimic_schemas"]]
        follower_drives = [name for name in MIMIC_JOINT_NAMES if mimic_schemas[name]["drive_schemas"]]
        if missing_mimic or follower_drives:
            raise RuntimeError(
                f"PhysX mimic schema validation failed: missing_mimic={missing_mimic}, follower_drives={follower_drives}"
            )

        print(f"[validation] joints={robot.data.joint_names}", flush=True)
        print(f"[validation] gripper contact bodies={runner.gripper_sensor.body_names}", flush=True)
        run_metadata = runner.run()
        summary.update(runner.summarize(mimic_schemas, collision_filters, camera_pose, run_metadata))
        exit_code = 0 if summary["passed"] else 2
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        traceback.print_exc()
        exit_code = 1
    finally:
        writer.release()
        if runner is not None:
            try:
                runner.cleanup_camera()
            except Exception:
                print("[validation] warning: failed to clean up camera resources", file=sys.stderr)
                traceback.print_exc()
        trace = runner.trace if runner is not None else []
        _write_csv(trace_csv_path, trace)
        _write_json(
            trace_json_path,
            {
                "metadata": summary.get("configuration", {}),
                "samples": trace,
            },
        )
        _write_json(summary_path, summary)
        print(f"[validation] status={summary['status']}", flush=True)
        print(f"[validation] video={video_path}", flush=True)
        print(f"[validation] trace_json={trace_json_path}", flush=True)
        print(f"[validation] trace_csv={trace_csv_path}", flush=True)
        print(f"[validation] summary={summary_path}", flush=True)

    return exit_code


if __name__ == "__main__":
    code = 1
    code = main()
    if args_cli.headless:
        # On this Isaac Sim 5 build, SimulationApp.close() can wait forever in
        # Replicator shutdown after a CameraCfg has streamed frames, even after
        # annotators and Hydra textures have been explicitly released.  All
        # artifacts are synchronously closed above, so terminate the dedicated
        # headless process without entering that faulty global shutdown path.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    else:
        # Camera annotators can leave Replicator's orchestrator in a non-stopped
        # state even though all frames have already been consumed and the video
        # writer is closed.  Waiting for that unrelated orchestrator can hang
        # shutdown indefinitely on Isaac Sim 5.
        simulation_app.close(wait_for_replicator=False)
    sys.exit(code)
