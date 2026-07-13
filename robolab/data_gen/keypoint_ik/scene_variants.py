# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene/task/job expansion for the 7 x 7 x 3 keypoint-IK dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shlex
from typing import Any

from robolab.constants import OBJECT_CATALOG_PATH, PACKAGE_DIR, SCENE_DIR, TASK_DIR


TARGET_OBJECT_VARIANT_COUNT = 7
CONTAINER_OBJECT_VARIANT_COUNT = 7
APPEARANCE_VARIANT_COUNT = 3

OBJECT_A_PRIM_NAME = "object_a"
OBJECT_B_PRIM_NAME = "object_b"

DEFAULT_TARGET_OBJECTS = (
    "banana",
    "orange_01",
    "lemon_02",
    "butter",
    "mustard",
    "rubiks_cube",
    "lime01",
)

DEFAULT_CONTAINER_OBJECTS = (
    "bowl",
    "plate_large",
    "raisin_box",
    "spoon_big",
    "butter",
    "mustard",
    "rubiks_cube",
)

FALLBACK_TARGET_OBJECTS = (
    "orange_02",
    "lemon_01",
    "pomegranate01",
    "tomato_soup_can",
    "sugar_box",
    "spam_can",
    "apple_01",
)

DEFAULT_APPEARANCES = (
    {"name": "original", "background_random": False, "background_seed_base": 0, "table_material": None},
    {
        "name": "random_background_lighting",
        "background_random": True,
        "background_seed_base": 1000,
        "table_material": None,
    },
    {
        "name": "random_table_material",
        "background_random": False,
        "background_seed_base": 0,
        "table_material": "Walnut_Planks",
    },
)

DEFAULT_OUTPUT_ROOT = "/mlp_vepfs/share/swy/cosmos3-framework"
DEFAULT_RAW_DATASET_DIR = f"{DEFAULT_OUTPUT_ROOT}/keypoint_ik_runs/robolabsim-147/raw"
DEFAULT_ATTEMPT_DATASET_DIR = f"{DEFAULT_OUTPUT_ROOT}/keypoint_ik_runs/robolabsim-147/attempts"
DEFAULT_LEROBOT_DIR = f"{DEFAULT_OUTPUT_ROOT}/lerobot/robolabsim-147"

TARGET_XY = (0.5097271203994751, -0.115806944668293)
CONTAINER_XY = (0.5611012578010559, 0.15261490643024445)
TABLETOP_CLEARANCE_M = 0.00285
DEFAULT_PROMPT_TEMPLATE = "pick up the {object_a} and place it near the {object_b}"


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    usd_path: str
    prim_path: str
    dims: tuple[float, float, float]
    object_class: str = ""


@dataclass(frozen=True)
class KeypointSceneVariant:
    variant_id: str
    target_object: str
    container_object: str
    target_asset_name: str
    container_asset_name: str
    target_label: str
    container_label: str
    scene_path: str
    task_path: str
    task_class: str
    target_dims: tuple[float, float, float]
    container_dims: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    container_xyz: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_trajectory_count(
    *,
    target_count: int = TARGET_OBJECT_VARIANT_COUNT,
    container_count: int = CONTAINER_OBJECT_VARIANT_COUNT,
    appearance_count: int = APPEARANCE_VARIANT_COUNT,
) -> int:
    """Return the requested target count, defaulting to 7 x 7 x 3."""

    return int(target_count) * int(container_count) * int(appearance_count)


def load_object_catalog(path: str | Path = OBJECT_CATALOG_PATH) -> dict[str, CatalogEntry]:
    """Load RoboLab's object catalog keyed by object name."""

    with Path(path).open() as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected object catalog list in {path}")

    catalog: dict[str, CatalogEntry] = {}
    for row in rows:
        if not isinstance(row, dict) or "name" not in row:
            continue
        name = str(row["name"])
        if name in catalog:
            continue
        dims = tuple(float(x) for x in row.get("dims", (0.0, 0.0, 0.0)))
        if len(dims) != 3:
            raise ValueError(f"Catalog entry {row.get('name')!r} has invalid dims: {row.get('dims')!r}")
        catalog[name] = CatalogEntry(
            name=name,
            usd_path=str(row["usd_path"]),
            prim_path=str(row.get("prim_path", f"/{row['name']}")),
            dims=dims,
            object_class=str(row.get("class", "")),
        )
    return catalog


def write_default_variants(
    *,
    scene_dir: str | Path = SCENE_DIR,
    task_dir: str | Path = Path(TASK_DIR) / "keypoint_pickplace",
    template_scene: str | Path = Path(SCENE_DIR) / "banana_bowl.usda",
    target_objects: tuple[str, ...] = DEFAULT_TARGET_OBJECTS,
    container_objects: tuple[str, ...] = DEFAULT_CONTAINER_OBJECTS,
    catalog_path: str | Path = OBJECT_CATALOG_PATH,
    clear_existing: bool = True,
) -> list[KeypointSceneVariant]:
    """Generate object-A/object-B scene and task files for the requested asset pairs."""

    catalog = load_object_catalog(catalog_path)
    text = Path(template_scene).read_text()
    scene_dir = Path(scene_dir)
    task_dir = Path(task_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        _remove_generated_files(scene_dir, "keypoint_pickplace_*.usda")
        _remove_generated_files(task_dir, "kp_*.py")
    init_path = task_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("\n")

    variants: list[KeypointSceneVariant] = []
    for target_asset in target_objects:
        for container_asset in container_objects:
            target_entry = _catalog_entry(catalog, target_asset)
            container_entry = _catalog_entry(catalog, container_asset)
            variant_id = f"{_slug(target_asset)}_to_{_slug(container_asset)}"
            scene_name = f"keypoint_pickplace_{variant_id}.usda"
            task_name = f"kp_{variant_id}.py"
            task_class = f"KeypointPickplace{_camel(target_asset)}To{_camel(container_asset)}Task"

            target_xyz = (*TARGET_XY, _tabletop_z(target_entry))
            container_xyz = (*CONTAINER_XY, _tabletop_z(container_entry))
            scene_text = render_scene_variant(
                text,
                target_entry=target_entry,
                container_entry=container_entry,
                target_xyz=target_xyz,
                container_xyz=container_xyz,
                target_prim_name=OBJECT_A_PRIM_NAME,
                container_prim_name=OBJECT_B_PRIM_NAME,
            )
            scene_path = scene_dir / scene_name
            scene_path.write_text(scene_text)

            task_path = task_dir / task_name
            target_label = _display_name(target_asset)
            container_label = _display_name(container_asset)
            task_path.write_text(
                render_task_variant(
                    task_class=task_class,
                    scene_name=scene_name,
                    target_object=OBJECT_A_PRIM_NAME,
                    container_object=OBJECT_B_PRIM_NAME,
                    target_label=target_label,
                    container_label=container_label,
                )
            )
            variants.append(
                KeypointSceneVariant(
                    variant_id=variant_id,
                    target_object=OBJECT_A_PRIM_NAME,
                    container_object=OBJECT_B_PRIM_NAME,
                    target_asset_name=target_asset,
                    container_asset_name=container_asset,
                    target_label=target_label,
                    container_label=container_label,
                    scene_path=str(scene_path),
                    task_path=str(task_path),
                    task_class=task_class,
                    target_dims=target_entry.dims,
                    container_dims=container_entry.dims,
                    target_xyz=target_xyz,
                    container_xyz=container_xyz,
                )
            )
    return variants


def render_scene_variant(
    template_text: str,
    *,
    target_entry: CatalogEntry,
    container_entry: CatalogEntry,
    target_xyz: tuple[float, float, float],
    container_xyz: tuple[float, float, float],
    target_prim_name: str = OBJECT_A_PRIM_NAME,
    container_prim_name: str = OBJECT_B_PRIM_NAME,
) -> str:
    """Render a scene by replacing the template banana and bowl prim blocks."""

    text = _replace_prim_block(
        template_text,
        old_name="bowl",
        new_name=container_prim_name,
        payload=_payload_for_scene_root(container_entry.usd_path),
        xyz=container_xyz,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    return _replace_prim_block(
        text,
        old_name="banana",
        new_name=target_prim_name,
        payload=_payload_for_scene_root(target_entry.usd_path),
        xyz=target_xyz,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def render_task_variant(
    *,
    task_class: str,
    scene_name: str,
    target_object: str,
    container_object: str,
    target_label: str,
    container_label: str,
) -> str:
    """Render a small task class for one generated scene."""

    termination_class = f"{task_class}Terminations"
    instruction = DEFAULT_PROMPT_TEMPLATE.format(object_a=target_label, object_b=container_label)
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task


@configclass
class {termination_class}:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class {task_class}(Task):
    contact_object_list = ["{target_object}", "{container_object}", "table"]
    scene = import_scene("{scene_name}", contact_object_list)
    terminations = {termination_class}
    instruction = {{
        "default": "{instruction}",
        "vague": "Move the picked object near the target object",
        "specific": "{instruction}",
    }}
    episode_length_s: int = 50
    attributes = ["semantics", "simple", "keypoint_ik"]
    subtasks = None
'''


def build_job_manifest(
    variants: list[KeypointSceneVariant],
    *,
    raw_dataset_dir: str | Path = DEFAULT_RAW_DATASET_DIR,
    appearances: tuple[dict[str, Any], ...] = DEFAULT_APPEARANCES,
    generator_script: str | Path = Path(PACKAGE_DIR) / "examples" / "generate_keypoint_ik_dataset.py",
    python_launcher: str = "/workspace/isaaclab/isaaclab.sh",
    ik_backend: str = "pinocchio",
    record: bool = True,
) -> list[dict[str, Any]]:
    """Expand scene variants into concrete generator jobs."""

    jobs: list[dict[str, Any]] = []
    raw_dataset_dir = Path(raw_dataset_dir)
    generator_script = Path(generator_script)
    for scene_index, variant in enumerate(variants):
        for appearance_index, appearance in enumerate(appearances):
            job_index = len(jobs)
            output_dir = raw_dataset_dir / f"{job_index:06d}_{variant.variant_id}_{appearance['name']}"
            background_random = bool(appearance.get("background_random", False))
            seed_base = int(appearance.get("background_seed_base", 0))
            background_seed = seed_base + scene_index
            table_material = appearance.get("table_material")
            command = [
                "TERM=xterm",
                str(python_launcher),
                "-p",
                str(generator_script),
                "--headless",
                "--ik-backend",
                ik_backend,
                "--num-episodes",
                "1",
                "--camera-preset",
                "wrist_left_right",
                "--task",
                variant.task_path,
                "--target-object",
                variant.target_object,
                "--container-object",
                variant.container_object,
                "--object-a-label",
                variant.target_label,
                "--object-b-label",
                variant.container_label,
                "--grasp-mode",
                "table_fixed",
                "--grasp-clearance-policy",
                "half_height_clamped",
                "--min-grasp-clearance-m",
                "0.018",
                "--max-grasp-clearance-m",
                "0.055",
                "--table-pre-grasp-clearance-m",
                "0.12",
                "--table-lift-clearance-m",
                "0.24",
                "--table-release-clearance-m",
                "0.12",
                "--table-tool0-tcp-offset-mode",
                "robotiq_2f85_open_pad_center",
                "--table-tcp-collision-margin-m",
                "0.005",
                "--success-mode",
                "near_target_xy",
                "--place-xy-tolerance-m",
                "0.10",
                "--prompt-template",
                DEFAULT_PROMPT_TEMPLATE,
                "--grasp-yaw-deg",
                "90",
                "--action-repeat",
                "4",
                "--settle-steps",
                "30",
                "--output-dir",
                str(output_dir),
            ]
            if record:
                command.append("--record")
            if background_random:
                command.extend(["--background-random", "--background-seed", str(background_seed)])
            if table_material:
                command.extend(["--table-material", str(table_material)])
            jobs.append(
                {
                    "job_index": job_index,
                    "scene_index": scene_index,
                    "appearance_index": appearance_index,
                    "appearance_name": appearance["name"],
                    "background_random": background_random,
                    "background_seed": background_seed if background_random else None,
                    "table_material": table_material,
                    "output_dir": str(output_dir),
                    "lerobot_output_dir": DEFAULT_LEROBOT_DIR,
                    "command": command,
                    "variant": variant.to_dict(),
                }
            )
    return jobs


def write_manifest(path: str | Path, jobs: list[dict[str, Any]]) -> None:
    """Write one JSON object per job."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True))
            f.write("\n")


def write_run_script(path: str | Path, jobs: list[dict[str, Any]]) -> None:
    """Write a sequential shell runner for the manifest jobs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "cd " + shlex.quote(PACKAGE_DIR),
        "",
    ]
    for job in jobs:
        lines.append(
            f"echo '[keypoint-ik-batch] job {job['job_index']:03d}: "
            f"{job['variant']['variant_id']} {job['appearance_name']}'"
        )
        lines.append(_shell_join(job["command"]))
        lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def _replace_prim_block(
    text: str,
    *,
    old_name: str,
    new_name: str,
    payload: str,
    xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> str:
    pattern = re.compile(
        rf'    def "{re.escape(old_name)}" \(\n'
        r"        prepend payload = @[^@]+@\n"
        r"    \)\n"
        r"    \{\n"
        r".*?"
        r"    \}",
        re.DOTALL,
    )
    replacement = _prim_block(new_name, payload=payload, xyz=xyz, quat_wxyz=quat_wxyz)
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Could not find prim block {old_name!r} in template scene")
    return new_text


def _prim_block(
    name: str,
    *,
    payload: str,
    xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
) -> str:
    q = ", ".join(_fmt(x) for x in quat_wxyz)
    p = ", ".join(_fmt(x) for x in xyz)
    return f'''    def "{name}" (
        prepend payload = @{payload}@
    )
    {{
        vector3f physics:angularVelocity = (0, 0, 0)
        vector3f physics:velocity = (0, 0, 0)
        quatf xformOp:orient = ({q})
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = ({p})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}'''


def _catalog_entry(catalog: dict[str, CatalogEntry], name: str) -> CatalogEntry:
    if name not in catalog:
        raise KeyError(f"Object {name!r} not found in {OBJECT_CATALOG_PATH}")
    return catalog[name]


def _remove_generated_files(directory: Path, pattern: str) -> None:
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def _payload_for_scene_root(catalog_usd_path: str) -> str:
    path = Path(catalog_usd_path)
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "assets" and parts[1] == "objects":
        return "../" + str(Path(*parts[1:]))
    return str(path)


def _tabletop_z(entry: CatalogEntry) -> float:
    return float(entry.dims[2] * 0.5 + TABLETOP_CLEARANCE_M)


def _display_name(value: str) -> str:
    return value.replace("_", " ")


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_").lower()


def _camel(value: str) -> str:
    parts = [part for part in re.split(r"[^0-9a-zA-Z]+", value) if part]
    text = "".join(part[:1].upper() + part[1:] for part in parts)
    if text and text[0].isdigit():
        text = "Obj" + text
    return text


def _fmt(value: float) -> str:
    return f"{float(value):.9g}"


def _shell_join(command: list[str]) -> str:
    if command and command[0].startswith("TERM="):
        return command[0] + " " + " ".join(shlex.quote(part) for part in command[1:])
    return " ".join(shlex.quote(part) for part in command)
