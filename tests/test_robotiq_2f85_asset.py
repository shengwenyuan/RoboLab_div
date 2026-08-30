from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robolab.constants import ASSET_DIR as ROBOLAB_ASSET_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(ROBOLAB_ASSET_DIR) / "robots" / "ur5e"
ARM_URDF = ASSET_DIR / "ur5e_mesh.urdf"
GRIPPER_URDF = ASSET_DIR / "robotiq_2f_85.urdf"
ASSEMBLY_URDF = ASSET_DIR / "ur5e_robotiq_2f_85.urdf"
BUILDER = REPO_ROOT / "scripts" / "build_ur5e_robotiq_2f85_urdf.py"

MENAGERIE_SHA256 = {
    "base.stl": "1019b87c1dcff4a08a2fdca2dfd4893d60a5e2fc53512c6b5fe2e372b75c9aa3",
    "base_coupling.stl": "6a2517f9d6d78f89d9edb617fb93a279b4d52ac61c12d9cb743c701676eeb06d",
    "c-a01-85-open.stl": "13ad73ef491f6f28b9ed6b8fbb3d6fb45896110ea7d415cda1689fc8daa5d925",
    "coupler.stl": "54949f7355c35c976d854fb77272feb92d9201213e343a8852429556fc81d416",
    "driver.stl": "baf8b4dde18ce59eeebc0928a289c69dccec9da81bb186e2838e2e304274e106",
    "follower.stl": "28811d3651345dbb5f2020c67d3bd05f754b5e3e791e379c3c4d1d87418bb9c5",
    "pad.stl": "c0f4d31e867a5b3634c102669b76ac5e8c026ede5fc645b751a5eb3d4bb0be02",
    "spring_link.stl": "56e9f28ce90841d654ab6d953161aeb62142b1459c210335d5862e7fcb281aab",
    "tongue.stl": "2bd5df8a2132542703d40d94be36a990bbc3eabe070283232871147081348d52",
}
V4_CLOSED_POSITION = 0.8 / (2.0 * 0.485)


def _root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _vec(text: str | None, width: int) -> np.ndarray:
    if text is None:
        return np.zeros(width, dtype=np.float64)
    result = np.fromstring(text, sep=" ", dtype=np.float64)
    assert result.shape == (width,)
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform(origin: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if origin is not None:
        result[:3, 3] = _vec(origin.get("xyz"), 3)
        result[:3, :3] = _rpy_matrix(_vec(origin.get("rpy"), 3))
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = c * np.eye(3) + (1.0 - c) * np.outer(axis, axis) + s * cross
    result = np.eye(4)
    result[:3, :3] = rotation
    return result


def _link_poses(root: ET.Element, master_position: float = 0.0) -> dict[str, np.ndarray]:
    links = {link.get("name") for link in root.findall("link")}
    joints = root.findall("joint")
    child_names = {joint.find("child").get("link") for joint in joints}
    roots = links - child_names
    assert len(roots) == 1
    poses = {roots.pop(): np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            position = 0.0
            if joint.get("type") in {"revolute", "continuous", "prismatic"}:
                mimic = joint.find("mimic")
                if joint.get("name") == "finger_joint":
                    position = master_position
                elif mimic is not None:
                    assert mimic.get("joint") == "finger_joint"
                    position = float(mimic.get("multiplier", "1")) * master_position + float(mimic.get("offset", "0"))
            motion = np.eye(4)
            if joint.get("type") in {"revolute", "continuous"}:
                motion = _axis_rotation(_vec(joint.find("axis").get("xyz"), 3), position)
            elif joint.get("type") == "prismatic":
                motion[:3, 3] = _vec(joint.find("axis").get("xyz"), 3) * position
            child = joint.find("child").get("link")
            poses[child] = poses[parent] @ _transform(joint.find("origin")) @ motion
            pending.remove(joint)
            progressed = True
        assert progressed, "URDF contains a disconnected link or joint cycle"
    assert poses.keys() == links
    return poses


def _pad_gap(root: ET.Element, master_position: float) -> tuple[float, float, float]:
    poses = _link_poses(root, master_position)
    intervals: list[tuple[float, float]] = []
    centers: list[float] = []
    for link_name in ("left_inner_finger_pad", "right_inner_finger_pad"):
        link = root.find(f"link[@name='{link_name}']")
        collision = link.find("collision[@name='pad_lower']")
        size = _vec(collision.find("geometry/box").get("size"), 3)
        world = poses[link_name] @ _transform(collision.find("origin"))
        half_x = float(np.abs(world[0, :3]) @ (size / 2.0))
        center_x = float(world[0, 3])
        intervals.append((center_x - half_x, center_x + half_x))
        centers.append(center_x)
    intervals.sort()
    return intervals[1][0] - intervals[0][1], centers[0], centers[1]


def _semantic(element: ET.Element) -> tuple:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_semantic(child) for child in element if isinstance(child.tag, str)),
    )


def _binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], byteorder="little")
    assert len(data) == 84 + count * 50
    values = np.ndarray((count, 12), dtype="<f4", buffer=data, offset=84, strides=(50, 4))
    return values[:, 3:12].reshape(-1, 3).astype(np.float64)


def test_meshes_are_pinned_and_every_reference_exists() -> None:
    root = _root(GRIPPER_URDF)
    referenced = set()
    for mesh in root.iter("mesh"):
        path = ASSET_DIR / mesh.get("filename")
        assert path.is_file(), path
        referenced.add(path.name)
    assert referenced == set(MENAGERIE_SHA256)

    mesh_dir = ASSET_DIR / "robotiq_2f_85" / "meshes" / "menagerie_v4"
    for name, expected in MENAGERIE_SHA256.items():
        assert hashlib.sha256((mesh_dir / name).read_bytes()).hexdigest() == expected
    assert {tuple(_vec(mesh.get("scale"), 3)) for mesh in root.iter("mesh")} == {(1.0, 1.0, 1.0)}


def test_mass_inertia_mount_and_open_center_of_mass() -> None:
    root = _root(GRIPPER_URDF)
    poses = _link_poses(root, 0.0)
    weighted_center = np.zeros(3)
    total_mass = 0.0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        assert inertial is not None
        mass = float(inertial.find("mass").get("value"))
        inertia = inertial.find("inertia")
        matrix = np.array(
            [
                [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
                [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
                [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert np.all(eigenvalues > 0.0), (link.get("name"), eigenvalues)
        assert eigenvalues[-1] <= eigenvalues[0] + eigenvalues[1] + 1e-12

        center = (poses[link.get("name")] @ _transform(inertial.find("origin")))[:3, 3]
        weighted_center += mass * center
        total_mass += mass

    assert total_mass == pytest.approx(0.89999986, abs=1e-9)
    # Menagerie v4 root is the tool mounting plane; there is no axial install offset.
    assembled_center = weighted_center / total_mass
    assert assembled_center[:2] == pytest.approx([0.0, 0.0], abs=1e-8)
    assert assembled_center[2] == pytest.approx(0.057, abs=1e-9)

    base = root.find("link[@name='robotiq_arg2f_base_link']")
    assert base.find("visual[@name='ur_coupling']/geometry/mesh").get("filename").endswith("base_coupling.stl")
    coupling_collision = base.find("collision[@name='ur_coupling']")
    assert coupling_collision.find("geometry/mesh").get("filename").endswith("base_coupling.stl")
    vertices = _binary_stl_vertices(ASSET_DIR / "robotiq_2f_85" / "meshes" / "menagerie_v4" / "base_coupling.stl")
    transform = _transform(coupling_collision.find("origin"))
    transformed = vertices @ transform[:3, :3].T + transform[:3, 3]
    # The -3 mm skirt wraps the mounting interface; +13.9 mm is exposed above it.
    assert [transformed[:, 2].min(), transformed[:, 2].max()] == pytest.approx([-0.003, 0.0139], abs=5e-8)


def test_one_driver_five_mimics_and_nonpenetrating_pad_sweep() -> None:
    root = _root(GRIPPER_URDF)
    movable = [joint for joint in root.findall("joint") if joint.get("type") == "revolute"]
    driver = root.find("joint[@name='finger_joint']")
    assert driver.find("mimic") is None
    assert float(driver.find("limit").get("effort")) == pytest.approx(5.0)
    assert float(driver.find("dynamics").get("damping")) == pytest.approx(0.1)

    mimics = [joint for joint in movable if joint.find("mimic") is not None]
    assert len(movable) == 6
    assert {float(joint.find("limit").get("lower")) for joint in movable} == {0.0}
    assert {float(joint.find("limit").get("upper")) for joint in movable} == {0.9}
    assert len(mimics) == 5
    assert {joint.find("mimic").get("joint") for joint in mimics} == {"finger_joint"}
    assert {joint.get("name"): float(joint.find("mimic").get("multiplier")) for joint in mimics} == {
        "right_outer_knuckle_joint": 1.0,
        "left_inner_knuckle_joint": 1.0,
        "right_inner_knuckle_joint": 1.0,
        "left_inner_finger_joint": 1.0,
        "right_inner_finger_joint": 1.0,
    }

    samples = [_pad_gap(root, q) for q in np.linspace(0.0, V4_CLOSED_POSITION, 101)]
    gaps = np.array([sample[0] for sample in samples])
    assert gaps[0] == pytest.approx(0.0868782314, abs=2e-9)
    assert gaps[-1] == pytest.approx(-0.0000927649, abs=2e-9)
    assert np.all(np.diff(gaps) < 0.0)
    assert np.all(gaps >= -0.0001)
    assert all(left_center < 0.0 < right_center for _, left_center, right_center in samples)


def test_generated_assembly_is_current_and_arm_is_semantically_unchanged() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--arm",
            str(ARM_URDF),
            "--gripper",
            str(GRIPPER_URDF),
            "--output",
            str(ASSEMBLY_URDF),
            "--check",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    arm = _root(ARM_URDF)
    gripper = _root(GRIPPER_URDF)
    assembly = _root(ASSEMBLY_URDF)
    for tag in ("material", "link", "joint"):
        assembled = {element.get("name"): element for element in assembly.findall(tag)}
        for source in (arm, gripper):
            for element in source.findall(tag):
                assert _semantic(assembled[element.get("name")]) == _semantic(element)

    mount = assembly.find("joint[@name='tool0_to_robotiq_arg2f_base_link']")
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == "tool0"
    assert mount.find("child").get("link") == "robotiq_arg2f_base_link"
    assert _vec(mount.find("origin").get("xyz"), 3) == pytest.approx([0.0, 0.0, 0.0])
    assert _vec(mount.find("origin").get("rpy"), 3) == pytest.approx([0.0, 0.0, math.pi / 2.0])
    assert assembly.find("link[@name='tcp']") is None
    assert assembly.find("link[@name='berkeley_tcp']") is None


def test_isaac_config_drives_only_master_and_parses_mimics() -> None:
    try:
        import omni.physics  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("Isaac Sim application is not running; covered by the USD integration validation")
    from robolab.robots.ur5 import GRIPPER_MIMIC_JOINT_REGEX, UR5E_LEFT_PAD_PRIM_PATH, UR5eCfg, contact_gripper
    from robolab.robots.ur5_spawn import (
        ROBOTIQ_COLLISION_FILTER_PAIRS,
        ROBOTIQ_INTERNAL_COLLISION_FILTER_PAIRS,
        ROBOTIQ_MOUNT_COLLISION_FILTER_PAIRS,
        ROBOTIQ_PAD_COLLIDER_PATHS,
        ROBOTIQ_PAD_COLLISION_ROOT_PATHS,
        ROBOTIQ_PAD_FRICTION_BY_SECTION,
        spawn_ur5e_robotiq_2f85,
    )

    cfg = UR5eCfg().robot
    assert cfg.spawn.convert_mimic_joints_to_normal_joints is True
    assert cfg.spawn.collider_type == "convex_hull"
    assert cfg.spawn.self_collision is True
    assert cfg.spawn.articulation_props.enabled_self_collisions is True
    assert cfg.spawn.func is spawn_ur5e_robotiq_2f85
    assert len(ROBOTIQ_INTERNAL_COLLISION_FILTER_PAIRS) == 6
    assert len(ROBOTIQ_MOUNT_COLLISION_FILTER_PAIRS) == 2
    assert len(ROBOTIQ_COLLISION_FILTER_PAIRS) == 8
    assert len(ROBOTIQ_PAD_COLLISION_ROOT_PATHS) == 2
    assert len(ROBOTIQ_PAD_COLLIDER_PATHS) == 4
    assert ROBOTIQ_PAD_FRICTION_BY_SECTION == {"pad_lower": 0.7, "pad_upper": 0.6}
    assert cfg.spawn.joint_drive.gains.natural_frequency == {GRIPPER_MIMIC_JOINT_REGEX: 0.0}
    assert cfg.spawn.joint_drive.gains.damping_ratio == {GRIPPER_MIMIC_JOINT_REGEX: 0.0}
    assert cfg.actuators["gripper"].joint_names_expr == ["finger_joint"]
    assert cfg.actuators["gripper"].effort_limit_sim == pytest.approx(5.0)
    assert cfg.actuators["gripper"].stiffness == pytest.approx(100.0)
    assert cfg.actuators["gripper"].damping == pytest.approx(10.0)
    assert contact_gripper == {"gripper": UR5E_LEFT_PAD_PRIM_PATH}
    assert UR5E_LEFT_PAD_PRIM_PATH.endswith("/left_inner_finger/left_inner_finger_pad")
