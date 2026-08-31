# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from robolab.registrations.ur5.initial_pose_presets import get_ur5_initial_pose_preset


def test_default_initial_pose_preserves_standard_ur5_home() -> None:
    preset = get_ur5_initial_pose_preset("default")

    assert preset.arm_joint_positions is None
    assert preset.gripper_close_fraction is None
    assert preset.root_rot_wxyz is None


def test_rh20t_initial_pose_uses_cfg4_aligned_safe_reset() -> None:
    preset = get_ur5_initial_pose_preset("rh20t_ur5")

    assert preset.arm_joint_positions == pytest.approx(
        (
            0.0,
            -1.57079632679,
            1.57079632679,
            -1.57079632679,
            -1.57079632679,
            -0.21157962083816528,
        )
    )
    assert preset.gripper_close_fraction == 1.0
    assert preset.root_rot_wxyz is None


def test_gripper_reset_initializes_the_complete_mimic_chain() -> None:
    from robolab.robots.ur5_profile import GRIPPER_CLOSED_POS, gripper_reset_joint_positions

    positions = gripper_reset_joint_positions(1.0)

    assert tuple(positions) == (
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    )
    assert set(positions.values()) == {GRIPPER_CLOSED_POS}


@pytest.mark.parametrize("close_fraction", [-0.01, 1.01])
def test_gripper_reset_rejects_out_of_range_fraction(close_fraction: float) -> None:
    from robolab.robots.ur5_profile import gripper_reset_joint_positions

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        gripper_reset_joint_positions(close_fraction)


def test_robomind_initial_pose_uses_representative_real_geometry() -> None:
    preset = get_ur5_initial_pose_preset("robomind_ur5")

    assert preset.arm_joint_positions == pytest.approx(
        (
            2.216296672821045,
            -1.8964468240737915,
            -1.6191987991333008,
            -1.1574769020080566,
            1.5983433723449707,
            1.0311790704727173,
        )
    )
    assert preset.root_rot_wxyz == pytest.approx(
        (0.8972531204891708, 0.0, 0.0, 0.44151652038450995)
    )


def test_initial_pose_preset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="choose one of: default, rh20t_ur5, robomind_ur5"):
        get_ur5_initial_pose_preset("missing")


def test_cosmos3_ur5_runner_wires_pose_preset_into_registration() -> None:
    source = (Path(__file__).parents[1] / "policies/cosmos3/run_ur5.py").read_text()

    assert '"--initial-pose-preset"' in source
    assert "get_ur5_initial_pose_preset" in source
    assert "initial_arm_joint_positions=initial_pose_preset.arm_joint_positions" in source
    assert "initial_gripper_close_fraction=initial_pose_preset.gripper_close_fraction" in source
    assert "initial_root_rot_wxyz=initial_pose_preset.root_rot_wxyz" in source


def test_camera_debug_runner_wires_pose_preset_into_registration() -> None:
    source = (Path(__file__).parents[1] / "examples/save_ur5_cosmos_initial_cameras.py").read_text()

    assert '"--initial-pose-preset"' in source
    assert "get_ur5_initial_pose_preset" in source
    assert "initial_arm_joint_positions=initial_pose_preset.arm_joint_positions" in source
    assert "initial_gripper_close_fraction=initial_pose_preset.gripper_close_fraction" in source
    assert "initial_root_rot_wxyz=initial_pose_preset.root_rot_wxyz" in source
