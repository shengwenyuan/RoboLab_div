# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from policies.cosmos3.client import Cosmos3UR5Client
from policies.cosmos3.specs import (
    JOINT_CURRENT_STATE_CONDITIONING,
    OBSERVATION_VIEW_ROLES,
    STATELESS_CONDITIONING,
    ClientCapability,
    ObservationCapability,
    ObservationContract,
    PolicyContract,
    DecoderAnchorContract,
)
from robolab.core.motion.eef import MotionBridgeResult
from robolab.eval.base_client import InferenceClient
from robolab.robots.ur5_profile import get_ur5_berkeley_eef_profile, get_ur5_eef_profile

_PRESET_METADATA = {
    "rh20t_vertical_pair": {
        "layout_id": "vertical_pair",
        "view_shape_hw": (360, 640),
        "canvas_shape_hw": (720, 640),
        "view_roles": ("primary", "aux_left"),
        "role_sources": {
            "primary": "over_shoulder_left_camera",
            "aux_left": "over_shoulder_right_camera",
        },
        "missing_view_policies": ("error",),
        "viewpoint": "concat_view",
    },
    "wrist_left_right": {
        "layout_id": "primary_top_aux_bottom_pair",
        "view_shape_hw": (360, 640),
        "canvas_shape_hw": (540, 640),
        "view_roles": OBSERVATION_VIEW_ROLES,
        "role_sources": {
            "primary": "wrist_cam",
            "aux_left": "over_shoulder_left_camera",
            "aux_right": "over_shoulder_right_camera",
        },
        "missing_view_policies": ("error", "black"),
        "viewpoint": "concat_view",
    },
    "berkeley_eef": {
        "layout_id": "primary_top_aux_bottom_pair_right_missing",
        "view_shape_hw": (360, 640),
        "canvas_shape_hw": (540, 640),
        "view_roles": OBSERVATION_VIEW_ROLES,
        "role_sources": {
            "primary": "wrist_cam",
            "aux_left": "over_shoulder_left_camera",
            "aux_right": None,
        },
        "missing_view_policies": ("black",),
        "viewpoint": "concat_view",
    },
    "robomind_single": {
        "layout_id": "primary_top_aux_bottom_pair",
        "view_shape_hw": (360, 640),
        "canvas_shape_hw": (540, 640),
        "view_roles": OBSERVATION_VIEW_ROLES,
        "role_sources": {
            "primary": "robomind_top_camera",
            "aux_left": None,
            "aux_right": None,
        },
        "missing_view_policies": ("black",),
        "viewpoint": "concat_view",
    },
}


def _observation_contract(preset_name: str) -> ObservationContract:
    metadata = _PRESET_METADATA[preset_name]
    return ObservationContract(
        layout_id=metadata["layout_id"],
        view_shape_hw=metadata["view_shape_hw"],
        canvas_shape_hw=metadata["canvas_shape_hw"],
        view_roles=metadata["view_roles"],
        missing_view_policy=(
            "error" if preset_name in {"wrist_left_right", "rh20t_vertical_pair"} else "black"
        ),
        viewpoint=metadata["viewpoint"],
        description=f"{preset_name} prose",
    )


def _present_view_roles(preset_name: str) -> tuple[str, ...]:
    metadata = _PRESET_METADATA[preset_name]
    return tuple(role for role in metadata["view_roles"] if metadata["role_sources"][role] is not None)


def _contract(
    *,
    action_space: str,
    fps: int = 15,
    chunk_size: int = 32,
    preset: str,
    eef_frame: str = "berkeley_tcp",
) -> PolicyContract:
    if action_space == "joint_position":
        if preset == "rh20t_vertical_pair":
            layout = (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "gripper",
            )
        else:
            layout = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper")
        eef_frame = quaternion_order = None
    else:
        layout = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
        quaternion_order = "xyzw"
    return PolicyContract(
        protocol_version=1,
        profile_id="diagnostic-label",
        dataset_source=preset,
        source_view_description=f"{preset} exact training view prompt",
        present_view_roles=_present_view_roles(preset),
        robot="ur5",
        action_space=action_space,
        policy_fps=fps,
        chunk_size=chunk_size,
        wire_action_dim=len(layout),
        action_layout=layout,
        gripper_indices=(len(layout) - 1,),
        gripper_semantics="close_fraction",
        eef_frame=eef_frame,
        quaternion_order=quaternion_order,
        pose_mode="absolute",
        conditioning=(JOINT_CURRENT_STATE_CONDITIONING if action_space == "joint_position" else STATELESS_CONDITIONING),
        observation=_observation_contract(preset),
        decoder_anchor=(
            None
            if action_space == "joint_position"
            else DecoderAnchorContract(kind="current_eef_pose", frame=eef_frame, quaternion_order="xyzw")
        ),
    )


def _bare_client(contract: PolicyContract) -> Cosmos3UR5Client:
    client = object.__new__(Cosmos3UR5Client)
    InferenceClient.__init__(client)
    client.policy_contract = contract
    client.control_fps = 15
    client.open_loop_horizon = contract.chunk_size
    client._image_h, client._image_w = contract.observation.view_shape_hw
    preset_metadata = next(
        metadata
        for metadata in _PRESET_METADATA.values()
        if metadata["layout_id"] == contract.observation.layout_id
        and metadata["view_roles"] == contract.observation.view_roles
        and tuple(role for role in metadata["view_roles"] if metadata["role_sources"][role] is not None)
        == contract.present_view_roles
    )
    joint_layout = (
        tuple(name for name in contract.action_layout[:-1])
        if contract.observation.layout_id == "vertical_pair"
        else ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    )
    client.capability = ClientCapability(
        robot="ur5",
        arm_dof=6,
        action_spaces=("joint_position", "eef_absolute"),
        joint_action_layout=joint_layout,
        conditioning_by_action_space={
            "joint_position": JOINT_CURRENT_STATE_CONDITIONING,
            "eef_absolute": STATELESS_CONDITIONING,
        },
        observation=ObservationCapability.from_mapping(preset_metadata),
    )
    client._profile = get_ur5_eef_profile()
    client._env_action_dim = client._profile.env_action_dim + 1
    client._eef_bridge = None
    client._last_diagnostics = {}
    return client


def _minimal_request_observation() -> dict:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    eef_pos = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    eef_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "canvas_views": (image, image, image),
        "joint_position": np.zeros(6, dtype=np.float32),
        "arm_joint_position": np.zeros(6, dtype=np.float32),
        "gripper_position": np.zeros(1, dtype=np.float32),
        "eef_observation": object(),
        "eef_pos": eef_pos,
        "eef_quat_xyzw": eef_quat,
        "eef_pose_xyzw": np.concatenate([eef_pos, eef_quat, np.zeros(1, dtype=np.float32)]),
    }


def test_camera_preset_carries_matching_observation_capability() -> None:
    standard_capability = ObservationCapability.from_mapping(_PRESET_METADATA["wrist_left_right"])
    berkeley_capability = ObservationCapability.from_mapping(_PRESET_METADATA["berkeley_eef"])
    robomind_capability = ObservationCapability.from_mapping(_PRESET_METADATA["robomind_single"])
    rh20t_capability = ObservationCapability.from_mapping(_PRESET_METADATA["rh20t_vertical_pair"])

    standard_capability.validate(_observation_contract("wrist_left_right"), _present_view_roles("wrist_left_right"))
    berkeley_capability.validate(_observation_contract("berkeley_eef"), _present_view_roles("berkeley_eef"))
    robomind_capability.validate(_observation_contract("robomind_single"), _present_view_roles("robomind_single"))
    rh20t_capability.validate(
        _observation_contract("rh20t_vertical_pair"), _present_view_roles("rh20t_vertical_pair")
    )
    assert standard_capability.missing_view_policies == ("error", "black")
    assert berkeley_capability.missing_view_policies == ("black",)
    assert robomind_capability.role_sources["primary"] == "robomind_top_camera"
    assert robomind_capability.role_sources["aux_left"] is None
    assert robomind_capability.role_sources["aux_right"] is None
    with pytest.raises(ValueError, match="missing role sources"):
        replace(robomind_capability, missing_view_policies=("black", "error"))
    with pytest.raises(ValueError, match="observation preset mismatch"):
        standard_capability.validate(_observation_contract("berkeley_eef"), _present_view_roles("berkeley_eef"))
    with pytest.raises(ValueError, match="present_view_roles"):
        standard_capability.validate(_observation_contract("robomind_single"), _present_view_roles("robomind_single"))


def test_camera_registration_is_the_single_ur5_preset_registry() -> None:
    source = (Path(__file__).parents[1] / "robolab/registrations/ur5/camera_presets.py").read_text()

    assert "COSMOS3_CAMERA_PRESETS" in source
    assert '"wrist_left_right"' in source
    assert '"berkeley_eef"' in source
    assert '"robomind_single"' in source
    assert '"rh20t_vertical_pair"' in source
    assert '"primary": "robomind_top_camera"' in source
    assert "ROBOMIND_SINGLE = [RoboMindGlobalCameraCfg, RoboMindTopCameraCfg]" in source
    assert "get_cosmos3_camera_preset" in source


def test_rh20t_vertical_pair_resizes_and_stacks_two_real_cameras() -> None:
    client = _bare_client(_contract(action_space="joint_position", preset="rh20t_vertical_pair"))
    client._image_h, client._image_w = 3, 5
    top = torch.full((1, 6, 10, 3), 17, dtype=torch.uint8)
    bottom = torch.full((1, 4, 8, 3), 29, dtype=torch.uint8)

    views = client._extract_canvas_views(
        {
            "over_shoulder_left_camera": top,
            "over_shoulder_right_camera": bottom,
        },
        env_id=0,
    )
    canvas = client._compose_canvas(views)
    visualization = client._build_visualization({"canvas_views": views})

    assert len(views) == 2
    assert views[0].shape == views[1].shape == (3, 5, 3)
    assert canvas.shape == (6, 5, 3)
    np.testing.assert_array_equal(canvas[:3], np.full((3, 5, 3), 17, dtype=np.uint8))
    np.testing.assert_allclose(canvas[3:], np.full((3, 5, 3), 29, dtype=np.uint8), atol=1)
    np.testing.assert_array_equal(visualization, canvas)


def test_ur5_canvas_preserves_selected_right_camera() -> None:
    client = _bare_client(_contract(action_space="joint_position", preset="wrist_left_right"))
    client._image_h = client._image_w = 4
    left = np.full((4, 4, 3), 10, dtype=np.uint8)
    wrist = np.full((4, 4, 3), 20, dtype=np.uint8)
    right = np.full((4, 4, 3), 30, dtype=np.uint8)

    canvas = client._compose_canvas((wrist, left, right))

    assert canvas.shape == (6, 4, 3)
    np.testing.assert_array_equal(canvas[:4], wrist)
    np.testing.assert_array_equal(canvas[4:, :2], np.full((2, 2, 3), 10, dtype=np.uint8))
    np.testing.assert_array_equal(canvas[4:, 2:], np.full((2, 2, 3), 30, dtype=np.uint8))


def test_ur5_observation_requires_all_declared_canvas_slots() -> None:
    client = _bare_client(_contract(action_space="joint_position", preset="wrist_left_right"))
    client._image_h = client._image_w = 4
    raw_obs = {
        "image_obs": {
            "over_shoulder_left_camera": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
            "wrist_cam": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 6)),
            "gripper_pos": torch.zeros((1, 1)),
        },
    }

    with pytest.raises(KeyError, match="over_shoulder_right_camera"):
        client._extract_observation(raw_obs)


def test_robomind_observation_maps_overhead_and_synthesizes_missing_views() -> None:
    client = _bare_client(_contract(action_space="joint_position", preset="robomind_single"))
    client._image_h = client._image_w = 4
    global_view = torch.full((1, 4, 6, 3), 7, dtype=torch.uint8)
    overhead = torch.full((1, 4, 4, 3), 42, dtype=torch.uint8)

    views = client._extract_canvas_views(
        {
            "robomind_global_camera": global_view,
            "robomind_top_camera": overhead,
        },
        env_id=0,
    )
    canvas = client._compose_canvas(views)

    np.testing.assert_array_equal(views[0], np.full((4, 4, 3), 42, dtype=np.uint8))
    np.testing.assert_array_equal(views[1], np.zeros((4, 4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(views[2], np.zeros((4, 4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(canvas[:4], np.full((4, 4, 3), 42, dtype=np.uint8))
    np.testing.assert_array_equal(canvas[4:], np.zeros((2, 4, 3), dtype=np.uint8))



def test_ur5_joint_contract_never_initializes_eef_bridge() -> None:
    client = _bare_client(_contract(action_space="joint_position", fps=5, chunk_size=12, preset="wrist_left_right"))
    action = np.zeros((12, 7), dtype=np.float32)
    action[:, -1] = 0.75

    chunk, diagnostics = client._convert_response_chunk(
        {"action": action}, {"arm_joint_position": np.zeros(6, dtype=np.float32)}, env_id=2
    )

    assert chunk.shape == (12, 7)
    np.testing.assert_array_equal(chunk[:, -1], np.ones(12, dtype=np.float32))
    assert diagnostics[0]["action_space"] == "joint_position"
    assert client._eef_bridge is None


def test_ur5_joint_guard_rejects_unsafe_first_step_without_clamping() -> None:
    client = _bare_client(_contract(action_space="joint_position", preset="rh20t_vertical_pair"))
    action = np.zeros((32, 7), dtype=np.float32)
    action[0, 0] = 0.5

    with pytest.raises(ValueError, match="Unsafe UR5 joint chunk rejected"):
        client._convert_response_chunk(
            {"action": action}, {"arm_joint_position": np.zeros(6, dtype=np.float32)}, env_id=4
        )

    assert client._last_diagnostics[4][0]["failures"] == ("velocity", "acceleration")


@pytest.mark.parametrize(
    ("eef_frame", "preset", "profile_factory"),
    [
        ("tool0", "wrist_left_right", get_ur5_eef_profile),
        ("berkeley_tcp", "berkeley_eef", get_ur5_berkeley_eef_profile),
    ],
)
def test_ur5_eef_contract_selects_profile_from_frame(eef_frame: str, preset: str, profile_factory) -> None:
    client = _bare_client(_contract(action_space="eef_absolute", preset=preset, eef_frame=eef_frame))

    client._validate_policy_contract(client.policy_contract)

    assert client._profile is profile_factory()
    assert client._env_action_dim == 7
    assert client._eef_bridge is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_layout", ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")),
        ("quaternion_order", "wxyz"),
        ("pose_mode", "delta"),
    ],
)
def test_ur5_eef_contract_rejects_noncanonical_wire_pose(field: str, value: object) -> None:
    client = _bare_client(_contract(action_space="eef_absolute", preset="wrist_left_right", eef_frame="tool0"))

    with pytest.raises(ValueError, match="UR5 EEF policies require"):
        client._validate_policy_contract(replace(client.policy_contract, **{field: value}))


def test_ur5_eef_contract_rejects_unknown_frame() -> None:
    client = _bare_client(_contract(action_space="eef_absolute", preset="wrist_left_right", eef_frame="flange"))

    with pytest.raises(ValueError, match="eef_frame='tool0' or 'berkeley_tcp'"):
        client._validate_policy_contract(client.policy_contract)


def test_tool0_eef_observation_has_no_berkeley_tcp_offset() -> None:
    client = _bare_client(_contract(action_space="eef_absolute", preset="wrist_left_right", eef_frame="tool0"))
    client._validate_policy_contract(client.policy_contract)
    client._image_h = client._image_w = 2
    half_sqrt_2 = np.float32(np.sqrt(0.5))
    raw_observation = {
        "image_obs": {
            "wrist_cam": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            "over_shoulder_left_camera": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            "over_shoulder_right_camera": torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
            "ee_pos": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "ee_quat": torch.tensor([[half_sqrt_2, 0.0, half_sqrt_2, 0.0]], dtype=torch.float32),
        },
    }

    extracted = client._extract_observation(raw_observation)

    np.testing.assert_allclose(extracted["eef_pos"], [0.1, 0.2, 0.3], atol=2e-6)
    np.testing.assert_allclose(extracted["eef_pose_xyzw"][:3], [0.1, 0.2, 0.3], atol=2e-6)
    assert extracted["eef_observation"].metadata["eef_frame"] == "tool0"


def test_ur5_eef_contract_requires_stateless_conditioning() -> None:
    observation = ObservationCapability.from_mapping(_PRESET_METADATA["berkeley_eef"])
    capability = ClientCapability(
        robot="ur5",
        arm_dof=6,
        action_spaces=("joint_position", "eef_absolute"),
        joint_action_layout=("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
        conditioning_by_action_space={
            "joint_position": JOINT_CURRENT_STATE_CONDITIONING,
            "eef_absolute": STATELESS_CONDITIONING,
        },
        observation=observation,
    )
    contract = _contract(action_space="eef_absolute", preset="berkeley_eef")

    capability.validate(contract)
    with pytest.raises(ValueError, match="conditioning mismatch"):
        capability.validate(replace(contract, conditioning=JOINT_CURRENT_STATE_CONDITIONING))


def test_ur5_eef_runs_ik_at_source_rate_before_zoh() -> None:
    client = _bare_client(_contract(action_space="eef_absolute", fps=5, chunk_size=12, preset="berkeley_eef"))
    action = np.zeros((12, 8), dtype=np.float32)
    action[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    action[:, -1] = 1.0

    class _Bridge:
        def __init__(self) -> None:
            self.horizon = None

        def convert_chunk(self, observation, action_chunk, *, env_id: int = 0) -> MotionBridgeResult:
            self.horizon = action_chunk.horizon
            arm = np.arange(12 * 6, dtype=np.float32).reshape(12, 6)
            return MotionBridgeResult(
                actions=arm,
                diagnostics=[{"env_id": env_id, "source_step": step} for step in range(12)],
                raw_gripper=action_chunk.gripper,
            )

    bridge = _Bridge()
    client._get_eef_bridge = lambda: bridge

    source_chunk, diagnostics = client._convert_response_chunk(
        {"action": action}, {"eef_observation": object()}, env_id=3
    )
    client._set_chunk(3, source_chunk)

    assert bridge.horizon == 12
    assert len(diagnostics) == 12
    assert source_chunk.shape == (12, 7)
    assert client._chunks[3].shape == (36, 7)
    np.testing.assert_array_equal(client._chunks[3], np.repeat(source_chunk, 3, axis=0))


def test_ur5_request_fields_follow_handshake_action_space() -> None:
    observation = _minimal_request_observation()
    joint_client = _bare_client(_contract(action_space="joint_position", preset="wrist_left_right"))
    eef_client = _bare_client(_contract(action_space="eef_absolute", preset="berkeley_eef"))
    joint_client._image_h = joint_client._image_w = 4
    eef_client._image_h = eef_client._image_w = 4

    joint_request = joint_client._pack_request(observation, "pick")
    eef_request = eef_client._pack_request(observation, "pick")

    assert not any(key.startswith("observation/eef_") for key in joint_request)
    assert "dataset_source" not in joint_request
    assert "source_view_description" not in joint_request
    assert eef_request["observation/eef_pose"].shape == (1, 8)
    assert eef_request["observation/eef_pos"].shape == (1, 3)
    assert eef_request["observation/eef_quat"].shape == (1, 4)


def test_ur5_reset_clears_cache_diagnostics_and_ik_state() -> None:
    client = _bare_client(_contract(action_space="joint_position", fps=5, chunk_size=12, preset="wrist_left_right"))
    client._set_chunk(6, np.zeros((12, 7), dtype=np.float32))
    client._last_diagnostics[6] = [{"source_step": 0}]

    class _Bridge:
        def __init__(self) -> None:
            self.reset_env_ids = []

        def reset(self, *, env_id=None) -> None:
            self.reset_env_ids.append(env_id)

    bridge = _Bridge()
    client._eef_bridge = bridge

    client.reset(env_id=6)

    assert 6 not in client._chunks
    assert 6 not in client._counters
    assert 6 not in client._last_diagnostics
    assert bridge.reset_env_ids == [6]
