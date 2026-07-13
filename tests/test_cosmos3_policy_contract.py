# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from policies.cosmos3.client import Cosmos3Client
from policies.cosmos3.specs import (
    CANONICAL_GRIPPER_SEMANTICS,
    JOINT_CURRENT_STATE_CONDITIONING,
    OBSERVATION_VIEW_ROLES,
    STANDARD_THREE_VIEW_OBSERVATION,
    ClientCapability,
    ConditioningContract,
    ObservationContract,
    PolicyContract,
    binarize_close_fraction,
    expand_action_chunk,
    read_server_contract,
    validate_same_policy_semantics,
    validate_server_metadata,
    validate_wire_action_chunk,
)


def _observation(*, description: str = "training prose", layout_id: str | None = None) -> ObservationContract:
    return ObservationContract(
        layout_id=layout_id or "primary_top_aux_bottom_pair",
        view_shape_hw=(360, 640),
        canvas_shape_hw=(540, 640),
        view_roles=OBSERVATION_VIEW_ROLES,
        missing_view_policy="error",
        viewpoint="concat_view",
        description=description,
    )


def _joint_contract(
    *,
    robot: str = "ur5",
    arm_dof: int = 6,
    fps: int = 15,
    chunk_size: int = 32,
    profile_id: str = "arbitrary-label",
    description: str = "training prose",
    joint_layout: tuple[str, ...] | None = None,
) -> PolicyContract:
    joint_layout = joint_layout or tuple(f"joint_{index}" for index in range(arm_dof))
    return PolicyContract(
        protocol_version=1,
        profile_id=profile_id,
        dataset_source="test-source",
        source_view_description="primary camera above two auxiliary cameras",
        present_view_roles=OBSERVATION_VIEW_ROLES,
        robot=robot,
        action_space="joint_position",
        policy_fps=fps,
        chunk_size=chunk_size,
        wire_action_dim=arm_dof + 1,
        action_layout=joint_layout + ("gripper",),
        gripper_indices=(arm_dof,),
        gripper_semantics=CANONICAL_GRIPPER_SEMANTICS,
        eef_frame=None,
        quaternion_order=None,
        pose_mode="absolute",
        conditioning=JOINT_CURRENT_STATE_CONDITIONING,
        observation=_observation(description=description),
    )


def _capability(
    *, robot: str = "ur5", arm_dof: int = 6, joint_layout: tuple[str, ...] | None = None
) -> ClientCapability:
    joint_layout = joint_layout or tuple(f"joint_{index}" for index in range(arm_dof))
    return ClientCapability(
        robot=robot,
        arm_dof=arm_dof,
        action_spaces=("joint_position",),
        joint_action_layout=joint_layout,
        conditioning_by_action_space={"joint_position": JOINT_CURRENT_STATE_CONDITIONING},
        observation=STANDARD_THREE_VIEW_OBSERVATION,
    )


def test_handshake_is_the_source_of_policy_timing_and_shape() -> None:
    advertised = _joint_contract(fps=5, chunk_size=12)
    payload = advertised.to_metadata()
    payload["future_extension"] = "ignored"

    actual = validate_server_metadata({"policy_contract": payload}, _capability())

    assert actual.policy_fps == 5
    assert actual.chunk_size == 12
    assert actual.wire_action_dim == 7
    assert actual.buffer_size(control_fps=15) == 36
    assert actual.dataset_source == "test-source"
    assert actual.present_view_roles == OBSERVATION_VIEW_ROLES


def test_contract_parser_requires_server_metadata() -> None:
    with pytest.raises(ValueError, match="missing required 'policy_contract'"):
        read_server_contract({})

    payload = _joint_contract().to_metadata()
    payload.pop("conditioning")
    with pytest.raises(ValueError, match="missing fields.*conditioning"):
        read_server_contract({"policy_contract": payload})

    payload = _joint_contract().to_metadata()
    payload.pop("dataset_source")
    with pytest.raises(ValueError, match="missing fields.*dataset_source"):
        read_server_contract({"policy_contract": payload})


def test_robot_capability_is_open_ended_but_identity_is_strict() -> None:
    x5 = _joint_contract(robot="x5", arm_dof=6)

    assert validate_server_metadata({"policy_contract": x5.to_metadata()}, _capability(robot="x5")) == x5
    with pytest.raises(ValueError, match="robot mismatch"):
        validate_server_metadata({"policy_contract": x5.to_metadata()}, _capability(robot="droid", arm_dof=7))


def test_joint_dof_and_observation_preset_are_client_capabilities() -> None:
    contract = _joint_contract(robot="ur5", arm_dof=6)

    with pytest.raises(ValueError, match="contain 7 arm joints"):
        _capability(arm_dof=7).validate(contract)

    wrong_observation = replace(
        STANDARD_THREE_VIEW_OBSERVATION,
        layout_id="primary_top_aux_bottom_pair_right_missing",
    )
    with pytest.raises(ValueError, match="observation preset mismatch"):
        replace(_capability(), observation=wrong_observation).validate(contract)

    with pytest.raises(ValueError, match="present_view_roles"):
        _capability().validate(replace(contract, present_view_roles=("primary",)))


def test_observation_roles_are_global_positional_slots() -> None:
    with pytest.raises(ValueError, match="positional slots"):
        replace(_observation(), view_roles=("wrist", "exterior_left", "exterior_right"))


def test_joint_action_layout_must_match_exactly() -> None:
    contract = _joint_contract(robot="ur5", arm_dof=6)
    wrong_order = replace(
        _capability(),
        joint_action_layout=("joint_1", "joint_0", "joint_2", "joint_3", "joint_4", "joint_5"),
    )

    with pytest.raises(ValueError, match="expected layout"):
        wrong_order.validate(contract)


@pytest.mark.parametrize(
    ("robot", "joint_layout"),
    [
        ("droid", tuple(f"joint_{index}" for index in range(7))),
        ("ur5", ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")),
        ("x5", tuple(f"joint{index}" for index in range(1, 7))),
    ],
)
def test_robot_joint_layout_capabilities_are_explicit(robot: str, joint_layout: tuple[str, ...]) -> None:
    arm_dof = len(joint_layout)
    contract = _joint_contract(robot=robot, arm_dof=arm_dof, joint_layout=joint_layout)

    _capability(robot=robot, arm_dof=arm_dof, joint_layout=joint_layout).validate(contract)


def test_conditioning_history_must_match_client_cache_capability() -> None:
    contract = replace(
        _joint_contract(),
        conditioning=ConditioningContract(state_rows=1, history_rows=2, source="current_state"),
    )

    with pytest.raises(ValueError, match="conditioning mismatch"):
        _capability().validate(contract)


def test_wire_gripper_must_be_canonical_close_fraction() -> None:
    contract = _joint_contract()

    with pytest.raises(ValueError, match="close_fraction"):
        replace(contract, gripper_semantics="open_fraction")


def test_profile_id_and_description_are_non_semantic() -> None:
    before = _joint_contract(profile_id="run-a", description="old prose")
    after = _joint_contract(profile_id="renamed-run", description="clearer prose")

    validate_same_policy_semantics(before, after)


def test_reconnect_rejects_actual_semantic_change() -> None:
    before = _joint_contract(fps=15, chunk_size=32)
    after = _joint_contract(fps=5, chunk_size=12)

    with pytest.raises(ValueError, match="changed across reconnect"):
        validate_same_policy_semantics(before, after)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_source", "different-source"),
        ("source_view_description", "different model prompt"),
        ("present_view_roles", ("primary",)),
    ],
)
def test_reconnect_rejects_selected_source_change(field: str, value: object) -> None:
    before = _joint_contract()
    after = replace(before, **{field: value})

    with pytest.raises(ValueError, match=f"changed across reconnect.*{field}"):
        validate_same_policy_semantics(before, after)


def test_zoh_expands_from_server_policy_rate_to_control_rate() -> None:
    contract = _joint_contract(fps=5, chunk_size=12)
    chunk = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)

    expanded = expand_action_chunk(chunk, contract, control_fps=15)

    assert expanded.shape == (36, 7)
    np.testing.assert_array_equal(expanded, np.repeat(chunk, 3, axis=0))


def test_equal_policy_and_control_rate_is_identity() -> None:
    contract = _joint_contract(fps=15, chunk_size=32)
    chunk = np.arange(32 * 7, dtype=np.float32).reshape(32, 7)

    np.testing.assert_array_equal(expand_action_chunk(chunk, contract, control_fps=15), chunk)


def test_non_integer_policy_to_control_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="must divide"):
        _joint_contract(fps=10).hold_ratio(15)


def test_wire_action_shape_and_finiteness_are_exact() -> None:
    contract = _joint_contract(fps=5, chunk_size=12)
    valid = np.zeros((12, 7), dtype=np.float32)

    assert validate_wire_action_chunk(valid, contract).shape == (12, 7)
    with pytest.raises(ValueError, match="wire action shape"):
        validate_wire_action_chunk(np.zeros((12, 8), dtype=np.float32), contract)
    valid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_wire_action_chunk(valid, contract)


def test_one_gripper_threshold_is_used_for_open_and_closed() -> None:
    chunk = np.array([[0.0, 0.49], [0.0, 0.5], [0.0, 0.51]], dtype=np.float32)

    processed = binarize_close_fraction(chunk)

    np.testing.assert_array_equal(processed[:, -1], [0.0, 0.0, 1.0])


def test_client_connects_without_a_profile_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    advertised = _joint_contract(robot="new_robot", arm_dof=4, fps=5, chunk_size=9, profile_id="any-name")

    class _Transport:
        def get_server_metadata(self) -> dict:
            return {"policy_contract": advertised.to_metadata()}

    monkeypatch.setattr(
        "policies.cosmos3.client.websocket_client_policy.WebsocketClientPolicy",
        lambda host, port: _Transport(),
    )
    capability = ClientCapability(
        robot="new_robot",
        arm_dof=4,
        action_spaces=("joint_position",),
        joint_action_layout=tuple(f"joint_{index}" for index in range(4)),
        conditioning_by_action_space={"joint_position": JOINT_CURRENT_STATE_CONDITIONING},
        observation=STANDARD_THREE_VIEW_OBSERVATION,
    )

    client = Cosmos3Client(remote_host="server", capability=capability, control_fps=15)

    assert client.policy_contract == advertised
    assert client.open_loop_horizon == 9


def test_client_cache_refresh_uses_expanded_buffer_length() -> None:
    client = object.__new__(Cosmos3Client)
    from robolab.eval.base_client import InferenceClient

    InferenceClient.__init__(client)
    client.policy_contract = _joint_contract(fps=5, chunk_size=12)
    client.control_fps = 15
    source = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)

    client._set_chunk(4, source)

    assert client._chunks[4].shape == (36, 7)
    for _ in range(36):
        assert not client._needs_refresh(4)
        client._next_action(4)
    assert client._needs_refresh(4)
