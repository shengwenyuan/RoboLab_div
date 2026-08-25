from __future__ import annotations

import numpy as np

from policies.cosmos3.client import Cosmos3EEFClient, IsaacLabAbsIKAdapter
from policies.cosmos3.specs import (
    OBSERVATION_VIEW_ROLES,
    STATELESS_CONDITIONING,
    DecoderAnchorContract,
    ObservationContract,
    PolicyContract,
)
from robolab.eval.base_client import InferenceClient


def _contract() -> PolicyContract:
    return PolicyContract(
        protocol_version=1,
        profile_id="droid-panda-link8-test",
        dataset_source="droid",
        source_view_description="three views",
        present_view_roles=OBSERVATION_VIEW_ROLES,
        robot="droid",
        action_space="eef_absolute",
        policy_fps=15,
        chunk_size=32,
        wire_action_dim=8,
        action_layout=("x", "y", "z", "qx", "qy", "qz", "qw", "gripper"),
        gripper_indices=(7,),
        gripper_semantics="close_fraction",
        eef_frame="panda_link8",
        quaternion_order="xyzw",
        pose_mode="absolute",
        conditioning=STATELESS_CONDITIONING,
        observation=ObservationContract(
            layout_id="primary_top_aux_bottom_pair",
            view_shape_hw=(360, 640),
            canvas_shape_hw=(540, 640),
            view_roles=OBSERVATION_VIEW_ROLES,
            missing_view_policy="error",
            viewpoint="concat_view",
            description="three views",
        ),
        decoder_anchor=DecoderAnchorContract(
            kind="current_eef_pose", frame="panda_link8", quaternion_order="xyzw"
        ),
    )


def test_droid_and_ur5_abs_ik_adapters_share_only_the_canonical_wire_chunk() -> None:
    droid = IsaacLabAbsIKAdapter(
        arm_dof=7,
        policy_frame="panda_link8",
        controller_frame="base_link",
        controller_in_policy_xyz=(0.0, 0.0, 0.018174022),
        controller_in_policy_quat_wxyz=(0.0, 2**-0.5, 0.0, 2**-0.5),
    )
    chunk = np.zeros((2, 8), dtype=np.float32)
    chunk[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    chunk[:, 7] = [0.25, 0.75]

    droid_result = droid.convert(chunk)
    ur5_result = IsaacLabAbsIKAdapter(
        arm_dof=6,
        policy_frame="tool0",
        controller_frame="tool0",
    ).convert(chunk)

    np.testing.assert_allclose(droid_result[:, :3], [[0.0, 0.0, 0.018174022]] * 2, atol=1e-6)
    np.testing.assert_allclose(droid_result[:, 3:7], [[2**-0.5, 0.0, 2**-0.5, 0.0]] * 2, atol=1e-6)
    np.testing.assert_allclose(ur5_result[:, 3:7], [[0.0, 0.0, 0.0, 1.0]] * 2, atol=1e-6)
    np.testing.assert_allclose(droid_result[:, 7], ur5_result[:, 7])


def test_abs_ik_adapter_rotates_the_fixed_controller_offset() -> None:
    adapter = IsaacLabAbsIKAdapter(
        arm_dof=7,
        policy_frame="panda_link8",
        controller_frame="base_link",
        controller_in_policy_xyz=(0.0, 0.0, 0.02),
    )
    chunk = np.zeros((1, 8), dtype=np.float32)
    chunk[0, :3] = [0.4, -0.2, 0.3]
    chunk[0, 3:7] = [0.0, 2**-0.5, 0.0, 2**-0.5]  # +90 degrees about Y

    result = adapter.convert(chunk)

    np.testing.assert_allclose(result[0, :3], [0.42, -0.2, 0.3], atol=1e-6)


def test_execute_horizon_requeries_before_ninth_source_action() -> None:
    client = object.__new__(Cosmos3EEFClient)
    InferenceClient.__init__(client)
    client.policy_contract = _contract()
    client.control_fps = 15
    client.execute_horizon = 8
    source = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)

    client._set_chunk(0, source)
    emitted = np.stack([client._next_action(0) for _ in range(8)])

    np.testing.assert_array_equal(emitted, source[:8])
    assert client._needs_refresh(0)
    assert len(client._chunks[0]) == 8
