# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven Cosmos3 wire contracts and client capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np

POLICY_CONTRACT_METADATA_KEY = "policy_contract"
POLICY_CONTRACT_VERSION = 1
DEFAULT_CONTROL_FPS = 15
CANONICAL_GRIPPER_SEMANTICS = "close_fraction"
OBSERVATION_VIEW_ROLES = ("primary", "aux_left", "aux_right")


@dataclass(frozen=True)
class ObservationContract:
    """Observation canvas advertised by the server-side policy manifest."""

    layout_id: str
    view_shape_hw: tuple[int, int]
    canvas_shape_hw: tuple[int, int]
    view_roles: tuple[str, ...]
    missing_view_policy: str
    viewpoint: str
    description: str

    def __post_init__(self) -> None:
        if not self.layout_id or not self.view_roles or not self.viewpoint:
            raise ValueError("Cosmos3 observation layout_id, view_roles, and viewpoint must be non-empty")
        if self.view_roles != OBSERVATION_VIEW_ROLES:
            raise ValueError(
                f"Cosmos3 observation view_roles must be positional slots {OBSERVATION_VIEW_ROLES!r}, "
                f"got {self.view_roles!r}"
            )
        if any(value <= 0 for value in (*self.view_shape_hw, *self.canvas_shape_hw)):
            raise ValueError("Cosmos3 observation shapes must contain positive dimensions")
        if self.missing_view_policy not in {"black", "error"}:
            raise ValueError(
                f"Cosmos3 observation missing_view_policy must be 'black' or 'error', got {self.missing_view_policy!r}"
            )

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> ObservationContract:
        if not isinstance(payload, Mapping):
            raise ValueError(f"Cosmos3 observation contract must be a mapping, got {type(payload).__name__}")
        _require_fields(payload, {field.name for field in fields(cls)}, "Cosmos3 observation contract")
        return cls(
            layout_id=str(payload["layout_id"]),
            view_shape_hw=_metadata_pair(payload, "view_shape_hw"),
            canvas_shape_hw=_metadata_pair(payload, "canvas_shape_hw"),
            view_roles=_metadata_strings(payload, "view_roles"),
            missing_view_policy=str(payload["missing_view_policy"]),
            viewpoint=str(payload["viewpoint"]),
            description=str(payload["description"]),
        )


@dataclass(frozen=True)
class ObservationCapability:
    """Canvas that a selected RoboLab camera preset can actually produce.

    Free-form training prose is deliberately absent: description changes do
    not alter runtime compatibility.
    """

    layout_id: str
    view_shape_hw: tuple[int, int]
    canvas_shape_hw: tuple[int, int]
    view_roles: tuple[str, ...]
    role_sources: Mapping[str, str | None]
    missing_view_policies: tuple[str, ...]
    viewpoint: str = "concat_view"

    def __post_init__(self) -> None:
        if self.view_roles != OBSERVATION_VIEW_ROLES:
            raise ValueError(
                f"Client observation view_roles must be positional slots {OBSERVATION_VIEW_ROLES!r}, "
                f"got {self.view_roles!r}"
            )
        if set(self.role_sources) != set(self.view_roles):
            raise ValueError("Client observation role_sources must describe every declared view role exactly once")
        if not any(source is not None for source in self.role_sources.values()):
            raise ValueError("Client observation capability requires at least one real camera source")
        if any(source == "" for source in self.role_sources.values()):
            raise ValueError("Client observation role_sources cannot contain an empty camera source")
        if not self.missing_view_policies or any(
            policy not in {"black", "error"} for policy in self.missing_view_policies
        ):
            raise ValueError("Client observation missing_view_policies must contain only 'black' or 'error'")
        if any(source is None for source in self.role_sources.values()) and "error" in self.missing_view_policies:
            raise ValueError("Client observation presets with missing role sources cannot support policy 'error'")

    def validate(self, actual: ObservationContract, present_view_roles: tuple[str, ...]) -> None:
        differences = [
            f"observation.{field.name}: client={getattr(self, field.name)!r}, server={getattr(actual, field.name)!r}"
            for field in fields(self)
            if field.name not in {"missing_view_policies", "role_sources"}
            and getattr(self, field.name) != getattr(actual, field.name)
        ]
        if actual.missing_view_policy not in self.missing_view_policies:
            differences.append(
                "observation.missing_view_policy: "
                f"client supports={self.missing_view_policies!r}, server={actual.missing_view_policy!r}"
            )
        expected_present_view_roles = tuple(role for role in self.view_roles if self.role_sources[role] is not None)
        if present_view_roles != expected_present_view_roles:
            differences.append(
                "present_view_roles: "
                f"client preset={expected_present_view_roles!r}, server source={present_view_roles!r}"
            )
        if differences:
            raise ValueError("Cosmos3 observation preset mismatch: " + "; ".join(differences))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ObservationCapability:
        missing_view_policies = payload.get("missing_view_policies")
        if missing_view_policies is None:
            missing_view_policies = (payload["missing_view_policy"],)
        return cls(
            layout_id=str(payload["layout_id"]),
            view_shape_hw=_value_pair(payload["view_shape_hw"], "view_shape_hw"),
            canvas_shape_hw=_value_pair(payload["canvas_shape_hw"], "canvas_shape_hw"),
            view_roles=_value_strings(payload["view_roles"], "view_roles"),
            role_sources=_role_sources(payload["role_sources"]),
            missing_view_policies=_value_strings(missing_view_policies, "missing_view_policies"),
            viewpoint=str(payload.get("viewpoint", "concat_view")),
        )


STANDARD_THREE_VIEW_OBSERVATION = ObservationCapability(
    layout_id="primary_top_aux_bottom_pair",
    view_shape_hw=(360, 640),
    canvas_shape_hw=(540, 640),
    view_roles=OBSERVATION_VIEW_ROLES,
    role_sources={
        "primary": "wrist_cam",
        "aux_left": "over_shoulder_left_camera",
        "aux_right": "over_shoulder_right_camera",
    },
    missing_view_policies=("error", "black"),
)


@dataclass(frozen=True)
class ConditioningContract:
    """State/history rows required by the server-side action policy."""

    state_rows: int
    history_rows: int
    source: str

    def __post_init__(self) -> None:
        if self.state_rows < 0 or self.history_rows < self.state_rows:
            raise ValueError("Cosmos3 conditioning requires history_rows >= state_rows >= 0")
        if (self.state_rows == 0) != (self.source == "none"):
            raise ValueError("Cosmos3 conditioning source must be 'none' exactly when state_rows is zero")

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> ConditioningContract:
        if not isinstance(payload, Mapping):
            raise ValueError(f"Cosmos3 conditioning contract must be a mapping, got {type(payload).__name__}")
        _require_fields(payload, {field.name for field in fields(cls)}, "Cosmos3 conditioning contract")
        return cls(
            state_rows=_metadata_int(payload, "state_rows"),
            history_rows=_metadata_int(payload, "history_rows"),
            source=str(payload["source"]),
        )


JOINT_CURRENT_STATE_CONDITIONING = ConditioningContract(
    state_rows=1,
    history_rows=1,
    source="current_state",
)
STATELESS_CONDITIONING = ConditioningContract(state_rows=0, history_rows=0, source="none")


@dataclass(frozen=True)
class DecoderAnchorContract:
    """Current pose supplied for server-side delta decoding, not conditioning."""

    kind: str
    frame: str
    quaternion_order: str

    def __post_init__(self) -> None:
        if self.kind != "current_eef_pose" or not self.frame or self.quaternion_order != "xyzw":
            raise ValueError("Cosmos3 decoder anchor must be current_eef_pose in an explicit xyzw frame")

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> DecoderAnchorContract:
        if not isinstance(payload, Mapping):
            raise ValueError("Cosmos3 decoder_anchor must be a mapping")
        _require_fields(payload, {field.name for field in fields(cls)}, "Cosmos3 decoder_anchor")
        return cls(
            kind=str(payload["kind"]),
            frame=str(payload["frame"]),
            quaternion_order=str(payload["quaternion_order"]),
        )


@dataclass(frozen=True)
class PolicyContract:
    """Wire contract and selected static dataset-source semantics advertised by a server."""

    protocol_version: int
    profile_id: str
    dataset_source: str
    source_view_description: str
    present_view_roles: tuple[str, ...]
    robot: str
    action_space: str
    policy_fps: int
    chunk_size: int
    wire_action_dim: int
    action_layout: tuple[str, ...]
    gripper_indices: tuple[int, ...]
    gripper_semantics: str
    eef_frame: str | None
    quaternion_order: str | None
    pose_mode: str | None
    conditioning: ConditioningContract
    observation: ObservationContract
    decoder_anchor: DecoderAnchorContract | None = None

    def __post_init__(self) -> None:
        if self.protocol_version != POLICY_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported Cosmos3 policy contract version {self.protocol_version}; "
                f"expected {POLICY_CONTRACT_VERSION}"
            )
        if (
            not self.profile_id.strip()
            or not self.dataset_source.strip()
            or not self.source_view_description.strip()
            or not self.robot.strip()
            or not self.action_space.strip()
        ):
            raise ValueError(
                "Cosmos3 profile_id, dataset_source, source_view_description, robot, and action_space must be non-empty"
            )
        if len(set(self.present_view_roles)) != len(self.present_view_roles):
            raise ValueError("Cosmos3 present_view_roles entries must be unique")
        if not self.present_view_roles or not set(self.present_view_roles) <= set(self.observation.view_roles):
            raise ValueError("Cosmos3 present_view_roles must be a non-empty ordered subset of observation.view_roles")
        expected_role_order = tuple(role for role in self.observation.view_roles if role in self.present_view_roles)
        if self.present_view_roles != expected_role_order:
            raise ValueError("Cosmos3 present_view_roles must follow observation.view_roles order")
        if self.policy_fps <= 0 or self.chunk_size <= 0 or self.wire_action_dim <= 0:
            raise ValueError("Cosmos3 policy_fps, chunk_size, and wire_action_dim must be positive")
        if len(self.action_layout) != self.wire_action_dim:
            raise ValueError(
                f"Cosmos3 action_layout width {len(self.action_layout)} does not match "
                f"wire_action_dim={self.wire_action_dim}"
            )
        if len(set(self.action_layout)) != len(self.action_layout):
            raise ValueError("Cosmos3 action_layout entries must be unique")
        if len(self.gripper_indices) != 1:
            raise ValueError("RoboLab Cosmos3 clients require exactly one wire gripper channel")
        gripper_index = self.gripper_indices[0]
        if gripper_index < 0 or gripper_index >= self.wire_action_dim:
            raise ValueError(f"Cosmos3 gripper index {gripper_index} is outside the wire action layout")
        if self.action_layout[gripper_index] != "gripper":
            raise ValueError("Cosmos3 action_layout at gripper_indices[0] must be named 'gripper'")
        if self.gripper_semantics != CANONICAL_GRIPPER_SEMANTICS:
            raise ValueError(
                "Cosmos3 wire gripper semantics must be close_fraction (0=open, 1=closed), "
                f"got {self.gripper_semantics!r}"
            )

        if self.action_space.startswith("eef_"):
            if self.eef_frame is None or self.quaternion_order is None or self.pose_mode is None:
                raise ValueError("Cosmos3 EEF contracts require frame, quaternion order, and pose mode")
            if self.decoder_anchor is None or self.decoder_anchor.frame != self.eef_frame:
                raise ValueError("Cosmos3 EEF contracts require a decoder anchor in the wire EEF frame")
        elif self.eef_frame is not None or self.quaternion_order is not None:
            raise ValueError("Cosmos3 non-EEF contracts must not define EEF frame or quaternion order")
        elif self.decoder_anchor is not None:
            raise ValueError("Cosmos3 non-EEF contracts must not define decoder_anchor")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> PolicyContract:
        if not isinstance(payload, Mapping):
            raise ValueError(f"Cosmos3 policy contract must be a mapping, got {type(payload).__name__}")
        _require_fields(payload, {field.name for field in fields(cls)}, "Cosmos3 policy contract")
        return cls(
            protocol_version=_metadata_int(payload, "protocol_version"),
            profile_id=str(payload["profile_id"]),
            dataset_source=str(payload["dataset_source"]),
            source_view_description=str(payload["source_view_description"]),
            present_view_roles=_metadata_strings(payload, "present_view_roles"),
            robot=str(payload["robot"]),
            action_space=str(payload["action_space"]),
            policy_fps=_metadata_int(payload, "policy_fps"),
            chunk_size=_metadata_int(payload, "chunk_size"),
            wire_action_dim=_metadata_int(payload, "wire_action_dim"),
            action_layout=_metadata_strings(payload, "action_layout"),
            gripper_indices=_metadata_ints(payload, "gripper_indices"),
            gripper_semantics=str(payload["gripper_semantics"]),
            eef_frame=_optional_string(payload.get("eef_frame")),
            quaternion_order=_optional_string(payload.get("quaternion_order")),
            pose_mode=_optional_string(payload.get("pose_mode")),
            decoder_anchor=(
                DecoderAnchorContract.from_metadata(payload["decoder_anchor"])
                if payload.get("decoder_anchor") is not None
                else None
            ),
            conditioning=ConditioningContract.from_metadata(payload["conditioning"]),
            observation=ObservationContract.from_metadata(payload["observation"]),
        )

    def hold_ratio(self, control_fps: int = DEFAULT_CONTROL_FPS) -> int:
        if control_fps <= 0:
            raise ValueError(f"control_fps must be positive, got {control_fps}")
        if control_fps < self.policy_fps or control_fps % self.policy_fps != 0:
            raise ValueError(
                f"Cosmos3 policy FPS {self.policy_fps} must divide client control FPS {control_fps} exactly"
            )
        return control_fps // self.policy_fps

    def buffer_size(self, control_fps: int = DEFAULT_CONTROL_FPS) -> int:
        return self.chunk_size * self.hold_ratio(control_fps)


@dataclass(frozen=True)
class ClientCapability:
    """Robot/backend facts known by a RoboLab runner, not by a checkpoint."""

    robot: str
    arm_dof: int
    action_spaces: tuple[str, ...]
    joint_action_layout: tuple[str, ...]
    conditioning_by_action_space: Mapping[str, ConditioningContract]
    observation: ObservationCapability

    def __post_init__(self) -> None:
        if not self.robot or self.arm_dof <= 0 or not self.action_spaces:
            raise ValueError("Client robot, arm_dof, and action_spaces must be explicit")
        if "joint_position" in self.action_spaces:
            if len(self.joint_action_layout) != self.arm_dof or len(set(self.joint_action_layout)) != self.arm_dof:
                raise ValueError("Client joint_action_layout must name each arm joint exactly once")
        elif self.joint_action_layout:
            raise ValueError("Client joint_action_layout is only valid when joint_position is supported")
        if set(self.conditioning_by_action_space) != set(self.action_spaces):
            raise ValueError("Client conditioning capabilities must be explicit for every supported action space")

    def validate(self, contract: PolicyContract) -> None:
        if contract.robot != self.robot:
            raise ValueError(f"Cosmos3 robot mismatch: client={self.robot!r}, server={contract.robot!r}")
        if contract.action_space not in self.action_spaces:
            raise ValueError(
                f"Cosmos3 action-space mismatch for robot {self.robot!r}: "
                f"client supports {self.action_spaces!r}, server={contract.action_space!r}"
            )
        expected_conditioning = self.conditioning_by_action_space[contract.action_space]
        if contract.conditioning != expected_conditioning:
            raise ValueError(
                f"Cosmos3 conditioning mismatch for {contract.action_space!r}: "
                f"client={expected_conditioning!r}, server={contract.conditioning!r}"
            )
        self.observation.validate(contract.observation, contract.present_view_roles)

        if contract.action_space == "joint_position":
            expected_dim = self.arm_dof + 1
            expected_layout = self.joint_action_layout + ("gripper",)
            if (
                contract.wire_action_dim != expected_dim
                or contract.action_layout != expected_layout
                or contract.gripper_indices != (self.arm_dof,)
                or contract.pose_mode != "absolute"
            ):
                raise ValueError(
                    f"Cosmos3 joint wire schema for {self.robot!r} must be absolute and contain "
                    f"{self.arm_dof} arm joints followed by one gripper (dim={expected_dim}); "
                    f"expected layout={expected_layout!r}; got dim={contract.wire_action_dim}, "
                    f"layout={contract.action_layout!r}, gripper_indices={contract.gripper_indices}, "
                    f"pose_mode={contract.pose_mode!r}"
                )


def read_server_contract(metadata: Mapping[str, Any]) -> PolicyContract:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Cosmos3 server metadata must be a mapping, got {type(metadata).__name__}")
    if POLICY_CONTRACT_METADATA_KEY not in metadata:
        raise ValueError("Cosmos3 server metadata is missing required 'policy_contract'")
    return PolicyContract.from_metadata(metadata[POLICY_CONTRACT_METADATA_KEY])


def validate_server_metadata(metadata: Mapping[str, Any], capability: ClientCapability) -> PolicyContract:
    contract = read_server_contract(metadata)
    capability.validate(contract)
    return contract


def validate_same_policy_semantics(expected: PolicyContract, actual: PolicyContract) -> None:
    """Reject changed semantics while ignoring diagnostic labels and generic observation prose."""

    ignored = {"profile_id"}
    differences = [
        f"{field.name}: before={getattr(expected, field.name)!r}, after={getattr(actual, field.name)!r}"
        for field in fields(PolicyContract)
        if field.name not in ignored
        and field.name != "observation"
        and getattr(expected, field.name) != getattr(actual, field.name)
    ]
    for field in fields(ObservationContract):
        if field.name == "description":
            continue
        if getattr(expected.observation, field.name) != getattr(actual.observation, field.name):
            differences.append(
                f"observation.{field.name}: before={getattr(expected.observation, field.name)!r}, "
                f"after={getattr(actual.observation, field.name)!r}"
            )
    if differences:
        raise ValueError("Cosmos3 server contract changed across reconnect: " + "; ".join(differences))


def validate_wire_action_chunk(value: Any, contract: PolicyContract) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    expected_shape = (contract.chunk_size, contract.wire_action_dim)
    if action.ndim != 2 or action.shape != expected_shape:
        raise ValueError(f"Expected Cosmos3 wire action shape {expected_shape}, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Cosmos3 wire action chunk contains non-finite values")
    return np.ascontiguousarray(action)


def binarize_close_fraction(chunk: np.ndarray, *, gripper_index: int = -1) -> np.ndarray:
    """Apply the one canonical RoboLab gripper threshold in one place."""

    result = np.asarray(chunk).copy()
    result[..., gripper_index] = (result[..., gripper_index] > 0.5).astype(result.dtype)
    return result


def expand_action_chunk(chunk: np.ndarray, contract: PolicyContract, *, control_fps: int) -> np.ndarray:
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[0] != contract.chunk_size:
        raise ValueError(f"Expected decoded action chunk with {contract.chunk_size} rows, got {chunk.shape}")
    repeats = contract.hold_ratio(control_fps)
    if repeats == 1:
        return np.ascontiguousarray(chunk)
    # Keep sim/controllers at 15 Hz; hold lower-rate targets instead of changing environment timing.
    return np.ascontiguousarray(np.repeat(chunk, repeats, axis=0))


def _require_fields(payload: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")


def _metadata_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Cosmos3 policy contract field {key!r} must be an integer, got {value!r}")
    return int(value)


def _metadata_ints(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = payload[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Cosmos3 policy contract field {key!r} must be a sequence")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"Cosmos3 policy contract field {key!r} must contain only integers")
        result.append(int(item))
    return tuple(result)


def _metadata_strings(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return _value_strings(payload[key], key)


def _value_strings(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Cosmos3 policy contract field {key!r} must be a non-empty sequence")
    return tuple(str(item) for item in value)


def _metadata_pair(payload: Mapping[str, Any], key: str) -> tuple[int, int]:
    return _value_pair(payload[key], key)


def _value_pair(value: Any, key: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Cosmos3 policy contract field {key!r} must contain exactly two integers")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
        raise ValueError(f"Cosmos3 policy contract field {key!r} must contain exactly two integers")
    return int(value[0]), int(value[1])


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _role_sources(value: Any) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError("Client observation role_sources must be a mapping")
    return {str(role): None if source is None else str(source) for role, source in value.items()}
