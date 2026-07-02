# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset-time pose randomization for task contact objects."""

import logging
from collections.abc import Iterable

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from robolab.core.events.reset_pose import reset_pose_uniform

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE_NAMES = ("table", "franka_table", "floor", "ground")


def _normalize_names(names: Iterable[str] | str | None) -> list[str] | None:
    if names is None:
        return None
    if isinstance(names, str):
        return [names]
    return [name for name in names if name]


def get_randomizable_contact_object_names(
    env: ManagerBasedEnv,
    *,
    include_names: Iterable[str] | str | None = None,
    exclude_names: Iterable[str] | str | None = _DEFAULT_EXCLUDE_NAMES,
) -> list[str]:
    """Return contact objects that are movable rigid objects in this scene."""
    contact_names = list(getattr(env.cfg, "contact_object_list", []) or [])
    rigid_names = set(env.scene.rigid_objects.keys())
    include = _normalize_names(include_names)
    exclude = set(_normalize_names(exclude_names) or [])

    if include is None:
        candidates = contact_names
    else:
        candidates = include

    selected = []
    seen = set()
    for name in candidates:
        if name in seen or name in exclude or name not in rigid_names:
            continue
        selected.append(name)
        seen.add(name)
    return selected


def reset_contact_object_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    include_names: Iterable[str] | str | None = None,
    exclude_names: Iterable[str] | str | None = _DEFAULT_EXCLUDE_NAMES,
    reset_to_default_otherwise: bool = False,
    use_collision_check: bool = True,
    collision_margin: float = 0.01,
    max_retries: int = 100,
):
    """Randomize movable contact-object poses at episode reset.

    The selector starts from ``env.cfg.contact_object_list`` and keeps only names
    that exist in ``env.scene.rigid_objects``. Static fixtures such as tables are
    therefore skipped automatically.
    """
    object_names = get_randomizable_contact_object_names(
        env,
        include_names=include_names,
        exclude_names=exclude_names,
    )
    if not object_names:
        logger.warning("No movable contact objects found for pose randomization; skipping reset event.")
        return

    if velocity_range is None:
        velocity_range = {}

    logger.debug(
        "Randomizing contact object poses: objects=%s pose_range=%s collision_check=%s margin=%s",
        object_names,
        pose_range,
        use_collision_check,
        collision_margin,
    )
    reset_pose_uniform(
        env=env,
        env_ids=env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=object_names,
        reset_to_default_otherwise=reset_to_default_otherwise,
        use_collision_check=use_collision_check,
        collision_margin=collision_margin,
        max_retries=max_retries,
    )


@configclass
class RandomizeContactObjectPoseUniform:
    """Config helper for reset-time contact-object pose randomization."""

    @classmethod
    def from_params(
        cls,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]] | None = None,
        include_names: Iterable[str] | str | None = None,
        exclude_names: Iterable[str] | str | None = _DEFAULT_EXCLUDE_NAMES,
        reset_to_default_otherwise: bool = False,
        use_collision_check: bool = True,
        collision_margin: float = 0.01,
        max_retries: int = 100,
    ):
        if velocity_range is None:
            velocity_range = {}

        class CustomRandomizeContactObjectPoseUniform(cls):
            randomize_contact_object_pose = EventTerm(
                func=reset_contact_object_pose_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": velocity_range,
                    "include_names": include_names,
                    "exclude_names": exclude_names,
                    "reset_to_default_otherwise": reset_to_default_otherwise,
                    "use_collision_check": use_collision_check,
                    "collision_margin": collision_margin,
                    "max_retries": max_retries,
                },
            )

        return CustomRandomizeContactObjectPoseUniform()
