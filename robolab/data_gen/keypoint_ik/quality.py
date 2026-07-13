# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality summaries for keypoint-IK batch runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from robolab.data_gen.keypoint_ik.lerobot import validate_keypoint_dataset


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL batch manifest."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_manifest(path: str | Path) -> dict[str, Any]:
    """Summarize all jobs in a keypoint-IK manifest."""

    jobs = load_manifest(path)
    rows = [summarize_job(job) for job in jobs]
    status_counts = Counter(row["status"] for row in rows)
    by_target = _group_counts(rows, "target_object")
    by_container = _group_counts(rows, "container_object")
    by_appearance = _group_counts(rows, "appearance_name")
    return {
        "manifest": str(Path(path)),
        "total_jobs": len(jobs),
        "status_counts": dict(sorted(status_counts.items())),
        "success_count": status_counts.get("success", 0),
        "failure_count": status_counts.get("failure", 0),
        "invalid_count": status_counts.get("invalid", 0),
        "missing_count": status_counts.get("missing", 0),
        "by_target": by_target,
        "by_container": by_container,
        "by_appearance": by_appearance,
        "jobs": rows,
    }


def summarize_job(job: dict[str, Any]) -> dict[str, Any]:
    """Summarize one manifest job's raw output directory."""

    variant = job.get("variant", {})
    output_dir = Path(job["output_dir"])
    metadata_path = output_dir / "episode_000000" / "metadata.json"
    metadata = _load_json(metadata_path)
    validation = validate_keypoint_dataset(output_dir)

    metadata_success = metadata.get("success") if isinstance(metadata, dict) else None
    if not output_dir.exists():
        status = "missing"
    elif not metadata_path.exists() or not validation["ok"]:
        status = "invalid"
    elif metadata_success is True and validation["successful_demo_count"] >= 1:
        status = "success"
    elif metadata_success is False:
        status = "failure"
    else:
        status = "invalid"

    return {
        "job_index": int(job["job_index"]),
        "status": status,
        "target_object": variant.get("target_asset_name", variant.get("target_object")),
        "container_object": variant.get("container_asset_name", variant.get("container_object")),
        "variant_id": variant.get("variant_id"),
        "appearance_name": job.get("appearance_name"),
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.exists(),
        "metadata_success": metadata_success,
        "hdf5_count": validation["hdf5_count"],
        "demo_count": validation["demo_count"],
        "ok_demo_count": validation["ok_demo_count"],
        "successful_demo_count": validation["successful_demo_count"],
        "validation_ok": validation["ok"],
        "validation_errors": _validation_errors(validation),
        "num_frames": _first_demo_field(validation, "num_frames"),
        "actions_shape": _first_demo_field(validation, "actions_shape"),
        "joint_position_shape": _first_demo_field(validation, "joint_position_shape"),
        "video_paths": _first_demo_field(validation, "video_paths") or {},
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _validation_errors(validation: dict[str, Any]) -> list[str]:
    errors = list(validation.get("errors", []))
    for report in validation.get("reports", []):
        errors.extend(report.get("errors", []))
    return errors


def _first_demo_field(validation: dict[str, Any], field: str):
    for report in validation.get("reports", []):
        demos = report.get("demos", [])
        if demos:
            return demos[0].get(field)
    return None


def _group_counts(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        groups[str(row.get(key))][row["status"]] += 1
        groups[str(row.get(key))]["total"] += 1
    return {name: dict(sorted(counter.items())) for name, counter in sorted(groups.items())}
