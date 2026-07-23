#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open and load every RoboLab USD asset and report unresolved dependencies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
DEFAULT_DIRS = ("objects", "scenes", "fixtures", "robots")
EXTERNAL_SCHEMES = {"http", "https", "omniverse"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--asset-storage-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dirs", nargs="*", default=list(DEFAULT_DIRS))
    parser.add_argument("--allow-absolute-root", action="append", default=[], type=Path)
    parser.add_argument("--progress-seconds", default=30.0, type=float)
    parser.add_argument("--min-output-free-gb", default=2.0, type=float)
    parser.add_argument("--min-asset-free-gb", default=50.0, type=float)
    return parser.parse_args()


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def asset_storage_root(args: argparse.Namespace, asset_root: Path) -> Path:
    return (args.asset_storage_root or asset_root).resolve()


def assert_disk_ok(args: argparse.Namespace, monitored_asset_root: Path) -> None:
    output_free = disk_free_gb(args.output_dir)
    asset_free = disk_free_gb(monitored_asset_root)
    if output_free < args.min_output_free_gb:
        raise RuntimeError(
            f"output filesystem low on space: {output_free:.2f} GiB free "
            f"< {args.min_output_free_gb:.2f} GiB"
        )
    if asset_free < args.min_asset_free_gb:
        raise RuntimeError(
            f"asset filesystem low on space: {asset_free:.2f} GiB free "
            f"< {args.min_asset_free_gb:.2f} GiB"
        )


def iter_usd_files(asset_root: Path, dirs: list[str]):
    seen_dirs: set[tuple[int, int]] = set()
    for rel_dir in dirs:
        start = asset_root / rel_dir
        if not start.exists():
            yield start, FileNotFoundError(start)
            continue
        for dirpath, dirnames, filenames in os.walk(start, followlinks=True):
            try:
                st = os.stat(dirpath)
            except OSError as exc:
                yield Path(dirpath), exc
                dirnames[:] = []
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen_dirs:
                dirnames[:] = []
                continue
            seen_dirs.add(key)
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.lower() in USD_EXTENSIONS:
                    yield path, None


def is_external_or_bad_path(path_value: str, allowed_absolute_roots: list[Path]) -> str | None:
    if not path_value:
        return None
    parsed = urlparse(path_value)
    if parsed.scheme.lower() in EXTERNAL_SCHEMES:
        return "external_uri"
    if os.path.isabs(path_value):
        try:
            resolved = Path(path_value).resolve()
        except OSError:
            return "absolute_unresolvable"
        for root in allowed_absolute_roots:
            try:
                resolved.relative_to(root)
                return None
            except ValueError:
                continue
        return "absolute_outside_allowed_roots"
    return None


def collect_asset_path_values(stage: Usd.Stage) -> list[str]:
    values: list[str] = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            try:
                value = attr.Get()
            except Exception:
                continue
            if isinstance(value, Sdf.AssetPath):
                values.append(value.path)
                if value.resolvedPath:
                    values.append(value.resolvedPath)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Sdf.AssetPath):
                        values.append(item.path)
                        if item.resolvedPath:
                            values.append(item.resolvedPath)
    return values


def dependency_strings(path: Path) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    unresolved_seen: set[str] = set()
    deps: list[str] = []
    unresolved_values: list[str] = []
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:
        return [], [f"ComputeAllDependencies failed: {exc}"]

    for layer in layers:
        value = getattr(layer, "identifier", None) or str(layer)
        if value not in seen:
            seen.add(value)
            deps.append(value)
    for asset in assets:
        value = str(asset)
        if value not in seen:
            seen.add(value)
            deps.append(value)
    for item in unresolved:
        value = str(item)
        if value not in unresolved_seen:
            unresolved_seen.add(value)
            unresolved_values.append(value)
    return deps, unresolved_values


def verify_one(path: Path, allowed_absolute_roots: list[Path]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    mark = Tf.Error.Mark()
    mark.SetMark()

    try:
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    except Exception as exc:
        return [{"kind": "stage_open_exception", "detail": str(exc)}]

    if stage is None:
        return [{"kind": "stage_open_failed", "detail": "Usd.Stage.Open returned None"}]

    try:
        stage.Load()
    except Exception as exc:
        issues.append({"kind": "stage_load_exception", "detail": str(exc)})

    deps, unresolved = dependency_strings(path)
    for value in unresolved:
        issues.append({"kind": "unresolved_dependency", "detail": value})

    for value in deps + collect_asset_path_values(stage):
        reason = is_external_or_bad_path(value, allowed_absolute_roots)
        if reason:
            issues.append({"kind": reason, "detail": value})

    if not mark.IsClean():
        for err in mark.GetErrors():
            issues.append({"kind": "usd_error", "detail": str(err)})

    return issues


def verify_assets(args: argparse.Namespace) -> int:
    asset_root = args.asset_root.resolve()
    monitored_asset_root = asset_storage_root(args, asset_root)
    asset_root_real = Path(os.path.realpath(monitored_asset_root))
    allowed_absolute_roots = [asset_root, asset_root_real]
    allowed_absolute_roots.extend(path.resolve() for path in args.allow_absolute_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not monitored_asset_root.exists():
        raise FileNotFoundError(monitored_asset_root)
    assert_disk_ok(args, monitored_asset_root)

    run_name = args.name or time.strftime("usd_%Y%m%dT%H%M%SZ", time.gmtime())
    failures_path = args.output_dir / f"{run_name}.failures.jsonl"
    summary_path = args.output_dir / f"{run_name}.summary.json"

    started = time.monotonic()
    last_progress = started
    checked = 0
    failed = 0
    missing_scan_errors = 0

    with failures_path.open("w", encoding="utf-8") as failures:
        for path, scan_error in iter_usd_files(asset_root, args.dirs):
            now = time.monotonic()
            if now - last_progress >= args.progress_seconds:
                elapsed = max(0.001, now - started)
                print(
                    f"PROGRESS usd_checked={checked} usd_failed={failed} "
                    f"elapsed_seconds={elapsed:.1f} output_free_gib={disk_free_gb(args.output_dir):.2f} "
                    f"asset_free_gib={disk_free_gb(monitored_asset_root):.2f}",
                    flush=True,
                )
                assert_disk_ok(args, monitored_asset_root)
                last_progress = now

            if scan_error is not None:
                missing_scan_errors += 1
                failed += 1
                failures.write(
                    json.dumps(
                        {
                            "file": str(path),
                            "relative": None,
                            "issues": [{"kind": "scan_error", "detail": str(scan_error)}],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                failures.flush()
                continue

            checked += 1
            issues = verify_one(path, allowed_absolute_roots)
            if issues:
                failed += 1
                failures.write(
                    json.dumps(
                        {
                            "file": str(path),
                            "relative": path.relative_to(asset_root).as_posix(),
                            "issues": issues,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                failures.flush()

    elapsed = time.monotonic() - started
    summary = {
        "asset_root": str(asset_root),
        "asset_root_real": str(asset_root_real),
        "asset_storage_root": str(monitored_asset_root),
        "allowed_absolute_roots": [str(path) for path in allowed_absolute_roots],
        "dirs": args.dirs,
        "usd_checked": checked,
        "usd_failed": failed,
        "scan_errors": missing_scan_errors,
        "failures": str(failures_path),
        "elapsed_seconds": elapsed,
        "output_free_gib": disk_free_gb(args.output_dir),
        "asset_free_gib": disk_free_gb(monitored_asset_root),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failed else 0


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        global Sdf, Tf, Usd, UsdUtils
        from pxr import Sdf, Tf, Usd, UsdUtils

        return verify_assets(args)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
