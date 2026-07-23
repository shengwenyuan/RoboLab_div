#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify RoboLab asset files are materialized by reading every byte.

This intentionally follows symlinked top-level directories, so a validation
entrypoint such as assets_0701 can point at a shared asset snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--asset-storage-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--progress-seconds", default=30.0, type=float)
    parser.add_argument("--min-output-free-gb", default=2.0, type=float)
    parser.add_argument("--min-asset-free-gb", default=50.0, type=float)
    parser.add_argument("--chunk-mb", default=16, type=int)
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


def iter_files(root: Path):
    seen_dirs: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
        except OSError as exc:
            yield Path(dirpath), None, exc
            dirnames[:] = []
            continue

        key = (st.st_dev, st.st_ino)
        if key in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(key)

        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                st_file = os.stat(path)
            except OSError as exc:
                yield path, None, exc
                continue
            if not os.path.isfile(path):
                continue
            yield path, st_file, None


def rel_for_manifest(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def hash_file(path: Path, chunk_size: int) -> tuple[str, bool, int]:
    hasher = hashlib.sha256()
    size = 0
    prefix = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            if len(prefix) < 256:
                prefix += chunk[: 256 - len(prefix)]
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), prefix.startswith(LFS_POINTER_PREFIX), size


def main() -> int:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not asset_root.exists():
        raise FileNotFoundError(asset_root)
    monitored_asset_root = asset_storage_root(args, asset_root)
    if not monitored_asset_root.exists():
        raise FileNotFoundError(monitored_asset_root)

    run_name = args.name or time.strftime("assets_%Y%m%dT%H%M%SZ", time.gmtime())
    manifest_path = args.output_dir / f"{run_name}.sha256"
    pointers_path = args.output_dir / f"{run_name}.lfs_pointers.txt"
    errors_path = args.output_dir / f"{run_name}.file_errors.txt"
    summary_path = args.output_dir / f"{run_name}.summary.json"
    tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    assert_disk_ok(args, monitored_asset_root)

    chunk_size = max(1, args.chunk_mb) * 1024 * 1024
    started = time.monotonic()
    last_progress = started
    total_files = 0
    total_bytes = 0
    pointer_count = 0
    error_count = 0

    with tmp_manifest_path.open("w", encoding="utf-8") as manifest, \
            pointers_path.open("w", encoding="utf-8") as pointers, \
            errors_path.open("w", encoding="utf-8") as errors:
        for path, st_file, err in iter_files(asset_root):
            now = time.monotonic()
            if now - last_progress >= args.progress_seconds:
                elapsed = max(0.001, now - started)
                gib = total_bytes / 1024**3
                rate = gib / elapsed * 3600
                print(
                    f"PROGRESS files={total_files} bytes_gib={gib:.2f} "
                    f"rate_gib_per_hour={rate:.2f} output_free_gib={disk_free_gb(args.output_dir):.2f} "
                    f"asset_free_gib={disk_free_gb(monitored_asset_root):.2f}",
                    flush=True,
                )
                assert_disk_ok(args, monitored_asset_root)
                last_progress = now

            if err is not None:
                error_count += 1
                errors.write(f"{path}\t{err}\n")
                errors.flush()
                continue

            rel = rel_for_manifest(asset_root, path)
            try:
                digest, is_pointer, size = hash_file(path, chunk_size)
            except OSError as exc:
                error_count += 1
                errors.write(f"{rel}\t{exc}\n")
                errors.flush()
                continue

            total_files += 1
            total_bytes += size
            manifest.write(f"{digest}  {rel}\n")

            if is_pointer:
                pointer_count += 1
                pointers.write(f"{rel}\n")
                pointers.flush()

    tmp_manifest_path.replace(manifest_path)
    elapsed = time.monotonic() - started
    summary = {
        "asset_root": str(asset_root),
        "asset_storage_root": str(monitored_asset_root),
        "manifest": str(manifest_path),
        "lfs_pointers": str(pointers_path),
        "errors": str(errors_path),
        "files": total_files,
        "bytes": total_bytes,
        "bytes_gib": total_bytes / 1024**3,
        "elapsed_seconds": elapsed,
        "pointer_count": pointer_count,
        "error_count": error_count,
        "output_free_gib": disk_free_gb(args.output_dir),
        "asset_free_gib": disk_free_gb(monitored_asset_root),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if pointer_count or error_count else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
