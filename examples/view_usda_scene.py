# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open a USD/USD[A] scene directly in Isaac Sim for lightweight viewing."""

import argparse
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated numbers, for example: 2.5,0,1.5")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


parser = argparse.ArgumentParser(description="Open a USDA/USD stage directly in Isaac Sim.")
parser.add_argument(
    "--usd-path",
    default=str(REPO_ROOT / "assets/scenes/banana_bowl.usda"),
    help="Path to the USDA/USD/USDZ file to open.",
)
parser.add_argument("--headless", action="store_true", help="Run without a local window.")
parser.add_argument(
    "--livestream",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="Enable WebRTC livestream. Use 2 for compatibility with Isaac Lab/Isaac Sim CLI style.",
)
parser.add_argument("--width", type=int, default=1280, help="Render width.")
parser.add_argument("--height", type=int, default=720, help="Render height.")
parser.add_argument("--stream-port", type=int, default=49100, help="WebRTC websocket/stream port.")
parser.add_argument("--renderer", default="RaytracedLighting", help="Isaac Sim renderer name.")
parser.add_argument("--eye", type=parse_vec3, default=(2.5, 0.0, 1.5), help="Viewport camera eye: x,y,z.")
parser.add_argument("--target", type=parse_vec3, default=(0.5, 0.0, 0.1), help="Viewport camera target: x,y,z.")
parser.add_argument("--no-light", action="store_true", help="Do not add a simple dome/distant light.")
parser.add_argument("--test-frames", type=int, default=-1, help="Exit after N frames. Use -1 to run until closed.")
args, _ = parser.parse_known_args()

usd_path = Path(args.usd_path).expanduser()
if not usd_path.is_absolute():
    usd_path = (Path.cwd() / usd_path).resolve()
if not usd_path.exists():
    print(f"[USDA Viewer] File not found: {usd_path}", file=sys.stderr)
    sys.exit(1)

config = {
    "width": args.width,
    "height": args.height,
    "window_width": args.width,
    "window_height": args.height,
    "headless": args.headless or args.livestream > 0,
    "hide_ui": False,
    "renderer": args.renderer,
    "sync_loads": True,
    "display_options": 3286,
}

kit = SimulationApp(launch_config=config)

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import UsdLux  # noqa: E402


def add_light() -> None:
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/ViewerDomeLight"):
        dome = UsdLux.DomeLight.Define(stage, "/ViewerDomeLight")
        dome.CreateIntensityAttr(500.0)
    if not stage.GetPrimAtPath("/ViewerDistantLight"):
        distant = UsdLux.DistantLight.Define(stage, "/ViewerDistantLight")
        distant.CreateIntensityAttr(3000.0)
        distant.CreateAngleAttr(0.5)


try:
    if args.livestream:
        kit.set_setting("/app/window/drawMouse", True)
        kit.set_setting("/app/livestream/port", args.stream_port)
        enable_extension("omni.kit.livestream.webrtc")
        kit.update()

    print(f"[USDA Viewer] Opening stage: {usd_path}", flush=True)
    open_result = omni.usd.get_context().open_stage(str(usd_path))
    if open_result is False:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    kit.update()
    kit.update()
    while is_stage_loading():
        kit.update()

    if not args.no_light:
        add_light()

    set_camera_view(eye=args.eye, target=args.target, camera_prim_path="/OmniverseKit_Persp")
    for _ in range(5):
        kit.update()

    print("[USDA Viewer] Stage ready.", flush=True)
    if args.livestream:
        print(f"[USDA Viewer] WebRTC signal/stream port: {args.stream_port}", flush=True)
    print("[USDA Viewer] Stop with Ctrl+C or close the viewer.", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    frame = 0
    while kit.is_running() and not kit.is_exiting():
        kit.update()
        frame += 1
        if args.test_frames >= 0 and frame >= args.test_frames:
            break
except Exception as exc:
    print(f"[USDA Viewer] Terminated with error: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    raise
finally:
    kit.close()
