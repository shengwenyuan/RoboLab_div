# RoboLab_div Isaac Sim 6.0.1 migration

## Baseline and target

- Frozen baseline: `e2038c7` (`Add Cosmos3 DROID EEF evaluation path`).
- Runtime: reuse `/home/lenovo/swy/RoboLab/edge_verify/envs/isaac601-robolab`.
- Reference: official RoboLab `edge-isaac601-migration` worktree; port only verified compatibility changes.
- Preserve policy semantics: DROID EEF absolute wire action, 15 Hz, predict 32, execute 8, current-pose decoder anchor.
- Acceptance task: one `BananaInBowlTask` episode with RoboLab viewport video.

## Compatibility work

1. Port Isaac Lab 3 interfaces: articulation/contact base classes, proxy-array tensor access, and XForm view handling.
2. Convert legacy USD/config `wxyz` quaternions only at Isaac Lab 3 boundaries; keep policy/network quaternions `xyzw` unchanged.
3. Resolve AppLauncher renderer argument ownership without duplicating CLI flags.
4. Reuse the local OpenPI protocol shim where Isaac 6's pinned `websockets` conflicts with newer clients.
5. Keep task definitions, success predicates, camera poses, IK frame transform, and action execution horizon unchanged.

## Gates

- G0 — Runtime: exact Python/Isaac versions, clean renderer startup, asset root and ffmpeg available.
- G1 — Static/unit: imports, quaternion tests, Cosmos EEF/UR5 tests, no formatting errors.
- G2 — Scene: register and construct one DROID absolute-IK Banana-in-Bowl environment; validate robot, banana, bowl, cameras, and viewport.
- G3 — Contract: connect to checkpoint 3500 server, validate manifest metadata, then complete one action request with finite `[32, 8]` output.
- G4 — Episode: execute predict-32/execute-8 rollout; save official viewport MP4 and validate codec, FPS, duration, and nonzero frames.

Stop immediately on renderer crash, GPU reset/Xid, invalid transforms, non-finite actions, or missing assets. Do not weaken task success criteria to obtain a video.

## Outputs

- Logs and videos: `output/cosmos3_eef_iter3500_banana_isaac601/`.
- Migration code remains small and version-boundary focused; commit only after G1, then record later gates separately.
