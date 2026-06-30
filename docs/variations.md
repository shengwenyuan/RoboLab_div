# Variation Map

RoboLab uses "variation" in a few different ways. The important distinction is
when the change is applied: while registering environments, while sweeping
evaluation runs, or at episode reset.

## 1. Registration-Time Configuration

Registration-time variations are fixed into the generated environment config.
They define what exists in the scene and what observations are exposed before an
episode starts.

Primary entry points:

| Area | Source | Effect |
|------|--------|--------|
| Cameras | `robolab/variations/camera.py` | Adds fixed scene cameras such as over-shoulder, head, and egocentric views |
| Camera presets | `robolab/registrations/droid/camera_presets.py` | Chooses which camera configs become policy image observations |
| Lighting | `robolab/variations/lighting.py` | Adds sphere, colored, or directional lights |
| Backgrounds | `robolab/variations/backgrounds.py` | Adds HDR/EXR dome-light backgrounds |
| DROID registration | `robolab/registrations/droid/auto_env_registrations_jointpos.py` | Combines task, robot, cameras, lighting, background, observations, and actions |
| ABS-IK registration | `robolab/registrations/droid/auto_env_registrations_abs_ik.py` | Same composition pattern for absolute IK actions |

The composition happens through:

- `robolab/core/environments/config.py`
- `robolab/core/environments/factory.py`

Use this layer when the variation should be part of the environment definition:
for example, which policy cameras exist, whether a viewport camera is attached,
or which background config a registered env uses.

## 2. Evaluation Sweep Layer

Sweep variations create or run multiple evaluation conditions for the same base
task. They usually live in policy runner scripts and may either register many
env variants or modify the scene after creating an env.

Primary entry points:

| Runner | Effect | Mechanism |
|--------|--------|-----------|
| `policies/pi0_family/run_lighting.py` | Sweeps lighting intensity, directional light, and colored lights | Registers env variants with different lighting/background configs |
| `policies/pi0_family/run_background_variation.py` | Sweeps a task x background matrix | Registers one env per task/background pair |
| `policies/pi0_family/run_camera_pose_variation.py` | Sweeps camera pose perturbation events | Reuses base envs and passes reset-time events into `create_env` |
| `policies/pi0_family/run_table_variation.py` | Sweeps table materials | Creates the base env, then changes USD material binding at runtime |

Supporting registration helpers:

- `robolab/registrations/droid/auto_env_registrations_bg_variations.py`
- `robolab/registrations/droid/auto_env_registrations_lighting_variations.py`

Use this layer when the experiment question is comparative: run the same task
under several lighting, background, camera, or material conditions and record
separate results for each condition.

## 3. Reset-Time Event Layer

Reset-time variations are sampled when an episode resets. They do not require
copying the whole task or registering a separate env unless a workflow chooses
to package them that way.

Primary entry points:

| Event | Source | Effect |
|-------|--------|--------|
| Camera pose randomization | `robolab/core/events/reset_camera.py` | Applies position/orientation deltas to selected camera sensors at reset |
| Object initial pose randomization | `robolab/core/events/reset_pose.py` | Applies root-pose and velocity deltas to selected scene assets at reset |
| Contact object pose randomization | `robolab/core/events/reset_contact_pose.py` | Selects movable task contact objects and applies reset-time pose deltas |
| Event merging | `robolab/core/events/utils.py` | Merges additional events into an existing env config |
| Runtime event injection | `robolab/core/environments/runtime.py` | Accepts `create_env(..., events=...)` and merges the event config |

Built-in task-copy examples live in:

- `robolab/tasks/randomize_initial_pose/`

Those task files demonstrate reset-time object randomization, but they hard-code
object names into task variants. For new reusable randomization, prefer adding a
small semantic event and injecting it through `create_env(..., events=...)` or a
runner-level option.

## How To Classify A Variation

Use this quick rule:

| Question | Layer |
|----------|-------|
| Does it decide what sensors, lights, backgrounds, observations, or robot/action configs exist in the env? | Registration-time configuration |
| Does it compare several named conditions and write separate eval results? | Evaluation sweep layer |
| Does it sample a new value each episode reset? | Reset-time event layer |

Some workflows use more than one layer. For example, camera pose variation is an
evaluation sweep over reset-time events: the runner chooses which camera group to
perturb, and the event samples the actual pose delta at each episode reset.

## Notes On Cameras And Videos

Policy image observations and video-only viewport cameras are separate. DROID
registration uses camera presets for policy observations, then attaches
`EgocentricMirroredCameraCfg` separately as `viewport_cam` for recording.

For Cosmos3, the model request is built from:

- `over_shoulder_left_camera`
- `over_shoulder_right_camera`
- `wrist_cam`

If the environment is registered with `WRIST_LEFT_RIGHT_HEAD`, the sensor video
may also include `head_camera`, even though the Cosmos3 client does not send it
to the model.
