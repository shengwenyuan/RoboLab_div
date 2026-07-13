# Cosmos 3

[**Cosmos 3**](https://huggingface.co/collections/nvidia/cosmos3) is a suite of omnimodal world models designed to jointly process and generate language, images, video, audio, and action sequences. This directory provides the RoboLab client for [**Cosmos3-Nano-Policy-DROID**](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), a World-Action Model (WAM) obtained by post-training Cosmos 3 on the [DROID](https://droid-dataset.github.io/) dataset.

[`client.py`](./client.py) provides the `Cosmos3Client` class that connects to a policy server hosting Cosmos3-Nano-Policy-DROID over the OpenPI WebSocket protocol. `Cosmos3Client` requires the [`openpi-client`](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client) package.

Below is a quickstart for bringing up the policy server and running an evaluation from a RoboLab client.

## Server

First, clone [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework):

```Shell
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Build the Docker image:

```Shell
docker build \
  -t cosmos-framework:latest \
  .
```

Set your Hugging Face token and launch the container, which installs the dependencies:

```Shell
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>

docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-train \
      --group=policy-server && \
    exec bash; \
  '
```

Inside the container, start the policy server:

```Shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --port 8000
```

## Policy contract

Training writes one versioned `action_policy.yaml` at each run root, and exported artifacts copy it to their export root. The server loads that file and publishes the wire contract plus the selected static dataset source in the WebSocket handshake. The client uses this handshake as the only source for policy FPS, chunk size, wire dimension/layout, action codec, EEF representation, gripper semantics, exact source view prompt, and which canvas slots were present during training. Dataset paths remain private. There is no client-side profile registry and no action-shape guessing.

RoboLab contributes only facts that are local to the selected runtime: robot identity, arm DOF, supported decoder, control FPS, and camera preset. It rejects an incompatible server before applying the first action. In particular, the preset's real/black slot mask must exactly match the selected source's `present_view_roles`. A server profile name and the generic observation description are labels for diagnostics; the selected `dataset_source`, `source_view_description`, and presence mask are reconnect compatibility keys. The source is fixed by the server process and is not repeated in inference requests.

Wire gripper semantics are always `close_fraction`: `0=open`, `1=closed`. The threshold used by all joint and EEF paths is shared. Lower-rate policies keep simulation/controllers at 15 Hz; for example, a 5 Hz policy action is held for three control steps. Environment decimation and physics timing are unchanged.

UR5 image layout is configured independently from the action contract. Every manifest and preset uses positional roles `primary`, `aux_left`, and `aux_right`: primary is placed on top, with the two auxiliaries below. Each preset maps those slots to a real camera stream or an explicit black view. `run_ur5.py` defaults to `--camera-preset wrist_left_right`, which maps wrist to primary and registers real left/right auxiliary views. For Berkeley models, select `--camera-preset berkeley_eef` for wrist primary, external aux-left, and black aux-right. For RoboMIND-single models, select `--camera-preset robomind_single` for overhead primary and two black auxiliaries. All presets preserve the same canvas shape, and the client does not discard a configured real-camera stream.

Start one server from a checkpoint carrying the matching manifest (or pass the manifest explicitly):

```Shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path <checkpoint>/model \
  --config-file <run>/config.yaml \
  --policy-config <run>/action_policy.yaml \
  --allow-dcp-checkpoint \
  --port 8000
```

`run.py`, `run_ur5.py`, and `run_x5.py` each declare their real robot capability. In particular, `run_x5.py` requires `robot: x5` in the handshake; it never reuses a DROID identity.

## Client

Clone [`RoboLab`](https://github.com/NVlabs/RoboLab):

```Shell
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
```

Build the Docker image:

```Shell
./docker/build_docker.sh latest
```

Launch the container:

```Shell
./docker/run_docker.sh latest
```

Run a task against the policy server. This opens a viewer window for real-time visualization of the simulation:

```Shell
python policies/cosmos3/run.py \
  --task BananaInBowlTask
```

To evaluate across multiple sub-environments in parallel in headless mode:

```Shell
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```
