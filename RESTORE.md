# Restoring this environment

This repo is code + training artifacts, not a frozen container image. The
underlying software stack (Isaac Sim, Isaac Lab) is large (tens of GB) and
lives permanently on NVIDIA's own registry — there's no need to personally
host a copy of it. This file is the recipe for getting an equivalent working
environment back from scratch.

## What was actually running

| Component | Version |
|---|---|
| Isaac Sim | `6.0.1` |
| Isaac Lab | `v3.0.0-beta2.patch1` |
| PyTorch | `2.10.0+cu128` |
| OS | Ubuntu 24.04 |
| GPU tested on | NVIDIA RTX 2070, 8GB VRAM |

ROS 2 is **not** installed in this container — the ROS 2 code in
`deployment/catbot_ws` runs on the Raspberry Pi 5 on the physical robot, not
in the Isaac Lab training environment. Nothing ROS-related needs to be built
here.

## Steps

Prerequisites on the host: Docker, the NVIDIA Container Toolkit, and a (free)
NVIDIA NGC account for `docker login nvcr.io` (needed to pull the Isaac Sim
base image).

1. Clone Isaac Lab at the pinned release tag:
   ```
   git clone --branch v3.0.0-beta2.patch1 https://github.com/isaac-sim/IsaacLab.git
   ```
   (Check `https://github.com/isaac-sim/IsaacLab/releases` first — a newer
   patch release may exist by the time you do this; there's no reason to
   deliberately stay pinned to this exact tag unless reproducing this exact
   setup matters more than being current.)

2. In `IsaacLab/docker/.env.base`, set:
   ```
   ISAACSIM_VERSION=6.0.1
   ```

3. Build and start the container:
   ```
   cd IsaacLab
   docker login nvcr.io
   ./docker/container.py start
   ```
   This pulls `nvcr.io/nvidia/isaac-sim:6.0.1`, layers Isaac Lab on top, and
   starts a fresh container. This is the slow step (multi-GB download).

4. Inside the new container, clone this repo and install the training
   project as an editable package:
   ```
   git clone https://github.com/Miguel-GCoca/ankybot.git /workspace/ankybot
   /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/ankybot/training/ankybot_v3/source/Ankybot_v3
   ```
   Point `training/ankybot_v3/source/Ankybot_v3`'s hard-coded USD asset path
   (see the training README) at `training/custom_assets/` in your clone, or
   recreate the original `/workspace/custom_assets/...` path.

5. Sanity-check the install:
   ```
   /workspace/isaaclab/isaaclab.sh -p training/ankybot_v3/scripts/list_envs.py
   ```
   `Ankybot-Base-v3`, `Ankybot-Rec1-v3`, etc. should be listed as registered
   tasks.

## Known caveat

Isaac Lab 3.0 was still in beta as of this snapshot (a "ground-up
architectural overhaul" per the project's own release notes — multi-backend
physics, pluggable renderers). If you're restoring this well after the fact,
the task/mdp code in `training/ankybot_v3/source` was written against
whichever exact beta build produced the checkpoints in `training/ankybot_v3/logs`
— a newer Isaac Lab release may have moved APIs out from under it. Diff the
release notes between this snapshot's version and whatever you're installing
before trusting a training resume; a smoke test (`--num_envs 1`, a few
iterations) is the fast way to check.
