# UBRobot

UBRobot is an experimental robot navigation and manipulation stack for building embodied AI agents that can see, reason, plan, and act in the real world. The project combines LeRobot-style robot interfaces, RGB-D perception, vision-language reasoning, navigation policy services, teleoperation tools, and a Gradio chat UI for issuing high-level natural-language instructions.

The repository is a work in progress, but the current codebase is organized around two main capabilities:

- Mobile robot navigation from language instructions, using RealSense RGB-D observations, odometry, VLM grounding/reasoning, InternNav/InternVLA-style policy inference, LogoPlanner components, and MPC/PID control.
- Robot arm manipulation and teleoperation, especially for the AgileX Piper arm, with LeRobot-compatible observation/action APIs, CAN control through `piper-sdk`, camera integration, point-cloud perception, and recording/evaluation examples.

## Features

- LeRobot-compatible robot abstractions for Piper, SO-101 follower, LeKiwi base, and Unitree Go2 experiments.
- RGB-D camera utilities for Intel RealSense and aligned color/depth observations.
- Natural-language navigation loop in `Go2Manager`, including instruction handling, policy-server calls, trajectory following, and annotated visual feedback.
- Flask services for navigation policy inference and vision-language reasoning.
- Gradio chat interface for text or microphone input, live robot observation display, and command routing.
- Teleoperation examples for Piper and SO-101 robots, including keyboard, gamepad, and networked workflows.
- ROS dependency workspace for Piper ROS, RTAB-Map/odometry, and related hardware integration.
- Assets for robot URDFs, meshes, camera/arm configuration, sample media, and local model checkpoints.

## Repository Layout

```text
.
|-- assets/                 # URDFs, meshes, icons, sample media, configs, model assets
|-- docs/                   # Setup notes for robots, datasets, and sensors
|-- examples/               # Teleoperation, recording, evaluation, and model-service examples
|-- ros_depends_ws/         # ROS/Catkin workspace for hardware and odometry dependencies
|-- src/
|   |-- chat_ui/            # Gradio/FastAPI user interface
|   |-- service/
|   |   |-- planning/       # InternVLA/InternNav navigation policy HTTP server
|   |   `-- reasoning/      # VLM reasoning, grasp planning, GraspNet/PointNet utilities
|   `-- ubrobot/
|       |-- cameras/        # RealSense, odometry, and camera utilities
|       |-- robots/         # Robot drivers/adapters and high-level robot manager
|       `-- teleoperators/  # Keyboard/gamepad teleoperation helpers
|-- third_party/            # Vendored external code, including GraspNet API
|-- install.sh              # Native dependency setup notes/script
|-- pyproject.toml          # Python package metadata and dependencies
`-- requirements.txt        # Pinned Python dependency set
```

## Main Components

### Robot Interfaces

The robot layer in `src/ubrobot/robots` provides hardware adapters and higher-level control code:

- `piper/` implements a LeRobot-compatible Piper arm interface with joint observations, actions, optional gripper control, RealSense/OpenCV camera support, and Piper SDK integration.
- `lekiwi/` contains the mobile-base interface used by the navigation manager.
- `so101_follower/` contains SO-101 host/client/follower control code.
- `unitree_go2_robot.py` wraps basic Unitree Go2 sport-client actions.
- `ubrobot.py` contains `Go2Manager`, the current high-level agent loop for observations, language instructions, navigation planning, and base motion commands.

### Planning And Reasoning Services

The project separates heavyweight model inference into HTTP services:

- `src/service/planning/http_internvla_server.py` starts a Flask server for InternVLA/InternNav navigation policy inference on port `5801`.
- `src/service/reasoning/http_reasoning_server.py` starts a Flask server for VLM reasoning on port `5802`.
- `src/service/reasoning/` also includes grasp planning, GraspNet models, PointNet/KNN extensions, and dataset utilities.

### Chat UI

`src/chat_ui/app.py` starts a Gradio interface served through FastAPI. It accepts text or microphone input, sends commands through `ChatPipeline`, and displays navigation/manipulation observations. By default it runs on port `7863` with the local TLS certificate files under `assets/`.

## Installation

UBRobot is currently developed for Linux robot machines, especially Ubuntu 20.04 style environments with Python 3.10, CUDA-capable PyTorch, ROS, RealSense, and robot-specific SDKs.

Create and activate a Python environment:

```bash
conda create -y -n ubrobot python=3.10
conda activate ubrobot
```

Install FFmpeg in the environment:

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

Install PyTorch. For CUDA 12.8:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only environments:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

Install common Python dependencies and the project in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Some hardware paths also require native dependencies such as ROS, RealSense, Piper CAN setup, CycloneDDS, Unitree SDK2 Python, and TRAC-IK/KDL libraries. See `install.sh`, `ros_depends_ws/README.md`, `docs/install_realsense.md`, and the scripts under `ros_depends_ws/` for the current setup notes.

## Running

Start the navigation policy service:

```bash
bash start_policy_server.bash
```

Start the reasoning service:

```bash
bash start_reasoning_server.bash
```

Start the robot/chat UI stack on the robot machine:

```bash
bash ubrobot_startup.sh
```

The UI is served by `src/chat_ui/app.py` and defaults to:

```text
https://0.0.0.0:7863
```

The startup script also sources ROS, activates Piper CAN support, sets `CYCLONEDDS_HOME`, and launches the Gradio/FastAPI application.

## Examples

Useful entry points include:

- `examples/piper/teleoperate.py` for Piper arm teleoperation.
- `examples/piper/record.py` for recording Piper demonstrations.
- `examples/piper/evaluate.py` for evaluating a policy on Piper.
- `examples/so101_to_so101/` for networked SO-101 teleoperation and recording.
- `examples/lerobot_record.py` and `examples/lerobot_eval.py` for LeRobot-style dataset/policy workflows.
- `examples/internnav_demo.py` and `examples/http_internvla_client.py` for navigation-model experiments.

## Related Projects

- [InternNav](https://github.com/InternRobotics/InternNav): Open platform for generalized navigation foundation models.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL): Vision-language foundation model family.
- [LeRobot](https://github.com/huggingface/lerobot): Robot learning datasets, policies, and robot abstractions used as a design reference.
- [Cosmos](https://github.com/nvidia-cosmos/cosmos-reason1): Physical-reasoning vision-language models for embodied decision making.
- [GraspNet API](https://github.com/graspnet/graspnetAPI): Grasp representation, visualization, and evaluation utilities.

## Status

This repository is actively evolving. Some scripts contain machine-specific paths, local model checkpoint paths, and hardware-specific assumptions. Treat the project as a research/development workspace rather than a packaged production release.

## TODO

- Add a system architecture diagram.
- Add Docker or reproducible environment setup.
- Normalize machine-specific paths and configuration.
- Expand hardware setup documentation.
