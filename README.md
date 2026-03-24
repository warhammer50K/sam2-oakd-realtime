<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Meta_Platforms_Inc._logo_%28cropped%29.svg/120px-Meta_Platforms_Inc._logo_%28cropped%29.svg.png" height="40" alt="Meta"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://www.luxonis.com/logo.svg" height="40" alt="Luxonis"/>
</p>

<h1 align="center">SAM2 + OAK-D Real-time Segmentation</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://docs.luxonis.com/"><img src="https://img.shields.io/badge/DepthAI-3.4-00B4D8?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQTEwIDEwIDAgMCAwIDIgMTJhMTAgMTAgMCAwIDAgMTAgMTAgMTAgMTAgMCAwIDAgMTAtMTBBMTAgMTAgMCAwIDAgMTIgMnoiLz48L3N2Zz4=&logoColor=white" alt="DepthAI"></a>
  <a href="https://github.com/facebookresearch/sam2"><img src="https://img.shields.io/badge/SAM_2.1-Meta_AI-blue?logo=meta&logoColor=white" alt="SAM2"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.133-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://ubuntu.com/"><img src="https://img.shields.io/badge/Ubuntu-22.04_LTS-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-RTX_4050-76B900?logo=nvidia&logoColor=white" alt="CUDA"></a>
</p>

<p align="center">
  Real-time object segmentation and tracking using <b>Meta's SAM 2.1</b> with a <b>Luxonis OAK-D Lite</b> depth camera, served through a FastAPI + WebSocket web interface.
</p>

---

## Features

| Mode | Description |
|------|-------------|
| **Image** | Live point-based segmentation — click on the camera feed to segment objects in real-time |
| **Video** | Record a clip → annotate the first frame → automatic multi-object tracking across all frames |
| **Streaming** | Click once → continuous real-time tracking on the live camera stream using SAM2 VideoPredictor |

- Multi-object tracking with color-coded masks and contour overlays
- FP16 inference for optimal GPU performance
- Live FPS counter and VRAM usage monitoring
- Supports SAM 2.1 Tiny / Small / Base+ / Large model variants
- Includes FPS benchmark script for performance evaluation

## Tested Environment

> **Note:** This project was developed and tested on the following specific setup. Other configurations may work but are not guaranteed.

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 22.04.5 LTS (Jammy) |
| **Kernel** | 6.8.0-90-generic |
| **Python** | 3.10.12 |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **Camera** | Luxonis OAK-D Lite |
| **PyTorch** | 2.10.0 |
| **torchvision** | 0.25.0 |
| **DepthAI** | 3.4.0 |
| **NumPy** | 2.2.6 |
| **OpenCV** | 4.13.0.92 |
| **FastAPI** | 0.133.1 |

## Prerequisites

- NVIDIA GPU with CUDA support
- [Luxonis OAK-D Lite](https://shop.luxonis.com/products/oak-d-lite-1) (or other OAK-D series camera)
- [DepthAI](https://docs.luxonis.com/) SDK installed with USB rules configured

## Installation

```bash
# Clone this repository (--recursive pulls the SAM 2 submodule automatically)
git clone --recursive https://github.com/warhammer50K/sam2-oakd-realtime.git
cd sam2-oakd-realtime

# Install SAM 2 in editable mode
cd sam2_repo && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt

# Download the SAM 2.1 Tiny checkpoint
cd sam2_repo/checkpoints
bash download_ckpts.sh
cd ../..
```

> **Already cloned without `--recursive`?** Run:
> ```bash
> git submodule update --init --recursive
> ```

## Usage

### Web Application

```bash
# Start the server (default: http://localhost:8000)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to `http://localhost:8000`. Select a mode (Image / Video / Streaming) and interact by clicking on the video feed.

### FPS Benchmark

```bash
# Benchmark with dummy images
python benchmark_fps.py --fp16

# Benchmark with OAK-D Lite camera
python benchmark_fps.py --fp16 --camera
```

## Project Structure

```
.
├── app.py              # Main FastAPI web application (Image/Video/Streaming modes)
├── benchmark_fps.py    # FPS benchmark script for SAM2 inference
├── requirements.txt    # Python dependencies
└── sam2_repo/          # Meta SAM 2 (git submodule → facebookresearch/sam2)
    └── checkpoints/    # Model weights (downloaded separately)
```

## How It Works

1. **Image Mode** — Each frame from the OAK-D camera is processed through SAM2's image predictor. Click points are used as prompts for instant segmentation.

2. **Video Mode** — Camera frames are recorded to disk, then SAM2's video predictor propagates annotations from the first frame across all recorded frames.

3. **Streaming Mode** — Extends SAM2's VideoPredictor for real-time use by dynamically injecting camera frames into the inference state and running per-frame tracking with memory management.

## Acknowledgements

- [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI (Apache 2.0 License)
- [DepthAI](https://github.com/luxonis/depthai) by Luxonis

## License

This project wraps Meta's SAM 2 (Apache 2.0). See [sam2_repo/LICENSE](sam2_repo/LICENSE) for the original license terms.
