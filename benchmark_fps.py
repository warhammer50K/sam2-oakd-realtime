"""
SAM2 FPS Benchmark Script
- Measure SAM2.1 Tiny model FPS on RTX 4050 Laptop GPU
- Test with OAK-D Lite camera or dummy images
"""

import time
import numpy as np
import torch
import argparse
from contextlib import nullcontext

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── Path config ──
SAM2_REPO = "sam2_repo"
CHECKPOINT = f"{SAM2_REPO}/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# ── Benchmark resolutions (OAK-D Lite default RGB: 1920x1080, 4K: 3840x2160) ──
RESOLUTIONS = [
    (640, 480, "VGA"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p (OAK-D Lite default)"),
]


def build_predictor(device: str, use_fp16: bool):
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def benchmark_image_encoder(predictor, image, use_fp16, n_warmup=5, n_iter=50):
    """Benchmark image encoder only (set_image)"""
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()
    with ctx:
        for _ in range(n_warmup):
            predictor.set_image(image)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            predictor.set_image(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return n_iter / elapsed


def benchmark_predict(predictor, image, use_fp16, n_warmup=5, n_iter=50):
    """Benchmark decoder only (predict) - single center point prompt"""
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()
    with ctx:
        predictor.set_image(image)
        h, w = image.shape[:2]
        point = np.array([[w // 2, h // 2]])
        label = np.array([1])

        for _ in range(n_warmup):
            predictor.predict(point_coords=point, point_labels=label, multimask_output=False)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return n_iter / elapsed


def benchmark_full_pipeline(predictor, image, use_fp16, n_warmup=3, n_iter=30):
    """Benchmark full pipeline (set_image + predict)"""
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()
    h, w = image.shape[:2]
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])

    with ctx:
        for _ in range(n_warmup):
            predictor.set_image(image)
            predictor.predict(point_coords=point, point_labels=label, multimask_output=False)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            predictor.set_image(image)
            predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return n_iter / elapsed


def benchmark_with_camera(predictor, duration=10):
    """Measure real-time FPS with OAK-D Lite camera"""
    import depthai as dai

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(30)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    h, w = 480, 640
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        print(f"\n  OAK-D Lite camera real-time benchmark ({duration}s)...")

        # warmup
        for _ in range(5):
            frame = q.get().getCvFrame()
            predictor.set_image(frame)
            predictor.predict(point_coords=point, point_labels=label, multimask_output=False)

        frame_count = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            frame = q.get().getCvFrame()
            predictor.set_image(frame)
            predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
            frame_count += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fps = frame_count / elapsed
        print(f"  Camera real-time pipeline: {fps:.1f} FPS ({frame_count} frames / {elapsed:.1f}s)")
        return fps


def main():
    parser = argparse.ArgumentParser(description="SAM2 FPS Benchmark")
    parser.add_argument("--camera", action="store_true", help="Real-time test with OAK-D Lite camera")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (half precision)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    print("=" * 60)
    print("SAM2.1 Tiny - FPS Benchmark")
    print(f"  Device: {args.device}")
    print(f"  FP16: {args.fp16}")
    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    predictor = build_predictor(args.device, args.fp16)

    # Dummy image benchmark
    for w, h, name in RESOLUTIONS:
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        print(f"\n── {name} ({w}x{h}) ──")

        enc_fps = benchmark_image_encoder(predictor, image, args.fp16)
        print(f"  Image Encoder (set_image):  {enc_fps:.1f} FPS")

        dec_fps = benchmark_predict(predictor, image, args.fp16)
        print(f"  Decoder (predict):          {dec_fps:.1f} FPS")

        full_fps = benchmark_full_pipeline(predictor, image, args.fp16)
        print(f"  Full Pipeline:              {full_fps:.1f} FPS")

    # VRAM usage
    if args.device == "cuda":
        print(f"\n  Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Camera benchmark
    if args.camera:
        benchmark_with_camera(predictor)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
