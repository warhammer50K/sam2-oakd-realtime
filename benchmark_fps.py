"""
SAM2 FPS Benchmark Script
- RTX 4050 Laptop GPU 에서 SAM2.1 Tiny 모델의 FPS 측정
- OAK-D Lite 카메라 또는 더미 이미지로 테스트 가능
"""

import time
import numpy as np
import torch
import argparse
from contextlib import nullcontext

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── 경로 설정 ──
SAM2_REPO = "sam2_repo"
CHECKPOINT = f"{SAM2_REPO}/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# ── 벤치마크 해상도 (OAK-D Lite 기본 RGB: 1920x1080, 4K: 3840x2160) ──
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
    """이미지 인코더만 측정 (set_image)"""
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
    """predict (디코더) 만 측정 - 중앙 포인트 1개 프롬프트"""
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
    """set_image + predict 전체 파이프라인 측정"""
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
    """OAK-D Lite 카메라로 실시간 FPS 측정"""
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
        print(f"\n  OAK-D Lite 카메라 실시간 벤치마크 ({duration}초)...")

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
        print(f"  카메라 실시간 파이프라인: {fps:.1f} FPS ({frame_count} frames / {elapsed:.1f}s)")
        return fps


def main():
    parser = argparse.ArgumentParser(description="SAM2 FPS Benchmark")
    parser.add_argument("--camera", action="store_true", help="OAK-D Lite 카메라로 실시간 테스트")
    parser.add_argument("--fp16", action="store_true", help="FP16 (half precision) 사용")
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

    # 더미 이미지 벤치마크
    for w, h, name in RESOLUTIONS:
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        print(f"\n── {name} ({w}x{h}) ──")

        enc_fps = benchmark_image_encoder(predictor, image, args.fp16)
        print(f"  Image Encoder (set_image):  {enc_fps:.1f} FPS")

        dec_fps = benchmark_predict(predictor, image, args.fp16)
        print(f"  Decoder (predict):          {dec_fps:.1f} FPS")

        full_fps = benchmark_full_pipeline(predictor, image, args.fp16)
        print(f"  Full Pipeline:              {full_fps:.1f} FPS")

    # VRAM 사용량
    if args.device == "cuda":
        print(f"\n  Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # 카메라 벤치마크
    if args.camera:
        benchmark_with_camera(predictor)

    print("\n" + "=" * 60)
    print("벤치마크 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
