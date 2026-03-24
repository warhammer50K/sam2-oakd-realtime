"""
OAK-D Lite + SAM2 Real-time Segmentation Web App
- Image Mode: Real-time point-based segmentation
- Video Mode: Record → Click → Automatic tracking playback
- Streaming Mode: Click once → Automatic tracking on live camera stream
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import depthai as dai
from PIL import Image as PILImage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── Config ──
SAM2_CHECKPOINT = "sam2_repo/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CAM_WIDTH, CAM_HEIGHT = 640, 480
MASK_ALPHA = 0.55

# Model variant mapping
_CFG_TO_VARIANT = {
    "sam2.1_hiera_t": "SAM2.1 Tiny",
    "sam2.1_hiera_s": "SAM2.1 Small",
    "sam2.1_hiera_b+": "SAM2.1 Base+",
    "sam2.1_hiera_l": "SAM2.1 Large",
}
_cfg_stem = os.path.splitext(os.path.basename(SAM2_CONFIG))[0]
MODEL_VARIANT = _CFG_TO_VARIANT.get(_cfg_stem, _cfg_stem)


def gpu_status_text() -> str:
    """Return GPU VRAM usage as a string"""
    if not torch.cuda.is_available():
        return "CPU"
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"VRAM {mem_used:.1f}/{mem_total:.0f}G"

# Per-object colors (BGR)
OBJ_COLORS = [
    (0, 120, 255),   # orange
    (255, 50, 50),    # blue
    (50, 220, 50),    # green
    (200, 50, 255),   # purple
    (0, 220, 220),    # yellow
    (255, 100, 180),  # pink
]


def overlay_masks_on_frame(frame, masks_dict, points_list=None):
    """Overlay masks and points onto a frame"""
    overlay = frame.copy()
    for obj_id, mask in masks_dict.items():
        color = OBJ_COLORS[obj_id % len(OBJ_COLORS)]
        color_layer = np.zeros_like(overlay)
        color_layer[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, MASK_ALPHA, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)

    if points_list:
        for obj_id, x, y, lbl in points_list:
            color = OBJ_COLORS[obj_id % len(OBJ_COLORS)]
            marker = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(overlay, (x, y), 6, marker, -1)
            cv2.circle(overlay, (x, y), 6, color, 2)

    return overlay


# ────────────────────────────────────────────
# Image Mode Engine
# ────────────────────────────────────────────
class ImageEngine:
    def __init__(self):
        self.predictor = None
        self.points = []
        self.lock = threading.Lock()
        self.current_frame = None
        self.current_overlay = None
        self.fps = 0.0
        self.running = False

    def start(self, cam_queue):
        print("[Image] Loading model...")
        model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
        self.predictor = SAM2ImagePredictor(model)
        self.cam_queue = cam_queue
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        frame_times = []
        while self.running:
            in_frame = self.cam_queue.tryGet()
            if in_frame is None:
                time.sleep(0.001)
                continue
            t0 = time.perf_counter()
            frame = in_frame.getCvFrame()
            self.current_frame = frame.copy()
            with self.lock:
                pts = list(self.points)
            overlay = frame.copy()
            if pts:
                coords = np.array([[p[0], p[1]] for p in pts])
                labels = np.array([p[2] for p in pts])
                with torch.autocast("cuda", dtype=torch.float16):
                    self.predictor.set_image(frame)
                    masks, scores, _ = self.predictor.predict(
                        point_coords=coords, point_labels=labels, multimask_output=False,
                    )
                masks_dict = {0: masks[0].astype(bool)}
                points_list = [(0, p[0], p[1], p[2]) for p in pts]
                overlay = overlay_masks_on_frame(frame, masks_dict, points_list)
            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            self.fps = len(frame_times) / sum(frame_times)
            cv2.putText(overlay, f"{self.fps:.1f} FPS | {gpu_status_text()} | {MODEL_VARIANT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            self.current_overlay = overlay

    def add_point(self, x, y, label=1):
        with self.lock:
            self.points.append((x, y, label))

    def clear_points(self):
        with self.lock:
            self.points.clear()

    def undo_point(self):
        with self.lock:
            if self.points:
                self.points.pop()

    def get_jpeg(self, quality=80):
        frame = self.current_overlay if self.current_overlay is not None else self.current_frame
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()


# ────────────────────────────────────────────
# Video Mode Engine
# ────────────────────────────────────────────
class VideoEngine:
    def __init__(self):
        self.predictor = None
        self.state = "idle"
        self.frames = []
        self.tracked_frames = []
        self.annotations = []
        self.next_obj_id = 1
        self.record_fps = 30
        self.playback_idx = 0
        self.lock = threading.Lock()
        self.tmp_dir = None

    def start(self, predictor):
        self.predictor = predictor
        print("[Video] Ready")

    def stop(self):
        self._cleanup_tmp()

    def _cleanup_tmp(self):
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self.tmp_dir = None

    def record(self, cam_queue, duration=5):
        self.state = "recording"
        self.frames.clear()
        self.tracked_frames.clear()
        self.annotations.clear()
        self.next_obj_id = 1
        self.playback_idx = 0
        print(f"[Video] Recording started ({duration}s)...")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            in_frame = cam_queue.tryGet()
            if in_frame is None:
                time.sleep(0.001)
                continue
            self.frames.append(in_frame.getCvFrame())
        actual_fps = len(self.frames) / (time.perf_counter() - t0)
        self.record_fps = actual_fps
        print(f"[Video] Recording done: {len(self.frames)} frames ({actual_fps:.1f} fps)")
        self._cleanup_tmp()
        self.tmp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        for i, frame in enumerate(self.frames):
            cv2.imwrite(os.path.join(self.tmp_dir, f"{i:05d}.jpg"), frame)
        self.state = "ready"
        return len(self.frames)

    def add_annotation(self, obj_id, x, y, label=1):
        with self.lock:
            self.annotations.append((obj_id, x, y, label))

    def get_next_obj_id(self):
        oid = self.next_obj_id
        self.next_obj_id += 1
        return oid

    def track(self):
        if not self.frames or not self.annotations:
            return
        self.state = "tracking"
        print(f"[Video] Tracking started: {len(self.annotations)} annotations, {len(self.frames)} frames")
        with torch.autocast("cuda", dtype=torch.float16):
            inference_state = self.predictor.init_state(video_path=self.tmp_dir)
            obj_points = {}
            for obj_id, x, y, label in self.annotations:
                if obj_id not in obj_points:
                    obj_points[obj_id] = {"points": [], "labels": []}
                obj_points[obj_id]["points"].append([x, y])
                obj_points[obj_id]["labels"].append(label)
            for obj_id, data in obj_points.items():
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state, frame_idx=0, obj_id=obj_id,
                    points=np.array(data["points"], dtype=np.float32),
                    labels=np.array(data["labels"], dtype=np.int32),
                )
            self.tracked_frames.clear()
            video_segments = {}
            for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(inference_state):
                masks_dict = {}
                for i, obj_id in enumerate(obj_ids):
                    mask = (masks[i] > 0.0).squeeze().cpu().numpy()
                    masks_dict[obj_id] = mask
                video_segments[frame_idx] = masks_dict
            points_on_0 = [(oid, x, y, l) for oid, x, y, l in self.annotations]
            for i, frame in enumerate(self.frames):
                masks_dict = video_segments.get(i, {})
                pts = points_on_0 if i == 0 else None
                overlay = overlay_masks_on_frame(frame, masks_dict, pts)
                cv2.putText(overlay, f"Frame {i}/{len(self.frames)-1} | {gpu_status_text()} | {MODEL_VARIANT}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(overlay, f"Tracking {len(obj_points)} object(s)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                self.tracked_frames.append(overlay)
            self.predictor.reset_state(inference_state)
        self.playback_idx = 0
        self.state = "done"
        print("[Video] Tracking done")

    def get_frame_jpeg(self, quality=80):
        frame = None
        if self.state == "ready" and self.frames:
            frame = self.frames[0].copy()
            with self.lock:
                pts = list(self.annotations)
            if pts:
                frame = overlay_masks_on_frame(frame, {}, pts)
            cv2.putText(frame, "Click to annotate, then Track", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif self.state == "done" and self.tracked_frames:
            frame = self.tracked_frames[self.playback_idx]
            self.playback_idx = (self.playback_idx + 1) % len(self.tracked_frames)
        elif self.state == "recording" and self.frames:
            frame = self.frames[-1].copy()
            cv2.putText(frame, f"Recording... {len(self.frames)} frames", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.state == "tracking":
            frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Tracking in progress...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def reset(self):
        with self.lock:
            self.state = "idle"
            self.frames.clear()
            self.tracked_frames.clear()
            self.annotations.clear()
            self.next_obj_id = 1
            self.playback_idx = 0
            self._cleanup_tmp()


# ────────────────────────────────────────────
# Streaming Mode Engine (real-time video tracking)
# ────────────────────────────────────────────
class StreamingEngine:
    """
    Real-time streaming tracker built on SAM2 VideoPredictor.
    - Builds inference_state from initial seed frames
    - Objects are designated by clicking on the first frame
    - Subsequent camera frames are dynamically injected into inference_state for tracking
    """

    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self):
        self.predictor = None
        self.inference_state = None
        self.state = "idle"  # idle, preview, annotating, tracking
        self.lock = threading.Lock()
        self.annotations = []  # [(obj_id, x, y, label)]
        self.next_obj_id = 1
        self.current_overlay = None
        self.fps = 0.0
        self.frame_idx = 0
        self.running = False
        self.first_frame = None  # First frame (for annotation)

    def start(self, predictor):
        self.predictor = predictor
        print("[Streaming] Ready")

    def stop(self):
        self.running = False

    def _preprocess_frame(self, frame_bgr):
        """BGR OpenCV frame -> SAM2 preprocessed tensor"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(frame_rgb)
        img_resized = img_pil.resize(
            (self.predictor.image_size, self.predictor.image_size)
        )
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        img_tensor = (img_tensor - self.IMG_MEAN) / self.IMG_STD
        return img_tensor

    def _build_initial_state(self, seed_frames):
        """Manually build inference_state from seed frames"""
        device = self.predictor.device

        # Preprocess frames
        images = {}
        for i, frame in enumerate(seed_frames):
            img_tensor = self._preprocess_frame(frame).to(device)
            images[i] = img_tensor

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(seed_frames)
        inference_state["offload_video_to_cpu"] = False
        inference_state["offload_state_to_cpu"] = False
        inference_state["video_height"] = seed_frames[0].shape[0]
        inference_state["video_width"] = seed_frames[0].shape[1]
        inference_state["device"] = device
        inference_state["storage_device"] = device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        # Warm up backbone on the first frame
        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    def _add_frame_to_state(self, frame_bgr):
        """Dynamically add a new camera frame to inference_state"""
        img_tensor = self._preprocess_frame(frame_bgr)
        if not self.inference_state["offload_video_to_cpu"]:
            img_tensor = img_tensor.to(self.inference_state["device"])
        idx = self.inference_state["num_frames"]
        self.inference_state["images"][idx] = img_tensor
        self.inference_state["num_frames"] = idx + 1
        return idx

    def _track_frame(self, frame_idx):
        """Run tracking on a single frame and return masks"""
        batch_size = self.predictor._get_obj_num(self.inference_state)
        if batch_size == 0:
            return {}

        masks_dict = {}
        for obj_idx in range(batch_size):
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            current_out, pred_masks = self.predictor._run_single_frame_inference(
                inference_state=self.inference_state,
                output_dict=obj_output_dict,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            self.inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                "reverse": False
            }

            obj_id = self.predictor._obj_idx_to_id(self.inference_state, obj_idx)
            _, video_res_masks = self.predictor._get_orig_video_res_output(
                self.inference_state, pred_masks
            )
            mask = (video_res_masks[0] > 0.0).squeeze().cpu().numpy()
            masks_dict[obj_id] = mask

        return masks_dict

    def begin_preview(self, cam_queue):
        """Start preview - wait for first frame capture"""
        self.state = "preview"
        self.annotations.clear()
        self.next_obj_id = 1
        self.inference_state = None
        self.first_frame = None
        self.current_overlay = None

        # Capture first frame
        print("[Streaming] Capturing preview frame...")
        while True:
            in_frame = cam_queue.tryGet()
            if in_frame is not None:
                self.first_frame = in_frame.getCvFrame()
                break
            time.sleep(0.001)

        # Build state from seed frame (single frame)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.inference_state = self._build_initial_state([self.first_frame])

        self.state = "annotating"
        print("[Streaming] Waiting for annotations (click to annotate)")

    def add_annotation(self, obj_id, x, y, label=1):
        with self.lock:
            self.annotations.append((obj_id, x, y, label))

    def get_next_obj_id(self):
        oid = self.next_obj_id
        self.next_obj_id += 1
        return oid

    def start_tracking(self, cam_queue):
        """Start real-time tracking after annotations are confirmed"""
        if not self.annotations or self.inference_state is None:
            return

        self.state = "tracking"
        self.running = True

        # Add annotations to frame 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            obj_points = {}
            for obj_id, x, y, label in self.annotations:
                if obj_id not in obj_points:
                    obj_points[obj_id] = {"points": [], "labels": []}
                obj_points[obj_id]["points"].append([x, y])
                obj_points[obj_id]["labels"].append(label)

            for obj_id, data in obj_points.items():
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array(data["points"], dtype=np.float32),
                    labels=np.array(data["labels"], dtype=np.int32),
                )

            # Preflight: run memory encoder
            self.predictor.propagate_in_video_preflight(self.inference_state)

        self.frame_idx = 0
        self._thread = threading.Thread(
            target=self._tracking_loop, args=(cam_queue,), daemon=True
        )
        self._thread.start()
        print(f"[Streaming] Real-time tracking started ({len(obj_points)} objects)")

    def _tracking_loop(self, cam_queue):
        """Main loop for tracking live camera frames"""
        frame_times = []
        MAX_MEMORY_FRAMES = 60

        print("[Streaming] Tracking loop started")
        while self.running:
            in_frame = cam_queue.tryGet()
            if in_frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            frame = in_frame.getCvFrame()

            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    new_idx = self._add_frame_to_state(frame)
                    masks_dict = self._track_frame(new_idx)

                    if new_idx > MAX_MEMORY_FRAMES:
                        old_idx = new_idx - MAX_MEMORY_FRAMES
                        self.inference_state["images"].pop(old_idx, None)
                        for obj_idx in range(self.predictor._get_obj_num(self.inference_state)):
                            obj_out = self.inference_state["output_dict_per_obj"][obj_idx]
                            obj_out["non_cond_frame_outputs"].pop(old_idx, None)
            except Exception as e:
                print(f"[Streaming] Tracking error: {e}")
                masks_dict = {}

            overlay = overlay_masks_on_frame(frame, masks_dict)

            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            self.fps = len(frame_times) / sum(frame_times)

            n_objs = self.predictor._get_obj_num(self.inference_state) if self.inference_state else 0
            cv2.putText(overlay, f"{self.fps:.1f} FPS | {gpu_status_text()} | {MODEL_VARIANT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(overlay, f"Tracking {n_objs} obj | Frame #{new_idx}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            self.current_overlay = overlay
            self.frame_idx = new_idx

    def get_jpeg(self, quality=80):
        if self.state == "annotating" and self.first_frame is not None:
            frame = self.first_frame.copy()
            with self.lock:
                pts = list(self.annotations)
            if pts:
                frame = overlay_masks_on_frame(frame, {}, pts)
            cv2.putText(frame, "Click object(s), then Start", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()

        if self.state == "tracking" and self.current_overlay is not None:
            _, buf = cv2.imencode(".jpg", self.current_overlay, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes()

        return None

    def reset(self):
        self.running = False
        time.sleep(0.05)
        with self.lock:
            if self.inference_state is not None:
                # Release GPU memory
                self.inference_state["images"].clear()
                self.inference_state["cached_features"].clear()
                for obj_idx in list(self.inference_state["output_dict_per_obj"].keys()):
                    self.inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"].clear()
                    self.inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].clear()
                self.inference_state = None
            self.state = "idle"
            self.annotations.clear()
            self.next_obj_id = 1
            self.first_frame = None
            self.current_overlay = None
            self.frame_idx = 0
            torch.cuda.empty_cache()


# ────────────────────────────────────────────
# Camera Manager
# ────────────────────────────────────────────
class CameraManager:
    def __init__(self):
        self.pipeline = None
        self.queue = None

    def start(self):
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera)
        cam.build(dai.CameraBoardSocket.CAM_A)
        out = cam.requestOutput((CAM_WIDTH, CAM_HEIGHT), dai.ImgFrame.Type.BGR888p)
        self.queue = out.createOutputQueue()
        self.pipeline.start()
        print("[Camera] OAK-D Lite connected")

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


# ── Global instances ──
camera = CameraManager()
image_engine = ImageEngine()
video_engine = VideoEngine()
streaming_engine = StreamingEngine()
current_mode = {"mode": "image"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    camera.start()
    image_engine.start(camera.queue)
    # Video and Streaming share the same predictor
    print("[Video/Streaming] Loading model...")
    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
    video_engine.start(video_predictor)
    streaming_engine.start(video_predictor)
    yield
    image_engine.stop()
    video_engine.stop()
    streaming_engine.stop()
    camera.stop()


app = FastAPI(lifespan=lifespan)


# ────────────────────────────────────────────
# HTML
# ────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAM2 Segmentation</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #eee;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 16px;
  }
  h1 { font-size: 1.3rem; margin-bottom: 10px; color: #e94560; }
  .mode-tabs {
    display: flex; gap: 0; margin-bottom: 12px; border-radius: 8px; overflow: hidden;
  }
  .mode-tab {
    padding: 10px 28px; border: none; font-size: 0.9rem; cursor: pointer;
    font-weight: 600; transition: background 0.2s;
    background: #16213e; color: #888;
  }
  .mode-tab.active { background: #e94560; color: #fff; }
  .toolbar {
    display: none; gap: 8px; margin-bottom: 10px; align-items: center;
    flex-wrap: wrap; justify-content: center;
  }
  .toolbar.active { display: flex; }
  button {
    padding: 7px 16px; border: none; border-radius: 6px;
    font-size: 0.85rem; cursor: pointer; font-weight: 600;
    transition: transform 0.1s, opacity 0.2s;
  }
  button:active { transform: scale(0.95); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-action { background: #0f3460; color: #fff; }
  .btn-danger { background: #e94560; color: #fff; }
  .btn-secondary { background: #555; color: #fff; }
  .btn-success { background: #1b6b2e; color: #fff; }
  .btn-obj { background: #333; color: #fff; min-width: 100px; }
  .btn-stream { background: #b8860b; color: #fff; }
  .container {
    position: relative; display: inline-block;
    border: 2px solid #333; border-radius: 8px; overflow: hidden;
    cursor: crosshair;
  }
  #canvas { display: block; }
  .status-bar {
    display: flex; gap: 12px; margin-top: 8px; font-size: 0.8rem; color: #888;
    align-items: center;
  }
  .status { padding: 3px 10px; border-radius: 10px; background: #0f3460; font-size: 0.8rem; }
  .status.connected { background: #1b6b2e; }
  .info {
    display: none; background: #16213e; padding: 8px 16px; border-radius: 6px;
    font-size: 0.8rem; color: #777; margin-top: 10px;
    max-width: 640px; text-align: center; line-height: 1.5;
  }
  .info.active { display: block; }
</style>
</head>
<body>
  <h1>SAM2 Real-time Segmentation</h1>

  <div class="mode-tabs">
    <button class="mode-tab active" data-mode="image" onclick="switchMode('image')">Image</button>
    <button class="mode-tab" data-mode="video" onclick="switchMode('video')">Video</button>
    <button class="mode-tab" data-mode="streaming" onclick="switchMode('streaming')">Streaming</button>
  </div>

  <!-- Image Mode Toolbar -->
  <div class="toolbar active" id="toolbar-image">
    <button class="btn-obj" id="modeBtn" onclick="toggleFgBg()">Foreground</button>
    <button class="btn-secondary" onclick="send('undo')">Undo (Z)</button>
    <button class="btn-danger" onclick="send('clear')">Clear (C)</button>
  </div>

  <!-- Video Mode Toolbar -->
  <div class="toolbar" id="toolbar-video">
    <button class="btn-action" id="recordBtn" onclick="videoAction('record')">Record 5s</button>
    <button class="btn-obj" id="objBtnV" onclick="send('video_new_object')" disabled>New Object</button>
    <button class="btn-secondary" id="undoVBtn" onclick="videoAction('undo')" disabled>Undo</button>
    <button class="btn-success" id="trackBtn" onclick="videoAction('track')" disabled>Track</button>
    <button class="btn-danger" onclick="videoAction('reset')">Reset</button>
  </div>

  <!-- Streaming Mode Toolbar -->
  <div class="toolbar" id="toolbar-streaming">
    <button class="btn-stream" id="captureBtn" onclick="send('stream_capture')">Capture Frame</button>
    <button class="btn-obj" id="objBtnS" onclick="send('stream_new_object')" disabled>New Object</button>
    <button class="btn-secondary" id="undoSBtn" onclick="send('stream_undo')" disabled>Undo</button>
    <button class="btn-success" id="startTrackBtn" onclick="send('stream_start')" disabled>Start Tracking</button>
    <button class="btn-danger" onclick="send('stream_reset')">Reset</button>
  </div>

  <div class="container">
    <canvas id="canvas" width="640" height="480"></canvas>
  </div>

  <div class="status-bar">
    <span class="status" id="status">Connecting...</span>
    <span id="modeInfo"></span>
    <span id="objInfo"></span>
  </div>

  <div class="info active" id="info-image">
    Left Click: foreground | Right Click: background | Z: undo | C: clear
  </div>
  <div class="info" id="info-video">
    1. Record → 2. Click objects on first frame → 3. Track → Playback<br>
    "New Object" for multi-object tracking
  </div>
  <div class="info" id="info-streaming">
    1. Capture Frame → 2. Click object(s) to track → 3. Start Tracking → Real-time tracking<br>
    Once tracking starts, the object is followed automatically across live camera frames
  </div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
let ws, mode = 'image', fgMode = 1, videoState = 'idle', streamState = 'idle', currentObjId = 1;

function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { statusEl.textContent = 'Connected'; statusEl.classList.add('connected'); };
  ws.onmessage = (e) => {
    if (typeof e.data === 'string') {
      const msg = JSON.parse(e.data);
      if (msg.type === 'video_state') { videoState = msg.state; updateVideoUI(); }
      if (msg.type === 'stream_state') { streamState = msg.state; updateStreamUI(); }
      if (msg.type === 'obj_id') {
        currentObjId = msg.obj_id;
        document.getElementById('objInfo').textContent = 'Object #' + currentObjId;
      }
      return;
    }
    const blob = new Blob([e.data], {type: 'image/jpeg'});
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, 0, 0); URL.revokeObjectURL(url); };
    img.src = url;
  };
  ws.onclose = () => {
    statusEl.textContent = 'Disconnected'; statusEl.classList.remove('connected');
    setTimeout(connect, 1000);
  };
}

function send(action, data) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({action, ...data}));
}

function switchMode(m) {
  mode = m;
  send('switch_mode', {mode: m});
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === m));
  ['image','video','streaming'].forEach(id => {
    document.getElementById('toolbar-' + id).classList.toggle('active', id === m);
    document.getElementById('info-' + id).classList.toggle('active', id === m);
  });
  document.getElementById('modeInfo').textContent = m.charAt(0).toUpperCase() + m.slice(1) + ' Mode';
}

function toggleFgBg() {
  fgMode = fgMode === 1 ? 0 : 1;
  const btn = document.getElementById('modeBtn');
  btn.textContent = fgMode === 1 ? 'Foreground' : 'Background';
  btn.style.background = fgMode === 1 ? '#333' : '#8b0000';
}

function videoAction(a) { send('video_' + a); }

function updateVideoUI() {
  document.getElementById('recordBtn').disabled = videoState === 'recording' || videoState === 'tracking';
  document.getElementById('objBtnV').disabled = videoState !== 'ready';
  document.getElementById('undoVBtn').disabled = videoState !== 'ready';
  document.getElementById('trackBtn').disabled = videoState !== 'ready';
  document.getElementById('recordBtn').textContent = videoState === 'recording' ? 'Recording...' : 'Record 5s';
}

function updateStreamUI() {
  const cap = document.getElementById('captureBtn');
  const obj = document.getElementById('objBtnS');
  const undo = document.getElementById('undoSBtn');
  const start = document.getElementById('startTrackBtn');
  cap.disabled = streamState !== 'idle';
  obj.disabled = streamState !== 'annotating';
  undo.disabled = streamState !== 'annotating';
  start.disabled = streamState !== 'annotating';
  start.textContent = streamState === 'tracking' ? 'Tracking...' : 'Start Tracking';
  cap.textContent = streamState === 'preview' ? 'Capturing...' : 'Capture Frame';
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
  if (mode === 'image') send('point', {x, y, label: fgMode});
  else if (mode === 'video' && videoState === 'ready')
    send('video_point', {x, y, label: fgMode, obj_id: currentObjId});
  else if (mode === 'streaming' && streamState === 'annotating')
    send('stream_point', {x, y, label: fgMode, obj_id: currentObjId});
});

canvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
  if (mode === 'image') send('point', {x, y, label: 0});
  else if (mode === 'video' && videoState === 'ready')
    send('video_point', {x, y, label: 0, obj_id: currentObjId});
  else if (mode === 'streaming' && streamState === 'annotating')
    send('stream_point', {x, y, label: 0, obj_id: currentObjId});
});

document.addEventListener('keydown', (e) => {
  if (mode === 'image') {
    if (e.key === 'z' || e.key === 'Z') send('undo');
    if (e.key === 'c' || e.key === 'C') send('clear');
  } else if (mode === 'video') {
    if (e.key === 'z' || e.key === 'Z') send('video_undo');
  } else if (mode === 'streaming') {
    if (e.key === 'z' || e.key === 'Z') send('stream_undo');
  }
});

connect();
switchMode('image');
</script>
</body>
</html>"""


@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    async def send_state(stype, state):
        await ws.send_text(json.dumps({"type": stype, "state": state}))

    async def send_obj_id(oid):
        await ws.send_text(json.dumps({"type": "obj_id", "obj_id": oid}))

    try:
        async def send_frames():
            while True:
                m = current_mode["mode"]
                jpg = None
                if m == "image":
                    jpg = image_engine.get_jpeg()
                elif m == "video":
                    jpg = video_engine.get_frame_jpeg()
                elif m == "streaming":
                    jpg = streaming_engine.get_jpeg()
                if jpg:
                    await ws.send_bytes(jpg)
                if m == "video" and video_engine.state == "done":
                    await asyncio.sleep(1.0 / max(video_engine.record_fps, 15))
                else:
                    await asyncio.sleep(0.033)

        send_task = asyncio.create_task(send_frames())

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            # ── Mode switch ──
            if action == "switch_mode":
                current_mode["mode"] = msg["mode"]
                if msg["mode"] == "video":
                    await send_state("video_state", video_engine.state)
                elif msg["mode"] == "streaming":
                    await send_state("stream_state", streaming_engine.state)

            # ── Image Mode ──
            elif action == "point":
                image_engine.add_point(msg["x"], msg["y"], msg.get("label", 1))
            elif action == "clear":
                image_engine.clear_points()
            elif action == "undo":
                image_engine.undo_point()

            # ── Video Mode ──
            elif action == "video_record":
                async def do_record():
                    await send_state("video_state", "recording")
                    loop = asyncio.get_event_loop()
                    image_engine.running = False
                    await loop.run_in_executor(None, video_engine.record, camera.queue, 5)
                    image_engine.running = True
                    image_engine._thread = threading.Thread(target=image_engine._loop, daemon=True)
                    image_engine._thread.start()
                    await send_state("video_state", "ready")
                    await send_obj_id(1)
                asyncio.create_task(do_record())
            elif action == "video_point":
                video_engine.add_annotation(msg.get("obj_id", 1), msg["x"], msg["y"], msg.get("label", 1))
            elif action == "video_new_object":
                await send_obj_id(video_engine.get_next_obj_id())
            elif action == "video_undo":
                with video_engine.lock:
                    if video_engine.annotations:
                        video_engine.annotations.pop()
            elif action == "video_track":
                async def do_track():
                    await send_state("video_state", "tracking")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, video_engine.track)
                    await send_state("video_state", "done")
                asyncio.create_task(do_track())
            elif action == "video_reset":
                video_engine.reset()
                await send_state("video_state", "idle")

            # ── Streaming Mode ──
            elif action == "stream_capture":
                async def do_capture():
                    await send_state("stream_state", "preview")
                    loop = asyncio.get_event_loop()
                    # Stop image engine (prevent camera queue contention)
                    image_engine.running = False
                    time.sleep(0.05)
                    # Flush queued frames
                    while camera.queue.tryGet() is not None:
                        pass
                    await loop.run_in_executor(None, streaming_engine.begin_preview, camera.queue)
                    await send_state("stream_state", "annotating")
                    await send_obj_id(1)
                asyncio.create_task(do_capture())
            elif action == "stream_point":
                streaming_engine.add_annotation(msg.get("obj_id", 1), msg["x"], msg["y"], msg.get("label", 1))
            elif action == "stream_new_object":
                await send_obj_id(streaming_engine.get_next_obj_id())
            elif action == "stream_undo":
                with streaming_engine.lock:
                    if streaming_engine.annotations:
                        streaming_engine.annotations.pop()
            elif action == "stream_start":
                async def do_stream():
                    await send_state("stream_state", "tracking")
                    loop = asyncio.get_event_loop()
                    # Ensure image engine is stopped
                    image_engine.running = False
                    await loop.run_in_executor(None, streaming_engine.start_tracking, camera.queue)
                asyncio.create_task(do_stream())
            elif action == "stream_reset":
                streaming_engine.reset()
                # Restart image engine
                if not image_engine.running:
                    image_engine.running = True
                    image_engine._thread = threading.Thread(target=image_engine._loop, daemon=True)
                    image_engine._thread.start()
                await send_state("stream_state", "idle")

    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
