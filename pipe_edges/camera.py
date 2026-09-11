"""Orbbec RGB-D capture with a rectified color grid and optical-axis depth."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from .progress import progress


def rectification_maps(color_intrinsic, distortion, target_intrinsic, cv2):
    def matrix(i):
        return np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]], dtype=float)
    model = distortion.model.name
    if model == "NONE":
        coefficients = np.zeros(8)
    elif model in ("BROWN_CONRADY", "BROWN_CONRADY_K6"):
        coefficients = np.array([distortion.k1, distortion.k2, distortion.p1, distortion.p2,
                                 distortion.k3, distortion.k4, distortion.k5, distortion.k6])
    else:
        raise RuntimeError(f"Unsupported camera distortion model: {model}. Cannot safely label this capture rectified.")
    return cv2.initUndistortRectifyMap(matrix(color_intrinsic), coefficients, np.eye(3),
                                      matrix(target_intrinsic),
                                      (target_intrinsic.width, target_intrinsic.height), cv2.CV_32FC1)


def capture(folder, *, preview=True, timeout=20., warmup=15, burst_frames=45):
    """Capture and robustly fuse a stationary synchronized RGB/depth burst."""
    try:
        import cv2
        import pyorbbecsdk as ob
    except ImportError as exc:
        raise RuntimeError("Camera mode needs pyorbbecsdk (SDK v2) and opencv-python in SAM3.") from exc
    if timeout <= 0:
        raise ValueError("Camera timeout must be positive.")
    if not 3 <= burst_frames <= 120:
        raise ValueError("Burst length must be between 3 and 120 frames.")
    pipeline = None
    started = False
    viewer = None
    capture_requested = False
    last_update = time.monotonic()
    burst, burst_rgb = [], []
    reference_gray = features = None
    motion_resets = 0
    try:
        progress("Camera", "Searching for the Gemini camera...")
        log_dir = Path(folder).resolve().parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        ob.Context.set_logger_to_file(ob.OBLogLevel.ERROR, str(log_dir))
        ctx = ob.Context()
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("No Orbbec camera found. Connect the Gemini 305 by USB and close other camera apps.")
        if devices.get_count() != 1:
            raise RuntimeError("Multiple Orbbec cameras found. Connect only the intended camera for this capture.")
        device = devices.get_device_by_index(0)
        name = device.get_device_info().get_name()
        progress("Camera", f"Opening {name}")
        pipeline = ob.Pipeline(device)
        config = ob.Config()
        colors = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        depths = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        # Same default profile choice as the working 3DMasks Orbbec backend.
        color_profile = colors.get_default_video_stream_profile()
        depth_profile = depths.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
        align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        # SDK defaults to a rectified target. Explicitly require this and avoid
        # copied depth across holes at silhouette boundaries.
        align.set_config_value("TargetDistortion", 0)
        align.set_config_value("GapFillCopy", 0)
        align.set_config_value("MatchTargetRes", 1)
        progress("Camera", "Starting synchronized RGB/depth streams...")
        pipeline.start(config)
        started = True
        deadline = time.monotonic() + timeout
        count, maps = 0, None
        progress("Camera", f"Warming up: waiting for {warmup} usable frames...")
        while True:
            if viewer is not None:
                action = viewer.poll()
                if action == "cancel":
                    raise KeyboardInterrupt
                capture_requested |= action == "capture"
            frames = pipeline.wait_for_frames(200)
            if not frames:
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out waiting for RGB/depth. Check USB connection and close other camera apps.")
                continue
            aligned = align.process(frames)
            if aligned is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("SDK did not produce an aligned RGB/depth frameset.")
                continue
            frameset = aligned.as_frame_set()
            color, depth = frameset.get_color_frame(), frameset.get_depth_frame()
            if color is None or depth is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("Camera frames are missing color or depth.")
                continue
            if depth.get_format() != ob.OBFormat.Y16:
                raise RuntimeError("Camera did not provide Y16 depth.")
            depth_m = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
                depth.get_height(), depth.get_width()).astype(np.float32) * (depth.get_depth_scale() * .001)
            from .reused_3dmasks import color_to_bgr
            color_data = cv2.cvtColor(color_to_bgr(color, ob.OBFormat), cv2.COLOR_BGR2RGB)
            target_intrinsic = depth.get_stream_profile().as_video_stream_profile().get_intrinsic()
            if maps is None:
                cp = color.get_stream_profile().as_video_stream_profile()
                maps = rectification_maps(cp.get_intrinsic(), cp.get_distortion(), target_intrinsic, cv2)
            rgb = cv2.remap(color_data, *maps, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if rgb.shape[:2] != depth_m.shape:
                raise RuntimeError("Aligned depth and rectified color grids do not match.")
            if not np.any(depth_m > 0):
                if time.monotonic() > deadline:
                    raise RuntimeError("Camera depth is entirely invalid. Check distance and lighting.")
                continue
            deadline = time.monotonic() + timeout
            count += 1
            valid_percent = 100*np.count_nonzero(depth_m)/depth_m.size
            if preview:
                if viewer is None:
                    from .gl_preview import GLPreview
                    viewer = GLPreview(rgb.shape[1], rgb.shape[0])
                    progress("OpenGL", f"Live preview: {viewer.renderer}")
                state = f"Warmup {count}/{warmup}" if count < warmup else "Ready"
                if burst:
                    state = f"Hold camera and pipe still: burst {len(burst)}/{burst_frames}"
                viewer.draw(rgb, f"{state} | valid depth {valid_percent:.0f}%")
            if count == warmup:
                progress("Camera", "Ready. Press SPACE/ENTER to capture; Q/ESC to cancel." if preview else "Warmup complete; capturing.")
            now = time.monotonic()
            if now-last_update >= 2.:
                progress("Camera", f"{count} frames received; valid depth {valid_percent:.1f}%; " +
                         (f"burst {len(burst)}/{burst_frames}" if burst else
                          "waiting for capture" if count >= warmup and preview else
                          "ready" if count >= warmup else "warming up"))
                last_update = now
            if count < warmup or (preview and not capture_requested):
                continue
            gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            if not burst:
                reference_gray = gray
                features = cv2.goodFeaturesToTrack(gray,250,.01,12)
                if features is None or len(features)<20:
                    raise RuntimeError("Not enough image features to verify stationary capture. Improve lighting or texture.")
                progress("Capture", f"Hold camera and pipe still: collecting {burst_frames} frames...")
            else:
                tracked,status,_ = cv2.calcOpticalFlowPyrLK(reference_gray,gray,features,None)
                movement = (np.linalg.norm(tracked-features,axis=2).ravel()[status.ravel()>0]
                            if tracked is not None and status is not None else np.array([]))
                if len(movement)<20 or np.median(movement)>.6 or np.percentile(movement,90)>1.5:
                    burst.clear()
                    burst_rgb.clear()
                    motion_resets += 1
                    progress("Capture", "Motion detected; restarting the burst. Hold both camera and pipe still.")
                    if motion_resets>=5:
                        raise RuntimeError("Repeated motion during capture. Stabilize the camera and pipe, then retry.")
                    continue
            burst.append(depth_m.copy())
            burst_rgb.append(rgb.copy())
            if len(burst)%10==0:
                progress("Capture", f"Burst {len(burst)}/{burst_frames} frames")
            if len(burst)<burst_frames:
                continue
            from .depth_fusion import fuse_depth
            progress("Capture", "Rejecting depth outliers and computing temporal uncertainty...")
            depth_m,stats = fuse_depth(burst)
            rgb = burst_rgb[len(burst)//2]
            progress("Capture", "Saving RGB, depth and calibration...")
            folder = Path(folder)
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(folder / "rgb.png")
            np.save(folder / "depth.npy", depth_m)
            np.save(folder / "depth_single.npy", burst[len(burst)//2])
            np.savez_compressed(folder / "depth_stats.npz", **stats)
            intr = target_intrinsic
            info = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                    "cx": intr.cx, "cy": intr.cy, "depth_scale": 1., "aligned_to": "color",
                    "rectified": True, "camera": name, "depth_units": "metres",
                    "alignment": "Orbbec software D2C, TargetDistortion=0, GapFillCopy=0",
                    "burst_frames":burst_frames,"motion_restarts":motion_resets,
                    "fusion":"temporal median/MAD rejection, trimmed mean; 60% minimum support"}
            (folder / "intrinsics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            progress("Capture", f"Saved: {folder.resolve()}")
            return folder / "rgb.png", folder / "depth.npy", folder / "intrinsics.json"
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise RuntimeError(f"Camera capture failed: {exc}") from exc
    finally:
        try:
            if started:
                pipeline.stop()
                progress("Camera", "Streams closed.")
        finally:
            if viewer is not None:
                viewer.close()
