"""Extracted from Desktop/3DMasks camera.py and segmentation.py.

Retains its format conversion, lazy model loading, BF16 inference and mask shape
checks. Copied locally so F2F does not depend on the other folder at runtime.
"""
from contextlib import nullcontext
from pathlib import Path
import warnings
import cv2
import numpy as np
from PIL import Image
from .progress import progress

def color_to_bgr(frame, formats):
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == formats.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Camera returned an invalid MJPEG frame.")
        return image
    if fmt == formats.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == formats.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    for name, conversion in (("YUYV", cv2.COLOR_YUV2BGR_YUY2),
                             ("UYVY", cv2.COLOR_YUV2BGR_UYVY)):
        if fmt == getattr(formats, name, None):
            return cv2.cvtColor(data.reshape(h, w, 2), conversion)
    for name, conversion in (("NV12", cv2.COLOR_YUV2BGR_NV12),
                             ("NV21", cv2.COLOR_YUV2BGR_NV21),
                             ("I420", cv2.COLOR_YUV2BGR_I420)):
        if fmt == getattr(formats, name, None):
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), conversion)
    raise RuntimeError(f"Unsupported color format: {fmt}. Try OpenCV capture.")


class Segmenter:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint
        self.processor = None

    def predict(self, bgr, prompt, threshold=0.5):
        if not prompt.strip():
            raise ValueError("Enter an item name, for example 'coffee mug'.")
        progress("SAM3", "Preparing GPU inference...")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3 requires a CUDA GPU here. Install CUDA-enabled PyTorch.")
        if self.processor is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"^Importing from timm\.models\.layers is deprecated, please import via timm\.layers$", category=FutureWarning)
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            if self.checkpoint and not Path(self.checkpoint).is_file():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
            progress("SAM3", "Loading model weights (local cache or Hugging Face download)...")
            model = build_sam3_image_model(
                checkpoint_path=self.checkpoint, load_from_HF=self.checkpoint is None,
                device="cuda", eval_mode=True, compile=False,
            )
            self.processor = Sam3Processor(model, device="cuda")
            progress("SAM3", "Model loaded.")
        self.processor.confidence_threshold = threshold
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if torch.cuda.is_bf16_supported() else nullcontext())
        with torch.inference_mode(), amp:
            progress("SAM3", "Encoding the captured RGB image...")
            state = self.processor.set_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            progress("SAM3", f"Extracting masks for {prompt.strip()!r}...")
            output = self.processor.set_text_prompt(prompt=prompt.strip(), state=state)
        masks = output["masks"].detach().cpu().numpy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.size == 0:
            masks = np.zeros((0, *bgr.shape[:2]), dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != bgr.shape[:2]:
            raise RuntimeError(f"Unexpected SAM mask dimensions: {masks.shape}")
        progress("SAM3", f"Finished: {len(masks)} matching instance(s).")
        return masks > 0.5, output["scores"].detach().float().cpu().numpy().reshape(-1)

