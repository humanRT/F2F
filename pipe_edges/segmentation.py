"""Lazy SAM3 integration so geometry/demo modes do not need a GPU or weights."""
from pathlib import Path

import numpy as np
from PIL import Image


def select_instance(masks, scores, *, instance=None, seed=None):
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("Expected SAM3 masks shaped (N,1,H,W) or (N,H,W).")
    candidates = np.flatnonzero(masks.reshape(len(masks), -1).any(axis=1)) if len(masks) else []
    if seed is not None:
        x, y = map(int, seed)
        if not (0 <= x < masks.shape[2] and 0 <= y < masks.shape[1]):
            raise ValueError("Seed pixel is outside the image.")
        candidates = [i for i in candidates if masks[i, y, x]]
    if instance is not None:
        if instance not in candidates:
            raise ValueError("Requested instance is absent, empty, or does not contain the seed.")
        chosen = instance
    elif len(candidates) == 1:
        chosen = int(candidates[0])
    elif not len(candidates):
        raise ValueError("SAM3 found no matching pipe. Try another prompt or threshold.")
    else:
        raise ValueError(f"SAM3 found {len(candidates)} instances. Select --instance INDEX or --seed X Y; "
                         "inspect sam3_candidates.npz. Masks are never silently merged.")
    return masks[chosen].astype(bool), int(chosen), float(scores[chosen])


def extract_mask(image_path, output_dir, *, prompt="pipe", checkpoint=None,
                 threshold=.5, instance=None, seed=None, interactive=False, candidates_file=None):
    if not 0 <= threshold <= 1:
        raise ValueError("SAM3 confidence threshold must be between 0 and 1.")
    from .sam3_worker import predict
    import cv2
    with Image.open(image_path) as im:
        bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        if candidates_file is not None:
            with np.load(candidates_file,allow_pickle=False) as data:
                masks,scores=data["masks"].copy(),data["scores"].copy()
        else:
            masks, scores = predict(bgr, prompt, threshold, checkpoint)
    except ImportError as exc:
        raise RuntimeError("SAM3 is not installed in the selected environment.") from exc
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "sam3_candidates.npz", masks=masks, scores=scores)
    if masks.ndim==4 and masks.shape[1]==1: masks=masks[:,0]
    masks=np.asarray(masks,dtype=bool)
    if masks.ndim!=3 or masks.shape[1:]!=bgr.shape[:2] or len(masks)!=len(scores):
        raise ValueError("SAM3 candidate dimensions do not match the captured RGB.")
    available=[i for i,m in enumerate(masks) if m.any()]
    if seed is not None:
        x,y=seed
        if not (0<=x<masks.shape[2] and 0<=y<masks.shape[1]):
            raise ValueError("Seed pixel is outside the image.")
        available=[i for i in available if masks[i,y,x]]
    if instance is None and len(available)>1 and interactive:
        from .mask_picker import pick_mask
        from .progress import progress
        progress("SAM3",f"{len(available)} candidates: select your pipe in the OpenGL window.")
        instance=pick_mask(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),masks,scores,available)
        progress("SAM3",f"Selected instance {instance}.")
    mask, chosen, score = select_instance(masks, scores, instance=instance, seed=seed)
    Image.fromarray(mask.astype(np.uint8)*255).save(folder / "mask.png")
    return mask, {"source": "sam3", "prompt": prompt, "instance": chosen, "score": score}
