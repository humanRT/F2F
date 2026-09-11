"""Robust temporal fusion for stationary aligned depth bursts, in metres."""
import warnings
import numpy as np


def fuse_depth(frames):
    stack = np.asarray(frames,dtype=np.float32)
    if stack.ndim != 3 or len(stack) < 3:
        raise ValueError("Depth fusion requires at least three aligned frames.")
    n,h,w = stack.shape
    fused = np.zeros((h,w),np.float32)
    spread = np.full((h,w),np.nan,np.float32)
    count = np.zeros((h,w),np.uint16)
    minimum = max(3,int(np.ceil(.6*n)))
    # Tile rows to bound temporary allocations at higher camera resolutions.
    for row in range(0,h,48):
        tile = stack[:,row:row+48].copy()
        tile[~np.isfinite(tile) | (tile <= 0)] = np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="All-NaN slice encountered",category=RuntimeWarning)
            median = np.nanmedian(tile,axis=0)
            mad = 1.4826*np.nanmedian(np.abs(tile-median),axis=0)
        accepted = np.isfinite(tile) & (np.abs(tile-median)<=np.maximum(.002,3*mad))
        counts = accepted.sum(axis=0)
        mean = np.where(accepted,tile,0).sum(axis=0)/np.maximum(counts,1)
        std = np.sqrt(np.where(accepted,(tile-mean)**2,0).sum(axis=0)/np.maximum(counts-1,1))
        reliable = counts >= minimum
        fused[row:row+48] = np.where(reliable,mean,0)
        spread[row:row+48] = np.where(reliable,std,np.nan)
        count[row:row+48] = counts
    return fused,{"std_m":spread,"valid_count":count,"frame_count":np.array(n),
                  "minimum_count":np.array(minimum)}
