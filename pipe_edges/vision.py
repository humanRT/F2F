"""Mask -> ordered side boundaries -> explicitly sampled, calibrated 3D points."""
from dataclasses import dataclass
import json

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        if config.get("aligned_to") != "color" or config.get("rectified") is not True:
            raise ValueError("Provide rectified RGB and depth aligned to color; set aligned_to='color', rectified=true.")
        intr = cls(**{key: config[key] for key in cls.__dataclass_fields__})
        values = np.array([intr.fx, intr.fy, intr.cx, intr.cy, intr.depth_scale], float)
        if not np.isfinite(values).all() or min(intr.fx, intr.fy, intr.depth_scale) <= 0:
            raise ValueError("Invalid intrinsics or depth_scale.")
        if intr.width <= 0 or intr.height <= 0:
            raise ValueError("Invalid image dimensions.")
        return intr

    def deproject(self, uv, z):
        uv, z = np.asarray(uv, float), np.asarray(z, float)
        return np.column_stack(((uv[:, 0]-self.cx)*z/self.fx,
                                (uv[:, 1]-self.cy)*z/self.fy, z))


def _resample(curve, count):
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if distance[-1] <= 0:
        raise ValueError("Mask path has zero length.")
    return np.column_stack([np.interp(np.linspace(0, distance[-1], count), distance, curve[:, j])
                            for j in range(2)])


def mask_path(mask, samples=240):
    """Longest skeleton path; reject substantial branches and closed silhouettes."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or mask.sum() < 25:
        raise ValueError("Need a nonempty 2D pipe mask with at least 25 pixels.")
    labels, count = ndimage.label(mask, np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    if count > 1 and np.count_nonzero(sizes > max(10, .01*sizes[largest])) > 1:
        raise ValueError("Mask contains disconnected pipe regions. Process each visible region separately.")
    mask = labels == largest
    skeleton = skeletonize(mask)
    yx = np.argwhere(skeleton)
    if len(yx) < 8:
        raise ValueError("Pipe mask is too short or small to trace.")
    lookup = {tuple(p): i for i, p in enumerate(yx)}
    rows, cols, weights = [], [], []
    for i, (y, x) in enumerate(yx):
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            j = lookup.get((y+dy, x+dx))
            if j is not None:
                rows.append(i)
                cols.append(j)
                weights.append(np.hypot(dx, dy))
    graph = csr_matrix((weights, (rows, cols)), shape=(len(yx), len(yx)))
    endpoints = np.flatnonzero(np.diff(graph.indptr) == 1)
    if len(endpoints) < 2:
        raise ValueError("No open pipe path found. Crop to a non-overlapping open section.")
    distances = dijkstra(graph, directed=False, indices=int(endpoints[0]))
    start = int(endpoints[np.argmax(distances[endpoints])])
    distances, predecessors = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    end = int(endpoints[np.argmax(distances[endpoints])])
    if not np.isfinite(distances[end]):
        raise ValueError("Disconnected skeleton; clean or crop the mask.")
    indices = [end]
    while indices[-1] != start:
        indices.append(int(predecessors[indices[-1]]))
    indices = indices[::-1]
    radius = ndimage.distance_transform_edt(mask)
    distance_to_path = dijkstra(graph, directed=False, indices=indices, min_only=True)
    if np.any(distance_to_path > 1.5 * radius[yx[:, 0], yx[:, 1]] + 3):
        raise ValueError("Mask has a substantial branch/overlap. Crop to one unbranched section.")
    path = yx[indices, ::-1].astype(float)
    path = _resample(path, max(samples, len(path)))
    path = ndimage.gaussian_filter1d(path, 2., axis=0, mode="nearest")
    # Exclude end regions where skeleton normals can hit artificial mask closures.
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    p0, p1 = np.rint(path[[0, -1]]).astype(int)
    trim0 = 1.2 * radius[p0[1], p0[0]]
    trim1 = 1.2 * radius[p1[1], p1[0]]
    selected = (s >= trim0) & (s <= s[-1]-trim1)
    if selected.sum() < 8:
        raise ValueError("Pipe section is too short relative to its width; use ordered edges directly.")
    return mask, _resample(path[selected], samples)


def select_mask_region(mask, *, region=None, seed=None, rgb=None, interactive=False):
    """Select one visible section, preserving occlusions instead of filling them."""
    mask=np.asarray(mask,bool)
    labels,count=ndimage.label(mask,np.ones((3,3)))
    sizes=np.bincount(labels.ravel())
    sizes[0]=0
    if count==0 or sizes.max()<25:
        raise ValueError("The selected pipe mask is empty or too small.")
    ordered=np.argsort(-sizes)
    components=[int(i) for i in ordered if sizes[i]>=max(25,.03*sizes.max())]
    chosen=region
    if chosen is None and seed is not None:
        x,y=seed
        if 0<=x<mask.shape[1] and 0<=y<mask.shape[0] and labels[y,x] in components:
            chosen=components.index(int(labels[y,x]))
    if chosen is None and len(components)==1:
        chosen=0
    if chosen is None and interactive:
        from .mask_picker import pick_mask
        from .progress import progress
        progress("Mask",f"{len(components)} disconnected visible sections: select one in the OpenGL window.")
        masks=np.stack([labels==i for i in components])
        chosen=pick_mask(rgb,masks,None,range(len(components)),label="section")
    if chosen is None:
        raise ValueError(f"Pipe has {len(components)} disconnected visible sections. Use --show to select one or --region INDEX (0 to {len(components)-1}).")
    if not 0<=chosen<len(components):
        raise ValueError(f"--region must be between 0 and {len(components)-1}.")
    selected=labels==components[chosen]
    return selected,{"visible_regions":len(components),"selected_region":int(chosen),
                     "region_pixels":int(selected.sum()),"excluded_mask_pixels":int(mask.sum()-selected.sum())}


def extract_edges(mask, samples=240):
    """Trace first boundary crossing on each side of the local 2D skeleton normal.

    These are initial image correspondences, NOT asserted 3D cross-sections.
    Boundaries at the image frame are rejected rather than treated as silhouettes.
    """
    mask, guide = mask_path(mask, samples)
    tangent = np.gradient(guide, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    edges = []
    h, w = mask.shape
    travel = np.arange(0, np.hypot(h, w), .25)
    for sign in (-1, 1):
        edge = np.full_like(guide, np.nan)
        for i, (p, n) in enumerate(zip(guide, normal)):
            ray = p + sign * travel[:, None] * n
            inside_frame = ((ray[:, 0] >= 0) & (ray[:, 0] <= w-1)
                            & (ray[:, 1] >= 0) & (ray[:, 1] <= h-1))
            values = ndimage.map_coordinates(mask.astype(float), [ray[:, 1], ray[:, 0]],
                                             order=1, mode="constant", cval=0)
            outside = np.flatnonzero((values < .5) | ~inside_frame)
            if not len(outside) or outside[0] == 0:
                continue
            k = int(outside[0])
            if not inside_frame[k]:
                continue
            fraction = (values[k-1]-.5) / max(values[k-1]-values[k], 1e-12)
            edge[i] = ray[k-1] + fraction * (ray[k]-ray[k-1])
        edges.append(edge)
    return mask, guide, edges[0], edges[1]


def lift_edge(edge_uv, guide_uv, mask, depth, intrinsics, *, inset_px=1.,
              max_depth_spread_m=.01):
    """Use actual in-mask pixel depth. Keep original boundary rays separately.

    No background filling, neighbour median assignment or invented silhouette Z.
    Inset samples are near-edge surface observations, not exact tangency points.
    """
    if inset_px < 0 or max_depth_spread_m <= 0:
        raise ValueError("inset_px must be >=0 and max_depth_spread_m >0.")
    if depth.shape != mask.shape or depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("Depth, mask and calibrated RGB dimensions must match exactly.")
    xyz = np.full((len(edge_uv), 3), np.nan)
    sampled_uv = np.full_like(edge_uv, np.nan)
    status = np.full(len(edge_uv), "invalid_boundary", dtype="U32")
    h, w = mask.shape
    for i, (edge, guide) in enumerate(zip(edge_uv, guide_uv)):
        if not np.isfinite(edge).all():
            continue
        direction = guide-edge
        direction /= max(np.linalg.norm(direction), 1e-12)
        x, y = np.rint(edge + inset_px*direction).astype(int)
        if not (0 <= x < w and 0 <= y < h and mask[y, x]):
            status[i] = "outside_mask"
            continue
        sampled_uv[i] = [x, y]
        z = float(depth[y, x]) * intrinsics.depth_scale
        if not np.isfinite(z) or z <= 0:
            status[i] = "invalid_depth"
            continue
        patch = depth[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)].astype(float)*intrinsics.depth_scale
        pmask = mask[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)]
        values = patch[pmask & np.isfinite(patch) & (patch > 0)]
        if np.ptp(values) > max_depth_spread_m:
            status[i] = "depth_discontinuity"
            continue
        xyz[i] = intrinsics.deproject(np.array([[x, y]]), np.array([z]))[0]
        status[i] = "ok"
    return xyz, sampled_uv, status


def local_depth_reference(z, neighbours=21, noise_floor_m=.012):
    """Robust local slope/intercept from surrounding stations, including endpoints.

    Used only to reject unsupported depth; predicted depths are never exported
    as measured geometry. The nearest valid stations handle sparse end samples.
    """
    z = np.asarray(z, float)
    good = np.flatnonzero(np.isfinite(z))
    predicted = np.full(len(z), np.nan)
    tolerance = np.full(len(z), np.nan)
    for i in range(len(z)):
        indices = good[np.argsort(abs(good-i))[:neighbours]]
        indices = indices[abs(indices-i) <= max(neighbours, int(np.ceil(.1*len(z))))]
        if len(indices) < 5:
            continue
        dx = indices[:, None] - indices[None, :]
        dz = z[indices, None] - z[indices][None, :]
        use = np.abs(dx) > 2
        if not np.any(use):
            continue
        slope = np.median(dz[use] / dx[use])
        intercept = np.median(z[indices] - slope*(indices-i))
        residual = z[indices] - (intercept + slope*(indices-i))
        predicted[i] = intercept
        tolerance[i] = max(noise_floor_m, 4*1.4826*np.median(abs(residual)))
    return predicted, tolerance


def lift_edge_adaptive(edge_uv, guide_uv, mask, depth, intrinsics, *, inset_px=1.,
                       max_inset_px=6., max_depth_spread_m=.01):
    """Select a supported actual depth pixel from a narrow inward search band.

    All candidates retain their real UV and Z. No depth is copied to the
    silhouette ray, and no gaps are interpolated. The band is restricted to
    35% of the boundary-to-image-guide distance to avoid crossing the pipe.
    """
    if inset_px < 0 or max_inset_px < inset_px:
        raise ValueError("Need 0 <= inset_px <= max_inset_px.")
    distances = np.arange(inset_px, max_inset_px+.001, 1.)
    attempts = [lift_edge(edge_uv, guide_uv, mask, depth, intrinsics, inset_px=float(d),
                          max_depth_spread_m=max_depth_spread_m) for d in distances]
    candidates = np.array([v[0] for v in attempts])
    band_limit = .35*np.linalg.norm(guide_uv-edge_uv, axis=1)
    for k, d in enumerate(distances):
        candidates[k, d > band_limit] = np.nan
    median_depth = np.ma.median(np.ma.masked_invalid(candidates[:, :, 2]), axis=0).filled(np.nan)
    reference, threshold = local_depth_reference(median_depth)
    xyz = np.full_like(candidates[0], np.nan)
    sampled_uv = np.full_like(edge_uv, np.nan)
    used_inset = np.full(len(edge_uv), np.nan)
    status = attempts[0][2].copy()
    has_candidate = np.isfinite(candidates[:, :, 2]).any(axis=0)
    status[has_candidate] = "depth_outlier"
    status[has_candidate & ~np.isfinite(reference)] = "unsupported_depth"
    for k, d in enumerate(distances):
        accept = (~np.isfinite(xyz[:, 2]) & np.isfinite(candidates[k, :, 2])
                  & (abs(candidates[k, :, 2]-reference) < threshold))
        xyz[accept] = candidates[k, accept]
        sampled_uv[accept] = attempts[k][1][accept]
        used_inset[accept] = d
        status[accept] = "ok"
    return xyz, sampled_uv, status, used_inset
