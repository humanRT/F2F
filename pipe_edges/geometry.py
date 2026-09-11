"""Rotation-invariant plane/polyline pairing, with explicit gaps and ambiguity.

This is a geometric initializer, not proof of true cross-section correspondence.
All 3D coordinates and tolerances use metres.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class PairingResult:
    left: np.ndarray
    right: np.ndarray
    midpoints: np.ndarray
    plane_origins: np.ndarray
    tangents: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    left_parameter: np.ndarray
    right_parameter: np.ndarray
    iterations: int
    converged: bool
    movement_m: float


def _validate_curve(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError("Each edge must be an ordered (N, 3) array with N >= 3.")
    if np.isinf(p).any() or np.isfinite(p).all(axis=1).sum() < 3:
        raise ValueError("Each edge needs >= 3 finite points; use NaN rows for gaps.")
    p = p.copy()
    p[~np.isfinite(p).all(axis=1)] = np.nan
    return p


def _filled(p):
    """Interpolation is for guide/parameter estimation ONLY, never edge geometry."""
    good = np.isfinite(p).all(axis=1)
    return np.column_stack([np.interp(np.arange(len(p)), np.flatnonzero(good), p[good, j])
                            for j in range(p.shape[1])])


def _parameter(p):
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(_filled(p), axis=0), axis=1))]
    if s[-1] <= 1e-12:
        raise ValueError("An edge has zero length.")
    return s / s[-1]


def _smooth_tangents(guide, sigma):
    smooth = gaussian_filter1d(guide, sigma, axis=0, mode="nearest") if sigma > 0 else guide
    tangents = np.gradient(smooth, axis=0)
    length = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(length[:, None], 1e-15)
    return tangents


def plane_intersections(edge, origin, normal, parameter=None, max_segment_m=np.inf,
                        tolerance=1e-9):
    """Return (point, normalized edge parameter) hits. Never bridge NaN gaps.

    A segment lying in the plane returns both ends: downstream marks ambiguity.
    Shared vertex hits are deduplicated by parameter, not spatial proximity.
    """
    edge = np.asarray(edge, float)
    parameter = _parameter(edge) if parameter is None else parameter
    a, b = edge[:-1], edge[1:]
    da, db = (a - origin) @ normal, (b - origin) @ normal
    ok = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
          & (np.linalg.norm(b - a, axis=1) <= max_segment_m))
    crossing = ok & (((da <= tolerance) & (db >= -tolerance))
                     | ((db <= tolerance) & (da >= -tolerance)))
    hits = []
    for k in np.flatnonzero(crossing):
        if abs(da[k]) <= tolerance and abs(db[k]) <= tolerance:
            fractions = (0., 1.)
        elif abs(da[k] - db[k]) > 1e-15:
            fractions = (float(np.clip(da[k] / (da[k] - db[k]), 0, 1)),)
        else:
            continue
        for f in fractions:
            t = parameter[k] + f * (parameter[k+1] - parameter[k])
            if not hits or abs(t - hits[-1][1]) > 1e-8:
                hits.append((a[k] + f * (b[k] - a[k]), t))
    return hits


def pair_edges(left, right, *, stations=100, iterations=15, smooth_sigma=2.,
               search_window=0.12, max_segment_m=np.inf, tolerance_m=1e-5,
               initial_guide=None, refine_guide=True, progress_callback=None, edge_parameters=None):
    """Pair two ordered 3D polylines using locally perpendicular slicing planes.

    Edges must cover approximately the same longitudinal interval. Direction is
    aligned automatically. NaN rows preserve depth gaps. search_window limits
    each intersection to a neighbourhood in normalized edge arc length.
    initial_guide may provide a better (K,3) guide in the same frame. Set
    refine_guide=False to keep that supplied guide fixed instead of replacing it
    with silhouette midpoints (which are biased by perspective).
    edge_parameters optionally supplies shared image-guide station coordinates,
    preserving initial correspondence instead of normalizing noisy 3D arc length.
    A valid result means unique, ordered intersections, NOT metrology confidence.
    """
    left, right = _validate_curve(left), _validate_curve(right)
    if stations < 8 or iterations < 1 or smooth_sigma < 0:
        raise ValueError("Need stations >= 8, iterations >= 1, smooth_sigma >= 0.")
    if not 0 < search_window <= 1 or max_segment_m <= 0 or tolerance_m <= 0:
        raise ValueError("Invalid search window, segment limit or tolerance.")
    if not refine_guide and initial_guide is None:
        raise ValueError("A fixed guide requires initial_guide.")
    sl, sr = _parameter(left), _parameter(right)
    if edge_parameters is not None:
        sl, sr = (np.asarray(v, float) for v in edge_parameters)
        for parameter, edge in ((sl, left), (sr, right)):
            if parameter.shape != (len(edge),) or not np.isfinite(parameter).all() or np.any(np.diff(parameter) <= 0):
                raise ValueError("Edge parameters must be finite, strictly increasing, and match each curve length.")
            if parameter[0] < 0 or parameter[-1] > 1:
                raise ValueError("Edge parameters must lie within [0,1].")
    lf, rf = _filled(left), _filled(right)
    if np.linalg.norm(lf[0]-rf[-1]) + np.linalg.norm(lf[-1]-rf[0]) < (
            np.linalg.norm(lf[0]-rf[0]) + np.linalg.norm(lf[-1]-rf[-1])):
        right, rf = right[::-1].copy(), rf[::-1].copy()
        sr = 1 - sr[::-1]
    stations_s = np.linspace(0, 1, stations)
    def sample(p, s):
        return np.column_stack([np.interp(stations_s, s, p[:, j]) for j in range(3)])
    guide = (sample(lf, sl) + sample(rf, sr)) / 2
    if initial_guide is not None:
        g = _validate_curve(initial_guide)
        if not np.isfinite(g).all():
            raise ValueError("initial_guide must be finite.")
        guide = sample(g, _parameter(g))
    expected_l, expected_r = stations_s.copy(), stations_s.copy()
    converged, movement = False, float("inf")
    previous_valid = None
    for iteration in range(iterations):
        origins = guide.copy()
        tangents = _smooth_tangents(guide, smooth_sigma)
        paired_l = np.full((stations, 3), np.nan)
        paired_r = paired_l.copy()
        tl = np.full(stations, np.nan)
        tr = tl.copy()
        status = np.full(stations, "missing_intersection", dtype="U32")
        last_l = last_r = -1.
        for k, (o, n) in enumerate(zip(origins, tangents)):
            if np.linalg.norm(n) < .5:
                status[k] = "degenerate_tangent"
                continue
            hl = [h for h in plane_intersections(left, o, n, sl, max_segment_m)
                  if abs(h[1]-expected_l[k]) <= search_window]
            hr = [h for h in plane_intersections(right, o, n, sr, max_segment_m)
                  if abs(h[1]-expected_r[k]) <= search_window]
            if len(hl) > 1 or len(hr) > 1:
                status[k] = "ambiguous_intersection"
                continue
            if not hl or not hr:
                continue
            if hl[0][1] <= last_l + 1e-10 or hr[0][1] <= last_r + 1e-10:
                status[k] = "nonmonotonic_pair"
                continue
            if np.linalg.norm(hl[0][0] - hr[0][0]) < 1e-10:
                status[k] = "zero_width"
                continue
            paired_l[k], tl[k] = hl[0]
            paired_r[k], tr[k] = hr[0]
            last_l, last_r = tl[k], tr[k]
            status[k] = "ok"
        valid = status == "ok"
        midpoints = (paired_l + paired_r) / 2
        if not refine_guide:
            movement, converged = 0., bool(valid.sum() >= 3)
            if progress_callback:
                progress_callback(iteration+1, iterations, int(valid.sum()), stations, movement)
            break
        if valid.sum() < 3:
            movement = float("inf")
            if progress_callback:
                progress_callback(iteration+1, iterations, int(valid.sum()), stations, movement)
            break
        movement = float(np.max(np.linalg.norm(midpoints[valid] - guide[valid], axis=1)))
        if progress_callback:
            progress_callback(iteration+1, iterations, int(valid.sum()), stations, movement)
        if previous_valid is not None and np.array_equal(valid, previous_valid) and movement < tolerance_m:
            converged = True
            break
        previous_valid = valid.copy()
        # Missing stations remain guide predictions; exported pairs remain NaN.
        guide[valid] = .5 * guide[valid] + .5 * midpoints[valid]
        expected_l[valid], expected_r[valid] = tl[valid], tr[valid]
    return PairingResult(paired_l, paired_r, midpoints, origins, tangents, valid,
                         status, tl, tr, iteration+1, converged, movement)
