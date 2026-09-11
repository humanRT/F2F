"""Continuous robust cubic spline model for two ordered pipe boundaries."""
import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator


def fit_pipe_edges(left, right, *, parameters=None, window_m=.04, samples=500):
    """Fit C(t)-D(t), C(t)+D(t), penalizing rapid changes in both C and D.

    C is a shared midpoint guide and D is the half-chord vector, not a radius.
    Different regularization strengths couple the edges without assuming a
    planar bend, constant apparent width, or exact silhouette tangency.
    Only the overlapping observed parameter interval is evaluated.
    """
    curves = [np.asarray(p, float) for p in (left, right)]
    if not np.isfinite(window_m) or window_m <= 0:
        raise ValueError("Spline smoothing length must be positive and finite.")
    ts = ([np.linspace(0, 1, len(p)) for p in curves] if parameters is None
          else [np.asarray(t, float) for t in parameters])
    for p, t in zip(curves, ts):
        if p.ndim != 2 or p.shape[1] != 3 or t.shape != (len(p),):
            raise ValueError("Each spline edge needs (N,3) points and N station parameters.")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Spline parameters must be finite and strictly increasing.")
    good = [np.isfinite(p).all(axis=1) for p in curves]
    if any(g.sum() < 6 for g in good):
        raise ValueError("Continuous spline fitting needs at least six depth samples on each edge.")
    ends = [p[g][[0, -1]] for p, g in zip(curves, good)]
    if np.linalg.norm(ends[0]-ends[1][::-1], axis=1).sum() < np.linalg.norm(ends[0]-ends[1], axis=1).sum():
        curves[1] = curves[1][::-1].copy()
        good[1] = good[1][::-1].copy()
        ts[1] = ts[1][0]+ts[1][-1]-ts[1][::-1]
    lo = max(t[g][0] for t, g in zip(ts, good))
    hi = min(t[g][-1] for t, g in zip(ts, good))
    if hi <= lo:
        raise ValueError("The two edges have no overlapping observed interval.")
    good = [g & (t >= lo) & (t <= hi) for t, g in zip(ts, good)]
    if any(g.sum() < 6 for g in good):
        raise ValueError("Too few overlapping edge samples for a supported spline fit.")
    # Estimate length on coarse median bins so depth spikes do not add knots.
    centers = []
    for p, t, g in zip(curves, ts, good):
        bins = np.linspace(lo, hi, 17)
        coarse = np.array([np.median(p[g & (t >= a) & (t <= b)], axis=0)
                           for a, b in zip(bins[:-1], bins[1:])
                           if np.any(g & (t >= a) & (t <= b))])
        centers.append(np.linalg.norm(np.diff(coarse, axis=0), axis=1).sum())
    segments = int(np.clip(np.ceil(np.mean(centers)/window_m), 3, 32))
    # Sparse captures cannot support many independent coefficients.
    segments = min(segments, max(1, min(g.sum() for g in good)//3-3))
    knots = np.r_[np.zeros(4), np.linspace(0, 1, segments+1)[1:-1], np.ones(4)]
    k = len(knots)-4
    basis = BSpline(knots, np.eye(k), 3, extrapolate=False)
    designs, targets = [], []
    for sign, p, t, g in zip((-1, 1), curves, ts, good):
        g &= (t >= lo) & (t <= hi)
        b = basis((t[g]-lo)/(hi-lo))
        designs.append(np.column_stack([b, sign*b]))
        targets.append(p[g])
    matrix = np.vstack(designs)
    target = np.vstack(targets)
    # Translation-invariant second differences; stronger chord regularization
    # encourages the two edges to bend together rather than wobble independently.
    d2 = np.diff(np.eye(k), n=2, axis=0)
    zero = np.zeros_like(d2)
    penalty = np.vstack([np.column_stack([d2*.5, zero]),
                         np.column_stack([zero, d2*2.])])
    weights = np.ones(len(target))
    for _ in range(12):
        w = np.sqrt(weights)
        coeff = np.linalg.lstsq(np.vstack([matrix*w[:, None], penalty]),
                               np.vstack([target*w[:, None], np.zeros((len(penalty), 3))]), rcond=None)[0]
        error = np.linalg.norm(matrix@coeff-target, axis=1)
        cutoff = max(.0015, 2*np.median(error))
        updated = np.minimum(1., cutoff/np.maximum(error, 1e-12))
        if np.max(np.abs(updated-weights)) < .001:
            break
        weights = updated
    grid = np.linspace(0, 1, samples)
    c, d = basis(grid)@coeff[:k], basis(grid)@coeff[k:]
    a, b = c-d, c+d
    if np.any(np.linalg.norm(d, axis=1) < 1e-6):
        raise ValueError("Spline edges collapse together; inspect mask and edge ordering.")
    supports = []
    offset = 0
    for t, g in zip(ts, good):
        n = g.sum()
        accepted = weights[offset:offset+n] >= .35
        observed = (t[g][accepted]-lo)/(hi-lo)
        step = np.median(np.diff(t))/(hi-lo)
        supports.append((observed, step))
        offset += n
    return {"left": a, "right": b, "grid": grid, "supports": supports,
            "domain": np.array([lo, hi]), "weights": weights,
            "rms_m": float(np.sqrt(np.mean(error**2)))}


def continuous_pairs(model, pairing):
    """Smooth monotone correspondence through solved pairs, including gaps.

    Where no plane pair was solved, correspondence is an estimate, explicitly
    flagged. The midpoint is always exactly between the two fitted edges.
    """
    n = len(pairing.valid)
    stations = np.linspace(0, 1, n)
    curves, inferred, parameters = [], [], []
    from scipy.interpolate import CubicSpline
    for side, solved in zip(("left", "right"), (pairing.left_parameter, pairing.right_parameter)):
        valid = pairing.valid & np.isfinite(solved)
        # Explicit endpoints cover the full overlapping measured interval.
        valid[0] = valid[-1] = False
        x = np.r_[0., stations[valid], 1.]
        y = np.r_[0., solved[valid], 1.]
        parameter = PchipInterpolator(x, y)(stations)
        curves.append(CubicSpline(model["grid"], model[side])(parameter))
        parameters.append(parameter)
    for parameter, (observed, step) in zip(parameters, model["supports"]):
        near = (np.min(np.abs(parameter[:, None]-observed[None, :]), axis=1)
                if len(observed) else np.full(n, np.inf))
        inferred.append((near > 1.5*step) | ~pairing.valid)
    a, b = curves
    return a, b, (a+b)/2, np.column_stack(inferred)
