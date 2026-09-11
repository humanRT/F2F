"""Constant-radius surface fit and watertight sphere-sweep mesh, in metres."""
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def masked_points(depth, mask, intrinsics, *, uncertainty=None, return_weights=False):
    z = np.asarray(depth, float)*intrinsics.depth_scale
    good = binary_erosion(mask, iterations=3) & np.isfinite(z) & (z > 0)
    y, x = np.nonzero(good)
    # Deterministic, image-wide sampling keeps the fit inexpensive.
    take = np.linspace(0, len(x)-1, min(4500, len(x)), dtype=int)
    x,y=x[take],y[take]
    points=intrinsics.deproject(np.column_stack([x,y]), z[y,x])
    weights=np.ones(len(points))
    if uncertainty is not None:
        std=np.asarray(uncertainty["std_m"])
        counts=np.asarray(uncertainty["valid_count"])
        if std.shape!=z.shape or counts.shape!=z.shape:
            raise ValueError("Depth uncertainty dimensions do not match depth.")
        sigma=std[y,x]
        support=counts[y,x]/float(uncertainty["frame_count"])
        # Temporal spread, not spread/sqrt(N): correlated errors do not vanish.
        weights=np.where(np.isfinite(sigma),np.clip((.002/np.maximum(sigma,.001))**2*support,.02,1),.02)
    return (points,weights) if return_weights else points


def nearest_stations(points, axis):
    """Closest projection on adjacent polyline segments, with continuous station."""
    _, nearest = cKDTree(axis).query(points)
    starts = np.column_stack([np.clip(nearest-1, 0, len(axis)-2),
                              np.clip(nearest, 0, len(axis)-2)])
    a, d = axis[starts], axis[starts+1]-axis[starts]
    u = np.clip(np.sum((points[:, None]-a)*d, axis=2)/
                np.maximum(np.sum(d*d, axis=2), 1e-15), 0, 1)
    projected = a+u[..., None]*d
    best = np.argmin(np.sum((points[:, None]-projected)**2, axis=2), axis=1)
    rows = np.arange(len(points))
    return (starts[rows, best]+u[rows, best])/(len(axis)-1)


def fit_pipe(left, right, *, surface=None, inferred=None, surface_weights=None):
    axis = (np.asarray(left)+np.asarray(right))/2
    if not np.isfinite(axis).all() or len(axis) < 8:
        raise ValueError("Mesh requires a continuous finite midpoint guide with at least eight stations.")
    radii = np.linalg.norm(np.asarray(right)-left, axis=1)/2
    supported = np.isfinite(radii) & (radii > 1e-5)
    if inferred is not None and np.count_nonzero(supported & ~inferred) >= 6:
        supported &= ~inferred
    if supported.sum() < 6:
        raise ValueError("Not enough supported edge widths to estimate a radius.")
    radius = float(np.median(radii[supported]))
    info = {"initial_radius_m": radius, "radius_m": radius,
            "method": "edge-width estimate", "surface_points": 0,
            "rounded_ends": "synthetic sphere-sweep ends; not observed pipe ends"}
    if surface is None or len(surface) < 100:
        info["refinement_note"] = "Insufficient surface depth; using the edge estimate."
        return axis, radius, info
    points = np.asarray(surface, float)
    finite=np.isfinite(points).all(axis=1)
    weights=np.ones(len(points)) if surface_weights is None else np.asarray(surface_weights,float)
    if weights.shape!=(len(points),) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("Surface weights must be positive finite values matching the points.")
    points,weights=points[finite],weights[finite]
    t0 = nearest_stations(points, axis)
    grid = np.linspace(0, 1, len(axis))
    initial = np.column_stack([np.interp(t0, grid, axis[:, j]) for j in range(3)])
    # Reject unrelated mask/depth surfaces and the artificial endpoint regions.
    keep = (t0 > .04) & (t0 < .96) & (np.linalg.norm(points-initial, axis=1) < radius*1.8)
    points = points[keep]
    weights=weights[keep]
    # Equalize longitudinal coverage so densely sampled areas cannot dominate.
    bins=np.clip((t0[keep]*20).astype(int),0,19)
    density=np.bincount(bins,minlength=20)
    weights*=np.median(density[density>0])/np.maximum(density[bins],1)
    weights=np.clip(weights,.02,3.)
    info["surface_points"] = len(points)
    if len(points) < 100:
        info["refinement_note"] = "Too few nearby surface samples; using the edge estimate."
        return axis, radius, info
    knots = np.r_[np.zeros(4), np.linspace(0,1,6)[1:-1], np.ones(4)]
    basis = BSpline(knots, np.eye(8), 3)
    bg = basis(grid)
    original = axis.copy()
    origin=original.mean(axis=0)
    d2=np.diff(np.eye(8),n=2,axis=0)
    controls=np.linalg.lstsq(np.vstack([bg,.1*d2]),
                           np.vstack([original-origin,np.zeros((6,3))]),rcond=None)[0]
    initial_controls=controls.copy()
    # Fit the actual axis; do not add the original wavy guide back afterward.
    params = np.r_[controls.ravel(), radius]
    # Spline coefficients are not axis points: bending can require coefficient
    # motion larger than the physical axis correction. Bound the evaluated
    # axis separately below rather than rejecting valid fits at tight controls.
    lower = np.r_[controls.ravel()-3*radius, radius*.7]
    upper = np.r_[controls.ravel()+3*radius, radius*1.8]
    axis=origin+bg@controls
    for _ in range(4):
        station = nearest_stations(points, axis)
        bp = basis(station)
        base = np.column_stack([np.interp(station, grid, original[:, j]) for j in range(3)])
        tangent = np.gradient(axis, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
        tangent = np.column_stack([np.interp(station, grid, tangent[:, j]) for j in range(3)])
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
        def residual(x):
            control = x[:-1].reshape(8, 3)
            fitted=origin+bp@control
            shift=fitted-base
            delta = points-fitted
            radial = delta-np.sum(delta*tangent, axis=1)[:, None]*tangent
            shift_grid=origin+bg@control-original
            # Endpoint depth has no data beyond the capture. Tie corrections
            # to nearby supported stations instead of allowing end overshoot.
            end_step=max(1,len(grid)//12)
            end_difference=shift_grid[[0,-1]]-shift_grid[[end_step,-end_step-1]]
            return np.r_[np.sqrt(weights)*(np.linalg.norm(radial, axis=1)-x[-1]),
                         .15*np.linalg.norm(np.diff(control, n=2, axis=0),axis=1),
                         1.5*np.linalg.norm(np.diff(control, n=3, axis=0),axis=1),
                         .04*np.linalg.norm(control-initial_controls,axis=1), .1*(x[-1]-radius),
                         10.*np.linalg.norm(end_difference,axis=1),
                         30.*np.maximum(np.linalg.norm(shift_grid,axis=1)-1.2*radius,0),
                         .15*np.sum(shift*tangent, axis=1)]
        result = least_squares(residual, params, bounds=(lower, upper),
                               loss="soft_l1", f_scale=.002, max_nfev=240,
                               ftol=1e-6,xtol=1e-6,gtol=1e-7)
        params = result.x
        axis = origin+bg@params[:-1].reshape(8, 3)
    fitted_radius = float(params[-1])
    final_station=nearest_stations(points,axis)
    closest=np.column_stack([np.interp(final_station,grid,axis[:,j]) for j in range(3)])
    signed_error=np.linalg.norm(points-closest,axis=1)-fitted_radius
    error = np.abs(signed_error)
    median_error = float(np.median(error))
    bounded = bool(np.any(result.active_mask))
    max_shift=float(np.max(np.linalg.norm(axis-original,axis=1)))
    accepted = bool(result.success and not bounded and max_shift<1.5*radius
                    and median_error < max(.004, .12*fitted_radius))
    info.update(surface_median_error_m=median_error, refinement_accepted=accepted,
                optimizer_success=bool(result.success), fit_at_bound=bounded,
                surface_signed_median_error_m=float(np.median(signed_error)),
                surface_p90_error_m=float(np.percentile(error,90)), axis_shift_max_m=max_shift)
    if not accepted:
        raise RuntimeError("Surface fit did not pass convergence/geometry checks; no fallback mesh was generated. "
                           f"Solver: {result.message}; median residual {median_error*1000:.2f} mm; "
                           f"parameter bound hit: {bounded}; maximum axis correction {max_shift*1000:.1f} mm.")
    info.update(radius_m=fitted_radius, method="direct smooth-axis fit; longitudinal-density weighted"+
                (" and temporal-uncertainty weighted" if surface_weights is not None else ""),
                uncertainty_weighted=surface_weights is not None,
                axis_shift_median_m=float(np.median(np.linalg.norm(axis-original, axis=1))))
    return axis, fitted_radius, info


def cross_section_rings(axis, radius, count=12, segments=96, *, positions=None):
    """Equally spaced cross-section circles normal to the fitted axis."""
    axis=np.asarray(axis,float)
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(axis,axis=0),axis=1))]
    if distance[-1]<=1e-10 or radius<=0:
        raise ValueError("Cross-sections require a nonzero axis length and positive radius.")
    good=np.r_[True,np.diff(distance)>1e-10]
    axis,distance=axis[good],distance[good]
    positions=np.linspace(0,1,count) if positions is None else np.asarray(positions,float)
    if positions.ndim!=1 or not np.isfinite(positions).all() or np.any((positions<0)|(positions>1)):
        raise ValueError("Ring positions must be finite fractions along the axis, between 0 and 1.")
    stations=positions*distance[-1]
    centers=np.column_stack([np.interp(stations,distance,axis[:,j]) for j in range(3)])
    tangents=np.gradient(axis,distance,axis=0)
    tangents=np.column_stack([np.interp(stations,distance,tangents[:,j]) for j in range(3)])
    tangents/=np.linalg.norm(tangents,axis=1)[:,None]
    reference=np.eye(3)[np.argmin(np.abs(tangents),axis=1)]
    u=np.cross(tangents,reference)
    u/=np.linalg.norm(u,axis=1)[:,None]
    v=np.cross(tangents,u)
    angle=np.linspace(0,2*np.pi,segments+1)
    return centers[:,None,:]+radius*(np.cos(angle)[None,:,None]*u[:,None,:]+
                                     np.sin(angle)[None,:,None]*v[:,None,:])


def sphere_sweep(axis, radius):
    """Extract the union of spheres swept along every axis segment.

    An exact segment-distance field gives capsule ends and handles bends
    without twisting rings or overlapping internal faces.
    """
    axis = np.asarray(axis, float)
    pitch = max(radius/12, np.ptp(axis, axis=0).max()/220)
    low, high = axis.min(axis=0)-radius-3*pitch, axis.max(axis=0)+radius+3*pitch
    shape = np.ceil((high-low)/pitch).astype(int)+1
    if np.prod(shape) > 5_000_000:
        pitch *= (np.prod(shape)/5_000_000)**(1/3)
        shape = np.ceil((high-low)/pitch).astype(int)+1
    field = np.full(tuple(shape), radius+pitch, dtype=np.float32)
    for a, b in zip(axis[:-1], axis[1:]):
        start = np.maximum(0, np.floor((np.minimum(a,b)-radius-2*pitch-low)/pitch).astype(int))
        stop = np.minimum(shape, np.ceil((np.maximum(a,b)+radius+2*pitch-low)/pitch).astype(int)+1)
        xyz = np.meshgrid(*[low[j]+np.arange(start[j],stop[j])*pitch for j in range(3)], indexing="ij")
        delta = np.stack(xyz, axis=-1)-a
        direction = b-a
        u = np.clip((delta@direction)/max(float(direction@direction),1e-15),0,1)
        distance = np.linalg.norm(delta-u[...,None]*direction,axis=-1)-radius
        region = tuple(slice(s,e) for s,e in zip(start,stop))
        np.minimum(field[region],distance,out=field[region])
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(pitch,)*3, allow_degenerate=False)
    vertices += low
    signed_volume = np.einsum('ij,ij->i', vertices[faces[:,0]],
                              np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6
    if signed_volume < 0:
        faces = faces[:,::-1]
    return vertices, faces, pitch


def simplify_mesh(vertices, faces, radius, target=1500):
    """Quadric decimation, checked for watertightness and sampled deviation."""
    import open3d as o3d
    original = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                        o3d.utility.Vector3iVector(faces))
    tolerance = max(.00075, radius*.03)
    def distance(source, destination):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(destination))
        return float(scene.compute_distance(o3d.core.Tensor(np.asarray(source.vertices),
                     dtype=o3d.core.Dtype.Float32)).numpy().max())
    for count in (target, target*2, target*4):
        if count >= len(faces):
            break
        reduced = original.simplify_quadric_decimation(count)
        if not reduced.is_watertight():
            continue
        deviation = max(distance(original,reduced),distance(reduced,original))
        if deviation <= tolerance:
            return np.asarray(reduced.vertices), np.asarray(reduced.triangles), {
                "original_triangles": len(faces), "triangles": len(reduced.triangles),
                "sampled_max_deviation_m": deviation, "deviation_tolerance_m": tolerance,
                "watertight": True}
    return vertices, faces, {"original_triangles": len(faces), "triangles": len(faces),
                            "note": "Reduction failed geometry checks; retained original mesh."}


def cut_open_ends(vertices, faces, axis, *, tangents=None):
    """Clip at endpoint cross-section planes without adding closing faces."""
    vertices, faces = np.asarray(vertices,float), np.asarray(faces,int)
    start,end=(axis[1]-axis[0],axis[-1]-axis[-2]) if tangents is None else tangents
    for origin, normal in ((axis[0],start), (axis[-1],-end)):
        normal = normal/np.linalg.norm(normal)
        distance = (vertices-origin)@normal
        new_vertices = vertices.tolist()
        intersections = {}
        def crossing(a,b):
            key = tuple(sorted((a,b)))
            if abs(distance[a]) < 1e-10: return a
            if abs(distance[b]) < 1e-10: return b
            if key not in intersections:
                u = distance[a]/(distance[a]-distance[b])
                intersections[key] = len(new_vertices)
                new_vertices.append((vertices[a]+u*(vertices[b]-vertices[a])).tolist())
            return intersections[key]
        clipped = []
        for triangle in faces:
            polygon = []
            for a,b in zip(triangle, np.roll(triangle,-1)):
                inside_a,inside_b = distance[a]>=-1e-10,distance[b]>=-1e-10
                if inside_a: polygon.append(int(a))
                if inside_a != inside_b: polygon.append(crossing(int(a),int(b)))
            polygon = list(dict.fromkeys(polygon))
            for j in range(1,len(polygon)-1):
                clipped.append([polygon[0],polygon[j],polygon[j+1]])
        if not clipped:
            raise ValueError("End cuts removed the entire mesh; inspect the fitted axis.")
        faces = np.asarray(clipped,int)
        vertices = np.asarray(new_vertices,float)
        area = np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                                       vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)
        faces = faces[area>1e-14]
        used, inverse = np.unique(faces,return_inverse=True)
        vertices,faces = vertices[used],inverse.reshape(-1,3)
    return vertices,faces


def save_mesh(folder, vertices, faces, filename="pipe_mesh.ply"):
    path = Path(folder)/filename
    with path.open("w", encoding="ascii") as f:
        f.write(f"ply\nformat ascii 1.0\ncomment units metres; open endpoint cuts\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        np.savetxt(f, vertices, fmt="%.8f")
        np.savetxt(f, np.column_stack([np.full(len(faces),3),faces]), fmt="%d")
