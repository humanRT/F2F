"""Inspectable numerical output plus 2D/3D plots (headless by default)."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import os

import numpy as np


def save_result(folder, result, left, right, *, metadata=None, extra=None, show=False, block=True, intrinsics=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in asdict(result).items() if isinstance(v, np.ndarray)}
    arrays.update(input_left=left, input_right=right)
    arrays.update(extra or {})
    np.savez_compressed(folder / "pairs.npz", **arrays)
    with (folder / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "valid", "status"] +
                        [f"{name}_{axis}_m" for name in ("left", "right", "midpoint") for axis in "xyz"])
        for i in range(len(result.valid)):
            writer.writerow([i, bool(result.valid[i]), result.status[i], *result.left[i],
                             *result.right[i], *result.midpoints[i]])
    statuses, counts = np.unique(result.status, return_counts=True)
    summary = {"units": "metres", "output_type": "edge pairing / midpoint guide, not fitted centerline",
               "stations": len(result.valid), "valid_pairs": int(result.valid.sum()),
               "statuses": dict(zip(statuses.tolist(), counts.tolist())),
               "iterations": result.iterations, "converged": result.converged,
               "last_movement_m": result.movement_m if np.isfinite(result.movement_m) else None,
               **(metadata or {})}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    display = tuple(arrays[k] for k in ("smoothed_left", "smoothed_right", "smoothed_midpoints")) if "smoothed_left" in arrays else None
    mesh = tuple(arrays[k] for k in ("mesh_vertices", "mesh_faces", "pipe_axis", "pipe_radius_m")) if "mesh_vertices" in arrays else None
    plot_3d(folder, result, left, right, show=show, block=block, intrinsics=intrinsics, display=display, mesh=mesh)
    return summary


def draw_camera(ax, intrinsics, points):
    """Draw the rectified RGB camera; the far plane is a display extent only."""
    positive_z = points[points[:, 2] > 0, 2]
    depth = float(positive_z.max()) * 1.1 if len(positive_z) else 1.0
    uv = np.array([[-.5, -.5], [intrinsics.width-.5, -.5],
                   [intrinsics.width-.5, intrinsics.height-.5],
                   [-.5, intrinsics.height-.5]])
    corners = intrinsics.deproject(uv, np.full(4, depth))
    color = "#9467bd"
    ax.scatter([0], [0], [0], color=color, marker="s", s=45, label="RGB camera (origin)")
    for corner in corners:
        ax.plot(*np.stack([np.zeros(3), corner]).T, color=color, alpha=.65, linewidth=.9)
    ax.plot(*corners[[0, 1, 2, 3, 0]].T, color=color, alpha=.65,
            linewidth=.9, label="Calibrated viewing frustum")
    ax.plot([0, 0], [0, 0], [0, depth], color=color, alpha=.5, linestyle=":")
    length = depth * .12
    for direction, label, axis_color in zip(np.eye(3), ("X right", "Y down", "Z forward"),
                                             ("#c43c39", "#39864a", "#3674bd")):
        endpoint = direction * length
        ax.quiver(0, 0, 0, *endpoint, color=axis_color, arrow_length_ratio=.2)
        ax.text(*endpoint, label, color=axis_color, fontsize=8)
    ax.text2D(.02, .02, f"Frustum drawn to Z = {depth:.2f} m (display extent)",
              transform=ax.transAxes, fontsize=8, color=color)
    return np.vstack([np.zeros(3), corners])


def plot_3d(folder, result, left, right, *, show=False, block=False, intrinsics=None, display=None, mesh=None):
    folder = Path(folder)
    # Keep font/cache writes beside the outputs in restricted environments.
    os.environ.setdefault("MPLCONFIGDIR", str((folder / ".mplcache").resolve()))
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title("F2F - 3D results")
    ax = fig.add_subplot(111, projection="3d")
    surface_artist = None
    if mesh is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        vertices, faces, pipe_axis, radius = mesh
        surface_artist = Poly3DCollection(vertices[faces], facecolor="#69b9d1", edgecolor="none",
                                          alpha=.25, linewidth=0, label="Sphere-sweep pipe surface")
        ax.add_collection3d(surface_artist)
        ax.plot(*pipe_axis.T, color="#9b2b94", linewidth=1.7, label="Estimated pipe axis")
        from .pipe_mesh import cross_section_rings
        for i,ring in enumerate(cross_section_rings(pipe_axis,float(radius))):
            ax.plot(*ring.T,color="#ed3030",linewidth=1.,label="Cross-sections" if i==0 else None)
    edge_a, edge_b, midpoint = display if display is not None else (left, right, result.midpoints)
    ax.plot(*edge_a.T, color="#2b6cb0", label="Edge A", linewidth=1.5)
    ax.plot(*edge_b.T, color="#dd6b20", label="Edge B", linewidth=1.5)
    ax.plot(*midpoint.T, color="#23875b", label="Midpoint guide (not centerline)")
    chord_a, chord_b = display[:2] if display is not None else (result.left, result.right)
    for a, b in zip(chord_a[result.valid][::3], chord_b[result.valid][::3]):
        ax.plot(*np.stack([a, b]).T, color="#777777", alpha=.5, linewidth=.7)
    points = np.concatenate([left, right])
    points = points[np.isfinite(points).all(axis=1)]
    if mesh is not None:
        points = np.vstack([points, mesh[0]])
    if intrinsics is not None:
        points = np.vstack([points, draw_camera(ax, intrinsics, points)])
    low, high = (points.min(axis=0), points.max(axis=0)) if len(points) else (np.zeros(3), np.ones(3))
    center, span = (low+high)/2, max(np.max(high-low), 1e-3)*.6
    ax.set(xlim=(center[0]-span, center[0]+span), ylim=(center[1]-span, center[1]+span),
           zlim=(center[2]-span, center[2]+span), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_box_aspect((1, 1, 1))
    # Look along camera +Z: screen right is +X and screen up is -Y.
    ax.view_init(elev=-90, azim=-90, roll=0)
    ax.set_title(f"3D pipe edge pairing | {result.valid.sum()}/{len(result.valid)} usable stations")
    if mesh is not None:
        ax.set_title(f"Estimated pipe | diameter {float(mesh[3])*2000:.1f} mm")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if surface_artist is not None and show:
        from matplotlib.widgets import Slider
        opacity = Slider(fig.add_axes([.73, .025, .2, .018]), "Mesh opacity", 0., 1., valinit=.25)
        def set_opacity(value):
            surface_artist.set_alpha(value)
            fig.canvas.draw_idle()
        opacity.on_changed(set_opacity)
        fig._mesh_opacity_slider = opacity
    fig.savefig(folder / "pairing_3d.png", dpi=160)
    if show:
        plt.show(block=block)
    if not show or block:
        plt.close(fig)
    return fig


def pump_3d():
    from matplotlib._pylab_helpers import Gcf
    for manager in list(Gcf.get_all_fig_managers()):
        manager.canvas.flush_events()


def wait_for_3d():
    import matplotlib.pyplot as plt
    if plt.get_fignums():
        plt.show(block=True)

