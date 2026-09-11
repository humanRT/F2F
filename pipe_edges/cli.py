import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import pair_edges
from .report import save_result, plot_3d, pump_3d, wait_for_3d
from .progress import progress, pairing_progress
from .pipe_mesh import cross_section_rings


def camera_burst_frames(override=None):
    """Resolve CLI override, then app-local setting.json, then the default."""
    value=override
    if value is None:
        path=Path(__file__).resolve().parents[1]/"setting.json"
        if path.exists():
            try:
                settings=json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError,OSError) as exc:
                raise ValueError(f"Cannot read {path}: {exc}") from exc
            if not isinstance(settings,dict):
                raise ValueError(f"{path} must contain a JSON object.")
            value=settings.get("burst_frames",45)
        else:
            value=45
    if type(value) is not int or not 3<=value<=120:
        raise ValueError("burst_frames must be an integer from 3 to 120 in setting.json or --burst-frames.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAM3 masks and iterative 3D pipe edge pairing")
    sub = parser.add_subparsers(dest="command", required=True)
    scene = sub.add_parser("scene", help="Open the OpenGL mesh and point-cloud viewer")
    scene.add_argument("--result", required=True)
    worker = sub.add_parser("worker", help="Manage the persistent local SAM3 GPU worker")
    worker.add_argument("action", choices=("start", "status", "stop", "restart"), default="status", nargs="?")
    overlay = sub.add_parser("overlay", help="Open saved edges and midpoint guide over RGB using OpenGL")
    overlay.add_argument("--result", required=True, help="Folder containing pairs.npz")
    overlay.add_argument("--rgb")
    overlay.add_argument("--intrinsics")
    overlay.add_argument("--no-show", action="store_true", help="Save OpenGL overlay without keeping a window open")
    overlay.add_argument("--matplotlib", action="store_true", help="Also open the Matplotlib plot")
    camera = sub.add_parser("camera", help="Preview Gemini camera, capture, then run SAM3 and pair edges")
    camera.add_argument("--no-preview", action="store_true", help="Capture automatically without a preview window")
    camera.add_argument("--capture-only", action="store_true", help="Save RGB/depth/calibration without processing")
    camera.add_argument("--timeout", type=float, default=20.)
    camera.add_argument("--burst-frames",type=int,default=None,help="Override setting.json burst_frames (3 to 120)")
    edges = sub.add_parser("edges", help="Use existing ordered 3D edges in metres")
    edges.add_argument("--input", required=True, help="NPZ containing left and right (N,3); optional guide (K,3)")
    edges.add_argument("--fixed-guide", action="store_true", help="Keep the supplied guide fixed; requires NPZ guide")
    rgbd = sub.add_parser("rgbd", help="SAM3 mask -> image edges -> aligned depth -> 3D pairs")
    rgbd.add_argument("--rgb", required=True)
    rgbd.add_argument("--depth", required=True, help="Raw depth NPY or single-channel PNG/TIFF")
    rgbd.add_argument("--intrinsics", required=True)
    rgbd.add_argument("--depth-stats",help="Optional temporal depth_stats.npz; otherwise auto-detected beside depth.npy")
    resume = sub.add_parser("resume",help="Continue an existing capture and select among cached SAM3 candidates")
    resume.add_argument("--result",required=True,help="Existing result folder containing capture/")
    for command in (camera, rgbd, resume):
        command.add_argument("--mask", help="Optional saved mask to rerun geometry without SAM3")
        command.add_argument("--prompt", default="pipe")
        command.add_argument("--checkpoint", help="Local SAM3 checkpoint; otherwise official HF download")
        command.add_argument("--threshold", type=float, default=.5)
        command.add_argument("--instance", type=int)
        command.add_argument("--region",type=int,help="Visible connected section within the selected mask (0 is largest)")
        command.add_argument("--seed", type=int, nargs=2, metavar=("X", "Y"))
        command.add_argument("--edge-samples", type=int, default=300)
        command.add_argument("--inset-px", type=float, default=1.)
        command.add_argument("--max-inset-px", type=float, default=6., help="Maximum inward search for supported depth")
        command.add_argument("--max-depth-spread-m", type=float, default=.01)
    for command in (camera, edges, rgbd, resume):
        command.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "F2F-results" / command.prog.split()[-1]))
        command.add_argument("--stations", type=int, default=100)
        command.add_argument("--iterations", type=int, default=30)
        command.add_argument("--smooth-sigma", type=float, default=2.)
        command.add_argument("--line-smoothing-mm", type=float, default=40.,
                             help="Continuous spline bend-detail scale in mm; larger is smoother, 0 disables")
        command.add_argument("--search-window", type=float, default=.12)
        command.add_argument("--max-segment-m", type=float, default=.02)
        command.add_argument("--tolerance-m", type=float, default=1e-5)
        command.add_argument("--show", action="store_true", help="Open RGB and 3D mesh/point-cloud OpenGL windows")
        command.add_argument("--matplotlib", action="store_true", help="Also open the Matplotlib plot")
    resume.set_defaults(show=True)
    resume.add_argument("--no-show",dest="show",action="store_false")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("--") and argv[0] != "--help":
        argv.insert(0, "camera")
    args = parser.parse_args(argv)
    try:
        progress("Start", f"Running {args.command}")
        if args.command=="camera":
            args.burst_frames=camera_burst_frames(args.burst_frames)
            progress("Settings",f"Burst capture: {args.burst_frames} frames.")
        if args.command == "scene":
            from .gl_scene import show_scene
            show_scene(args.result)
            return 0
        if args.command == "worker":
            from .sam3_worker import control
            control(args.action)
            return 0
        if args.command == "overlay":
            from .gl_overlay import render_overlay
            from .vision import Intrinsics
            folder = Path(args.result)
            summary_path = folder / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            capture_folder = folder / "capture"
            if not capture_folder.exists():
                capture_folder = folder.parent / "capture"
            rgb_path = args.rgb or next((p for p in [Path(summary.get("rgb_file", "__missing_rgb__")), capture_folder / "rgb.png"] if p.is_file()), capture_folder / "rgb.png")
            intr_path = args.intrinsics or next((p for p in [Path(summary.get("intrinsics_file", "__missing_intrinsics__")), capture_folder / "intrinsics.json"] if p.is_file()), capture_folder / "intrinsics.json")
            intr = Intrinsics.load(intr_path)
            with Image.open(rgb_path) as image:
                rgb = image.convert("RGB")
            mask_path = next((p for p in [folder / "mask_selected.png", folder / "mask.png",
                                            Path(summary.get("mask_file", "__missing_mask__")),
                                            folder / "mask_used.png"] if p.is_file()), None)
            overlay_mask = None
            if mask_path is not None:
                with Image.open(mask_path) as image:
                    overlay_mask = np.asarray(image.convert("L")) > 127
            with np.load(folder / "pairs.npz", allow_pickle=False) as data:
                from .gl_scene import launch, wait
                scene_process = launch(folder) if not args.no_show and "mesh_vertices" in data else None
                if not args.no_show and args.matplotlib:
                    from types import SimpleNamespace
                    plot_3d(folder, SimpleNamespace(**{k: data[k] for k in ("midpoints", "left", "right", "valid")}),
                            data["input_left"], data["input_right"], show=True, intrinsics=intr,
                            display=tuple(data[k] for k in ("smoothed_left", "smoothed_right", "smoothed_midpoints"))
                            if "smoothed_left" in data else None,
                            mesh=tuple(data[k] for k in ("mesh_vertices", "mesh_faces", "pipe_axis", "pipe_radius_m"))
                            if "mesh_vertices" in data else None)
                left_uv = data["left_uv"]
                right_uv = data["right_uv"]
                if "smoothed_left" in data:
                    from .gl_overlay import project_points
                    left_uv = project_points(data["smoothed_left"], intr)
                    right_uv = project_points(data["smoothed_right"], intr)
                renderer = render_overlay(rgb, intr, left_uv, right_uv,
                                          data["smoothed_midpoints"] if "smoothed_midpoints" in data else data["midpoints"],
                                          paired_left=data["smoothed_left"] if "smoothed_left" in data else data["left"],
                                          paired_right=data["smoothed_right"] if "smoothed_right" in data else data["right"],
                                          converged=summary.get("converged", False),
                                          output=folder / "rgb_overlay.png", show=not args.no_show, mask=overlay_mask,
                                          inferred_count=summary.get("inferred_stations"),
                                          fit_status=("Surface fit accepted" if summary.get("pipe_mesh",{}).get("refinement_accepted") else
                                                      "UNREFINED MODEL: edge estimate only; surface alignment not validated") if "mesh_vertices" in data else None,
                                          mesh=(data["mesh_vertices"], data["mesh_faces"]) if "mesh_vertices" in data else None,
                                          rings=cross_section_rings(data["pipe_axis"],float(data["pipe_radius_m"])) if "pipe_axis" in data else None,
                                          event_pump=pump_3d if not args.no_show and args.matplotlib else None)
            wait(scene_process, pump_3d if args.matplotlib else None)
            if not args.no_show and args.matplotlib:
                wait_for_3d()
            print(f"OpenGL: {renderer} | Saved {folder / 'rgb_overlay.png'}")
            return 0
        if args.command == "camera":
            args.out = str(Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if args.command == "resume":
            args.out = args.result
            capture_folder=Path(args.result)/"capture"
            args.rgb,args.depth,args.intrinsics=(capture_folder/name for name in ("rgb.png","depth.npy","intrinsics.json"))
            if not all(Path(p).is_file() for p in (args.rgb,args.depth,args.intrinsics)):
                raise ValueError("Resume requires capture/rgb.png, depth.npy and intrinsics.json in the result folder.")
        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        if args.command == "camera":
            from .camera import capture
            args.rgb, args.depth, args.intrinsics = capture(folder / "capture", preview=not args.no_preview,
                                                           timeout=args.timeout,burst_frames=args.burst_frames)
            if args.capture_only:
                return 0
            print("Running SAM3 mask extraction and edge pairing...", flush=True)
        extra, meta, guide = {}, {}, None
        edge_parameters = None
        uncertainty = None
        if args.command == "edges":
            with np.load(args.input, allow_pickle=False) as data:
                left, right = data["left"], data["right"]
                guide = data["guide"] if "guide" in data else None
                if "edge_station_parameter" in data:
                    edge_parameters = (data["edge_station_parameter"], data["edge_station_parameter"])
            meta = {"source": str(Path(args.input).resolve())}
        else:
            from .vision import Intrinsics, extract_edges, lift_edge_adaptive, select_mask_region
            progress("Input", "Loading RGB, depth and calibration...")
            intr = Intrinsics.load(args.intrinsics)
            with Image.open(args.rgb) as image:
                rgb = image.convert("RGB")
            if rgb.size != (intr.width, intr.height):
                raise ValueError("RGB dimensions disagree with intrinsics.")
            if Path(args.depth).suffix.lower() == ".npy":
                depth = np.load(args.depth, allow_pickle=False)
            else:
                with Image.open(args.depth) as image:
                    depth = np.asarray(image).copy()
            if depth.shape != (intr.height, intr.width):
                raise ValueError("Depth must be single-channel and aligned to the calibrated RGB grid.")
            stats_path = Path(getattr(args,"depth_stats",None) or Path(args.depth).with_name("depth_stats.npz"))
            if stats_path.exists():
                with np.load(stats_path,allow_pickle=False) as stats:
                    uncertainty={k:stats[k].copy() for k in ("std_m","valid_count","frame_count")}
                if uncertainty["std_m"].shape!=depth.shape or uncertainty["valid_count"].shape!=depth.shape:
                    raise ValueError("Depth statistics must match the depth image dimensions.")
                if float(uncertainty["frame_count"])<3:
                    raise ValueError("Invalid depth burst frame count.")
                progress("Depth",f"Using temporal uncertainty from {int(uncertainty['frame_count'])} frames.")
            elif getattr(args,"depth_stats",None):
                raise ValueError(f"Depth statistics file not found: {stats_path}")
            else:
                progress("Depth","Single-frame input; temporal uncertainty unavailable.")
            if args.mask:
                progress("Mask", f"Loading saved mask: {args.mask}")
                with Image.open(args.mask) as image:
                    mask = np.asarray(image.convert("L")) > 127
                meta = {"source": "saved mask", "mask_file": str(Path(args.mask).resolve())}
            else:
                from .segmentation import extract_mask
                mask, meta = extract_mask(args.rgb, folder, prompt=args.prompt, checkpoint=args.checkpoint,
                                          threshold=args.threshold, instance=args.instance, seed=args.seed,
                                          interactive=args.show,
                                          candidates_file=folder/"sam3_candidates.npz"
                                          if args.command=="resume" and (folder/"sam3_candidates.npz").is_file() else None)
            if mask.shape != depth.shape:
                raise ValueError("Mask and depth dimensions differ.")
            if args.edge_samples < 16:
                raise ValueError("--edge-samples must be >=16.")
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/"mask_instance.png")
            mask,region_info=select_mask_region(mask,region=args.region,seed=args.seed,rgb=rgb,interactive=args.show)
            meta.update(region_info)
            if region_info["excluded_mask_pixels"]:
                progress("Mask",f"Using visible section {region_info['selected_region']}; excluded {region_info['excluded_mask_pixels']} pixels outside it.")
            selected_mask = mask.copy()
            Image.fromarray(selected_mask.astype(np.uint8)*255).save(folder / "mask_selected.png")
            progress("Edges", "Tracing the two pipe boundaries...")
            mask, guide_uv, uv_l, uv_r = extract_edges(mask, args.edge_samples)
            Image.fromarray(mask.astype(np.uint8)*255).save(folder / "mask_used.png")
            progress("Depth", "Mapping edge pixels into 3D...")
            left, sample_l, status_l, inset_l = lift_edge_adaptive(uv_l, guide_uv, mask, depth, intr,
                                                inset_px=args.inset_px, max_inset_px=max(args.max_inset_px, args.inset_px), max_depth_spread_m=args.max_depth_spread_m)
            right, sample_r, status_r, inset_r = lift_edge_adaptive(uv_r, guide_uv, mask, depth, intr,
                                                 inset_px=args.inset_px, max_inset_px=max(args.max_inset_px, args.inset_px), max_depth_spread_m=args.max_depth_spread_m)
            # Preserve image-space gaps/discontinuities in the 3D polyline.
            for uv, xyz, status in ((uv_l, left, status_l), (uv_r, right, status_r)):
                steps = np.linalg.norm(np.diff(uv, axis=0), axis=1)
                typical = np.nanmedian(steps)
                jumps = np.flatnonzero(steps > max(5., 5*typical)) + 1
                xyz[jumps] = np.nan
                status[jumps] = "image_edge_jump"
            parameter = np.linspace(0., 1., len(guide_uv))
            edge_parameters = (parameter, parameter)
            progress("Depth", f"Supported samples: left {np.isfinite(left[:,2]).sum()}/{len(left)}, "
                     f"right {np.isfinite(right[:,2]).sum()}/{len(right)}; "
                     f"background/outlier rejections {np.count_nonzero(status_l == 'depth_outlier') + np.count_nonzero(status_r == 'depth_outlier')}")
            extra = {"edge_station_parameter": parameter, "left_inset_px": inset_l, "right_inset_px": inset_r, "left_uv": uv_l, "right_uv": uv_r, "guide_uv": guide_uv,
                     "left_sample_uv": sample_l, "right_sample_uv": sample_r,
                     "left_depth_status": status_l, "right_depth_status": status_r,
                     "left_rays": intr.deproject(uv_l, np.ones(len(uv_l))),
                     "right_rays": intr.deproject(uv_r, np.ones(len(uv_r)))}
            # Save extraction even if insufficient depth prevents pairing.
            np.savez_compressed(folder / "edges.npz", left=left, right=right, **extra)
            meta.update(rgb_file=str(Path(args.rgb).resolve()), intrinsics_file=str(Path(args.intrinsics).resolve()),
                        frame="color camera: X right, Y down, Z forward", inset_px=args.inset_px,
                        observation="supported near-edge surface samples; not exact silhouette tangency",
                        max_inset_px=max(args.max_inset_px, args.inset_px),
                        parameterization="shared image-guide stations",
                        left_depth_status_counts=dict(zip(*[v.tolist() for v in np.unique(status_l, return_counts=True)])),
                        right_depth_status_counts=dict(zip(*[v.tolist() for v in np.unique(status_r, return_counts=True)])),
                        valid_left_depth=int(np.isfinite(left).all(axis=1).sum()),
                        valid_right_depth=int(np.isfinite(right).all(axis=1).sum()))
        if not np.isfinite(args.line_smoothing_mm) or args.line_smoothing_mm < 0:
            raise ValueError("--line-smoothing-mm must be finite and nonnegative.")
        fitted = None
        pair_left, pair_right = left, right
        if args.line_smoothing_mm > 0:
            from .smoothing import fit_pipe_edges
            progress("Spline", "Fitting continuous coupled edges to all available depth samples...")
            fitted = fit_pipe_edges(left, right, parameters=edge_parameters,
                                    window_m=args.line_smoothing_mm/1000)
            pair_left, pair_right = fitted["left"], fitted["right"]
            edge_parameters = (fitted["grid"], fitted["grid"])
            extra.update(spline_left=pair_left, spline_right=pair_right,
                         spline_parameter_domain=fitted["domain"], spline_sample_weights=fitted["weights"])
        progress("Pairing", "Refining cross-section edge pairs...")
        result = pair_edges(pair_left, pair_right, stations=args.stations, iterations=args.iterations,
                            smooth_sigma=args.smooth_sigma, search_window=args.search_window,
                            max_segment_m=args.max_segment_m, initial_guide=guide,
                            tolerance_m=args.tolerance_m, refine_guide=not getattr(args, "fixed_guide", False), progress_callback=pairing_progress, edge_parameters=edge_parameters)
        progress("Export", "Saving paired points and 3D plot...")
        if fitted is not None:
            from .smoothing import continuous_pairs
            display_left, display_right, display_mid, inferred = continuous_pairs(fitted, result)
        else:
            display_left, display_right, display_mid = result.left, result.right, result.midpoints
            inferred = np.zeros((len(result.valid), 2), dtype=bool)
        smoothed = np.full(len(result.valid), fitted is not None)
        extra.update(smoothed_left=display_left, smoothed_right=display_right,
                     smoothed_midpoints=display_mid, smoothing_applied=smoothed,
                     spline_inferred_edges=inferred, spline_inferred_midpoints=inferred.any(axis=1))
        meta.update(line_smoothing_window_mm=args.line_smoothing_mm,
                    smoothed_stations=int(smoothed.sum()),
                    inferred_stations=int(inferred.any(axis=1).sum()),
                    output_type="continuous spline edges / approximate midpoint guide" if fitted is not None else "measured edge pairing / midpoint guide",
                    smoothing="robust coupled cubic splines from all edge samples" if fitted is not None else "disabled",
                    pairing_source="fitted edges" if fitted is not None else "measured edges")
        progress("Spline", f"{smoothed.sum()} fitted stations; {inferred.any(axis=1).sum()} span missing depth or estimated correspondence. Raw edge samples retained.")
        if fitted is not None:
            from .pipe_mesh import fit_pipe, masked_points, sphere_sweep, save_mesh, simplify_mesh, cut_open_ends
            progress("Pipe", "Estimating radius and refining the axis against masked depth...")
            surface, surface_weights = (masked_points(depth,selected_mask,intr,uncertainty=uncertainty,return_weights=True)
                                        if args.command!="edges" else (None,None))
            pipe_axis, radius, pipe_info = fit_pipe(display_left, display_right,
                surface=surface, inferred=inferred.any(axis=1),surface_weights=surface_weights if uncertainty is not None else None)
            if uncertainty is not None:
                pipe_info["depth_burst_frames"]=int(uncertainty["frame_count"])
            progress("Pipe", f"Radius {radius*1000:.2f} mm / diameter {radius*2000:.2f} mm; {pipe_info['method']}.")
            if pipe_info.get("refinement_note"):
                progress("Pipe", pipe_info["refinement_note"])
            progress("Mesh", "Sweeping the sphere along the axis and extracting the surface...")
            vertices, faces, pitch = sphere_sweep(pipe_axis, radius)
            full_vertices, full_faces = cut_open_ends(vertices, faces, pipe_axis)
            save_mesh(folder, full_vertices, full_faces, "pipe_mesh_full.ply")
            progress("Mesh", f"Simplifying {len(faces)} triangles with geometry checks...")
            vertices, faces, simplification = simplify_mesh(vertices, faces, radius)
            progress("Mesh", "Cutting away the rounded ends; leaving two open cross-sections...")
            vertices, faces = cut_open_ends(vertices, faces, pipe_axis)
            simplification["stage"] = "before end cuts"
            pipe_info.pop("rounded_ends", None)
            pipe_info.update(ends="open cuts perpendicular to the fitted endpoint tangents", closed=False)
            pipe_info["simplification"] = simplification
            extra.update(pipe_axis=pipe_axis, pipe_radius_m=np.array(radius),
                         mesh_vertices=vertices, mesh_faces=faces)
            pipe_info.update(mesh_voxel_size_m=pitch, mesh_vertices=len(vertices), mesh_triangles=len(faces))
            meta["pipe_mesh"] = pipe_info
            save_mesh(folder, vertices, faces)
            progress("Mesh", f"Saved {len(faces)} triangles to pipe_mesh.ply (metres).")
        if args.command != "edges":
            from .gl_scene import point_cloud
            extra.update(point_cloud(depth, rgb, selected_mask, intr))
            capture_info=json.loads(Path(args.intrinsics).read_text(encoding="utf-8-sig"))
            meta["sampling_time_s"]=capture_info.get("sampling_time_s")
            meta["sampling_time_definition"]="Burst acquisition including motion restarts; excludes warmup, user wait, fusion and model processing."
        summary = save_result(folder, result, left, right, metadata=meta, extra=extra,
                              show=args.show and args.matplotlib, block=False,
                              intrinsics=intr if args.command != "edges" else None)
        print(json.dumps(summary, indent=2))
        progress("Output", str(folder.resolve()))
        progress("Output", f"Latest results: {Path(__file__).resolve().parents[1] / 'results.json'}")
        if result.valid.sum() < 3:
            print("Insufficient paired stations. Inspect status arrays, mask and depth alignment.", flush=True)
        if not result.converged:
            print("Pairing did not converge within the iteration budget; inspect before use.")
        progress("Processing", "Finished. Results ready.")
        from .gl_scene import launch, wait
        scene_process = launch(folder) if args.show and "mesh_vertices" in extra else None
        if args.command != "edges":
            from .gl_overlay import render_overlay, project_points
            progress("OpenGL", "RGB and mesh/point-cloud windows ready; close both to finish." if args.show else "Saving RGB overlay...")
            renderer = render_overlay(rgb, intr, project_points(display_left, intr),
                                      project_points(display_right, intr), display_mid,
                                      paired_left=display_left, paired_right=display_right,
                                      converged=result.converged, output=folder / "rgb_overlay.png", show=args.show, mask=selected_mask,
                                      inferred_count=int(inferred.any(axis=1).sum()) if fitted is not None else None,
                                      fit_status=("Surface fit accepted" if meta.get("pipe_mesh",{}).get("refinement_accepted") else
                                                  "UNREFINED MODEL: edge estimate only; surface alignment not validated") if "mesh_vertices" in extra else None,
                                      mesh=(extra["mesh_vertices"], extra["mesh_faces"]) if "mesh_vertices" in extra else None,
                                      rings=cross_section_rings(extra["pipe_axis"],float(extra["pipe_radius_m"])) if "pipe_axis" in extra else None,
                                      event_pump=pump_3d if args.show and args.matplotlib else None)
            print(f"OpenGL overlay renderer: {renderer}")
        wait(scene_process, pump_3d if args.matplotlib else None)
        if args.show and args.matplotlib:
            wait_for_3d()
        progress("Done", "Processing finished.")
        return 0 if result.valid.sum() >= 3 else 2
    except KeyboardInterrupt:
        print("Capture cancelled.")
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
