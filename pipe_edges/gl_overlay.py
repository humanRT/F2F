"""OpenGL RGB texture with calibrated image-space edge and midpoint overlays.

Pygame owns the window/context only. RGB and all overlay geometry are drawn by
OpenGL. Invalid samples split curves; they are never connected across gaps.
"""
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from .gl_hud import InstructionsPanel


def project_points(points, intrinsics):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Projection expects an (N,3) array.")
    uv = np.full((len(points), 2), np.nan)
    good = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-8)
    p = points[good]
    uv[good, 0] = intrinsics.fx*p[:, 0]/p[:, 2] + intrinsics.cx
    uv[good, 1] = intrinsics.fy*p[:, 1]/p[:, 2] + intrinsics.cy
    return uv


def curve_segments(uv):
    """Consecutive valid segments only; do not compact away invalid samples."""
    uv = np.asarray(uv, dtype=float)
    pairs = np.stack([uv[:-1], uv[1:]], axis=1)
    return pairs[np.isfinite(pairs).all(axis=(1, 2))]


def image_viewport(window_width, window_height, image_width, image_height):
    scale = min(window_width/image_width, window_height/image_height)
    w, h = max(1, round(image_width*scale)), max(1, round(image_height*scale))
    return (window_width-w)//2, (window_height-h)//2, w, h


def render_overlay(rgb, intrinsics, left_uv, right_uv, midpoints, *,
                   paired_left=None, paired_right=None, converged=True,
                   output=None, show=False, mask=None, event_pump=None, inferred_count=None, mesh=None, fit_status=None, rings=None):
    """Save a full-resolution GPU-rendered PNG; optionally keep the viewer open.

    K toggles the translucent mask; E/M/P toggle edges/midpoint/pair chords; O toggles all overlays; S saves the
    current viewport; Escape/Q closes. Raw silhouette pixels draw the edges,
    whereas midpoint/pair points are projected from camera-frame 3D coordinates.
    """
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"^pkg_resources is deprecated as an API\.", category=UserWarning, module=r"^pygame\.pkgdata$")
            import pygame
        from OpenGL import GL as gl
    except ImportError as exc:
        raise RuntimeError("OpenGL overlays require pygame and PyOpenGL in SAM3.") from exc
    pixels = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    h, w = pixels.shape[:2]
    if (w, h) != (intrinsics.width, intrinsics.height):
        raise ValueError("Overlay RGB dimensions disagree with camera intrinsics.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError("Mask dimensions disagree with RGB; mask will not be resized.")
    midpoint_uv = project_points(midpoints, intrinsics)
    ring_lines=np.empty((0,2,2))
    if rings is not None:
        ring_uv=project_points(np.asarray(rings).reshape(-1,3),intrinsics).reshape(*np.asarray(rings).shape[:2],2)
        ring_lines=np.concatenate([curve_segments(r) for r in ring_uv])
    pair_segments = np.empty((0, 2, 2))
    if paired_left is not None and paired_right is not None:
        a, b = project_points(paired_left, intrinsics), project_points(paired_right, intrinsics)
        pair_segments = np.stack([a, b], axis=1)
        pair_segments = pair_segments[np.isfinite(pair_segments).all(axis=(1, 2))]
    layers = [curve_segments(left_uv), curve_segments(right_uv), curve_segments(midpoint_uv)]
    midpoint_dots = midpoint_uv[np.isfinite(midpoint_uv).all(axis=1)]
    mesh_vertices = mesh_lines = None
    if mesh is not None:
        vertices, faces = mesh
        triangles = np.asarray(vertices, float)[np.asarray(faces, int)]
        if not np.isfinite(triangles).all():
            raise ValueError("Mesh must contain finite camera-frame coordinates.")
        mesh_vertices = np.ascontiguousarray(triangles.reshape(-1,3), dtype=np.float32)
        faces = np.asarray(faces, int)
        edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]], faces[:,[1,2]],
                                                   faces[:,[2,0]]]),axis=1),axis=0)
        mesh_lines = np.ascontiguousarray(np.asarray(vertices)[edges].reshape(-1,3), dtype=np.float32)
    texture = None
    mask_texture = None
    panel = None
    pygame.display.init()
    try:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.RESIZABLE if show else pygame.HIDDEN)
        pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("F2F - RGB results")
        panel = InstructionsPanel(pygame, gl)
        renderer = gl.glGetString(gl.GL_RENDERER).decode("utf-8", errors="replace")
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
        if mask is not None:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = [170, 65, 245]
            rgba[..., 3] = mask.astype(np.uint8) * 80
            mask_texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, mask_texture)
            for setting in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, setting, gl.GL_NEAREST)
            for setting in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, setting, gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba)
        edges_on, mid_on, pairs_on, overlays_on, mask_on = True, True, False, True, True
        mesh_on, mesh_alpha = True, .7
        rings_on=True

        def draw_mesh():
            # Camera intrinsics define perspective; camera +Y down/+Z forward
            # becomes OpenGL +Y up/-Z forward. Bounds use pixel centers.
            near = .001
            far = max(1., float(np.max(mesh_vertices[:,2]))*2)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPushMatrix()
            gl.glLoadIdentity()
            gl.glFrustum((-intrinsics.cx-.5)*near/intrinsics.fx,
                         (w-.5-intrinsics.cx)*near/intrinsics.fx,
                         -(h-.5-intrinsics.cy)*near/intrinsics.fy,
                         (intrinsics.cy+.5)*near/intrinsics.fy, near, far)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
            gl.glLoadIdentity()
            gl.glScalef(1., -1., -1.)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3,gl.GL_FLOAT,0,mesh_vertices)
            # Filled depth-only pass hides rear edges. A small polygon offset
            # lets coplanar front edges pass without depth-fighting artifacts.
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glColorMask(False,False,False,False)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glPolygonOffset(1., 1.)
            gl.glDrawArrays(gl.GL_TRIANGLES,0,len(mesh_vertices))
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glColorMask(True,True,True,True)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glColor4f(.15,.85,1.,mesh_alpha)
            gl.glLineWidth(1.)
            gl.glVertexPointer(3,gl.GL_FLOAT,0,mesh_lines)
            gl.glDrawArrays(gl.GL_LINES,0,len(mesh_lines))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_MODELVIEW)

        def lines(segments, color, thickness):
            # Triangles give consistent widths on drivers that limit GL_LINES.
            gl.glColor4f(*color)
            gl.glBegin(gl.GL_TRIANGLES)
            for a, b in segments:
                delta = b-a
                length = np.linalg.norm(delta)
                if length < 1e-8:
                    continue
                offset = np.array([-delta[1], delta[0]]) * (thickness/2/length)
                for vertex in (a-offset, a+offset, b+offset, a-offset, b+offset, b-offset):
                    gl.glVertex2f(*vertex)
            gl.glEnd()

        def draw():
            ww, wh = pygame.display.get_window_size()
            gl.glViewport(0, 0, ww, wh)
            gl.glClearColor(.06, .06, .06, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glViewport(*image_viewport(ww, wh, w, h))
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(-.5, w-.5, h-.5, -.5, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glColor4f(1, 1, 1, 1)
            gl.glBegin(gl.GL_QUADS)
            for tex, vertex in [((0,0),(-.5,-.5)), ((1,0),(w-.5,-.5)),
                                ((1,1),(w-.5,h-.5)), ((0,1),(-.5,h-.5))]:
                gl.glTexCoord2f(*tex)
                gl.glVertex2f(*vertex)
            gl.glEnd()
            if overlays_on and mask_on and mask_texture is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, mask_texture)
                gl.glBegin(gl.GL_QUADS)
                for tex, vertex in [((0,0),(-.5,-.5)), ((1,0),(w-.5,-.5)),
                                    ((1,1),(w-.5,h-.5)), ((0,1),(-.5,h-.5))]:
                    gl.glTexCoord2f(*tex)
                    gl.glVertex2f(*vertex)
                gl.glEnd()
            gl.glDisable(gl.GL_TEXTURE_2D)
            if overlays_on:
                if mesh_on and mesh_vertices is not None:
                    draw_mesh()
                if pairs_on:
                    lines(pair_segments, (1,1,1,.45), 1.)
                if edges_on:
                    lines(layers[0], (.15,.65,1,1), 2.5)
                    lines(layers[1], (1,.48,.08,1), 2.5)
                if mid_on:
                    lines(layers[2], (.12,1,.3,1), 3.)
                    gl.glPointSize(4.)
                    gl.glColor4f(.12,1,.3,1)
                    gl.glBegin(gl.GL_POINTS)
                    for p in midpoint_dots:
                        gl.glVertex2f(*p)
                    gl.glEnd()
            state = "stable" if converged else "NOT CONVERGED"
            if overlays_on and rings_on:
                lines(ring_lines,(1.,.12,.12,1.),1.8)
            panel.draw([
                "X: rings   T: wireframe   [ / ]: opacity   K: mask   E: edges   M: midpoint   P: pairs   O: overlays   S: save   ESC / Q: close",
                "Cyan: visible mesh edges | Purple: SAM3 mask | Blue/orange: edges | Green: midpoint guide",
                f"Pairing: {state} | {len(midpoint_dots)}/{len(midpoints)} drawn"
                + (f" ({inferred_count} inferred)" if inferred_count is not None else "") + " | "
                f"Mask {'on' if mask_on else 'off'} | Edges {'on' if edges_on else 'off'} | "
                f"Midpoint {'on' if mid_on else 'off'} | Wireframe {f'{mesh_alpha:.0%}' if mesh_on and mesh_vertices is not None else 'off'}",
            ] + ([fit_status] if fit_status else []), ww, wh)
            return ww, wh

        def save(path, dimensions):
            gl.glReadBuffer(gl.GL_BACK)
            data = gl.glReadPixels(0, 0, *dimensions, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            image = Image.frombytes("RGB", dimensions, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)

        dimensions = draw()
        if output:
            save(output, dimensions)
        pygame.display.flip()
        clock = pygame.time.Clock()
        running = show
        while running:
            if event_pump is not None:
                event_pump()
            save_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_k:
                        mask_on = not mask_on
                    elif event.key == pygame.K_t:
                        mesh_on = not mesh_on
                    elif event.key == pygame.K_x:
                        rings_on = not rings_on
                    elif event.key == pygame.K_LEFTBRACKET:
                        mesh_alpha = max(0., mesh_alpha-.1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        mesh_alpha = min(1., mesh_alpha+.1)
                    elif event.key == pygame.K_e:
                        edges_on = not edges_on
                    elif event.key == pygame.K_m:
                        mid_on = not mid_on
                    elif event.key == pygame.K_p:
                        pairs_on = not pairs_on
                    elif event.key == pygame.K_o:
                        overlays_on = not overlays_on
                    elif event.key == pygame.K_s:
                        save_requested = True
            dimensions = draw()
            if save_requested and output:
                save(Path(output).with_name("rgb_overlay_view.png"), dimensions)
            pygame.display.flip()
            clock.tick(30)
        return renderer
    except pygame.error as exc:
        raise RuntimeError(f"Cannot create OpenGL overlay window: {exc}") from exc
    finally:
        if panel is not None:
            panel.close()
        if mask_texture is not None:
            gl.glDeleteTextures([mask_texture])
        if texture is not None:
            gl.glDeleteTextures([texture])
        pygame.display.quit()
