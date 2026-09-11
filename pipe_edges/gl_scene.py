"""Independent OpenGL mesh/point-cloud window; shares saved camera coordinates."""
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
import json
import numpy as np
from .progress import progress


def point_cloud(depth, rgb, mask, intrinsics):
    z = np.asarray(depth,float)*intrinsics.depth_scale
    y,x = np.indices(z.shape)
    good = np.isfinite(z) & (z > 0) & ((x % 2)==0) & ((y % 2)==0)
    y,x = y[good],x[good]
    return {"cloud_points": intrinsics.deproject(np.column_stack([x,y]),z[y,x]),
            "cloud_colors": np.asarray(rgb)[y,x,:3].astype(np.uint8),
            "cloud_pipe_mask": np.asarray(mask,bool)[y,x]}


def launch(folder):
    main = Path(__file__).resolve().parents[1]/"main.py"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen([sys.executable,"-u","-B",str(main),"scene","--result",str(Path(folder).resolve())],
                            creationflags=flags)


def wait(process, pump=None):
    if process is None:
        return
    while process.poll() is None:
        if pump:
            pump()
        time.sleep(.03)
    if process.returncode:
        raise RuntimeError(f"OpenGL 3D viewer exited with code {process.returncode}.")


def show_scene(folder, *, visible=True, test_frames=None):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message=r"^pkg_resources is deprecated as an API\.",category=UserWarning)
        import pygame as pg
    from OpenGL import GL as gl, GLU as glu
    from .gl_hud import InstructionsPanel
    folder = Path(folder)
    summary=json.loads((folder/"summary.json").read_text()) if (folder/"summary.json").exists() else {}
    fit_info=summary.get("pipe_mesh",{})
    fit_status=("Surface fit accepted" if fit_info.get("refinement_accepted") else
                "UNREFINED MODEL: edge estimate only; surface alignment not validated")
    with np.load(folder/"pairs.npz",allow_pickle=False) as data:
        v,f = data["mesh_vertices"],data["mesh_faces"]
        axis = data["pipe_axis"]
        from .pipe_mesh import cross_section_rings
        rings=cross_section_rings(axis,float(data["pipe_radius_m"]))
        cloud = np.asarray(data["cloud_points"],np.float32) if "cloud_points" in data else np.concatenate([data["input_left"],data["input_right"]]).astype(np.float32)
        colors = data["cloud_colors"].astype(np.float32)/255 if "cloud_colors" in data else np.full(cloud.shape,.85,np.float32)
        pipe = data["cloud_pipe_mask"] if "cloud_pipe_mask" in data else np.ones(len(cloud),bool)
    good = np.isfinite(cloud).all(axis=1)
    cloud,colors,pipe = cloud[good],colors[good],pipe[good]
    cloud,colors = np.ascontiguousarray(cloud),np.ascontiguousarray(colors)
    pipe_cloud,pipe_colors = np.ascontiguousarray(cloud[pipe]),np.ascontiguousarray(colors[pipe])
    triangles = np.ascontiguousarray(v[f].reshape(-1,3),dtype=np.float32)
    e = np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    edges = np.ascontiguousarray(v[e].reshape(-1,3),dtype=np.float32)
    ring_lines=np.ascontiguousarray(np.stack([rings[:,:-1],rings[:,1:]],axis=2).reshape(-1,3),dtype=np.float32)
    focus = (v.min(axis=0)+v.max(axis=0))/2
    extent = max(float(np.linalg.norm(np.ptp(v,axis=0))),.05)
    center = focus.copy()
    distance,yaw,pitch = extent*1.6,0.,0.
    cloud_on,mesh_on,context,solid = True,True,False,False
    rings_on=True
    panel = None
    pg.display.init()
    try:
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION,2)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION,1)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE,24)
        pg.display.set_mode((1100,760),pg.OPENGL|pg.DOUBLEBUF|(pg.RESIZABLE if visible else pg.HIDDEN))
        pg.display.set_caption("F2F - Mesh and point cloud")
        panel = InstructionsPanel(pg,gl)
        progress("OpenGL 3D", f"{len(f)} triangles, {len(pipe_cloud)} pipe points / {len(cloud)} scene points. "
                 +gl.glGetString(gl.GL_RENDERER).decode())
        def vertices(points, mode):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3,gl.GL_FLOAT,0,points)
            gl.glDrawArrays(mode,0,len(points))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        clock = pg.time.Clock()
        running,frames = True,0
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running=False
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_ESCAPE,pg.K_q): running=False
                    elif event.key == pg.K_p: cloud_on=not cloud_on
                    elif event.key == pg.K_m: mesh_on=not mesh_on
                    elif event.key == pg.K_c: context=not context
                    elif event.key == pg.K_w: solid=not solid
                    elif event.key == pg.K_x: rings_on=not rings_on
                    elif event.key == pg.K_r: center,distance,yaw,pitch=focus.copy(),extent*1.6,0.,0.
                elif event.type == pg.MOUSEWHEEL:
                    distance=float(np.clip(distance*np.exp(-event.y*.12),extent*.12,extent*30))
                elif event.type == pg.MOUSEMOTION:
                    dx,dy=event.rel
                    if event.buttons[0]:
                        yaw-=dx*.006
                        pitch=float(np.clip(pitch-dy*.006,-1.5,1.5))
                    elif event.buttons[2]:
                        right=np.array([np.cos(yaw),0,np.sin(yaw)])
                        center-=right*dx*distance*.0015
                        center[1]-=dy*distance*.0015
            w,h=pg.display.get_window_size()
            gl.glViewport(0,0,w,h)
            gl.glClearColor(.045,.055,.07,1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDisable(gl.GL_BLEND)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            glu.gluPerspective(45,w/max(h,1),max(.0001,distance/1000),distance+extent*50)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            eye=center+distance*np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),-np.cos(yaw)*np.cos(pitch)])
            glu.gluLookAt(*eye,*center,0,-1,0)
            if cloud_on:
                points,rgba=(cloud,colors) if context else (pipe_cloud,pipe_colors)
                gl.glPointSize(2.)
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)
                gl.glColorPointer(3,gl.GL_FLOAT,0,rgba)
                vertices(points,gl.GL_POINTS)
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            if mesh_on:
                # Self-occluded wireframe, independent of RGB depth noise.
                gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
                gl.glColorMask(solid,solid,solid,solid)
                gl.glColor3f(.25,.55,.65)
                gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glPolygonOffset(1,1)
                vertices(triangles,gl.GL_TRIANGLES)
                gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glColorMask(True,True,True,True)
                gl.glColor3f(.15,.85,1.)
                gl.glLineWidth(1.)
                vertices(edges,gl.GL_LINES)
            gl.glDisable(gl.GL_DEPTH_TEST)
            if rings_on:
                gl.glColor3f(1.,.12,.12)
                gl.glLineWidth(2.)
                vertices(ring_lines,gl.GL_LINES)
            panel.draw(["Left drag: orbit | Right drag: pan | Wheel: zoom | R: camera orientation",
                        "X: rings | P: point cloud | M: mesh | W: solid / wireframe | C: pipe / whole scene | ESC / Q: close",
                        f"{len(f):,} triangles | {'Scene' if context else 'Pipe'} cloud: {len(cloud) if context else len(pipe_cloud):,} points | Mesh {'on' if mesh_on else 'off'} | Cloud {'on' if cloud_on else 'off'}",
                        fit_status],w,h)
            if frames == 0:
                from PIL import Image
                gl.glPixelStorei(gl.GL_PACK_ALIGNMENT,1)
                raw=gl.glReadPixels(0,0,w,h,gl.GL_RGB,gl.GL_UNSIGNED_BYTE)
                Image.frombytes("RGB",(w,h),raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(folder/"mesh_cloud_3d.png")
            pg.display.flip()
            frames+=1
            if test_frames is not None and frames>=test_frames: running=False
            clock.tick(30)
    finally:
        if panel is not None: panel.close()
        pg.display.quit()
