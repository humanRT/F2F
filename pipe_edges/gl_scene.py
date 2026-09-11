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


class EndRingDrag:
    """Screen-space picking with motion constrained to axis arc length."""

    def __init__(self, axis, radius):
        self.axis=np.asarray(axis,float)
        self.radius=radius
        self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.axis,axis=0),axis=1))]
        self.arc/=self.arc[-1]
        self.ends=np.array([0.,1.])
        self.active=None
        self.project=None
        self.update()

    def update(self):
        from .pipe_mesh import cross_section_rings
        self.positions=np.linspace(*self.ends,12)
        self.rings=cross_section_rings(self.axis,self.radius,positions=self.positions)
        self.lines=np.ascontiguousarray(np.stack([self.rings[:,:-1],self.rings[:,1:]],axis=2).reshape(-1,3),dtype=np.float32)

    def centers(self, positions):
        return np.column_stack([np.interp(positions,self.arc,self.axis[:,j]) for j in range(3)])

    def cut_mesh(self, vertices, faces):
        from .pipe_mesh import cut_open_ends
        if np.array_equal(self.ends,[0.,1.]):
            return vertices,faces
        tangent=np.gradient(self.axis,self.arc,axis=0)
        tangent=np.column_stack([np.interp(self.ends,self.arc,tangent[:,j]) for j in range(3)])
        return cut_open_ends(vertices,faces,self.centers(self.ends),tangents=tangent)

    def set_view(self, model, projection, viewport, height):
        # PyOpenGL returns column-major matrices; row vectors multiply them directly.
        matrix=np.asarray(model)@np.asarray(projection)
        x,y,w,h=viewport
        def project(points):
            clip=np.column_stack([points,np.ones(len(points))])@matrix
            with np.errstate(divide="ignore",invalid="ignore"):
                ndc=clip[:,:3]/clip[:,3,None]
            screen=np.column_stack([x+(ndc[:,0]+1)*w/2,height-y-(ndc[:,1]+1)*h/2])
            screen[(clip[:,3]<=0)|(np.abs(ndc[:,2])>1)]=np.nan
            return screen
        self.project=project

    def begin(self, mouse):
        if self.project is None: return False
        distances=[]
        for ring in self.rings[[0,-1]]:
            p=self.project(ring)
            a,b=p[:-1],p[1:]
            ab=b-a
            u=np.clip(np.sum((mouse-a)*ab,axis=1)/np.maximum(np.sum(ab*ab,axis=1),1e-12),0,1)
            d=np.linalg.norm(a+u[:,None]*ab-mouse,axis=1)
            distances.append(np.min(np.where(np.isfinite(d),d,np.inf)))
        index=int(np.argmin(distances))
        if distances[index]>12: return False
        center=self.project(self.centers([self.ends[index]]))[0]
        if not np.isfinite(center).all(): return False
        self.active=index
        self.offset=np.asarray(mouse)-center
        return True

    def move(self, mouse):
        if self.active is None: return
        # Dense arc samples avoid depth-dependent interpolation of projected segments.
        stations=np.unique(np.r_[np.linspace(0,1,2001),self.ends[self.active]])
        points=self.project(self.centers(stations))
        error=np.linalg.norm(points-(np.asarray(mouse)-self.offset),axis=1)
        error=np.where(np.isfinite(error),error,np.inf)
        if not np.isfinite(error).any(): return
        # At projected crossings, prefer the nearby branch instead of jumping along the tube.
        candidates=np.flatnonzero(error<=np.min(error)+.5)
        station=stations[candidates[np.argmin(np.abs(stations[candidates]-self.ends[self.active]))]]
        low,high=(0,self.ends[1]-.005) if self.active==0 else (self.ends[0]+.005,1)
        self.ends[self.active]=np.clip(station,low,high)
        self.update()


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
    rgb_pixels=intrinsics=None
    from PIL import Image
    from .vision import Intrinsics
    rgb_path=next((p for p in [Path(summary.get("rgb_file","__missing__")),folder/"capture"/"rgb.png",folder.parent/"capture"/"rgb.png"] if p.is_file()),None)
    intr_path=next((p for p in [Path(summary.get("intrinsics_file","__missing__")),folder/"capture"/"intrinsics.json",folder.parent/"capture"/"intrinsics.json"] if p.is_file()),None)
    if rgb_path is not None and intr_path is not None:
        intrinsics=Intrinsics.load(intr_path)
        with Image.open(rgb_path) as image:
            rgb_pixels=np.asarray(image.convert("RGB")).copy()
        if rgb_pixels.shape[:2]!=(intrinsics.height,intrinsics.width):
            raise ValueError("Saved RGB dimensions do not match calibration.")
    fit_status=("Surface fit accepted" if fit_info.get("refinement_accepted") else
                "UNREFINED MODEL: edge estimate only; surface alignment not validated")
    with np.load(folder/"pairs.npz",allow_pickle=False) as data:
        v,f = data["mesh_vertices"],data["mesh_faces"]
        axis = data["pipe_axis"]
        ring_drag=EndRingDrag(axis,float(data["pipe_radius_m"]))
        cloud = np.asarray(data["cloud_points"],np.float32) if "cloud_points" in data else np.concatenate([data["input_left"],data["input_right"]]).astype(np.float32)
        colors = data["cloud_colors"].astype(np.float32)/255 if "cloud_colors" in data else np.full(cloud.shape,.85,np.float32)
        pipe = data["cloud_pipe_mask"] if "cloud_pipe_mask" in data else np.ones(len(cloud),bool)
    good = np.isfinite(cloud).all(axis=1)
    cloud,colors,pipe = cloud[good],colors[good],pipe[good]
    cloud,colors = np.ascontiguousarray(cloud),np.ascontiguousarray(colors)
    pipe_cloud,pipe_colors = np.ascontiguousarray(cloud[pipe]),np.ascontiguousarray(colors[pipe])
    triangles = np.ascontiguousarray(v[f].reshape(-1,3),dtype=np.float32)
    from .gl_lighting import triangle_normals, enable_solid_lighting, disable_solid_lighting
    normals=triangle_normals(v,f)
    e = np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    edges = np.ascontiguousarray(v[e].reshape(-1,3),dtype=np.float32)
    original_v,original_f=v.copy(),f.copy()
    applied_ends=ring_drag.ends.copy()
    save_cut=False
    from .surface_probe import SurfaceProbe
    probe=SurfaceProbe(v,f)
    placed_points=[None,None]
    selected_point=0
    editing_point=False
    mouse=None
    orbit_drag=False
    focus = (v.min(axis=0)+v.max(axis=0))/2
    extent = max(float(np.linalg.norm(np.ptp(v,axis=0))),.05)
    center = focus.copy()
    distance,yaw,pitch = extent*1.6,0.,0.
    cloud_on,mesh_on,context,solid = True,True,False,False
    rings_on=True
    rgb_on=camera_view=rgb_pixels is not None
    rgb_texture=None
    if intrinsics is not None:
        image_depth=max(float(v[:,2].max())*1.15,.1)
        image_plane=intrinsics.deproject(np.array([[-.5,-.5],[intrinsics.width-.5,-.5],
                      [intrinsics.width-.5,intrinsics.height-.5],[-.5,intrinsics.height-.5]]),np.full(4,image_depth))
    panel = None
    dimension_labels=None
    pg.display.init()
    try:
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION,2)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION,1)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE,24)
        from .gl_context import create_window
        create_window(pg,gl,(1100,760),pg.OPENGL|pg.DOUBLEBUF|(pg.RESIZABLE if visible else pg.HIDDEN))
        pg.display.set_caption("F2F - Mesh and point cloud")
        panel = InstructionsPanel(pg,gl)
        from .gl_labels import DimensionLabels
        dimension_labels=DimensionLabels(pg,gl)
        if rgb_pixels is not None:
            rgb_texture=gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D,rgb_texture)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
            for parameter in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D,parameter,gl.GL_LINEAR)
            for parameter in (gl.GL_TEXTURE_WRAP_S,gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D,parameter,gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGB8,intrinsics.width,intrinsics.height,
                            0,gl.GL_RGB,gl.GL_UNSIGNED_BYTE,rgb_pixels)
        progress("OpenGL 3D", f"{len(f)} triangles, {len(pipe_cloud)} pipe points / {len(cloud)} scene points. "
                 +gl.glGetString(gl.GL_RENDERER).decode())
        def vertices(points, mode):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3,gl.GL_FLOAT,0,points)
            gl.glDrawArrays(mode,0,len(points))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        clock = pg.time.Clock()
        running,frames = True,0
        hud_on=False
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running=False
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_ESCAPE,pg.K_q): running=False
                    elif event.key == pg.K_F1: hud_on=not hud_on
                    elif event.key in (pg.K_1,pg.K_2): selected_point=event.key-pg.K_1
                    elif event.key == pg.K_p: cloud_on=not cloud_on
                    elif event.key == pg.K_m: mesh_on=not mesh_on
                    elif event.key == pg.K_c: context=not context
                    elif event.key == pg.K_w: solid=not solid
                    elif event.key == pg.K_x:
                        rings_on=not rings_on
                        ring_drag.active=None
                    elif event.key == pg.K_HOME:
                        ring_drag.ends[:]=[0,1]
                        ring_drag.active=None
                        ring_drag.update()
                        save_cut=True
                    elif event.key == pg.K_i: rgb_on=not rgb_on
                    elif event.key == pg.K_v and intrinsics is not None: camera_view=True
                    elif event.key == pg.K_r:
                        center,distance,yaw,pitch=focus.copy(),extent*1.6,0.,0.
                        camera_view=False
                elif event.type == pg.MOUSEWHEEL:
                    if ring_drag.active is not None or probe.active: continue
                    camera_view=False
                    distance=float(np.clip(distance*np.exp(-event.y*.12),extent*.12,extent*30))
                elif event.type == pg.MOUSEBUTTONDOWN and event.button==1:
                    mouse=event.pos
                    orbit_drag=False
                    editing_point=False
                    nearby=[]
                    if mesh_on and ring_drag.project is not None:
                        for index,point in enumerate(placed_points):
                            if point is not None:
                                screen=ring_drag.project(point[0][None])[0]
                                distance_px=np.linalg.norm(screen-np.asarray(event.pos))
                                if np.isfinite(distance_px) and distance_px<=14:
                                    nearby.append((distance_px,index))
                    if nearby:
                        selected_point=min(nearby)[1]
                        editing_point=True
                        probe.active=True
                    elif pg.key.get_mods() & pg.KMOD_ALT:
                        probe.active=True
                        if mesh_on and probe.pick(event.pos):
                            placed_points[selected_point]=tuple(value.copy() for value in probe.hit)
                            if selected_point==0 and placed_points[1] is None:
                                selected_point=1
                    elif rings_on and ring_drag.begin(event.pos):
                        probe.active=False
                    else:
                        orbit_drag=True
                elif event.type == pg.MOUSEBUTTONUP and event.button==1:
                    save_cut=ring_drag.active is not None
                    ring_drag.active=None
                    probe.active=False
                    editing_point=False
                    orbit_drag=False
                elif event.type == pg.WINDOWFOCUSLOST:
                    save_cut=ring_drag.active is not None
                    ring_drag.active=None
                    probe.active=False
                    editing_point=False
                    orbit_drag=False
                    mouse=None
                elif event.type == pg.MOUSEMOTION:
                    mouse=event.pos
                    dx,dy=event.rel
                    if event.buttons[0] and ring_drag.active is not None:
                        ring_drag.move(event.pos)
                    elif event.buttons[0] and probe.active:
                        if editing_point and mesh_on and probe.pick(event.pos):
                            placed_points[selected_point]=tuple(value.copy() for value in probe.hit)
                    elif event.buttons[0]:
                        camera_view=False
                        yaw-=dx*.006
                        pitch=float(np.clip(pitch-dy*.006,-1.5,1.5))
                    elif event.buttons[2]:
                        camera_view=False
                        right=np.array([np.cos(yaw),0,np.sin(yaw)])
                        center-=right*dx*distance*.0015
                        center[1]-=dy*distance*.0015
            if not np.array_equal(applied_ends,ring_drag.ends):
                try:
                    v,f=ring_drag.cut_mesh(original_v,original_f)
                except ValueError as exc:
                    progress("Mesh cut",str(exc))
                    ring_drag.ends[:]=applied_ends
                    ring_drag.update()
                else:
                    applied_ends=ring_drag.ends.copy()
                    triangles=np.ascontiguousarray(v[f].reshape(-1,3),dtype=np.float32)
                    normals=triangle_normals(v,f)
                    e=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
                    edges=np.ascontiguousarray(v[e].reshape(-1,3),dtype=np.float32)
                    probe.set_mesh(v,f)
            if save_cut or (not running and not np.array_equal(applied_ends,[0.,1.])):
                from .pipe_mesh import save_mesh
                save_mesh(folder,v,f,"pipe_mesh_trimmed.ply")
                (folder/"mesh_trim.json").write_text(json.dumps({"start_fraction":float(applied_ends[0]),
                    "end_fraction":float(applied_ends[1]),"mesh_file":"pipe_mesh_trimmed.ply",
                    "vertices":len(v),"triangles":len(f),"ends":"open"},indent=2),encoding="utf-8")
                progress("Mesh cut",f"Saved {len(f)} triangles: {folder / 'pipe_mesh_trimmed.ply'}")
                save_cut=False
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
            if camera_view and intrinsics is not None:
                from .gl_overlay import image_viewport
                gl.glViewport(*image_viewport(w,h,intrinsics.width,intrinsics.height))
                gl.glMatrixMode(gl.GL_PROJECTION)
                gl.glLoadIdentity()
                near,far=.001,max(image_depth*3,1.)
                gl.glFrustum((-intrinsics.cx-.5)*near/intrinsics.fx,
                    (intrinsics.width-.5-intrinsics.cx)*near/intrinsics.fx,
                    -(intrinsics.height-.5-intrinsics.cy)*near/intrinsics.fy,
                    (intrinsics.cy+.5)*near/intrinsics.fy,near,far)
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glLoadIdentity()
                gl.glScalef(1,-1,-1)
            ring_drag.set_view(gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX),gl.glGetDoublev(gl.GL_PROJECTION_MATRIX),
                               gl.glGetIntegerv(gl.GL_VIEWPORT),h)
            probe.set_view(gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX),gl.glGetDoublev(gl.GL_PROJECTION_MATRIX),
                           gl.glGetIntegerv(gl.GL_VIEWPORT),h)
            if mesh_on and mouse is not None and ring_drag.active is None and not orbit_drag and pg.key.get_mods() & pg.KMOD_ALT and not probe.active:
                probe.pick(mouse)
            else:
                probe.hit=placed_points[selected_point] if mesh_on else None
            if rgb_on and rgb_texture is not None:
                # A calibrated reference image plane, not reconstructed scene geometry.
                gl.glDepthMask(gl.GL_FALSE)
                gl.glEnable(gl.GL_TEXTURE_2D)
                gl.glBindTexture(gl.GL_TEXTURE_2D,rgb_texture)
                gl.glColor3f(1,1,1)
                gl.glBegin(gl.GL_QUADS)
                for uv,xyz in zip(((0,0),(1,0),(1,1),(0,1)),image_plane):
                    gl.glTexCoord2f(*uv)
                    gl.glVertex3f(*xyz)
                gl.glEnd()
                gl.glDisable(gl.GL_TEXTURE_2D)
                gl.glDepthMask(gl.GL_TRUE)
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
                if solid:
                    enable_solid_lighting(gl)
                    gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
                    gl.glNormalPointer(gl.GL_FLOAT,0,normals)
                vertices(triangles,gl.GL_TRIANGLES)
                if solid:
                    gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
                    disable_solid_lighting(gl)
                gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glColorMask(True,True,True,True)
                gl.glColor3f(.15,.85,1.)
                gl.glLineWidth(1.)
                if not solid:
                    vertices(edges,gl.GL_LINES)
            gl.glDisable(gl.GL_DEPTH_TEST)
            if probe.hit is not None:
                circle,dot,arrow=probe.marker(ring_drag.radius*.16)
                gl.glColor3f(1.,.85,.1)
                gl.glLineWidth(1.2)
                vertices(circle,gl.GL_LINE_STRIP)
                gl.glPointSize(4.)
                vertices(dot,gl.GL_POINTS)
                vertices(arrow,gl.GL_LINES)
            if mesh_on:
                current_hit=probe.hit
                for index,point in enumerate(placed_points):
                    if point is None: continue
                    probe.hit=point
                    circle,dot,arrow=probe.marker(ring_drag.radius*.16)
                    gl.glColor3f(1.,.85,.1) if index==0 else gl.glColor3f(1.,.4,1.)
                    gl.glLineWidth(1.2)
                    vertices(circle,gl.GL_LINE_STRIP)
                    gl.glPointSize(4.)
                    vertices(dot,gl.GL_POINTS)
                    vertices(arrow,gl.GL_LINES)
                probe.hit=current_hit
            gl.glDisable(gl.GL_DEPTH_TEST)
            measurement_status=""
            labels=[]
            if mesh_on and all(p is not None for p in placed_points):
                a,b=placed_points[0][0],placed_points[1][0]
                delta=b-a
                def dimension_arrow(start,end,color):
                    direction=end-start
                    length=np.linalg.norm(direction)
                    if length<1e-10: return
                    direction/=length
                    side=np.cross(direction,np.eye(3)[np.argmin(np.abs(direction))])
                    side/=np.linalg.norm(side)
                    head=min(ring_drag.radius*.2,length*.22)
                    points=np.array([start,end,end,end-head*direction+head*.4*side,
                                     end,end-head*direction-head*.4*side])
                    gl.glColor3f(*color)
                    gl.glLineWidth(1.2)
                    vertices(np.ascontiguousarray(points,dtype=np.float32),gl.GL_LINES)
                dimension_arrow(a,b,(1.,1.,1.))
                labels.append(((a+b)/2,f"Total: {np.linalg.norm(delta)*1000:.2f} mm",(1.,1.,1.)))
                corner=a.copy()
                for axis_index,color in enumerate(((1.,1.,0.),(1.,0.,1.),(0.,1.,1.))):
                    end=corner.copy()
                    end[axis_index]=b[axis_index]
                    dimension_arrow(corner,end,color)
                    labels.append(((corner+end)/2,f"{'XYZ'[axis_index]}: {abs(delta[axis_index])*1000:.2f} mm",color))
                    corner=end
                measurement_status=(f"P1 -> P2: {np.linalg.norm(delta)*1000:.2f} mm | "
                    f"X (yellow): {delta[0]*1000:+.2f} | Y (magenta): {delta[1]*1000:+.2f} | Z (cyan): {delta[2]*1000:+.2f} mm")
            if rings_on:
                if mesh_on and solid:
                    gl.glEnable(gl.GL_DEPTH_TEST)
                    gl.glDepthFunc(gl.GL_LEQUAL)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glColor3f(1.,.12,.12)
                gl.glLineWidth(1.)
                vertices(ring_drag.lines,gl.GL_LINES)
                for index,ring in enumerate(ring_drag.rings[[0,-1]]):
                    gl.glColor3f(1.,.65 if ring_drag.active==index else .12,.12)
                    gl.glLineWidth(1.5)
                    vertices(np.ascontiguousarray(ring,dtype=np.float32),gl.GL_LINE_STRIP)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_DEPTH_TEST)
            probe_status=f"Point {selected_point+1} selected | Click / drag point: select / edit | Alt + click: place | Drag elsewhere: orbit"
            if probe.hit is not None:
                point,normal=probe.hit
                probe_status+=f" | XYZ (mm): {point[0]*1000:.1f}, {point[1]*1000:.1f}, {point[2]*1000:.1f} | N: {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}"
            if hud_on:
                panel.draw(["F1: hide HUD | I: RGB plane | V: calibrated RGB view | Left drag background: orbit | Right drag: pan | Wheel: zoom | R: reset orbit",
                        probe_status,
                        measurement_status or "Place P1 and P2 to measure | Camera axes: X right, Y down, Z forward",
                        f"Drag end rings to cut mesh | Release: save trimmed PLY | Home: restore full mesh | Span: {ring_drag.ends[0]*100:.1f}% - {ring_drag.ends[1]*100:.1f}%",
                        "X: rings | P: point cloud | M: mesh | W: solid / wireframe | C: pipe / whole scene | ESC / Q: close",
                        f"{len(f):,} triangles | {'Scene' if context else 'Pipe'} cloud: {len(cloud) if context else len(pipe_cloud):,} points | Mesh {'on' if mesh_on else 'off'} | Cloud {'on' if cloud_on else 'off'}",
                        fit_status+f" | RGB {'on' if rgb_on and rgb_texture is not None else 'off'} | {'Camera view' if camera_view else 'Orbit view'}"],w,h)
            dimension_labels.draw(labels,ring_drag.project,w,h)
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
        if dimension_labels is not None: dimension_labels.close()
        if rgb_texture is not None: gl.glDeleteTextures([rgb_texture])
        if panel is not None: panel.close()
        pg.display.quit()


