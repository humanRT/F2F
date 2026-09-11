"""Smooth mesh normals and view-relative studio lighting for solid surfaces."""
import numpy as np


def triangle_normals(vertices, faces):
    vertices=np.asarray(vertices,float)
    faces=np.asarray(faces,int)
    p=vertices[faces]
    area_normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    normals=np.zeros_like(vertices)
    for column in range(3):
        np.add.at(normals,faces[:,column],area_normals)
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-20)
    return np.ascontiguousarray(normals[faces].reshape(-1,3),dtype=np.float32)


def enable_solid_lighting(gl):
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_NORMALIZE)
    gl.glShadeModel(gl.GL_SMOOTH)
    gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT,(.12,.12,.14,1))
    gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE,gl.GL_TRUE)
    gl.glLightModeli(gl.GL_LIGHT_MODEL_LOCAL_VIEWER,gl.GL_TRUE)
    # Set positions in eye coordinates so orbiting never leaves the model unlit.
    gl.glPushMatrix()
    gl.glLoadIdentity()
    for light,position,color in (
        (gl.GL_LIGHT0,(-.6,.8,1.,0),(.85,.82,.78,1)),
        (gl.GL_LIGHT1,(.8,.1,.5,0),(.28,.34,.42,1)),
        (gl.GL_LIGHT2,(.1,.7,-1.,0),(.25,.3,.35,1))):
        gl.glEnable(light)
        gl.glLightfv(light,gl.GL_POSITION,position)
        gl.glLightfv(light,gl.GL_DIFFUSE,color)
        gl.glLightfv(light,gl.GL_SPECULAR,color if light==gl.GL_LIGHT0 else (0,0,0,1))
    gl.glPopMatrix()
    gl.glMaterialfv(gl.GL_FRONT_AND_BACK,gl.GL_AMBIENT_AND_DIFFUSE,(.24,.55,.65,1))
    gl.glMaterialfv(gl.GL_FRONT_AND_BACK,gl.GL_SPECULAR,(.3,.3,.3,1))
    gl.glMaterialf(gl.GL_FRONT_AND_BACK,gl.GL_SHININESS,40)


def disable_solid_lighting(gl):
    gl.glDisable(gl.GL_LIGHTING)
    gl.glDisable(gl.GL_NORMALIZE)
    for light in (gl.GL_LIGHT0,gl.GL_LIGHT1,gl.GL_LIGHT2):
        gl.glDisable(light)
