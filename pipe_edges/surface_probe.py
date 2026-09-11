"""Pick the nearest visible mesh triangle and display its geometric normal."""
import numpy as np


class SurfaceProbe:
    def __init__(self, vertices, faces):
        self.inverse=None
        self.hit=None
        self.active=False
        self.set_mesh(vertices,faces)

    def set_mesh(self, vertices, faces):
        triangles=np.asarray(vertices,float)[faces]
        self.a=triangles[:,0]
        self.e1=triangles[:,1]-self.a
        self.e2=triangles[:,2]-self.a
        normal=np.cross(self.e1,self.e2)
        self.normals=normal/np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-20)
        self.hit=None

    def set_view(self, model, projection, viewport, height):
        self.inverse=np.linalg.inv(np.asarray(model)@np.asarray(projection))
        self.viewport=viewport
        self.height=height

    def pick(self, mouse):
        self.hit=None
        if self.inverse is None: return False
        x,y,w,h=self.viewport
        mx,my=mouse[0],self.height-mouse[1]
        if not (x<=mx<=x+w and y<=my<=y+h): return False
        ndc=[2*(mx-x)/w-1,2*(my-y)/h-1]
        ends=np.array([[*ndc,-1,1],[*ndc,1,1]])@self.inverse
        ends=ends[:,:3]/ends[:,3,None]
        origin=ends[0]
        direction=ends[1]-origin
        far=np.linalg.norm(direction)
        direction/=far
        p=np.cross(direction,self.e2)
        det=np.sum(self.e1*p,axis=1)
        valid=np.abs(det)>1e-14
        inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
        delta=origin-self.a
        u=np.sum(delta*p,axis=1)*inv
        q=np.cross(delta,self.e1)
        v=q@direction*inv
        t=np.sum(self.e2*q,axis=1)*inv
        valid&=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>=0)&(t<=far)
        if not valid.any(): return False
        index=int(np.argmin(np.where(valid,t,np.inf)))
        self.hit=(origin+t[index]*direction,self.normals[index].copy())
        return True

    def marker(self, radius):
        point,normal=self.hit
        u=np.cross(normal,np.eye(3)[np.argmin(np.abs(normal))])
        u/=np.linalg.norm(u)
        v=np.cross(normal,u)
        center=point+normal*radius*.015
        theta=np.linspace(0,2*np.pi,65)
        circle=center+radius*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
        tip=center+normal*radius*5
        arrow=np.array([center,tip,tip,tip-normal*radius+u*radius*.5,
                        tip,tip-normal*radius-u*radius*.5])
        return [np.ascontiguousarray(p,dtype=np.float32) for p in (circle,center[None],arrow)]
