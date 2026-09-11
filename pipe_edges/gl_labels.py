"""Small screen-space labels anchored to projected 3D dimensions."""
import numpy as np


class DimensionLabels:
    def __init__(self, pg, gl):
        self.pg,self.gl=pg,gl
        pg.font.init()
        self.font=pg.font.SysFont("Segoe UI",14,bold=True)
        self.cache={}

    def draw(self, labels, project, width, height):
        pg,gl=self.pg,self.gl
        keys={(text,color) for _,text,color in labels}
        for key in list(self.cache):
            if key not in keys:
                gl.glDeleteTextures([self.cache.pop(key)[0]])
        gl.glViewport(0,0,width,height)
        gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity()
        gl.glOrtho(0,width,height,0,-1,1)
        gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadIdentity()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_TEXTURE_2D)
        for position,text,color in labels:
            screen=project(np.asarray(position)[None])[0]
            if not np.isfinite(screen).all(): continue
            key=(text,color)
            if key not in self.cache:
                rendered=self.font.render(text,True,tuple(round(c*255) for c in color))
                size=(rendered.get_width()+10,rendered.get_height()+6)
                surface=pg.Surface(size,pg.SRCALPHA)
                surface.fill((10,14,20,205)); surface.blit(rendered,(5,3))
                texture=gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
                for parameter in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
                    gl.glTexParameteri(gl.GL_TEXTURE_2D,parameter,gl.GL_LINEAR)
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
                gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA8,*size,0,gl.GL_RGBA,gl.GL_UNSIGNED_BYTE,pg.image.tostring(surface,"RGBA"))
                self.cache[key]=(texture,size)
            texture,(w,h)=self.cache[key]
            x=float(np.clip(screen[0]+8,0,max(0,width-w)))
            y=float(np.clip(screen[1]-h-8,0,max(0,height-h)))
            gl.glBindTexture(gl.GL_TEXTURE_2D,texture); gl.glColor4f(1,1,1,1)
            gl.glBegin(gl.GL_QUADS)
            for uv,xy in (((0,0),(x,y)),((1,0),(x+w,y)),((1,1),(x+w,y+h)),((0,1),(x,y+h))):
                gl.glTexCoord2f(*uv); gl.glVertex2f(*xy)
            gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

    def close(self):
        for texture,_ in self.cache.values(): self.gl.glDeleteTextures([texture])
        self.cache.clear()
