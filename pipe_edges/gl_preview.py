"""Live RGB preview rendered as a streaming OpenGL texture."""
import os
import warnings
import numpy as np

from .gl_overlay import image_viewport
from .gl_hud import InstructionsPanel


class GLPreview:
    def __init__(self, width, height, *, visible=True):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"^pkg_resources is deprecated as an API\.", category=UserWarning, module=r"^pygame\.pkgdata$")
            import pygame
        from OpenGL import GL
        self.pg, self.gl = pygame, GL
        self.width, self.height = width, height
        self.texture = None
        self.panel = None
        pygame.display.init()
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            scale = min(1., 1200/width, 800/height)
            flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.RESIZABLE if visible else pygame.HIDDEN)
            pygame.display.set_mode((round(width*scale), round(height*scale)), flags)
            pygame.display.set_caption("F2F - Live camera")
            self.panel = InstructionsPanel(pygame, GL)
            self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            for setting in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, setting, GL.GL_LINEAR)
            for setting in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, setting, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, width, height, 0,
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        except Exception:
            self.close()
            raise

    def poll(self):
        action = None
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return "cancel"
            if event.type == self.pg.KEYDOWN:
                if event.key in (self.pg.K_ESCAPE, self.pg.K_q):
                    return "cancel"
                if event.key in (self.pg.K_SPACE, self.pg.K_RETURN):
                    action = "capture"
        return action

    def draw(self, rgb, status, *, instructions=None, overlay=None):
        gl, pg = self.gl, self.pg
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError("Preview frame dimensions changed during capture.")
        ww, wh = pg.display.get_window_size()
        gl.glViewport(0, 0, ww, wh)
        gl.glClearColor(.06, .06, .06, 1.)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glViewport(*image_viewport(ww, wh, self.width, self.height))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, self.height, 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                          gl.GL_RGB, gl.GL_UNSIGNED_BYTE, rgb)
        gl.glColor4f(1, 1, 1, 1)
        gl.glBegin(gl.GL_QUADS)
        for uv, xy in [((0,0),(0,0)), ((1,0),(self.width,0)),
                       ((1,1),(self.width,self.height)), ((0,1),(0,self.height))]:
            gl.glTexCoord2f(*uv)
            gl.glVertex2f(*xy)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)
        if overlay is not None:
            overlay()
        self.panel.draw([instructions or "SPACE / ENTER: capture    Q / ESC: quit", status], ww, wh)
        pg.display.flip()

    def close(self):
        try:
            if self.panel is not None:
                self.panel.close()
            if self.texture is not None:
                self.gl.glDeleteTextures([self.texture])
                self.texture = None
        finally:
            self.pg.display.quit()
