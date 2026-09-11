"""Reusable instructions panel, composited inside the OpenGL framebuffer."""


class InstructionsPanel:
    def __init__(self, pygame, gl):
        self.pg, self.gl = pygame, gl
        pygame.font.init()
        self.font = pygame.font.SysFont("Segoe UI", 17)
        self.texture = None
        self.key = None

    def draw(self, lines, width, height):
        pg, gl = self.pg, self.gl
        available = max(40, min(820, width-24))
        key = (tuple(lines), available)
        if key != self.key:
            wrapped = []
            for line in lines:
                current = ""
                for word in line.split():
                    candidate = f"{current} {word}".strip()
                    if current and self.font.size(candidate)[0] > available-24:
                        wrapped.append(current)
                        current = word
                    else:
                        current = candidate
                wrapped.append(current)
            self.size = (min(available, max(self.font.size(s)[0] for s in wrapped)+24),
                         len(wrapped)*self.font.get_linesize()+20)
            surface = pg.Surface(self.size, pg.SRCALPHA)
            surface.fill((12, 17, 24, 218))
            for i, line in enumerate(wrapped):
                color = (255, 208, 98) if "NOT CONVERGED" in line else (238, 243, 250)
                surface.blit(self.font.render(line, True, color), (12, 10+i*self.font.get_linesize()))
            if self.texture is None:
                self.texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            for setting in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, setting, gl.GL_LINEAR)
            for setting in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, setting, gl.GL_CLAMP_TO_EDGE)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, *self.size, 0,
                            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pg.image.tostring(surface, "RGBA"))
            self.key = key
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, height, 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glColor4f(1, 1, 1, 1)
        pw, ph = self.size
        x, y = 12, max(0, height-ph-12)
        gl.glBegin(gl.GL_QUADS)
        for uv, xy in [((0,0),(x,y)), ((1,0),(x+pw,y)),
                       ((1,1),(x+pw,y+ph)), ((0,1),(x,y+ph))]:
            gl.glTexCoord2f(*uv)
            gl.glVertex2f(*xy)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

    def close(self):
        if self.texture is not None:
            self.gl.glDeleteTextures([self.texture])
            self.texture = None
