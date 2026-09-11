"""Shared OpenGL context creation with verified multisample anti-aliasing."""
from .progress import progress


def create_window(pg, gl, size, flags):
    for requested in (8,4,2,0):
        try:
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION,2)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION,1)
            pg.display.gl_set_attribute(pg.GL_DOUBLEBUFFER,1)
            pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE,24)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS,int(requested>0))
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES,requested)
            pg.display.set_mode(size,flags)
        except pg.error:
            if requested==0:
                raise
            pg.display.quit()
            pg.display.init()
            continue
        samples=int(gl.glGetIntegerv(gl.GL_SAMPLES)) if int(gl.glGetIntegerv(gl.GL_SAMPLE_BUFFERS)) else 0
        if samples:
            gl.glEnable(gl.GL_MULTISAMPLE)
        progress("OpenGL",f"Anti-aliasing: {samples}x MSAA" if samples else
                 "Anti-aliasing: unavailable; using standard rendering.")
        return samples
