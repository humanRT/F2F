"""Select a SAM3 instance over RGB in an OpenGL window."""
import numpy as np
from .gl_preview import GLPreview
from .gl_overlay import image_viewport


def pick_mask(rgb, masks, scores, candidates, *, visible=True, label="instance"):
    rgb=np.asarray(rgb,dtype=np.uint8)
    h,w=rgb.shape[:2]
    candidates=list(map(int,candidates))
    selected=candidates[0]
    view=GLPreview(w,h,visible=visible)
    pg,gl=view.pg,view.gl
    texture=None
    cached=None
    hint="Click the intended pipe, then press ENTER."
    try:
        pg.display.set_caption("F2F - Select visible pipe section" if label=="section" else "F2F - Select pipe")
        texture=gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
        for setting in (gl.GL_TEXTURE_MIN_FILTER,gl.GL_TEXTURE_MAG_FILTER):
            gl.glTexParameteri(gl.GL_TEXTURE_2D,setting,gl.GL_NEAREST)
        def overlay():
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
            gl.glColor4f(1,1,1,1)
            gl.glBegin(gl.GL_QUADS)
            for uv,xy in [((0,0),(0,0)),((1,0),(w,0)),((1,1),(w,h)),((0,1),(0,h))]:
                gl.glTexCoord2f(*uv)
                gl.glVertex2f(*xy)
            gl.glEnd()
            gl.glDisable(gl.GL_TEXTURE_2D)
        clock=pg.time.Clock()
        while True:
            for event in pg.event.get():
                if event.type==pg.QUIT:
                    raise KeyboardInterrupt
                if event.type==pg.KEYDOWN:
                    if event.key in (pg.K_ESCAPE,pg.K_q): raise KeyboardInterrupt
                    if event.key in (pg.K_RETURN,pg.K_KP_ENTER): return selected
                    if event.key in (pg.K_RIGHT,pg.K_TAB,pg.K_LEFT):
                        selected=candidates[(candidates.index(selected)+(-1 if event.key==pg.K_LEFT else 1))%len(candidates)]
                    if event.unicode.isdigit() and int(event.unicode) in candidates:
                        selected=int(event.unicode)
                if event.type==pg.MOUSEBUTTONDOWN and event.button==1:
                    ww,wh=pg.display.get_window_size()
                    vx,vy,vw,vh=image_viewport(ww,wh,w,h)
                    mx,my=event.pos
                    x,y=int((mx-vx)*w/vw),int((my-(wh-vy-vh))*h/vh)
                    if 0<=x<w and 0<=y<h:
                        hits=[i for i in candidates if masks[i,y,x]]
                        if hits:
                            selected=hits[(hits.index(selected)+1)%len(hits)] if selected in hits and len(hits)>1 else hits[0]
                            hint="Overlapping masks: click again or use arrows to cycle." if len(hits)>1 else "ENTER confirms the highlighted pipe."
                        else:
                            hint="No candidate at that pixel. Click a colored region or use arrows."
            if selected!=cached:
                rgba=np.zeros((h,w,4),np.uint8)
                for i in candidates:
                    rgba[masks[i]]=[190,80,230,45]
                rgba[masks[selected]]=[40,230,100,115]
                gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA8,w,h,0,gl.GL_RGBA,gl.GL_UNSIGNED_BYTE,rgba)
                cached=selected
            detail=f"score {float(scores[selected]):.3f}" if scores is not None else f"{int(masks[selected].sum())} pixels"
            view.draw(rgb,f"{len(candidates)} candidates | Green: {label} {selected}, {detail} | {hint}",
                      instructions="Click: select | Left / Right / Tab: cycle | Number: select | ENTER: use highlighted region | ESC: cancel",
                      overlay=overlay)
            clock.tick(30)
    finally:
        if texture is not None: gl.glDeleteTextures([texture])
        view.close()
