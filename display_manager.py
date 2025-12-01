# display_manager.py

import moderngl
import settings
import renderer

# Conditional import to prevent crashes on headless servers
if not settings.SERVER_MODE:
    try:
        import glfw
    except ImportError:
        glfw = None
        print("Warning: GLFW not found. Local display mode will fail.")


class DisplayState:
    def __init__(self):
        self.run_mode = True
        self.needs_update = False
        self.fullscreen = False
        self.image_size = (0, 0)


class HeadlessWindow:
    """
    Simple wrapper for a headless FBO-based rendering target.

    .use() binds the FBO so subsequent draws go into it.
    .size is (width, height) for capture reshape.
    """
    def __init__(self, ctx: moderngl.Context, fbo: moderngl.Framebuffer, size: tuple[int, int]):
        self.ctx = ctx
        self.fbo = fbo
        self.size = size

    def use(self) -> None:
        self.fbo.use()


def display_init(state: DisplayState):
    """
    Initialize either:

    - SERVER_MODE + HEADLESS_USE_GL = False  -> return None (pure CPU compositing)
    - SERVER_MODE + HEADLESS_USE_GL = True   -> headless ModernGL FBO (HeadlessWindow)
    - not SERVER_MODE (dev / Mac)            -> GLFW window + ModernGL context

    The caller decides how to use the returned object.
    """

    # --- Headless server mode ------------------------------------------------
    if settings.SERVER_MODE:
        use_gl = getattr(settings, "HEADLESS_USE_GL", False)

        if not use_gl:
            print("[DISPLAY] SERVER_MODE: GL disabled (HEADLESS_USE_GL=False). Using CPU-only compositor.")
            return None

        print(f"[DISPLAY] Initializing headless ModernGL "
              f"(port={getattr(settings, 'STREAM_PORT', 8080)})...")

        try:
            backend = getattr(settings, "HEADLESS_BACKEND", None)
            create_kwargs = {"standalone": True}
            if backend:
                create_kwargs["backend"] = backend   # e.g. "egl", "osmesa", "x11"
            ctx = moderngl.create_context(**create_kwargs)
        except Exception as e:
            print(f"[DISPLAY] Failed to create headless GL context: {e}")
            print("[DISPLAY] Falling back to CPU-only compositing.")
            return None

        width, height = getattr(settings, "HEADLESS_RES", (1280, 720))
        state.image_size = (width, height)

        tex = ctx.texture((width, height), components=4)
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()

        # Viewport + aspect info for renderer (headless path)
        ctx.viewport = (0, 0, width, height)

        renderer.initialize(ctx)
        renderer.set_viewport_size(width, height)

        print(f"[DISPLAY] Headless GL context ready at {width}x{height}")
        return HeadlessWindow(ctx, fbo, (width, height))

    # --- Local windowed mode (dev / Mac) ------------------------------------
    # (unchanged below here)
    ...
