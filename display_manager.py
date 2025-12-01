import moderngl
import settings
import renderer

# Conditional import to prevent crashes on headless servers
if not settings.SERVER_MODE:
    try:
        import glfw
    except ImportError:
        print("Warning: GLFW not found. Local display mode will fail.")


class DisplayState:
    def __init__(self):
        self.run_mode = True
        self.needs_update = False
        self.fullscreen = False
        self.image_size = (0, 0)


class HeadlessWindow:
    def __init__(self, ctx, fbo, size):
        self.ctx = ctx
        self.fbo = fbo
        self.size = size

    def use(self):
        self.fbo.use()


def display_init(state: DisplayState):
    """
    Initialize either:
    - CPU-only headless mode (SERVER_MODE + HEADLESS_USE_GL=False) -> returns None
    - Headless ModernGL FBO (SERVER_MODE + HEADLESS_USE_GL=True)
    - Local GLFW window (not SERVER_MODE)
    """
    # --- Headless server mode ---
    if settings.SERVER_MODE:
        use_gl = getattr(settings, "HEADLESS_USE_GL", False)

        if not use_gl:
            # Pure CPU mode: no GL context, no Xvfb, no XOpenDisplay.
            print("[DISPLAY] SERVER_MODE: GL disabled (HEADLESS_USE_GL=False). Using CPU-only compositor.")
            return None

        # Headless GL (for when you're testing on a machine with a GPU)
        print(f"Initializing Headless Mode (Port {getattr(settings, 'STREAM_PORT', 8080)})...")
        ctx = moderngl.create_context(standalone=True)

        width, height = getattr(settings, "HEADLESS_RES", (1280, 720))
        state.image_size = (width, height)

        texture = ctx.texture((width, height), 3)
        fbo = ctx.framebuffer(color_attachments=[texture])
        fbo.use()

        renderer.initialize(ctx)
        return HeadlessWindow(ctx, fbo, (width, height))

    # --- Local windowed mode (Mac/dev) ---
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.VISIBLE, True)
    glfw.window_hint(glfw.RESIZABLE, False)

    if state.fullscreen:
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        width, height = mode.size.width, mode.size.height
    else:
        width, height = 1280, 720

    window = glfw.create_window(width, height, "Portrait Generator", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if getattr(settings, "VSYNC", False) else 0)

    ctx = moderngl.create_context()
    renderer.initialize(ctx)
    return window
