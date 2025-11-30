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


def display_init(state):
    if settings.SERVER_MODE:
        print(f"Initializing Headless Mode (Port {getattr(settings, 'STREAM_PORT', 8080)})...")
        ctx = moderngl.create_context(standalone=True)
        width, height = getattr(settings, 'HEADLESS_RES', (1280, 720))
        state.image_size = (width, height)
        texture = ctx.texture((width, height), 3)
        fbo = ctx.framebuffer(color_attachments=[texture])
        fbo.use()
        renderer.initialize(ctx)
        return HeadlessWindow(ctx, fbo, (width, height))
    else:
        if not glfw.init(): raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.VISIBLE, True)
        glfw.window_hint(glfw.RESIZABLE, False)

        if state.fullscreen:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height
        else:
            width, height = 1280, 720

        window = glfw.create_window(width, height, "Portrait Generator", None, None)
        if not window: glfw.terminate(); raise RuntimeError("Window failed")

        glfw.make_context_current(window)
        glfw.swap_interval(1 if settings.VSYNC else 0)
        ctx = moderngl.create_context()
        renderer.initialize(ctx)
        return window