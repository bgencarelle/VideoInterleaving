try:
    import glfw
except ImportError:
    glfw = None

def register_callbacks(window, state):
    """Register GLFW callbacks for key presses and window resize to update state."""
    if glfw is None:
        # GLFW not available - skip event handling (e.g., headless mode or Pi 2 without GLFW)
        return
    
    def on_key(win, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
                state.run_mode = False    # Quit the display loop
            elif key == glfw.KEY_F:
                state.fullscreen = not state.fullscreen
                state.needs_update = True  # Toggle fullscreen
            elif key == glfw.KEY_R:
                state.rotation = (state.rotation + 90) % 360
                state.needs_update = True  # Rotate display
            elif key == glfw.KEY_M:
                state.mirror = 0 if state.mirror else 1
                state.needs_update = True  # Mirror mode

    def on_window_size(win, width, height):
        if not state.fullscreen:
            # Trigger reconfiguration but don't adjust anything directly
            state.needs_update = True

    glfw.set_key_callback(window, on_key)
    glfw.set_window_size_callback(window, on_window_size)
