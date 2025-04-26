import glfw

def register_callbacks(window, state):
    """Register GLFW callbacks for key presses and window resize to update state."""
    def on_key(win, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
                state.run_mode = False    # Quit the display loop
            elif key == glfw.KEY_F:
                state.fullscreen = not state.fullscreen
                state.needs_update = True  # Trigger re-init (toggle fullscreen)
            elif key == glfw.KEY_R:
                state.rotation = (state.rotation + 45) % 360
                state.needs_update = True  # Update rotation immediately
            elif key == glfw.KEY_M:
                state.mirror = 0 if state.mirror else 1
                state.needs_update = True  # Toggle mirror mode
    def on_window_size(win, width, height):
        # Maintain original aspect ratio when window is resized
        aspect = state.image_size[0] / state.image_size[1]
        if state.rotation % 180 == 90:
            aspect = 1.0 / aspect  # swap aspect if rotated 90 or 270
        target_w = width
        target_h = int(width / aspect)
        if target_h > height:
            target_h = height
            target_w = int(height * aspect)
        if target_w != width or target_h != height:
            # Adjust window to closest aspect-correct size
            glfw.set_window_size(window, target_w, target_h)
        # Update state to new window size and flag for recalculation
        state.image_size = (target_w, target_h)
        state.needs_update = True
    glfw.set_key_callback(window, on_key)
    glfw.set_window_size_callback(window, on_window_size)
