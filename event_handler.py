try:
    import glfw
except ImportError:
    glfw = None

def jiggle_mouse_for_focus(window):
    """
    Jiggle mouse cursor slightly to help compositors grab window focus.
    Useful for Wayland and other compositors that need mouse movement to grab focus.
    
    Args:
        window: GLFW window object or None (no-op if invalid)
    """
    if glfw is None or window is None:
        return
    
    try:
        # Get current cursor position
        x, y = glfw.get_cursor_pos(window)
        
        # Move cursor dramatically (50 pixels) to make it clearly visible
        glfw.set_cursor_pos(window, x + 50, y + 90)
        
        # Process events to ensure the movement is registered
        glfw.poll_events()
        
        # Move cursor back to original position
        glfw.set_cursor_pos(window, x, y)
        
        # Process events again to ensure the return movement is registered
        glfw.poll_events()
    except Exception:
        # Silently fail - don't break the application if mouse jiggle fails
        pass

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
        if state.fullscreen:
            # Check if fullscreen was lost (OS kicked us out)
            try:
                from display_manager import _is_wayland_session
                is_wayland = _is_wayland_session()
                
                if is_wayland:
                    # Wayland: Compare window size to monitor size
                    # If window is significantly smaller (< 90% of monitor), fullscreen was lost
                    primary_monitor = glfw.get_primary_monitor()
                    if primary_monitor:
                        mode = glfw.get_video_mode(primary_monitor)
                        if mode:
                            monitor_w, monitor_h = mode.size.width, mode.size.height
                            # Only restore if window is clearly smaller than monitor (not just a small difference)
                            if width < monitor_w * 0.9 or height < monitor_h * 0.9:
                                state.needs_update = True
                else:
                    # X11: Check if window is no longer on a monitor
                    current_monitor = glfw.get_window_monitor(win)
                    if current_monitor is None:
                        # Window not on monitor = fullscreen lost
                        state.needs_update = True
            except Exception:
                pass  # Ignore errors - don't break on callback errors
        else:
            # Windowed mode: trigger reconfiguration for normal resize
            state.needs_update = True

    glfw.set_key_callback(window, on_key)
    glfw.set_window_size_callback(window, on_window_size)
