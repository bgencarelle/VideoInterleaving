"""Minimal GLFW/ModernGL viewer for the standalone V7 receiver."""
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from animation_modem.imaging import values_image


VERTEX_SHADER = '''#version 330
out vec2 uv;
void main() {
    vec2 position;
    if (gl_VertexID == 0) position = vec2(-1.0, -1.0);
    else if (gl_VertexID == 1) position = vec2(3.0, -1.0);
    else position = vec2(-1.0, 3.0);
    gl_Position = vec4(position, 0.0, 1.0);
    uv = vec2((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
}
'''

FRAGMENT_SHADER = '''#version 330
uniform sampler2D image;
in vec2 uv;
out vec4 color;
void main() {
    color = texture(image, uv);
}
'''


def fit_viewport(framebuffer_size, image_aspect):
    """Return a centered letterbox viewport preserving the transmitted aspect."""
    width, height = (max(0, int(value)) for value in framebuffer_size)
    if width == 0 or height == 0 or image_aspect <= 0:
        return 0, 0, 0, 0
    window_aspect = width/height
    if window_aspect > image_aspect:
        view_height = height
        view_width = max(1, round(height*image_aspect))
    else:
        view_width = width
        view_height = max(1, round(width/image_aspect))
    return ((width-view_width)//2, (height-view_height)//2,
            view_width, view_height)


def title_for_status(meter, details=False):
    """Keep receiver state visible without putting diagnostic widgets over video."""
    status = str(meter.get('status') or 'acquiring').upper()
    parts = ['V7 Receiver']
    device = meter.get('device')
    if device is not None:
        rate = float(meter.get('capture_rate') or 0)/1000
        channels = meter.get('input_channels') or 1
        mode = meter.get('mode') or ('mono-input' if channels == 1 else 'M/S')
        parts.append(f'{device} · {rate:g} kHz · {channels} ch · {mode}')
    parts.append(status)
    index = meter.get('source_index')
    if index is not None:
        parts.append(f'index {index}')
    lag_ms = meter.get('lag_ms')
    if lag_ms is not None:
        parts.append(f'lag {lag_ms:+.0f} ms')
    if details:
        incoming = meter.get('input_fps') or 0.0
        gain = meter.get('auto_gain') or 1.0
        dropped = meter.get('dropped') or 0
        parts.extend((f'{incoming:.1f} fps', f'gain {gain:.1f}x',
                      f'drops {dropped}'))
    return '  |  '.join(parts)


def _diagnostic_image(size, diagnostics):
    """Build the four compact diagnostic cards at a low, fixed refresh rate."""
    width, height = size
    image = Image.new('RGBA', (max(1, width), max(1, height)),
                      (9, 16, 24, 224))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSansMono.ttf', 14)
        heading_font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError:
        try:
            font = ImageFont.load_default(size=14)
            heading_font = ImageFont.load_default(size=13)
        except TypeError:
            font = heading_font = ImageFont.load_default()

    padding = 14
    gap = 12
    heading_height = 20
    card_width = max(1, (width-2*padding-gap)//2)
    card_height = max(1, (height-2*padding-heading_height-gap)//2)
    status = diagnostics.get('status', ('ACQUIRING',))[0].upper()
    draw.text((padding, 3),
              f'{status}  ·  F fullscreen  ·  I toggle info  ·  Esc exit',
              fill=(132, 153, 173, 255), font=heading_font)
    specs = (
        ('SYNC  /  INDEX', 'sync'),
        ('DECODE  /  FLOW', 'decode'),
        ('INPUT  /  LEVEL', 'input'),
        ('PICTURE  /  SIGNAL', 'signal'),
    )
    for index, (heading, key) in enumerate(specs):
        col, row = index % 2, index // 2
        x = padding + col*(card_width+gap)
        y = padding+heading_height + row*(card_height+gap)
        box = (x, y, x+card_width, y+card_height)
        draw.rounded_rectangle(box, radius=5, fill=(19, 30, 41, 232),
                               outline=(50, 70, 89, 230), width=1)
        draw.text((x+10, y+7), heading, fill=(137, 162, 184, 255),
                  font=heading_font)
        line_y = y+26
        for line in diagnostics.get(key, ()):
            if line_y + 16 > y+card_height:
                break
            draw.text((x+10, line_y), str(line), fill=(227, 237, 245, 255),
                      font=font)
            line_y += 17
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def run(frame_source, status_source, aspect_ratios, fullscreen=False,
        show_diagnostics=True, diagnostics_source=None,
        profile_cpu=False):
    """Display new frames on a vsynced GL window, sleeping between events.

    GLFW and ModernGL are imported here so headless receive stays independent
    of the graphics stack. The context is owned by this thread; decoded frames
    arrive through the receiver's latest-frame mailbox.
    """
    import glfw
    import moderngl

    if not glfw.init():
        raise RuntimeError('GLFW initialization failed')

    window = None
    texture = None
    overlay = None
    context = None
    program = None
    vertex_array = None
    overlay_program = None
    overlay_array = None
    try:
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        monitor = glfw.get_primary_monitor() if fullscreen else None
        if fullscreen:
            mode = glfw.get_video_mode(monitor) if monitor else None
            width = mode.size.width if mode else 1280
            height = mode.size.height if mode else 720
        else:
            width, height = 960, 720
        window = glfw.create_window(width, height, 'V7 Receiver', monitor, None)
        if not window:
            raise RuntimeError('GLFW could not create the V7 receiver window')
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        context = moderngl.create_context(require=330)
        program = context.program(vertex_shader=VERTEX_SHADER,
                                  fragment_shader=FRAGMENT_SHADER)
        program['image'].value = 0
        vertex_array = context.vertex_array(program, [])
        overlay_program = context.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER)
        overlay_program['image'].value = 0
        overlay_array = context.vertex_array(overlay_program, [])

        windowed_bounds = {'position': (80, 80), 'size': (960, 720)}
        is_fullscreen = bool(fullscreen)
        show_details = bool(show_diagnostics)
        last_frame_generation = None
        last_viewport = None
        last_title = None
        overlay_key = None
        last_overlay_update = 0.0
        last_window_size = (width, height)
        profile_wall = time.monotonic()
        profile_process = time.process_time()
        profile_thread = time.thread_time()

        def toggle_fullscreen():
            nonlocal is_fullscreen
            primary = glfw.get_primary_monitor()
            if primary is None:
                return
            if is_fullscreen:
                x, y = windowed_bounds['position']
                w, h = windowed_bounds['size']
                glfw.set_window_monitor(window, None, x, y, w, h,
                                        glfw.DONT_CARE)
                is_fullscreen = False
            else:
                windowed_bounds['position'] = glfw.get_window_pos(window)
                windowed_bounds['size'] = glfw.get_window_size(window)
                mode = glfw.get_video_mode(primary)
                glfw.set_window_monitor(window, primary, 0, 0,
                                        mode.size.width, mode.size.height,
                                        mode.refresh_rate)
                is_fullscreen = True

        def on_key(_window, key, _scancode, action, _mods):
            nonlocal show_details, last_title
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_F:
                toggle_fullscreen()
            elif key == glfw.KEY_I:
                show_details = not show_details
                last_title = None
            elif key == glfw.KEY_ESCAPE:
                if is_fullscreen:
                    toggle_fullscreen()
                else:
                    glfw.set_window_should_close(window, True)

        glfw.set_key_callback(window, on_key)
        dirty = True

        def on_refresh(_window):
            nonlocal dirty
            dirty = True

        glfw.set_window_refresh_callback(window, on_refresh)
        while not glfw.window_should_close(window):
            # Wait for input/resize events; this caps polling at 60 Hz without
            # consuming a CPU core while the decoder has no new picture.
            glfw.wait_events_timeout(1/60)

            meter = status_source()
            title = title_for_status(meter, show_details)
            if title != last_title:
                glfw.set_window_title(window, title)
                last_title = title

            frame = frame_source()
            if frame is not None and frame.generation != last_frame_generation:
                image = values_image(frame.values, frame.shapes)
                pixels = np.ascontiguousarray(np.asarray(image.convert('RGB'),
                                                         dtype=np.uint8))
                size = (pixels.shape[1], pixels.shape[0])
                if texture is None or texture.size != size:
                    if texture is not None:
                        texture.release()
                    texture = context.texture(size, 3, data=pixels.tobytes(),
                                              dtype='f1')
                    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    texture.repeat_x = False
                    texture.repeat_y = False
                else:
                    texture.write(pixels.tobytes())
                last_frame_generation = frame.generation
                dirty = True

            now = time.monotonic()
            window_size = glfw.get_window_size(window)
            if show_details and diagnostics_source is not None and (
                    now-last_overlay_update >= .2 or
                    window_size != last_window_size):
                last_overlay_update = now
                last_window_size = window_size
                diagnostics = diagnostics_source()
                key = tuple((name, tuple(lines))
                            for name, lines in diagnostics.items())
                if key != overlay_key:
                    logical_w = max(320, int(window_size[0]))
                    logical_h = max(160, round(window_size[1]*.36))
                    overlay_pixels = _diagnostic_image(
                        (logical_w, logical_h), diagnostics)
                    if overlay is not None:
                        overlay.release()
                    overlay = context.texture(
                        (logical_w, logical_h), 4,
                        data=overlay_pixels.tobytes(), dtype='f1')
                    overlay.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    overlay.repeat_x = False
                    overlay.repeat_y = False
                    overlay_key = key
                    dirty = True
            elif not show_details and overlay is not None:
                overlay.release()
                overlay = None
                overlay_key = None
                dirty = True

            fb_size = glfw.get_framebuffer_size(window)
            ratio = (aspect_ratios[frame.aspect & 7]
                     if frame is not None else 4/3)
            panel_height = (round(fb_size[1]*.36)
                            if show_details and overlay is not None else 0)
            picture_area = (fb_size[0], max(0, fb_size[1]-panel_height))
            picture_viewport = fit_viewport(picture_area, ratio)
            viewport = ((picture_viewport[0],
                         panel_height+picture_viewport[1],
                         picture_viewport[2], picture_viewport[3]))
            if viewport != last_viewport:
                last_viewport = viewport
                dirty = True
            if dirty and viewport[2] and viewport[3]:
                context.viewport = (0, 0, *fb_size)
                context.clear(.025, .032, .045, 1.0)
                if texture is not None:
                    context.viewport = viewport
                    texture.use(location=0)
                    vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
                if show_details and overlay is not None and panel_height:
                    context.enable(moderngl.BLEND)
                    context.blend_func = (moderngl.SRC_ALPHA,
                                          moderngl.ONE_MINUS_SRC_ALPHA)
                    context.viewport = (0, 0, fb_size[0], panel_height)
                    overlay.use(location=0)
                    overlay_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    context.disable(moderngl.BLEND)
                glfw.swap_buffers(window)
                dirty = False

            if profile_cpu:
                elapsed = time.monotonic()-profile_wall
                if elapsed >= 5.0:
                    process_cpu = time.process_time()-profile_process
                    thread_cpu = time.thread_time()-profile_thread
                    print(
                        'V7 viewer CPU over '
                        f'{elapsed:.1f}s: process {100*process_cpu/elapsed:.1f}% '
                        '(all threads), UI {100*thread_cpu/elapsed:.1f}% '
                        '(viewer thread; event waits excluded)',
                        file=sys.stderr, flush=True)
                    profile_wall = time.monotonic()
                    profile_process = time.process_time()
                    profile_thread = time.thread_time()
    finally:
        if overlay_array is not None:
            overlay_array.release()
        if overlay_program is not None:
            overlay_program.release()
        if vertex_array is not None:
            vertex_array.release()
        if program is not None:
            program.release()
        if overlay is not None:
            overlay.release()
        if texture is not None:
            texture.release()
        if context is not None:
            context.release()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
