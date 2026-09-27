"""Minimal GLFW/ModernGL viewer for the standalone V7 receiver."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from animation_modem.imaging import values_image


DISPLAY_MODES = ('nearest', 'bilinear')
FULLSCREEN_TOOLBAR_HIDE_SECONDS = 2.0
FULLSCREEN_TOOLBAR_EDGE = 14
DISPLAY_LABELS = {'nearest': 'Nearest', 'bilinear': 'Bilinear'}


def toolbar_layout(width, open_dropdown=None):
    """Logical-pixel hit regions for the compact in-viewer toolbar."""
    width = max(320, int(width))
    upscale = (12, 7, 204, 42)
    panel = (224, 7, 382, 42)
    fullscreen = (width-102, 7, width-12, 42)
    hits = {'upscale_button': upscale, 'panel_button': panel}
    if width >= 500:
        hits['fullscreen_button'] = fullscreen
    if width >= 720:
        hits['save_default_button'] = (width-226, 7, width-112, 42)
    if open_dropdown == 'upscale':
        for index, mode in enumerate(DISPLAY_MODES):
            y = 48 + index*29
            hits[f'mode:{mode}'] = (upscale[0], y, upscale[2], y+28)
    elif open_dropdown == 'panel':
        for index, visible in enumerate((True, False)):
            y = 48 + index*29
            hits[f'panel:{int(visible)}'] = (panel[0], y, panel[2], y+28)
    return hits


def _contains(rect, x, y):
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def _preference_path():
    root = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')
    return root/'modemTest'/'v7_display.json'


def _load_display_default():
    try:
        data = json.loads(_preference_path().read_text())
        if (isinstance(data, dict) and data.get('version') == 1 and
                data.get('mode') in DISPLAY_MODES):
            return data['mode']
    except (OSError, ValueError, TypeError):
        pass
    return 'nearest'


def _save_display_default(mode):
    if mode not in DISPLAY_MODES:
        raise ValueError(f'unsupported display mode: {mode}')
    path = _preference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version': 1, 'mode': mode},
                                    sort_keys=True)+'\n')
    os.replace(temporary, path)


def _toolbar_image(size, mode, show_details, open_dropdown, notice=''):
    """Build a small, dark toolbar and optional dropdown using the viewer style."""
    width = max(320, int(size[0]))
    height = 48
    if open_dropdown == 'upscale':
        height += 2*29 + 4
    elif open_dropdown == 'panel':
        height += 2*29 + 4
    image = Image.new('RGBA', (width, height), (9, 16, 24, 246))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
        small = ImageFont.truetype('DejaVuSans.ttf', 12)
    except OSError:
        try:
            font = ImageFont.load_default(size=14)
            small = ImageFont.load_default(size=12)
        except TypeError:
            font = small = ImageFont.load_default()

    draw.rectangle((0, 46, width, 47), fill=(50, 70, 89, 255))
    draw.rounded_rectangle((12, 7, 204, 41), radius=5,
                           fill=(25, 39, 52, 255),
                           outline=(66, 94, 116, 255), width=1)
    draw.text((22, 15), f'Upscale  {DISPLAY_LABELS[mode]}  ▾',
              fill=(232, 240, 246, 255), font=font)
    panel_label = 'Info panel  On' if show_details else 'Info panel  Off'
    draw.rounded_rectangle((224, 7, 382, 41), radius=5,
                           fill=(25, 39, 52, 255),
                           outline=(66, 94, 116, 255), width=1)
    draw.text((234, 15), f'{panel_label}  ▾',
              fill=(232, 240, 246, 255), font=small)
    if width >= 920:
        draw.text((404, 17), 'F fullscreen  ·  I info  ·  Esc exit',
                  fill=(137, 162, 184, 255), font=small)
    save_box = (width-226, 7, width-112, 41)
    if width >= 500:
        draw.rounded_rectangle((width-102, 7, width-12, 41), radius=5,
                               fill=(25, 39, 52, 255),
                               outline=(66, 94, 116, 255), width=1)
        draw.text((width-90, 15), 'Fullscreen',
                  fill=(232, 240, 246, 255), font=small)
    if width >= 720:
        draw.rounded_rectangle(save_box, radius=5,
                               fill=(25, 39, 52, 255),
                               outline=(66, 94, 116, 255), width=1)
        draw.text((save_box[0]+8, 15), notice or 'Save default',
                  fill=(232, 240, 246, 255), font=small)

    if open_dropdown:
        hits = toolbar_layout(width, open_dropdown)
        prefix = 'mode:' if open_dropdown == 'upscale' else 'panel:'
        rows = [key for key in hits if key.startswith(prefix)]
        active = f'mode:{mode}' if open_dropdown == 'upscale' else f'panel:{int(show_details)}'
        anchor = (12, 224) if open_dropdown == 'upscale' else (224, 382)
        draw.rounded_rectangle((anchor[0], 47, anchor[1], height-3), radius=4,
                               fill=(18, 29, 40, 255),
                               outline=(66, 94, 116, 255), width=1)
        for key in rows:
            box = hits[key]
            if key == active:
                draw.rectangle((box[0]+1, box[1], box[2]-1, box[3]),
                               fill=(45, 76, 98, 255))
            if key.startswith('mode:'):
                label = DISPLAY_LABELS[key.split(':', 1)[1]]
            else:
                label = 'Show diagnostics' if key.endswith(':1') else 'Hide diagnostics'
            draw.text((box[0]+10, box[1]+6), label,
                      fill=(232, 240, 246, 255), font=small)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


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


def title_for_status(meter, details=False, display_mode='nearest'):
    """Keep receiver state visible without putting diagnostic widgets over video."""
    status = str(meter.get('status') or 'acquiring').upper()
    parts = ['V7 Receiver']
    parts.append(f'view {DISPLAY_LABELS.get(display_mode, display_mode)}')
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
              f'{status}  ·  F fullscreen  ·  I toggle diagnostics  ·  Esc exit',
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
        profile_cpu=False, display_mode=None, image_only=False):
    """Display new frames on a vsynced GL window, sleeping between events.

    GLFW and ModernGL are imported here so headless receive stays independent
    of the graphics stack. The context is owned by this thread; decoded frames
    arrive through the receiver's latest-frame mailbox.
    """
    import glfw
    import moderngl

    if not glfw.init():
        raise RuntimeError('GLFW initialization failed')

    image_only = bool(image_only)
    fullscreen = bool(fullscreen or image_only)

    window = None
    texture = None
    overlay = None
    toolbar = None
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
        if image_only:
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        monitor = glfw.get_primary_monitor() if fullscreen else None
        if fullscreen:
            mode = glfw.get_video_mode(monitor) if monitor else None
            width = mode.size.width if mode else 1280
            height = mode.size.height if mode else 720
        else:
            width, height = 960, 720
        title = 'V7 · Image only' if image_only else 'V7 Receiver'
        window = glfw.create_window(width, height, title, monitor, None)
        if not window:
            raise RuntimeError('GLFW could not create the V7 receiver window')
        glfw.make_context_current(window)
        if image_only:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
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
        show_details = bool(show_diagnostics) and not image_only
        last_frame_generation = None
        last_viewport = None
        last_title = None
        overlay_key = None
        toolbar_key = None
        display_mode = display_mode or _load_display_default()
        if display_mode not in DISPLAY_MODES:
            display_mode = 'nearest'
        open_dropdown = None
        save_notice = ''
        save_notice_until = 0.0
        toolbar_visible = not is_fullscreen
        last_ui_activity = time.monotonic()
        toolbar_hits = toolbar_layout(width)
        last_overlay_update = 0.0
        last_window_size = (width, height)
        profile_wall = time.monotonic()
        profile_process = time.process_time()
        profile_thread = time.thread_time()

        def toggle_fullscreen():
            nonlocal is_fullscreen, toolbar_visible, last_ui_activity
            nonlocal toolbar_key, dirty
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
            toolbar_visible = not is_fullscreen
            last_ui_activity = time.monotonic()
            toolbar_key = None
            dirty = True

        def on_cursor_position(_window, _x, y):
            nonlocal toolbar_visible, last_ui_activity, toolbar_key, dirty
            if not is_fullscreen or image_only:
                return
            now = time.monotonic()
            if toolbar_visible:
                last_ui_activity = now
            elif y <= FULLSCREEN_TOOLBAR_EDGE:
                toolbar_visible = True
                last_ui_activity = now
                toolbar_key = None
                dirty = True

        def on_key(_window, key, _scancode, action, _mods):
            nonlocal show_details, last_title, display_mode, open_dropdown
            nonlocal toolbar_key, dirty, toolbar_visible, last_ui_activity
            if action != glfw.PRESS:
                return
            if image_only:
                if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                    glfw.set_window_should_close(window, True)
                return
            was_hidden = not toolbar_visible
            toolbar_visible = True
            last_ui_activity = time.monotonic()
            if was_hidden:
                toolbar_key = None
                dirty = True
            if key == glfw.KEY_ESCAPE and open_dropdown is not None:
                open_dropdown = None
                last_title = None
            elif key == glfw.KEY_F:
                toggle_fullscreen()
            elif key == glfw.KEY_I:
                show_details = not show_details
                last_title = None
                open_dropdown = None
            elif key == glfw.KEY_U:
                step = -1 if _mods & glfw.MOD_SHIFT else 1
                display_mode = DISPLAY_MODES[
                    (DISPLAY_MODES.index(display_mode)+step) % len(DISPLAY_MODES)]
                if texture is not None:
                    filtering = (moderngl.NEAREST if display_mode == 'nearest'
                                 else moderngl.LINEAR)
                    texture.filter = (filtering, filtering)
                open_dropdown = None
                last_title = None
            elif key in (glfw.KEY_1, glfw.KEY_2):
                display_mode = DISPLAY_MODES[key-glfw.KEY_1]
                if texture is not None:
                    filtering = (moderngl.NEAREST if display_mode == 'nearest'
                                 else moderngl.LINEAR)
                    texture.filter = (filtering, filtering)
                open_dropdown = None
                last_title = None
            elif key == glfw.KEY_ESCAPE:
                if is_fullscreen:
                    toggle_fullscreen()
                else:
                    glfw.set_window_should_close(window, True)
            toolbar_key = None
            dirty = True

        glfw.set_key_callback(window, on_key)
        glfw.set_cursor_pos_callback(window, on_cursor_position)
        dirty = True

        def on_mouse_button(_window, button, action, _mods):
            nonlocal show_details, display_mode, open_dropdown
            nonlocal toolbar_key, last_title, dirty
            nonlocal save_notice, save_notice_until
            nonlocal toolbar_visible, last_ui_activity
            if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
                return
            if not image_only:
                was_hidden = not toolbar_visible
                toolbar_visible = True
                last_ui_activity = time.monotonic()
                if was_hidden:
                    toolbar_key = None
                    dirty = True
                    return
            x, y = glfw.get_cursor_pos(window)
            key = next((name for name, rect in toolbar_hits.items()
                        if _contains(rect, x, y)), None)
            if key == 'upscale_button':
                open_dropdown = None if open_dropdown == 'upscale' else 'upscale'
            elif key == 'panel_button':
                open_dropdown = None if open_dropdown == 'panel' else 'panel'
            elif key == 'fullscreen_button':
                open_dropdown = None
                toggle_fullscreen()
            elif key == 'save_default_button':
                try:
                    _save_display_default(display_mode)
                    save_notice = 'Saved'
                except OSError as exc:
                    print(f'Could not save V7 display preference: {exc}',
                          file=sys.stderr, flush=True)
                    save_notice = 'Save failed'
                save_notice_until = time.monotonic()+1.5
            elif key and key.startswith('mode:'):
                display_mode = key.split(':', 1)[1]
                filtering = (moderngl.NEAREST if display_mode == 'nearest'
                             else moderngl.LINEAR)
                if texture is not None:
                    texture.filter = (filtering, filtering)
                open_dropdown = None
                last_title = None
            elif key and key.startswith('panel:'):
                show_details = key.endswith(':1')
                open_dropdown = None
                last_title = None
            elif open_dropdown is not None:
                open_dropdown = None
            toolbar_key = None
            dirty = True

        glfw.set_mouse_button_callback(window, on_mouse_button)

        def on_refresh(_window):
            nonlocal dirty
            dirty = True

        glfw.set_window_refresh_callback(window, on_refresh)
        while not glfw.window_should_close(window):
            # Wait for input/resize events; this caps polling at 60 Hz without
            # consuming a CPU core while the decoder has no new picture.
            glfw.wait_events_timeout(1/60)

            meter = status_source()
            title = title_for_status(meter, show_details, display_mode)
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
                    filtering = (moderngl.NEAREST if display_mode == 'nearest'
                                 else moderngl.LINEAR)
                    texture.filter = (filtering, filtering)
                    texture.repeat_x = False
                    texture.repeat_y = False
                else:
                    texture.write(pixels.tobytes())
                last_frame_generation = frame.generation
                dirty = True

            now = time.monotonic()
            if is_fullscreen and not image_only:
                cursor_y = glfw.get_cursor_pos(window)[1]
                toolbar_was_visible = toolbar_visible
                if open_dropdown is not None or cursor_y <= FULLSCREEN_TOOLBAR_EDGE:
                    toolbar_visible = True
                elif (toolbar_visible and
                      now-last_ui_activity >= FULLSCREEN_TOOLBAR_HIDE_SECONDS):
                    toolbar_visible = False
                if toolbar_visible != toolbar_was_visible:
                    toolbar_key = None
                    dirty = True
            details_visible = (show_details and
                               (not is_fullscreen or toolbar_visible))
            window_size = glfw.get_window_size(window)
            if details_visible and diagnostics_source is not None and (
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
            elif not details_visible and overlay is not None:
                overlay.release()
                overlay = None
                overlay_key = None
                dirty = True

            fb_size = glfw.get_framebuffer_size(window)
            window_size = glfw.get_window_size(window)
            if save_notice and time.monotonic() >= save_notice_until:
                save_notice = ''
                toolbar_key = None
            toolbar_fb_height = 0
            show_toolbar = (not image_only and
                            (not is_fullscreen or toolbar_visible or
                             open_dropdown is not None))
            if show_toolbar:
                toolbar_state = (window_size, display_mode,
                                 show_details, open_dropdown, save_notice,
                                 toolbar_visible, is_fullscreen)
                toolbar_height = 48 + (62 if open_dropdown else 0)
                toolbar_size = (max(320, int(window_size[0])), toolbar_height)
                if (toolbar is None or toolbar.size != toolbar_size or
                        toolbar_key != toolbar_state):
                    toolbar_pixels = _toolbar_image(
                        window_size, display_mode, show_details, open_dropdown,
                        save_notice)
                    if toolbar is not None:
                        toolbar.release()
                    toolbar = context.texture(
                        toolbar_size, 4, data=toolbar_pixels.tobytes(), dtype='f1')
                    toolbar.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    toolbar.repeat_x = False
                    toolbar.repeat_y = False
                    toolbar_key = toolbar_state
                    toolbar_hits = toolbar_layout(window_size[0], open_dropdown)
                    dirty = True
                toolbar_fb_height = (
                    round(fb_size[1]*toolbar_size[1]/window_size[1])
                    if window_size[1] else 0)
            else:
                toolbar_hits = {}
            ratio = (aspect_ratios[frame.aspect & 7]
                     if frame is not None else 4/3)
            panel_height = (round(fb_size[1]*.36)
                            if details_visible and overlay is not None else 0)
            picture_area = (fb_size[0], max(
                0, fb_size[1]-panel_height-toolbar_fb_height))
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
                if details_visible and overlay is not None and panel_height:
                    context.enable(moderngl.BLEND)
                    context.blend_func = (moderngl.SRC_ALPHA,
                                          moderngl.ONE_MINUS_SRC_ALPHA)
                    context.viewport = (0, 0, fb_size[0], panel_height)
                    overlay.use(location=0)
                    overlay_array.render(mode=moderngl.TRIANGLES, vertices=3)
                    context.disable(moderngl.BLEND)
                if show_toolbar and toolbar is not None and toolbar_fb_height:
                    context.enable(moderngl.BLEND)
                    context.blend_func = (moderngl.SRC_ALPHA,
                                          moderngl.ONE_MINUS_SRC_ALPHA)
                    context.viewport = (0, fb_size[1]-toolbar_fb_height,
                                        fb_size[0], toolbar_fb_height)
                    toolbar.use(location=0)
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
                        f'(all threads), UI {100*thread_cpu/elapsed:.1f}% '
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
        if toolbar is not None:
            toolbar.release()
        if texture is not None:
            texture.release()
        if context is not None:
            context.release()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
