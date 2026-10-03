#!/usr/bin/env python3
"""Pop-out window for the V7 sender GUI's preview.

A small process the sender GUI starts for the "Pop-out window" preview. It
shows the same pictures the pane on the Live page would show: the GUI
receives the sender's preview datagrams (tools/v7_preview_protocol.py) and
forwards the ones of the selected stage to this process's loopback UDP port,
so the window follows pause, seek and resume like the pane does.

Talks to the GUI on its pipes, one JSON object per line on stdout:
  {"status": "popout_ready", "port": N}      the window is open, send to N
  {"status": "popout_error", "message": ...} no window; the process exits 2
The process ends when its window is closed or when stdin reaches end of file
(the GUI closed the pipe, or the GUI is gone).
"""
import io
import json
from pathlib import Path
import socket
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v7_preview_protocol import parse_preview_datagram


WINDOW_SIZE = (512, 640)
BACKGROUND = (8, 14, 20)
STAGE_NAMES = {'source': 'Source', 'resized': 'Encoder input'}
TITLE = 'V7 Sender · Preview'


class PopoutFrames:
    """The newest forwarded picture, and how it is drawn in a window.

    No window or GL here, so it is tested without a display.
    """

    def __init__(self):
        self.image = None
        self.counter = None
        self.aspect = None
        self.stage = None
        self.received = 0

    def feed(self, datagram):
        """Take one preview datagram; True when it became the picture."""
        parsed = parse_preview_datagram(datagram)
        if parsed is None:
            return False
        counter, aspect, _handoff_ns, stage, jpeg = parsed
        from PIL import Image
        try:
            with Image.open(io.BytesIO(jpeg)) as image:
                image.load()
                decoded = image.convert('RGB')
        except (OSError, ValueError):
            return False                  # keep the last good picture
        self.image, self.counter = decoded, int(counter)
        self.aspect, self.stage = int(aspect), stage
        self.received += 1
        return True

    def title(self):
        if self.image is None:
            return TITLE+' · waiting for the sender'
        return (f'{TITLE} · {STAGE_NAMES.get(self.stage, self.stage)} · '
                f'packet {self.counter}')

    def compose(self, size):
        """The window contents: the picture scaled to fit, centred."""
        from PIL import Image, ImageOps
        size = (max(1, int(size[0])), max(1, int(size[1])))
        canvas = Image.new('RGB', size, BACKGROUND)
        if self.image is not None:
            picture = ImageOps.contain(self.image, size,
                                       method=Image.Resampling.BILINEAR)
            canvas.paste(picture, ((size[0]-picture.width)//2,
                                   (size[1]-picture.height)//2))
        return canvas


def _say(output, **message):
    try:
        output.write(json.dumps(message)+'\n')
        output.flush()
    except (OSError, ValueError):
        pass


def _receive(receiver, frames, lock, state, wake):
    while not state['closing']:
        try:
            datagram, _address = receiver.recvfrom(65507)
        except socket.timeout:
            continue
        except OSError:
            return
        with lock:
            if frames.feed(datagram):
                state['dirty'] = True
        wake()


def _watch_parent(control, state, wake):
    try:
        while control.readline():
            pass
    except (OSError, ValueError):
        pass
    state['closing'] = True
    wake()


def run(control=None, output=None):
    """Open the window and show forwarded pictures until told to close."""
    control = sys.stdin if control is None else control
    output = sys.stdout if output is None else output
    window = context = program = vertex_array = texture = receiver = None
    glfw = None
    try:
        import glfw
        import moderngl
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed (no display?)')
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        window = glfw.create_window(*WINDOW_SIZE, TITLE, None, None)
        if not window:
            raise RuntimeError('GLFW could not create the preview window')
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = moderngl.create_context(require=330)
        program = context.program(
            vertex_shader='''#version 330
            out vec2 uv;
            void main() {
                vec2 p[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0),
                                     vec2(-1.0, 3.0));
                vec2 pos = p[gl_VertexID];
                uv = (pos + 1.0) * 0.5;
                gl_Position = vec4(pos, 0.0, 1.0);
            }''',
            fragment_shader='''#version 330
            uniform sampler2D image;
            in vec2 uv;
            out vec4 color;
            void main() { color = texture(image, vec2(uv.x, 1.0-uv.y)); }
            ''')
        program['image'].value = 0
        vertex_array = context.vertex_array(program, [])
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256*1024)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.2)
    except Exception as exc:              # headless, no GL, no socket
        _say(output, status='popout_error',
             message=f'Pop-out window unavailable: {exc}')
        if receiver is not None:
            receiver.close()
        if window:
            glfw.destroy_window(window)
        if glfw is not None:
            try:
                glfw.terminate()
            except Exception:
                pass
        return 2

    frames = PopoutFrames()
    lock = threading.Lock()
    state = {'dirty': True, 'closing': False}

    def wake():
        try:
            glfw.post_empty_event()
        except Exception:
            pass

    def redraw(*_args):
        state['dirty'] = True

    glfw.set_window_size_callback(window, redraw)
    glfw.set_framebuffer_size_callback(window, redraw)
    glfw.set_window_refresh_callback(window, redraw)
    threading.Thread(target=_receive, name='v7-popout-receive', daemon=True,
                     args=(receiver, frames, lock, state, wake)).start()
    threading.Thread(target=_watch_parent, name='v7-popout-parent',
                     daemon=True, args=(control, state, wake)).start()
    _say(output, status='popout_ready', port=receiver.getsockname()[1])
    try:
        while not (glfw.window_should_close(window) or state['closing']):
            glfw.wait_events_timeout(.25)
            if not state['dirty']:
                continue
            size = glfw.get_window_size(window)
            framebuffer_size = glfw.get_framebuffer_size(window)
            if not all(size) or not all(framebuffer_size):
                continue
            with lock:
                state['dirty'] = False
                canvas = frames.compose(size)
                title = frames.title()
            if texture is None or texture.size != canvas.size:
                if texture is not None:
                    texture.release()
                texture = context.texture(canvas.size, 3, canvas.tobytes(),
                                          dtype='f1', alignment=1)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                texture.repeat_x = False
                texture.repeat_y = False
            else:
                texture.write(canvas.tobytes(), alignment=1)
            context.viewport = (0, 0, *framebuffer_size)
            texture.use(location=0)
            vertex_array.render(mode=moderngl.TRIANGLES, vertices=3)
            glfw.swap_buffers(window)
            glfw.set_window_title(window, title)
    except KeyboardInterrupt:
        pass
    finally:
        state['closing'] = True
        receiver.close()
        for resource in (texture, vertex_array, program, context):
            if resource is not None:
                resource.release()
        glfw.destroy_window(window)
        glfw.terminate()
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
