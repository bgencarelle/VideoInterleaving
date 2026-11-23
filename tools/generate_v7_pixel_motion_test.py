#!/usr/bin/env python3
"""Build the 10-second H.264 pixel-motion fixtures used for V7 testing.

One clip per picture shape.  Every element is laid out on the grid the modem
reduces a frame to (80 columns by 96 rows), so a pitch of "2" means two
reduced samples per cycle whatever the source size.  ``frame_image`` gives
the reference frame without decoding the mp4 and ``read_picture_index`` reads
the binary strip back from a picture of any size.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'modem_tests'/'fixtures'
FFMPEG = 'ffmpeg'
FPS, SECONDS = 30, 10
FRAMES = FPS*SECONDS
SHAPES = {
    '4x3': (640, 480),
    '16x9': (854, 480),
    '3x4': (480, 640),
    '5x6': (400, 480),
}
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
CAP_HEIGHT = .729          # capital height of the font, in ems

# The modem's reduced picture, and its picture rate: one per audio packet.
GRID_COLUMNS, GRID_ROWS = 80, 96
PICTURE_RATE_NUMERATOR, PICTURE_RATE_DENOMINATOR = 48000, 3920

# Layout boxes in reduced samples: (column0, row0, column1, row1).
TITLE_BOX = (1, 1, 79, 5.5)
ZONE_BOX = (1, 6, 27, 32)
PICTURE_BOX = (29, 7.5, 79, 19.5)
SOURCE_BOX = (29, 20.5, 79, 32)
BAR_LABEL_ROWS = (32.5, 35.5)
BAR_ROWS = (36, 47)
VERTICAL_BARS = (.5, 9, 1)        # first column, group width, gap
HORIZONTAL_BARS = (50.5, 5, 1)
CHECK_BOXES = ((1, 48, 13, 58, 2), (14, 48, 26, 58, 4))   # last: square size
VERTICAL_EDGE_BOX = (28, 48, 44, 58)
HORIZONTAL_EDGE_BOX = (45.5, 48, 61.5, 58)
DIAGONAL_BOX = (63, 48, 79, 58)
PATCH_BOX = (1, 59, 79, 65)
COLOUR_RAMP_BOX = (1, 65.5, 79, 68)
GREY_RAMP_BOX = (1, 68, 79, 70.5)
TEXT_BOX = (1, 71.5, 79, 87.5)
STRIP_BOX = (2, 88.5, 78, 95)

PITCHES = (8, 6, 4, 3, 2)          # reduced samples per cycle
TEXT = 'R5'
TEXT_ROWS = (16, 12, 8, 6, 4)      # capital height in reduced rows
PATCHES = (
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 128, 128), (224, 172, 138),
)
BACKGROUND = (40, 40, 40)

# Motion, per source frame.
ZONE_PHASE_STEP = .04              # cycles
BAR_STEP = .1                      # reduced samples
CHECK_STEP = .1
EDGE_STEP = .08
DIAGONAL_WIDTH = .7                # reduced samples

# Strip: white, black, STRIP_BITS data bits (most significant first), an
# even-parity bit, black, white.  Each cell is 4.75 by 6.5 reduced samples.
STRIP_BITS = 11
STRIP_CELLS = STRIP_BITS+5
STRIP_MIN_CONTRAST = 40

_FONTS = {}
_STATIC = {}


def font(size):
    size = max(4, int(round(size)))
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(FONT_PATH, size)
    return _FONTS[size]


def picture_index(index):
    """Return the 12.245 fps picture index shown during source frame `index`."""
    return (index*PICTURE_RATE_NUMERATOR)//(FPS*PICTURE_RATE_DENOMINATOR)


def pixels(size, box):
    """Convert a box in reduced samples to whole source pixels."""
    width, height = size
    c0, r0, c1, r1 = box[:4]
    return (round(c0*width/GRID_COLUMNS), round(r0*height/GRID_ROWS),
            round(c1*width/GRID_COLUMNS), round(r1*height/GRID_ROWS))


def reduced_grid(size, box):
    """Return reduced-sample coordinates of every pixel centre in a box."""
    x0, y0, x1, y1 = pixels(size, box)
    xs = (np.arange(x0, x1)+.5)*GRID_COLUMNS/size[0]
    ys = (np.arange(y0, y1)+.5)*GRID_ROWS/size[1]
    return xs[None, :], ys[:, None]


def square_wave(position, period, pixels_per_sample):
    """A 0..1 line-pair wave whose edges are one source pixel soft."""
    wave = np.sin(2*np.pi*position/period)
    return np.clip(.5+wave*period*pixels_per_sample/(2*np.pi), 0, 1)


def paste_grey(image, size, box, values):
    x0, y0, _x1, _y1 = pixels(size, box)
    grey = np.round(np.broadcast_to(
        values, (values.shape[0], values.shape[1]))*255).astype(np.uint8)
    image.paste(Image.fromarray(grey, 'L').convert('RGB'), (x0, y0))


def draw_fitted(draw, size, box, text, fill):
    """Draw text centred in a box, capitals as tall as the box allows."""
    x0, y0, x1, y1 = pixels(size, box)
    points = (y1-y0)/CAP_HEIGHT
    width = font(points).getlength(text)
    if width > x1-x0:
        points *= (x1-x0)/width
    use_font = font(points)
    cap = use_font.size*CAP_HEIGHT
    draw.text(((x0+x1)/2, (y0+y1)/2+cap/2), text, font=use_font, fill=fill,
              anchor='ms')


def bar_groups(layout):
    first, width, gap = layout
    for number, pitch in enumerate(PITCHES):
        c0 = first+number*(width+gap)
        yield pitch, c0, c0+width


def static_image(shape):
    """Everything that does not move: title, labels, colours and text."""
    if shape in _STATIC:
        return _STATIC[shape]
    size = SHAPES[shape]
    image = Image.new('RGB', size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_fitted(draw, size, TITLE_BOX, f'V7 PIXEL MOTION {shape}', 'white')

    for layout, prefix in ((VERTICAL_BARS, 'V'), (HORIZONTAL_BARS, 'H')):
        for pitch, c0, c1 in bar_groups(layout):
            label = f'{prefix}{pitch}' if c1-c0 > 6 else str(pitch)
            draw_fitted(draw, size, (c0, BAR_LABEL_ROWS[0], c1,
                                     BAR_LABEL_ROWS[1]), label, 'white')

    c0, r0, c1, r1 = PATCH_BOX
    step = (c1-c0)/len(PATCHES)
    for number, colour in enumerate(PATCHES):
        draw.rectangle(_inside(pixels(size, (
            c0+number*step, r0, c0+(number+1)*step, r1))), fill=colour)

    x0, y0, x1, y1 = pixels(size, COLOUR_RAMP_BOX)
    hue = np.linspace(0, 255, x1-x0).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], 1)
    ramp = Image.fromarray(np.broadcast_to(
        hsv[None], (y1-y0, x1-x0, 3)).copy(), 'HSV').convert('RGB')
    image.paste(ramp, (x0, y0))
    x0, y0, x1, y1 = pixels(size, GREY_RAMP_BOX)
    grey = np.round(np.linspace(0, 255, x1-x0)).astype(np.uint8)
    image.paste(Image.fromarray(np.broadcast_to(
        grey[None], (y1-y0, x1-x0)).copy(), 'L').convert('RGB'), (x0, y0))

    # The same string at stepped capital heights, sharing one baseline; the
    # 8-row copy sits above the 6- and 4-row copies to fit the narrow shapes.
    x0, y0, _x1, y1 = pixels(size, TEXT_BOX)
    row = size[1]/GRID_ROWS
    gap = 1.5*size[0]/GRID_COLUMNS
    fonts = [font(rows*row/CAP_HEIGHT) for rows in TEXT_ROWS]
    x = x0
    for use_font in fonts[:2]:
        draw.text((x, y1), TEXT, font=use_font, fill='white', anchor='ls')
        x += use_font.getlength(TEXT)+gap
    draw.text((x, y0+TEXT_ROWS[2]*row), TEXT, font=fonts[2], fill='white',
              anchor='ls')
    for use_font in fonts[3:]:
        draw.text((x, y1), TEXT, font=use_font, fill='white', anchor='ls')
        x += use_font.getlength(TEXT)+gap
    _STATIC[shape] = image
    return image


def _inside(box):
    """PIL rectangles include both corners; keep to the half-open box."""
    return (box[0], box[1], box[2]-1, box[3]-1)


def draw_zone_plate(image, size, index):
    # cos(k r^2): the ring frequency is k r / pi cycles per reduced sample,
    # so k = pi / (2 R) reaches one cycle per two samples at the nearest
    # panel edge (r = R).  The corners go past that limit.
    c0, r0, c1, r1 = ZONE_BOX
    xs, ys = reduced_grid(size, ZONE_BOX)
    radius = min(c1-c0, r1-r0)/2
    squared = (xs-(c0+c1)/2)**2+(ys-(r0+r1)/2)**2
    phase = np.pi/(2*radius)*squared-2*np.pi*ZONE_PHASE_STEP*index
    paste_grey(image, size, ZONE_BOX, .5+.5*np.cos(phase))


def draw_bars(image, size, index):
    per_column, per_row = size[0]/GRID_COLUMNS, size[1]/GRID_ROWS
    slide = BAR_STEP*index
    for pitch, c0, c1 in bar_groups(VERTICAL_BARS):
        box = (c0, BAR_ROWS[0], c1, BAR_ROWS[1])
        xs, ys = reduced_grid(size, box)
        paste_grey(image, size, box,
                   square_wave(xs-slide, pitch, per_column)+0*ys)
    for pitch, c0, c1 in bar_groups(HORIZONTAL_BARS):
        box = (c0, BAR_ROWS[0], c1, BAR_ROWS[1])
        xs, ys = reduced_grid(size, box)
        paste_grey(image, size, box,
                   square_wave(ys-slide, pitch, per_row)+0*xs)


def draw_checkers(image, size, index):
    per_column, per_row = size[0]/GRID_COLUMNS, size[1]/GRID_ROWS
    slide = CHECK_STEP*index
    for c0, r0, c1, r1, square in CHECK_BOXES:
        box = (c0, r0, c1, r1)
        xs, ys = reduced_grid(size, box)
        across = 2*square_wave(xs-slide, 2*square, per_column)-1
        down = 2*square_wave(ys-slide, 2*square, per_row)-1
        paste_grey(image, size, box, .5+.5*across*down)


def draw_edges(image, size, index):
    per_column, per_row = size[0]/GRID_COLUMNS, size[1]/GRID_ROWS
    c0, r0, c1, r1 = VERTICAL_EDGE_BOX
    xs, ys = reduced_grid(size, VERTICAL_EDGE_BOX)
    edge = c0+(EDGE_STEP*index) % (c1-c0)
    paste_grey(image, size, VERTICAL_EDGE_BOX,
               np.clip((edge-xs)*per_column+.5, 0, 1)+0*ys)

    c0, r0, c1, r1 = HORIZONTAL_EDGE_BOX
    xs, ys = reduced_grid(size, HORIZONTAL_EDGE_BOX)
    edge = r0+(EDGE_STEP*index) % (r1-r0)
    paste_grey(image, size, HORIZONTAL_EDGE_BOX,
               np.clip((edge-ys)*per_row+.5, 0, 1)+0*xs)

    c0, r0, c1, r1 = DIAGONAL_BOX
    xs, ys = reduced_grid(size, DIAGONAL_BOX)
    span = (c1-c0)+(r1-r0)
    offset = ((xs-c0)+(ys-r0)-EDGE_STEP*index) % span
    distance = np.minimum(offset, span-offset)
    paste_grey(image, size, DIAGONAL_BOX, np.clip(
        (DIAGONAL_WIDTH/2-distance)*min(per_column, per_row)+.5, 0, 1))


def strip_cells(value):
    """Return the strip's cells, True for white, for a picture index."""
    value %= 1 << STRIP_BITS
    bits = [bool(value >> shift & 1) for shift in range(STRIP_BITS-1, -1, -1)]
    return [True, False]+bits+[sum(bits) % 2 == 1, False, True]


def draw_counter(image, size, index):
    draw = ImageDraw.Draw(image)
    picture = picture_index(index)
    draw_fitted(draw, size, PICTURE_BOX, f'P{picture:03d}', 'white')
    draw_fitted(draw, size, SOURCE_BOX, f'F{index:03d}', (255, 210, 70))

    c0, r0, c1, r1 = STRIP_BOX
    draw.rectangle(_inside(pixels(size, (c0-1, r0-.5, c1+1, r1+.5))),
                   fill='black')
    step = (c1-c0)/STRIP_CELLS
    for cell, white in enumerate(strip_cells(picture)):
        if white:
            draw.rectangle(_inside(pixels(size, (
                c0+cell*step, r0, c0+(cell+1)*step, r1))), fill='white')


def frame_image(shape, index):
    """Return the PIL RGB image of source frame `index` for a shape tag.

    `shape` is one of SHAPES ('4x3', '16x9', '3x4', '5x6') and `index` counts
    30 fps source frames from zero.  The result is the picture that went into
    the encoder, so it is the reference for a decoded or received frame.
    """
    size = SHAPES[shape]
    image = static_image(shape).copy()
    draw_zone_plate(image, size, index)
    draw_bars(image, size, index)
    draw_checkers(image, size, index)
    draw_edges(image, size, index)
    draw_counter(image, size, index)
    return image


def read_picture_index(rgb_array):
    """Read the binary strip of an RGB picture; return the index or None.

    `rgb_array` is height x width x 3 at any size (a full source frame or an
    80x96 received picture) showing the whole frame.  Returns None when the
    start/end markers lack contrast, are the wrong way round, or the parity
    bit disagrees.  The index wraps at 2**STRIP_BITS.
    """
    array = np.asarray(rgb_array, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    luma = array[..., :3]@np.array([.299, .587, .114])
    height, width = luma.shape
    c0, r0, c1, r1 = STRIP_BOX
    step = (c1-c0)/STRIP_CELLS
    y0 = int(round((r0+.25*(r1-r0))*height/GRID_ROWS))
    y1 = max(y0+1, int(round((r0+.75*(r1-r0))*height/GRID_ROWS)))
    levels = []
    for cell in range(STRIP_CELLS):
        x0 = int(round((c0+(cell+.25)*step)*width/GRID_COLUMNS))
        x1 = max(x0+1, int(round((c0+(cell+.75)*step)*width/GRID_COLUMNS)))
        patch = luma[y0:y1, x0:x1]
        if not patch.size:
            return None
        levels.append(float(patch.mean()))
    white = (levels[0]+levels[-1])/2
    black = (levels[1]+levels[-2])/2
    if white-black < STRIP_MIN_CONTRAST:
        return None
    threshold = (white+black)/2
    cells = [level > threshold for level in levels]
    if cells[:2] != [True, False] or cells[-2:] != [False, True]:
        return None
    bits = cells[2:2+STRIP_BITS]
    if (sum(bits) % 2 == 1) != cells[2+STRIP_BITS]:
        return None
    value = 0
    for bit in bits:
        value = value << 1 | bit
    return value


def encode_video(shape):
    width, height = SHAPES[shape]
    output = FIXTURES/f'v7_pixel_motion_{shape}.mp4'
    command = [
        FFMPEG, '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
        f'{width}x{height}', '-framerate', str(FPS), '-i', 'pipe:0',
        '-map', '0:v:0', '-an', '-frames:v', str(FRAMES),
        '-c:v', 'libx264', '-preset', 'fast', '-threads', '1',
        '-crf', '18', '-g', str(FPS), '-pix_fmt', 'yuv420p',
        '-profile:v', 'high', '-movflags', '+faststart',
        '-fflags', '+bitexact', '-flags:v', '+bitexact',
        '-metadata', f'title=V7 Pixel Motion Test {shape}',
        '-metadata', 'comment=Pitches are reduced samples (80x96) per cycle',
        str(output),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(FRAMES):
            process.stdin.write(frame_image(shape, index).tobytes())
        process.stdin.close()
    except (BrokenPipeError, OSError):
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    stderr = process.stderr.read() if process.stderr else b''
    return_code = process.wait()
    if return_code:
        raise RuntimeError(stderr.decode(errors='replace'))
    return output


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if not Path(FONT_PATH).is_file():
        raise FileNotFoundError(FONT_PATH)
    for shape in SHAPES:
        output = encode_video(shape)
        print(f'Created {output}')
        print(f'Size: {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise
