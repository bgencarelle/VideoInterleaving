#!/usr/bin/env python3
"""Build the 30-second H.264 A/V sync fixture used for V7 testing."""
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT/'tmp'
OUTPUT = ROOT/'modem_tests'/'fixtures'/'v7_robot_count_sync_test.mp4'
AUDIO = TMP/'v7-robot-count-sync-test.wav'
FFMPEG = 'ffmpeg'
WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 30, 30
SAMPLE_RATE = 44100
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'

PALETTES = (
    ((12, 33, 67), (38, 178, 255)),
    ((68, 18, 25), (255, 100, 82)),
    ((9, 55, 39), (76, 255, 157)),
    ((48, 20, 72), (208, 127, 255)),
    ((72, 43, 8), (255, 210, 70)),
    ((8, 50, 69), (60, 226, 255)),
)

ONES = (
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
)


def spoken_number(value):
    if value < len(ONES):
        return ONES[value]
    tens, units = divmod(value, 10)
    return ('twenty' if tens == 2 else 'thirty') + (
        '' if units == 0 else ' '+ONES[units])


def synthesize(word, tempo):
    expression = f"flite=text='{word}':voice=kal"
    filters = f'atempo={tempo},tremolo=f=24:d=0.08,alimiter=limit=0.8'
    command = [
        FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', expression, '-af', filters,
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 'f32le', 'pipe:1',
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def make_audio():
    total_samples = SECONDS*SAMPLE_RATE
    output = np.zeros(total_samples, dtype=np.float32)
    start_offset = round(.02*SAMPLE_RATE)
    max_phrase_samples = round(.98*SAMPLE_RATE)
    for number in range(SECONDS):
        word = spoken_number(number)
        phrase = None
        for tempo in (1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35,
                      1.4, 1.45, 1.5, 1.6, 1.7):
            candidate = synthesize(word, tempo)
            if len(candidate) <= max_phrase_samples:
                phrase = candidate
                break
        if phrase is None:
            raise RuntimeError(f'Speech for {number} does not fit in one second.')

        peak = float(np.max(np.abs(phrase))) if len(phrase) else 0.0
        if peak:
            phrase *= min(.75/peak, 2.0)
        fade_in = min(round(.006*SAMPLE_RATE), len(phrase)//2)
        fade_out = min(round(.012*SAMPLE_RATE), len(phrase)//2)
        if fade_in:
            phrase[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        if fade_out:
            phrase[-fade_out:] *= np.linspace(
                1, 0, fade_out, dtype=np.float32)
        start = number*SAMPLE_RATE+start_offset
        output[start:start+len(phrase)] += phrase
        print(f'{number:02d}: {word:>12}  {len(phrase)/SAMPLE_RATE:.3f}s')

    pcm = np.clip(output, -1, 1)
    pcm = np.round(pcm*32767).astype('<i2')
    with wave.open(str(AUDIO), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def draw_center(draw, xy, text, use_font, fill, **kwargs):
    draw.text(xy, text, font=use_font, fill=fill, anchor='mm', **kwargs)


def render_frame(index):
    second, frame = divmod(index, FPS)
    background, accent = PALETTES[second//5]
    image = Image.new('RGB', (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)

    # A thick outer frame keeps the five-second palette changes visible after
    # the 640x480 picture is reduced to the modem's tiny picture grid.
    draw.rounded_rectangle((18, 18, 622, 462), radius=28,
                           outline=accent, width=10)
    draw.rounded_rectangle((43, 86, 597, 395), radius=24,
                           fill=(10, 15, 23), outline=(238, 244, 250),
                           width=5)

    draw_center(draw, (320, 47), 'V7 ROBOT A/V SYNC', font(31), 'white')
    draw_center(draw, (320, 119), f'SPOKEN: {spoken_number(second).upper()}',
                font(33), accent)
    draw_center(draw, (320, 226), f'{second:02d}', font(205), 'white')

    # SS.FF.NNN = second, frame number within that second, absolute frame.
    timecode = f'{second:02d}.{frame:02d}.{index:03d}'
    draw_center(draw, (320, 333), timecode, font(69), accent)
    draw_center(draw, (320, 374), 'SEC   FRAME   SUBFRAME',
                font(20), (236, 240, 246))

    # Thirty broad ticks make the current frame-in-second easy to see in a
    # heavily downscaled or partially damaged decode.
    left, gap, tick_width, tick_height = 48, 5, 14, 20
    for tick in range(FPS):
        x0 = left+tick*(tick_width+gap)
        color = 'white' if tick == frame else accent
        draw.rounded_rectangle(
            (x0, 415, x0+tick_width, 415+tick_height), radius=4, fill=color)
    draw_center(draw, (320, 451), f'FRAME {frame:02d} / 29',
                font(20), 'white')
    return image


def encode_video():
    command = [
        FFMPEG, '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
        f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS), '-i', 'pipe:0',
        '-i', str(AUDIO), '-map', '0:v:0', '-map', '1:a:0',
        '-frames:v', str(FPS*SECONDS), '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
        '-c:a', 'aac', '-b:a', '128k', '-ar', str(SAMPLE_RATE),
        '-ac', '1', '-t', str(SECONDS), '-movflags', '+faststart',
        '-metadata', 'title=V7 Robot Count A/V Sync Test',
        '-metadata', 'comment=SS.FF.NNN: second, frame, absolute frame number',
        str(OUTPUT),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(FPS*SECONDS):
            process.stdin.write(render_frame(index).tobytes())
        process.stdin.close()
    except (BrokenPipeError, OSError):
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    stderr = process.stderr.read() if process.stderr else b''
    return_code = process.wait()
    if return_code:
        raise RuntimeError(stderr.decode(errors='replace'))


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not Path(FONT_PATH).is_file():
        raise FileNotFoundError(FONT_PATH)
    make_audio()
    encode_video()
    print(f'Created {OUTPUT}')
    print(f'Size: {OUTPUT.stat().st_size:,} bytes')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise
