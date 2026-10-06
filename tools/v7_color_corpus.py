"""Repository assets and frozen train/validation/test schedules for V7 trials."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from animation_modem import v7


ROOT = Path(__file__).resolve().parents[1]
FACE = ROOT/'images_sbs/face/00_C_BG_faceSource_960'
FIXTURES = ROOT/'modem_tests/fixtures'
PARTITIONS = {'train': (0, 599), 'validation': (700, 1299), 'test': (1400, 2220)}
MOVIES = {'validation': ('v7_pixel_motion_4x3.mp4', 'v7_pixel_motion_5x6.mp4'),
          'test': ('v7_pixel_motion_16x9.mp4', 'v7_pixel_motion_3x4.mp4',
                   'v7_robot_count_sync_test.mp4')}
PACKET_SECONDS = v7.PULSE_FRAME/v7.RATE


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def indices(partition, count):
    first, last = PARTITIONS[partition]
    if not 1 <= count <= last-first+1:
        raise ValueError('image count outside partition bounds')
    return np.rint(np.linspace(first, last, count)).astype(int).tolist()


def face_path(index):
    return FACE/f'benFaceSource{index:04d}.jpg'


@lru_cache(maxsize=12)
def face_image(index):
    with Image.open(face_path(index)) as image:
        data = np.asarray(image.convert('RGB'))
    if data.shape[1] % 2:
        raise ValueError('invalid SBS face width')
    half = data.shape[1]//2
    alpha = data[:, half:, 0:1].astype(float)/255
    rgb = data[:, :half].astype(float)*alpha+4*(1-alpha)
    result = np.uint8(np.rint(rgb))
    result.setflags(write=False)
    return result


class Movie:
    def __init__(self, path, cache):
        self.path = Path(path)
        self.sha256 = digest_file(path)
        self.cache = Path(cache)/self.sha256
        self.cache.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height:frame=best_effort_timestamp_time',
            '-of', 'json', str(path)], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.size = (data['streams'][0]['width'], data['streams'][0]['height'])
        self.pts = np.array([float(f['best_effort_timestamp_time']) for f in data['frames']])
        if len(self.pts) < 2 or np.any(np.diff(self.pts) <= 0):
            raise ValueError(f'non-monotonic video timestamps: {path}')
        self.duration = self.pts[-1]+np.median(np.diff(self.pts))
        self._prepare()

    def _prepare(self):
        marker = self.cache/'complete.json'
        if marker.exists() and all((self.cache/f'{i:06d}.png').is_file()
                                   for i in range(len(self.pts))):
            return
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                        '-i', str(self.path), '-map', '0:v:0', '-fps_mode', 'passthrough',
                        '-start_number', '0', str(self.cache/'%06d.png')], check=True)
        if not all((self.cache/f'{i:06d}.png').is_file() for i in range(len(self.pts))):
            raise RuntimeError('video frame extraction count mismatch')
        marker.write_text(json.dumps({'sha256': self.sha256, 'pts': self.pts.tolist()}))

    def frame_index(self, timestamp):
        return int(np.clip(np.searchsorted(self.pts, timestamp+1e-9, side='right')-1,
                           0, len(self.pts)-1))

    @lru_cache(maxsize=12)
    def image(self, index):
        with Image.open(self.cache/f'{index:06d}.png') as image:
            return np.array(image.convert('RGB'))

    def record(self):
        return {'path': str(self.path.relative_to(ROOT)), 'sha256': self.sha256,
                'size': self.size, 'pts': self.pts.tolist(), 'duration': self.duration}


@dataclass
class Exposure:
    name: str
    kind: str
    source_indices: list
    partition: str
    movie: object = None

    def image(self, packet):
        index = self.source_indices[packet]
        return face_image(index) if self.movie is None else self.movie.image(index)

    @property
    def layout(self):
        image = self.image(0)
        code = v7.aspect_wire_code(image.shape[1::-1]) & 7
        from aspect_fold import layout_for_aspect_code
        return layout_for_aspect_code(code)

    def record(self):
        source = (self.movie.record() if self.movie else {
            'paths': [{'index': i, 'path': str(face_path(i).relative_to(ROOT)),
                       'sha256': digest_file(face_path(i))}
                      for i in sorted(set(self.source_indices))],
            'sbs': 'left RGB, right red alpha; background (4,4,4)'})
        return {'name': self.name, 'kind': self.kind, 'partition': self.partition,
                'layout': self.layout, 'source_indices': self.source_indices,
                'sample_timestamps': [i*PACKET_SECONDS for i in range(len(self.source_indices))],
                'source': source}


def exposures(partition, cache, assets=('stills', 'motion', 'movies'),
              stills=24, repeats=24, max_packets=None):
    result = []
    if 'stills' in assets:
        for index in indices(partition, stills):
            result.append(Exposure(f'face-static-{index:04d}', 'static',
                                   [index]*repeats, partition))
    if 'motion' in assets:
        first, last = PARTITIONS[partition]
        sequence = list(range(first, last+1))
        for hold in (1, 2):
            result.append(Exposure(f'face-motion-hold{hold}', 'motion',
                                   [i for i in sequence for _ in range(hold)], partition))
    if 'movies' in assets:
        for name in MOVIES[partition]:
            movie = Movie(FIXTURES/name, cache)
            times = np.arange(0, movie.duration-1e-9, PACKET_SECONDS)
            result.append(Exposure(Path(name).stem, 'movie',
                                   [movie.frame_index(t) for t in times], partition, movie))
    if max_packets is not None:
        for exposure in result:
            exposure.source_indices = exposure.source_indices[:max_packets]
    if not result:
        raise ValueError('no exposures selected')
    return result
