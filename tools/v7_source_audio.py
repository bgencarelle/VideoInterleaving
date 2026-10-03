"""Low-latency mono source audio for the V7 side-channel sender."""
from collections import deque
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SampleBuffer:
    """Bounded float32 mono FIFO; overflow sheds oldest audio to cap latency."""

    def __init__(self, max_samples):
        self.max_samples = max(1, int(max_samples))
        self._chunks = deque()
        self._offset = 0
        self._samples = 0
        self._condition = threading.Condition()
        self.dropped = 0

    def push(self, samples):
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(values):
            return
        values = values.copy()
        with self._condition:
            if len(values) >= self.max_samples:
                self.dropped += self._samples + len(values)-self.max_samples
                values = values[-self.max_samples:]
                self._chunks.clear()
                self._offset = 0
                self._samples = 0
            overflow = self._samples+len(values)-self.max_samples
            while overflow > 0 and self._chunks:
                first = self._chunks[0]
                available = len(first)-self._offset
                take = min(overflow, available)
                self._offset += take
                self._samples -= take
                self.dropped += take
                overflow -= take
                if self._offset == len(first):
                    self._chunks.popleft()
                    self._offset = 0
            self._chunks.append(values)
            self._samples += len(values)
            self._condition.notify_all()

    def read(self, count):
        return self.read_with_status(count)[0]

    def read_with_status(self, count):
        """Read mono samples and report how many came from the FIFO."""
        count = max(0, int(count))
        result = np.zeros(count, dtype=np.float32)
        with self._condition:
            written = 0
            while written < count and self._chunks:
                first = self._chunks[0]
                available = len(first)-self._offset
                take = min(count-written, available)
                result[written:written+take] = first[
                    self._offset:self._offset+take]
                written += take
                self._offset += take
                self._samples -= take
                if self._offset == len(first):
                    self._chunks.popleft()
                    self._offset = 0
        return result, written

    def wait_for(self, count, timeout, ended=lambda: False):
        deadline = time.monotonic()+max(0.0, float(timeout))
        with self._condition:
            while self._samples < count and not ended():
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._samples >= count

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    def clear(self):
        with self._condition:
            self._chunks.clear()
            self._offset = 0
            self._samples = 0

    @property
    def available(self):
        with self._condition:
            return self._samples


class PacketAudioDelay:
    """Delay source audio by one emitted picture packet plus an optional trim."""

    def __init__(self, extra_delay_ms=0.0):
        self.extra_delay_ms = max(0.0, float(extra_delay_ms))
        self._pending = None
        self.delay_samples = None

    def apply(self, samples, packet_samples, sample_rate):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._pending is None:
            self.delay_samples = max(0, int(packet_samples) + round(
                self.extra_delay_ms*float(sample_rate)/1000.0))
            self._pending = np.zeros(self.delay_samples, dtype=np.float32)
        combined = np.concatenate((self._pending, samples))
        output = combined[:len(samples)].copy()
        self._pending = combined[len(samples):].copy()
        return output


class ClockMatchedReader:
    """Read a nominal-rate FIFO while slowly correcting independent clocks.

    The fill-level servo changes the source/output ratio by a configurable
    amount, defaulting to 0.1%, without allowing a separate capture clock to
    build unbounded monitoring latency.
    """

    MAX_CORRECTION = 0.001
    PROPORTIONAL_GAIN = 0.1

    def __init__(self, buffer, target_samples, nominal_ratio=1.0,
                 correction_limit=None, smoothing_samples=None,
                 servo_rate=None, servo_kp=0.2, servo_ki=0.01):
        self.buffer = buffer
        # Optional PI servo; `servo_rate` is source samples per second. The
        # integral carries a steady clock offset with no standing fill error,
        # so the proportional term can stay gentle: 0.2/s maps a 5 ms fill
        # error to 0.1% instead of the full correction limit. Without it the
        # proportional-only servo below is used (the sender's behaviour).
        self.servo_rate = None if servo_rate is None else float(servo_rate)
        if self.servo_rate is not None and not (
                np.isfinite(self.servo_rate) and self.servo_rate > 0):
            raise ValueError('servo rate must be positive')
        self.servo_kp = float(servo_kp)
        self.servo_ki = float(servo_ki)
        self._integral = 0.0
        self.target_samples = max(1, int(target_samples))
        self.nominal_ratio = float(nominal_ratio)
        if not np.isfinite(self.nominal_ratio) or self.nominal_ratio <= 0:
            raise ValueError('nominal sample-rate ratio must be positive')
        self.correction_limit = float(
            self.MAX_CORRECTION if correction_limit is None
            else correction_limit)
        if (not np.isfinite(self.correction_limit) or
                not 0.0 <= self.correction_limit < 1.0):
            raise ValueError('clock correction limit must be in [0, 1)')
        # Optional low-pass on the fill level, in source samples. Blocks
        # arrive in bursts (a 1,024-sample capture callback is ~13% of an
        # 80 ms target), so an unfiltered servo swings the ratio by the full
        # correction limit at the callback rate: audible pitch roughness.
        self.smoothing_samples = (None if smoothing_samples is None
                                  else max(1.0, float(smoothing_samples)))
        self._fill = None
        self._phase = 0.0
        self._pending = np.empty(0, dtype=np.float32)
        self.max_correction = 0.0
        self.current_correction = 0.0
        self.underflow = False
        self.valid_output_samples = 0

    def reset(self):
        self._fill = None
        self._phase = 0.0
        self._pending = np.empty(0, dtype=np.float32)
        self.current_correction = 0.0
        self.underflow = False
        self.valid_output_samples = 0

    def read(self, count):
        count = max(0, int(count))
        self.underflow = False
        self.valid_output_samples = 0
        if count == 0:
            return np.empty(0, dtype=np.float32)
        queued = self.buffer.available+len(self._pending)
        if self.smoothing_samples is not None:
            if self._fill is None:
                self._fill = float(queued)
            else:
                alpha = min(1.0, count*self.nominal_ratio/self.smoothing_samples)
                self._fill += alpha*(queued-self._fill)
            queued = self._fill
        if self.servo_rate is not None:
            error_seconds = (queued-self.target_samples)/self.servo_rate
            dt = count*self.nominal_ratio/self.servo_rate
            self._integral += error_seconds*dt
            if self.servo_ki > 0:
                # Anti-windup: the integral alone never exceeds the limit.
                bound = self.correction_limit/self.servo_ki
                self._integral = float(np.clip(self._integral, -bound, bound))
            correction = float(np.clip(
                self.servo_kp*error_seconds+self.servo_ki*self._integral,
                -self.correction_limit, self.correction_limit))
        else:
            error = (queued-self.target_samples)/self.target_samples
            correction = float(np.clip(
                error*self.PROPORTIONAL_GAIN,
                -self.correction_limit, self.correction_limit))
        self.current_correction = correction
        self.max_correction = max(self.max_correction, abs(correction))
        step = self.nominal_ratio*(1.0+correction)

        positions = self._phase+np.arange(count, dtype=np.float64)*step
        advanced = self._phase+count*step
        consumed = int(advanced)
        needed = max(int(np.floor(positions[-1]))+2, consumed)
        valid_source_count = len(self._pending)
        if needed > len(self._pending):
            appended, appended_count = self.buffer.read_with_status(
                needed-len(self._pending))
            valid_source_count += appended_count
            self._pending = np.concatenate((
                self._pending,
                appended))
        indexes = np.floor(positions).astype(np.intp)
        fraction = (positions-indexes).astype(np.float32)
        output = (self._pending[indexes]*(1.0-fraction) +
                  self._pending[indexes+1]*fraction)

        valid_output_count = max(0, int(np.floor(
            (valid_source_count-1-self._phase)/step))+1)
        self.valid_output_samples = min(count, valid_output_count)
        self.underflow = self.valid_output_samples < count
        if self.underflow:
            self._phase = 0.0
            self._pending = np.empty(0, dtype=np.float32)
        else:
            self._phase = advanced-consumed
            remaining = max(0, valid_source_count-consumed)
            self._pending = self._pending[
                consumed:consumed+remaining].copy()
        return output.astype(np.float32, copy=False)


class FFmpegSourceAudio:
    """Fallback paced mono PCM from a separate FFmpeg input process.

    POSIX video-file sends use SharedVideoAudioSource instead, which shares the
    demux clock and keeps embedded A/V on one input process.
    """

    def __init__(self, source, sample_rate, live=None, buffer_seconds=2.0,
                 target_samples=None, start=0.0):
        from tools.v7_capture import _LIVE_SCHEMES, _is_stream_url

        source = os.path.expanduser(str(source))
        is_stream = _is_stream_url(source)
        if not is_stream and not os.path.isfile(source):
            raise ValueError(f'No such video file: {source}')
        scheme = source.split(':', 1)[0].lower() if is_stream else ''
        is_live = (scheme in _LIVE_SCHEMES if live is None else bool(live))
        self._source = source
        self._is_live = is_live
        self._is_stream = is_stream
        self.sample_rate = int(sample_rate)
        self.buffer = SampleBuffer(round(self.sample_rate*buffer_seconds))
        self.clock_match = ClockMatchedReader(
            self.buffer, target_samples or round(self.sample_rate*.08))
        temp_dir = ROOT/'tmp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.errors = tempfile.TemporaryFile(dir=temp_dir)
        self._running = False
        self._priming = False
        self._needs_reset = False
        try:
            self._launch(start)
        except BaseException:
            self.errors.close()
            raise

    def _command(self, start=0.0):
        from tools.v7_capture import _realtime_input_options

        start = max(0.0, float(start or 0.0))
        command = ['ffmpeg', '-nostdin', '-loglevel', 'error']
        if not self._is_live:
            # A pass from a seek position plays to the end once; see
            # tools.v7_capture.FilePlayback.
            if start <= 0:
                command += ['-stream_loop', '-1']
        elif self._is_stream:
            command += ['-rw_timeout', '10000000']
        command += _realtime_input_options(self._source, self._is_live)
        if start > 0:
            command += ['-ss', f'{start:.3f}']
        command += [
            '-i', self._source, '-map', '0:a:0?', '-vn', '-sn', '-dn',
            '-af', 'aresample=async=1:first_pts=0',
            '-ac', '1', '-ar', str(self.sample_rate), '-c:a', 'pcm_f32le',
            '-f', 'f32le', 'pipe:1',
        ]
        return command

    def _launch(self, start):
        self.proc = subprocess.Popen(
            self._command(start), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=self.errors, bufsize=0)
        self._stop = threading.Event()
        self._ended = False
        self._thread = threading.Thread(
            target=self._pump, args=(self.proc, self._stop), daemon=True)
        self._running = True
        self._thread.start()

    def _pump(self, proc, stop):
        remainder = b''
        try:
            while not stop.is_set():
                data = proc.stdout.read(65536)
                if not data:
                    break
                data = remainder+data
                usable = len(data)-(len(data) % 4)
                if usable:
                    self.buffer.push(np.frombuffer(data[:usable], dtype='<f4'))
                remainder = data[usable:]
        except (OSError, ValueError):
            pass
        finally:
            self._ended = True
            self.buffer.wake()

    def read(self, count):
        silence = _transport_silence(self, count)
        return self.clock_match.read(count) if silence is None else silence

    def wait_for_samples(self, count, timeout):
        return self.buffer.wait_for(count, timeout, lambda: self._ended)

    def _end_process(self):
        self._running = False
        self._stop.set()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        self._thread.join(timeout=2)
        if self.proc.stdout is not None:
            self.proc.stdout.close()

    def halt(self):
        """Stop reading (pause or seek); read() is silent until restart."""
        if not self._running:
            return
        self._end_process()
        self.buffer.clear()
        self._needs_reset = True

    def restart(self, start=0.0):
        """Read again from a position, in step with a reopened video."""
        self.halt()
        self._priming = True
        self._launch(start)

    def close(self):
        self._end_process()
        self.errors.close()


def _transport_silence(source, count):
    """Zeros while a file soundtrack is halted or refilling, else None.

    Runs on the thread that reads the soundtrack, which is also the only
    thread that touches the clock-matched reader: a pause or seek only sets
    flags, and the reader is reset here before its next use.
    """
    if not source._running:
        return np.zeros(max(0, int(count)), dtype=np.float32)
    if source._priming:
        # Refill to the usual fill level before reading again, as at start.
        if (source.buffer.available < source.clock_match.target_samples and
                not source._ended):
            return np.zeros(max(0, int(count)), dtype=np.float32)
        source._priming = False
    if source._needs_reset:
        source._needs_reset = False
        source.clock_match.reset()
    return None


def _probe_stream_types(text):
    """Stream types named in ffprobe's JSON (or older csv) output, or None
    when nothing recognisable is there."""
    import json
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    found = set()
    if isinstance(data, dict):
        streams = list(data.get('streams') or [])
        for program in data.get('programs') or []:
            streams += list(program.get('streams') or [])
        for stream in streams:
            kind = str((stream or {}).get('codec_type') or '').strip().lower()
            if kind:
                found.add(kind)
        return found
    for line in str(text).replace('|', ',').splitlines():
        for field in line.split(','):
            field = field.strip().lower()
            if field in ('video', 'audio', 'subtitle', 'data', 'attachment'):
                found.add(field)
    return found or None


class SharedVideoAudioSource:
    """One FFmpeg demux/clock feeding the video and embedded-audio paths."""

    def __init__(self, source, sample_rate, width=320, scale_flags='bicubic',
                 live=None, target_samples=None, buffer_seconds=2.0,
                 preserve_size=False, start=0.0):
        from tools.v7_capture import _LIVE_SCHEMES, _is_stream_url

        if not hasattr(os, 'mkfifo'):
            raise NotImplementedError('shared FFmpeg pipes require POSIX FIFOs')
        source = os.path.expanduser(str(source))
        is_stream = _is_stream_url(source)
        if not is_stream and not os.path.isfile(source):
            raise ValueError(f'No such video file: {source}')
        if int(width) < 1:
            raise ValueError('capture width must be positive')
        if scale_flags not in ('neighbor', 'area', 'bilinear', 'bicubic',
                               'lanczos'):
            raise ValueError(f'Unsupported FFmpeg scale flags: {scale_flags}')

        ffmpeg = shutil.which('ffmpeg')
        ffprobe = shutil.which('ffprobe')
        if not ffmpeg or not ffprobe:
            raise RuntimeError('FFmpeg and ffprobe are required for embedded '
                               'video audio capture.')
        # JSON, not csv: csv output varies between FFmpeg versions and with
        # stream side data ("video," for a rotated or HDR clip, extra lines
        # for programs), and a misread list refuses a good file.
        probe = [ffprobe, '-v', 'error', '-show_entries',
                 'stream=codec_type', '-of', 'json']
        if is_stream:
            probe += ['-rw_timeout', '10000000']
        probe.append(source)
        try:
            result = subprocess.run(
                probe, capture_output=True, text=True, encoding='utf-8',
                errors='replace', timeout=15, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Timed out checking the video soundtrack.') from exc
        if result.returncode:
            detail = (result.stderr or '').strip()
            raise RuntimeError('Could not inspect the video soundtrack.'+
                               (f'\n{detail}' if detail else ''))
        stream_types = _probe_stream_types(result.stdout or '')
        if stream_types is None:
            # The list could not be read at all: do not refuse the file on
            # that; FFmpeg itself reports a source it cannot play.
            stream_types = {'video', 'audio'}
        if 'video' not in stream_types:
            found = ', '.join(sorted(stream_types)) or 'none'
            raise ValueError('video source contains no video stream '
                             f'(streams found: {found})')
        self.has_audio = 'audio' in stream_types

        scheme = source.split(':', 1)[0].lower() if is_stream else ''
        is_live = (scheme in _LIVE_SCHEMES if live is None else bool(live))
        if is_live and not is_stream:
            raise ValueError('live capture requires a stream URL')
        directory = ROOT/'tmp'
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix='v7-av-', dir=directory)
        self.video_path = Path(self._temporary.name)/'video.ppm'
        self.audio_path = Path(self._temporary.name)/'audio.f32'
        self.video_pipe = None
        self.audio_pipe = None
        self.proc = None
        self.errors = tempfile.TemporaryFile(dir=directory)
        self.buffer = (SampleBuffer(round(int(sample_rate)*buffer_seconds))
                       if self.has_audio else None)
        self.clock_match = (ClockMatchedReader(
            self.buffer, target_samples or round(int(sample_rate)*.08))
                           if self.has_audio else None)
        self.sample_rate = int(sample_rate)
        self._source = source
        self._ffmpeg = ffmpeg
        self._is_live = is_live
        self._is_stream = is_stream
        self._width = int(width)
        self._scale_flags = scale_flags
        self._preserve_size = bool(preserve_size)
        self._stop = threading.Event()
        self._ended = not self.has_audio
        self._closed = False
        self._close_lock = threading.RLock()
        self._audio_thread = None
        self._launch_id = 0
        self._running = False
        self._priming = False
        self._needs_reset = False
        self._reader = None
        try:
            os.mkfifo(self.video_path)
            if self.has_audio:
                os.mkfifo(self.audio_path)
            grab = self._launch(start)
        except BaseException:
            self.errors.close()
            self._temporary.cleanup()
            raise
        grab.close = self.close
        self.video_grab = grab

    def _command(self, start=0.0):
        from tools.v7_capture import (_realtime_input_options,
                                      video_scale_filter)

        start = max(0.0, float(start or 0.0))
        command = [self._ffmpeg, '-nostdin', '-y', '-loglevel', 'error']
        if not self._is_live:
            # FFmpeg cannot combine an input seek with -stream_loop: a pass
            # from a seek position plays to the end once, and FilePlayback
            # then reopens the looping reader from the beginning.
            if start <= 0:
                command += ['-stream_loop', '-1']
        elif self._is_stream:
            command += ['-rw_timeout', '10000000']
        command += _realtime_input_options(self._source, self._is_live)
        if start > 0:
            command += ['-ss', f'{start:.3f}']
        command += ['-i', self._source, '-map', '0:v:0']
        video_filter = video_scale_filter(
            self._source, self._width, self._scale_flags, self._preserve_size)
        if video_filter is not None:
            command += ['-vf', video_filter]
        command += ['-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
                    '-c:v', 'ppm', '-f', 'image2pipe',
                    str(self.video_path)]
        if self.has_audio:
            command += [
                '-map', '0:a:0', '-af', 'aresample=async=1:first_pts=0',
                '-ac', '1', '-ar', str(self.sample_rate),
                '-c:a', 'pcm_f32le', '-f', 'f32le', str(self.audio_path),
            ]
        return command

    def _launch(self, start=0.0, prime=False):
        """Start one FFmpeg process for picture and soundtrack at `start`.

        Returns its frame reader. With `prime` (a reopen after a pause or
        seek) the first picture waits until the soundtrack has refilled to
        its usual level, so the two resume together as they start together.
        """
        from tools.v7_capture import CaptureEndOfStream, _read_ppm

        start = max(0.0, float(start or 0.0))
        tail = start > 0 and not self._is_live
        opened = []
        proc = None
        try:
            proc = subprocess.Popen(
                self._command(start), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=self.errors, bufsize=0)
            # Each FIFO open rendezvous with FFmpeg's matching output open.
            # Open in command order; FFmpeg opens all outputs before encoding.
            opened.append(os.open(self.video_path, os.O_RDONLY))
            if self.has_audio:
                opened.append(os.open(self.audio_path, os.O_RDONLY))
            video_pipe = os.fdopen(opened[0], 'rb', buffering=0)
            audio_pipe = (os.fdopen(opened[1], 'rb', buffering=0)
                          if self.has_audio else None)
        except BaseException:
            for descriptor in opened:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
            raise
        stop = threading.Event()
        with self._close_lock:
            if self._closed:
                video_pipe.close()
                if audio_pipe is not None:
                    audio_pipe.close()
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=2)
                raise RuntimeError('shared video capture is closed')
            self._launch_id += 1
            launch_id = self._launch_id
            self.proc = proc
            self.video_pipe = video_pipe
            self.audio_pipe = audio_pipe
            self._stop = stop
            self._ended = not self.has_audio
            self._priming = bool(prime and self.has_audio)
            self._audio_thread = None
            if self.has_audio:
                self._audio_thread = threading.Thread(
                    target=self._pump_audio, args=(audio_pipe, stop),
                    daemon=True)
                self._audio_thread.start()
            self._running = True
        waiting = [bool(prime and self.has_audio)]

        def grab():
            if waiting[0]:
                waiting[0] = False
                target = self.clock_match.target_samples
                self.buffer.wait_for(
                    target, target/self.sample_rate+.5,
                    lambda: self._ended or stop.is_set())
            try:
                return _read_ppm(video_pipe)
            except RuntimeError as exc:
                if tail:
                    code = proc.poll()
                    if code is None:
                        try:
                            code = proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
                    if code == 0:
                        raise CaptureEndOfStream from exc
                detail = ''
                if proc.poll() is not None and not self._closed:
                    self.errors.flush()
                    self.errors.seek(0)
                    detail = self.errors.read().decode(
                        'utf-8', errors='replace').strip()
                message = f'FFmpeg shared video capture failed: {exc}'
                if detail:
                    message += f'\n{detail}'
                raise RuntimeError(message) from exc

        # The shared FFmpeg input is already paced; drain it continuously like
        # video_source so its output pipe cannot stall the audio stream.
        grab.paced = True

        def reader():
            return grab()

        def release():
            try:
                video_pipe.close()
            except OSError:
                pass

        reader.paced = True
        reader.close = lambda: self._halt(launch_id)
        reader.release = release
        self._reader = reader
        return grab

    def playback_reader(self):
        """The running process as a tools.v7_capture.FilePlayback reader:
        closing it stops this process only, and the source can reopen."""
        return self._reader

    def reopen(self, start=0.0):
        """Restart picture and soundtrack together at a position."""
        with self._close_lock:
            if self._closed:
                raise RuntimeError('shared video capture is closed')
            launch_id = self._launch_id
            video_pipe = self.video_pipe
        self._halt(launch_id)
        if video_pipe is not None:
            video_pipe.close()
        # Not under the lock: the FIFO opens wait for FFmpeg.
        self._launch(start, prime=True)
        return self._reader

    def _halt(self, launch_id):
        """Stop one process (pause, seek, end of a pass); read() is silent
        until the next one runs. A call for an older process does nothing."""
        with self._close_lock:
            if launch_id != self._launch_id or not self._running:
                return
            self._running = False
            self._stop.set()
            proc = self.proc
            ended_by_itself = proc is None or proc.poll() is not None
            if not ended_by_itself:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            thread, self._audio_thread = self._audio_thread, None
            pipe, self.audio_pipe = self.audio_pipe, None
            if thread is not None:
                thread.join(timeout=2)
            if pipe is not None:
                pipe.close()
            if self.buffer is not None:
                if not ended_by_itself:
                    # Interrupted: what is queued belongs to the old position.
                    # A pass that reached the end of the file keeps its last
                    # samples; they play before the repeat.
                    self.buffer.clear()
                self._needs_reset = True

    def _pump_audio(self, pipe, stop):
        remainder = b''
        try:
            while not stop.is_set():
                data = pipe.read(65536)
                if not data:
                    break
                data = remainder+data
                usable = len(data)-(len(data) % 4)
                if usable:
                    self.buffer.push(np.frombuffer(data[:usable], dtype='<f4'))
                remainder = data[usable:]
        except (OSError, ValueError):
            pass
        finally:
            self._ended = True
            if self.buffer is not None:
                self.buffer.wake()

    def read(self, count):
        if not self.has_audio:
            return np.zeros(max(0, int(count)), dtype=np.float32)
        silence = _transport_silence(self, count)
        return self.clock_match.read(count) if silence is None else silence

    def wait_for_samples(self, count, timeout):
        if not self.has_audio:
            return False
        return self.buffer.wait_for(count, timeout, lambda: self._ended)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._running = False
            self._stop.set()
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
            thread = self._audio_thread
        if thread is not None:
            thread.join(timeout=2)
        for pipe in (self.video_pipe, self.audio_pipe):
            if pipe is not None:
                pipe.close()
        self.errors.close()
        self._temporary.cleanup()


class DeviceSourceAudio:
    """Selected sounddevice input with silence and reconnect on device loss."""

    def __init__(self, device, sample_rate, input_side='mix',
                 buffer_seconds=2.0, target_samples=None,
                 sounddevice_module=None, status_callback=None):
        if sounddevice_module is None:
            import sounddevice as sounddevice_module
        from tools.v7_device_recovery import (
            device_identity, DEVICE_RATE_FRAME_INTERVAL)

        self.sd = sounddevice_module
        self.device = device
        self.sample_rate = int(sample_rate)
        self.input_side = input_side
        self.buffer_seconds = float(buffer_seconds)
        self.target_seconds = (float(target_samples)/self.sample_rate
                               if target_samples else .08)
        self.identity = device_identity(self.sd, device, 'input')
        self.status_callback = status_callback
        self.error = None
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self.stream = None
        self.channels = 0
        self.input_rate = self.sample_rate
        self.device_default_rate = None
        self.buffer = SampleBuffer(round(self.sample_rate*self.buffer_seconds))
        self.clock_match = ClockMatchedReader(
            self.buffer, round(self.sample_rate*self.target_seconds))
        try:
            self._open_stream(device)
        except Exception as exc:
            raise ValueError(f'cannot open source-audio input: {exc}') from exc
        self._monitor_interval = DEVICE_RATE_FRAME_INTERVAL
        self._monitor_thread = threading.Thread(
            target=self._monitor_device,
            name='v7-source-audio-device-monitor', daemon=True)
        if callable(getattr(self.sd, 'query_hostapis', None)):
            self._monitor_thread.start()
        else:
            self._monitor_thread = None

    def _report(self, error):
        error = None if error is None else str(error)
        with self._lock:
            if error == self.error:
                return
            self.error = error
        if self.status_callback is not None:
            try:
                self.status_callback(error)
            except Exception:
                pass

    def _callback(self, indata, _frames, _timing, _status):
        if self._closed.is_set() or self.error is not None:
            return
        values = np.asarray(indata, dtype=np.float32)
        if values.ndim == 1:
            mono = values
        elif self.channels == 1:
            mono = values[:, 0]
        elif self.input_side == 'left':
            mono = values[:, 0]
        elif self.input_side == 'right':
            mono = values[:, 1]
        else:
            mono = (values[:, 0]+values[:, 1])*0.5
        self.buffer.push(mono)

    def _close_stream(self):
        with self._lock:
            stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _open_stream(self, device):
        info = self.sd.query_devices(device, 'input')
        maximum = int(info.get('max_input_channels') or 0)
        if maximum < 1:
            raise ValueError('selected source-audio device has no input channels')
        channels = min(maximum, 2)
        if self.input_side == 'right' and channels < 2:
            raise ValueError('right input selection requires a stereo audio device')
        native_rate = float(info.get('default_samplerate') or self.sample_rate)
        rates = [self.sample_rate]
        if abs(native_rate-self.sample_rate) > .5:
            rates.append(native_rate)
        last_error = None
        for rate in rates:
            stream = None
            try:
                stream = self.sd.InputStream(
                    device=device, channels=channels,
                    samplerate=rate, dtype='float32', callback=self._callback)
                stream.start()
                actual_rate = float(getattr(stream, 'samplerate', rate) or rate)
                if not np.isfinite(actual_rate) or actual_rate <= 0:
                    raise ValueError('input device reported an invalid sample rate')
            except Exception as exc:
                last_error = exc
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                continue
            with self._lock:
                self.device = device
                self.channels = channels
                self.input_rate = actual_rate
                self.device_default_rate = native_rate
                self.buffer.max_samples = max(
                    1, int(round(actual_rate*self.buffer_seconds)))
                self.buffer.clear()
                self.clock_match = ClockMatchedReader(
                    self.buffer,
                    round(actual_rate*self.target_seconds),
                    nominal_ratio=actual_rate/self.sample_rate)
                self.stream = stream
            self._report(None)
            return
        raise ValueError(f'cannot open source-audio input at '
                         f'{self.sample_rate} Hz: {last_error}')

    def _monitor_device(self):
        from tools.v7_device_recovery import (
            DeviceRateDebouncer, query_device_snapshot)

        stable = DeviceRateDebouncer()
        while not self._closed.wait(self._monitor_interval):
            with self._lock:
                stream = self.stream
                device = self.device
                old_rate = self.device_default_rate
            try:
                index, _info, identity, rate = query_device_snapshot(
                    self.sd, device, self.identity, 'input')
            except Exception as exc:
                self._close_stream()
                self.buffer.clear()
                stable.reset()
                self._report(f'source-audio input unavailable: {exc}')
                continue
            try:
                active = bool(stream is not None and stream.active)
            except Exception:
                active = False
            changed_rate = (old_rate is not None and
                            abs(rate-old_rate) > .5)
            if active and not changed_rate and index == device:
                stable.reset()
                continue
            if changed_rate:
                self._close_stream()
                self.buffer.clear()
                self._report(
                    f'source-audio sample rate changed to {rate:g}Hz; '
                    'waiting to reconnect')
            elif not active:
                self._close_stream()
                self.buffer.clear()
                self._report('source-audio input stopped; waiting to reconnect')
            elif index != device:
                self._close_stream()
                self.buffer.clear()
                self._report('source-audio device index changed; reconnecting')
            if stable.observe(identity, rate):
                if self._closed.is_set():
                    return
                try:
                    self._open_stream(index)
                except Exception as exc:
                    self._report(f'source-audio input reconnect failed: {exc}')
                stable.reset()

    def read(self, count):
        with self._lock:
            reader = self.clock_match
        return reader.read(count)

    def wait_for_samples(self, count, timeout):
        return self.buffer.wait_for(count, timeout)

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self._close_stream()
        monitor = getattr(self, '_monitor_thread', None)
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=1)
