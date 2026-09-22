"""Live V7 input policy (animation_modem/v7_live_input.py): one decode per
arriving frame, a two-frame buffer, honest rates and stored-audio polarity."""
import unittest

import numpy as np
from PIL import Image

from animation_modem import v7
from animation_modem.v7_core import speed_resample
from animation_modem.v7_live_input import LiveInput, windowed_rate

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
FRAMES = 14
BLOCK = 1024


class V7LiveInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            values = v7.image_values(v7.prepare_image(source, encode_filter='nearest'),
                                     cls.model.coder.grids, encode_filter='nearest')
        cls.wire = v7.encode_pulse_stream(cls.model, [values]*FRAMES, 1, [6]*FRAMES,
                                          source_indices=list(range(FRAMES)))

    def _run(self, audio, rate=v7.RATE):
        live = LiveInput(rate=rate)
        taken, indices = [], []
        for start in range(0, len(audio), BLOCK):
            live.add(np.array(audio[start:start+BLOCK], copy=True))
            chunk = live.take(start/rate)
            if chunk is None:
                continue
            taken.append(chunk.copy())
            results, _ = v7.decode_pulse_stream(self.model, chunk, latest_only=True)
            live.decoded()
            indices += [r.diag.get('source_index') for r in results]
        return live, taken, indices, len(audio)/rate

    def test_one_decode_per_frame_and_every_frame_decoded(self):
        live, taken, indices, _ = self._run(self.wire)
        # Every frame but the last (no following header) is decoded once.
        self.assertEqual(sorted(i for i in indices if i is not None),
                         list(range(FRAMES-1)))
        self.assertLessEqual(len(taken), FRAMES)
        self.assertLessEqual(max(len(t) for t in taken), live.cap())
        self.assertLessEqual(live.cap(), 2.3*v7.PULSE_FRAME)   # locked at 1x

    def test_incoming_rate_follows_playback_speed(self):
        for speed in (1.0, .8):
            audio = v7.speed_pulse_stream(self.wire, speed).astype(np.float32)
            live, _, _, seconds = self._run(audio)
            with self.subTest(speed=speed):
                self.assertAlmostEqual(live.incoming_fps(seconds), v7.PULSE_FPS*speed, delta=.02)

    def test_device_rate_capture(self):
        """A 96 kHz device carries a 2x wire whole (carriers to 25.5 kHz);
        captured at 96 kHz the frames decode like 1x at 48 kHz, and 1x at
        96 kHz sits at the top of the accepted scale range."""
        for speed in (2.0, 1.0):
            device = np.concatenate([
                speed_resample(self.wire[i:i+v7.PULSE_FRAME], 96000, speed)
                for i in range(0, len(self.wire), v7.PULSE_FRAME)])
            live, _, indices, seconds = self._run(device, rate=96000)
            with self.subTest(speed=speed):
                self.assertEqual(sorted(i for i in indices if i is not None),
                                 list(range(FRAMES-1)))
                self.assertAlmostEqual(live.incoming_fps(seconds), v7.PULSE_FPS*speed,
                                       delta=.03)

    def test_speed_limit_follows_output_rate(self):
        self.assertAlmostEqual(v7.max_wire_speed(48000), 48000/28000)
        self.assertAlmostEqual(v7.max_wire_speed(96000), 96000/28000)
        speed = 3.4                                    # just under the 96 kHz limit
        device = np.concatenate([
            speed_resample(self.wire[i:i+v7.PULSE_FRAME], 96000, speed)
            for i in range(0, len(self.wire), v7.PULSE_FRAME)])
        _, _, indices, _ = self._run(device, rate=96000)
        self.assertEqual(sorted(i for i in indices if i is not None),
                         list(range(FRAMES-1)))

    def test_inverted_leg_is_corrected_in_the_stored_audio(self):
        live, taken, indices, _ = self._run(self.wire*np.float32([1, -1]))
        self.assertEqual(live.polarity, -1)
        for chunk in taken:
            self.assertGreater(np.dot(chunk[:, 0], chunk[:, 1]), 0)
        self.assertGreaterEqual(len([i for i in indices if i is not None]), FRAMES-2)

    def test_silence_never_decodes_and_stays_bounded(self):
        live = LiveInput()
        for step in range(40):
            live.add(np.zeros((BLOCK, 2), np.float32))
            self.assertIsNone(live.take(step*BLOCK/v7.RATE))
        self.assertLessEqual(live.buffered(), live.cap())
        self.assertEqual(live.incoming_fps(1.0), 0.0)

    def test_windowed_rate(self):
        times = [i/12 for i in range(30)]
        self.assertAlmostEqual(windowed_rate(times, times[-1]), 12.0)
        self.assertEqual(windowed_rate(times, times[-1] + 5), 0.0)   # stalled reads 0
        self.assertEqual(windowed_rate([], 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()
