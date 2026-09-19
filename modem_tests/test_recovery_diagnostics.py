"""Recovery traces must explain losses without changing the decode decisions."""
import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from animation_modem import engines as ENG
from animation_modem.audio_common import InputLevel, pcm
from utilities import modem_v3_check as check


class RecoveryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.engine = ENG.get_engine('v5')
        self.coder, _ = self.engine.coder_for('hd-dwt')
        self.values = np.full(self.coder.source_count, .1)

    def receiver(self, diagnostics=True):
        return self.engine.receiver(coder=self.coder, diagnostics=diagnostics,
                                    input_rate=48000, coders=ENG.coders_for(self.engine.wire))

    def packet(self, number):
        return self.engine.encode(self.values, self.coder, number, 1, 1)

    def test_trace_is_transparent_and_recovers_after_either_leg_is_muted(self):
        for channel in (0, 1):
            packets = []
            for n in range(1, 10):
                packet = self.packet(n)
                if 3 <= n <= 5:
                    packet[:, channel] = 0
                packets.append(packet)
            audio = np.concatenate(packets)
            traced, normal = self.receiver(), self.receiver(False)
            a, b = [], []
            for offset in range(0, len(audio), 256):
                a.extend(traced.feed(audio[offset:offset+256]))
                b.extend(normal.feed(audio[offset:offset+256]))
            a += traced.flush()
            b += normal.flush()
            self.assertEqual([r.absolute for r in a], [r.absolute for r in b])
            self.assertEqual([r.absolute for r in a[-3:]], [7, 8, 9])
            for x, y in zip(a, b):
                self.assertEqual(x.status, y.status)
                np.testing.assert_array_equal(x.values, y.values)
                self.assertIn('decode_attempts', x.extra)
                self.assertNotIn('decode_attempts', y.extra)
            state = traced.diagnostic_state()
            self.assertGreater(state['decode_counts']['verified'], 0)
            self.assertEqual(state['last_decode']['frame'], 9)
            self.assertEqual(len(state['pulse_channels']), 2)
            json.dumps(state)

    def test_silence_reports_no_preamble_and_reset_is_visible(self):
        rx = self.receiver()
        self.assertEqual(rx.feed(np.zeros((48000, 2), np.float32)), [])
        state = check.recovery_record(rx, InputLevel())
        self.assertGreater(state['decode_counts']['no_preamble'], 0)
        self.assertIsNone(state['last_decode'])
        self.assertEqual(state['input_gain'], [1., 1.])
        self.assertEqual(state['receiver_resets'], 0)
        rx.reset()
        self.assertEqual(rx.diagnostic_state()['receiver_resets'], 1)

    def test_failed_packet_attempts_remain_visible(self):
        rx = self.receiver()
        audio = self.packet(1)
        from animation_modem.core import SYNC_LEN
        audio[SYNC_LEN:] = 0  # intact preamble, no usable header or picture
        self.assertEqual(rx.feed(audio) + rx.flush(), [])
        state = rx.diagnostic_state()
        self.assertGreater(state['decode_counts']['lost'], 0)
        self.assertEqual(state['last_decode']['identity'], 'unknown')
        self.assertEqual(len(state['last_decode']['attempts']), 3)

    def test_offline_verbose_heartbeat_without_any_decoded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'silence.wav'
            with wave.open(str(path), 'wb') as sink:
                sink.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
                sink.writeframes(pcm(np.zeros((60000, 2), np.float32)))
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                check.main(['read', '--wav', str(path), '--verbose'])
            records = [json.loads(line) for line in stderr.getvalue().splitlines() if line.startswith('{')]
            self.assertGreaterEqual(len(records), 2)
            self.assertTrue(records[-1]['final'])
            self.assertEqual(records[-1]['decode_counts']['pictures'], 0)
            self.assertGreater(records[-1]['decode_counts']['no_preamble'], 0)
