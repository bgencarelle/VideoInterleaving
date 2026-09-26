"""Native and browser X-only scope output; no audio hardware required."""
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

import scope_out
from scope_out import Scope, beam_is_parked
from tests.test_scope_web import _extract_fn


class XOnlyNativeOutputTests(unittest.TestCase):
    def test_portaudio_stream_is_opened_mono(self):
        opened = {}

        class FakeStream:
            latency = 0.0

        class FakeAudio:
            @staticmethod
            def query_devices(*_args, **_kwargs):
                return {"name": "Test DAC", "default_samplerate": 48000}

            @staticmethod
            def OutputStream(**kwargs):
                opened.update(kwargs)
                return FakeStream()

        with patch.object(scope_out, "sd", FakeAudio):
            scope = Scope(device=0, samplerate=48000, samples=96,
                          x_only=True, trigger=False)
        self.assertEqual(opened["channels"], 1)
        self.assertEqual(scope.output_channels, 1)

    def test_scope_opens_one_output_channel_and_emits_only_x(self):
        scope = Scope(device="null", samplerate=48000, samples=96,
                      x_only=True, trigger=True)
        try:
            self.assertEqual(scope.output_channels, 1)
            self.assertEqual(scope.stream._buf.shape[1], 1)
            frame = np.column_stack((np.linspace(-0.7, 0.7, 96),
                                     np.linspace(0.4, -0.4, 96))).astype(np.float32)
            scope.show_frame(frame)
            expected = scope._pending.copy()
            scope._frame, scope._pending = expected, None
            out = np.empty((len(expected), 1), dtype=np.float32)
            scope._callback(out, len(out), None, None)
            np.testing.assert_array_equal(out[:, 0], expected[:, 0])
            self.assertEqual(len(np.flatnonzero(
                (out[:-1, 0] < 0.95) & (out[1:, 0] >= 0.95))), 1)
        finally:
            scope.stream.close()

    def test_x_only_park_guard_checks_the_emitted_axis(self):
        frame = np.column_stack((np.zeros(64), np.linspace(-0.8, 0.8, 64)))
        self.assertFalse(beam_is_parked(frame))
        self.assertTrue(beam_is_parked(frame, x_only=True))

    def test_device_filter_accepts_mono_hardware_only_for_x_only(self):
        class FakeAudio:
            @staticmethod
            def query_hostapis():
                return [{"name": "Test API"}]

            @staticmethod
            def query_devices():
                return [
                    {"name": "Mono DAC", "max_output_channels": 1,
                     "hostapi": 0, "default_samplerate": 48000},
                    {"name": "Stereo DAC", "max_output_channels": 2,
                     "hostapi": 0, "default_samplerate": 48000},
                ]

        with patch.object(scope_out, "sd", FakeAudio):
            self.assertEqual([d[1] for d in scope_out.list_output_devices()],
                             ["Stereo DAC"])
            self.assertEqual([d[1] for d in scope_out.list_output_devices(1)],
                             ["Mono DAC", "Stereo DAC"])
            self.assertEqual(scope_out.resolve_device("Mono DAC", min_channels=1), 0)
            with self.assertRaises(SystemExit):
                scope_out.resolve_device("Mono DAC")


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser parity")
class XOnlyBrowserOutputTests(unittest.TestCase):
    def test_browser_reduces_interleaved_xy_to_the_x_channel(self):
        page = (Path(__file__).resolve().parents[1] / "static" /
                "scope_renderer.js").read_text()
        helper = _extract_fn(page, "xOnlyTrace")
        script = helper + "\n" + (
            "const x=xOnlyTrace(new Float32Array([1,10,2,20,-3,30]));"
            "process.stdout.write(JSON.stringify(Array.from(x)));"
        )
        result = subprocess.run(["node", "-e", script], check=True,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout, "[1,2,-3]")

    def test_audio_worklet_accepts_mono_and_stereo_buffers(self):
        page = (Path(__file__).resolve().parents[1] / "static" /
                "scope_renderer.js").read_text()
        source = re.search(r"const WORKLET_SRC = `([\s\S]*?)`;", page).group(1)
        script = """
          class AudioWorkletProcessor { constructor(){ this.port = {}; } }
          let Processor;
          function registerProcessor(_name, klass){ Processor = klass; }
        """ + source + """
          const mono = new Processor();
          mono.buf = new Float32Array([3,-2,7]);
          const m = [[new Float32Array(4)]];
          mono.process([], m);
          const stereo = new Processor();
          stereo.buf = new Float32Array([1,10,2,20]);
          const s = [[new Float32Array(3),new Float32Array(3)]];
          stereo.process([], s);
          process.stdout.write(JSON.stringify({mono:Array.from(m[0][0]),
            left:Array.from(s[0][0]),right:Array.from(s[0][1])}));
        """
        result = subprocess.run(["node", "-e", script], check=True,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout,
                         '{"mono":[3,-2,7,3],"left":[1,2,1],"right":[10,20,10]}')


if __name__ == "__main__":
    unittest.main()
