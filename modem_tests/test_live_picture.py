"""Fixed-format image handling and display-loss policies."""
import unittest
import numpy as np
from PIL import Image
from animation_modem.core import Decoded, SourceCoder
from animation_modem import transport3 as v
from animation_modem.imaging import plane_shapes, image_values, values_image
from animation_modem.live_picture import LivePicture


class LivePictureTests(unittest.TestCase):
    def test_hold_black_and_damaged_are_distinct(self):
        good = Decoded('received', values=np.ones(2), extra={'complete': True, 'frame_seconds': .1})
        bad = Decoded('degraded', values=np.zeros(2), extra={'complete': False, 'frame_seconds': .1})
        for choice, expected in [('hold', good), ('black', None), ('damaged', bad)]:
            state = LivePicture(choice)
            state.push(good, 0)
            state.push(bad, .1)
            self.assertIs(state.current(.11), expected)
            self.assertIs(state.current(1), good if choice == 'hold' else None)

    def test_loss_timeout_tracks_playback_speed(self):
        state = LivePicture('black')
        good = Decoded('received', values=np.ones(2), extra={'complete': True, 'frame_seconds': .4})
        state.push(good, 0)
        self.assertIs(state.current(.5), good)
        self.assertIsNone(state.current(.7))

    def test_grayscale_uses_unchanged_color_wire_format(self):
        layout = v.ALL_PRESETS['lean-v3']
        coder = SourceCoder(plane_shapes('color-lean'))
        grey = np.tile(np.linspace(20, 230, 40, dtype=np.uint8), (48, 1))
        image = Image.fromarray(grey).convert('RGB')
        audio = v.encode(image_values(image, coder.shapes), layout, coder, 1, 1, 1)
        self.assertEqual(audio.shape, (2768, 2))
        rx = v.Receiver(layout, coder)
        result = (rx.feed(audio)+rx.flush())[0]
        decoded = np.asarray(values_image(result.values, coder.shapes), float)
        self.assertEqual(result.identity, 'verified_header')
        self.assertLess(np.max(np.ptp(decoded, axis=2)), 5)

    def test_no_payload_does_not_show_fake_damaged_pixels(self):
        state = LivePicture('damaged')
        state.push(Decoded('lost'), 0)
        self.assertIsNone(state.current(.01))
