import unittest
import numpy as np
from PIL import Image
from animation_modem.transport2 import Decoded
from animation_modem.imaging import values_image
from animation_modem.preview_overlay import PreviewOverlay


class PreviewOverlayTests(unittest.TestCase):
    def setUp(self):
        self.shapes = [(2, 2)]
        self.overlay = PreviewOverlay(self.shapes)

    def result(self, value, packet, received=4, complete=True, start=False):
        r = Decoded('received', values=np.full(4, value))
        r.extra.update(packet_id=packet, received_values=received,
                       complete=complete, frame_start=start)
        return r

    def test_partial_overlays_held_picture_without_mutating_values(self):
        background = self.overlay.render(self.result(1., 1))
        partial = self.result(-1., 2, received=1, complete=False)
        original = partial.values.copy()
        got = self.overlay.render(partial)
        expected = Image.blend(background, values_image(original, self.shapes), .25)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(partial.values, original)
        np.testing.assert_array_equal(background, np.full((2, 2, 3), 255))

    def test_refinement_uses_fixed_background_and_full_frame_replaces_it(self):
        background = self.overlay.render(self.result(1., 1))
        self.overlay.render(self.result(-1., 2, 1, False))
        refined = self.result(0., 2, 2, False)
        expected = Image.blend(background, values_image(refined.values, self.shapes), .5)
        np.testing.assert_array_equal(self.overlay.render(refined), expected)
        full = self.result(-.5, 2)
        np.testing.assert_array_equal(self.overlay.render(full), values_image(full.values, self.shapes))

    def test_erased_tail_and_silence_do_not_force_sparse_image_opaque(self):
        self.overlay.render(self.result(1., 1))
        held = self.overlay.render(self.result(-1., 2, 1, False))
        np.testing.assert_array_equal(self.overlay.render(self.result(-1., 2, 1, True)), held)
        self.assertIs(self.overlay.render(Decoded('lost')), self.overlay.image)
        np.testing.assert_array_equal(self.overlay.image, held)

    def test_first_picture_and_reset_identity(self):
        first = self.result(.5, 0, 1, False, True)
        displayed = self.overlay.render(first)
        np.testing.assert_array_equal(displayed, values_image(first.values, self.shapes))
        # A reset can reuse the packet offset, even after a truncated frame.
        following = self.result(-.5, 0, 2, False, True)
        expected = Image.blend(displayed, values_image(following.values, self.shapes), .5)
        np.testing.assert_array_equal(self.overlay.render(following), expected)


if __name__ == '__main__':
    unittest.main()
