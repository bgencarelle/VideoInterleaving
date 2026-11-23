import re
import unittest

from ascii_delta import TerminalDelta, read_picture


def picture(rows, colours=None):
    parts = ['\x1b[?2026h\x1b[40m']
    for y, row in enumerate(rows):
        parts.append(f'\x1b[{y+1};1H\x1b[37m')
        for x, char in enumerate(row):
            colour = colours[y][x] if colours else (50, 90, 130)
            parts.append('\x1b[38;2;%d;%d;%dm%s' % (*colour, char))
    parts.append('\x1b[0m\x1b[?2026l')
    return ''.join(parts).encode()


def apply(payload, size, old=None):
    """Independent small terminal emulator; retain prior cells for delta tests."""
    cells = list(old) if old is not None else [None] * (size[0] * size[1])
    x = y = 0
    colour = b'\x1b[37m'
    for token in re.findall(rb'\x1b\[[0-9;?]*[A-Za-z]|[^\x1b]+', payload):
        if token.startswith(b'\x1b['):
            if token.endswith(b'H'):
                y, x = (int(v)-1 for v in token[2:-1].split(b';'))
            elif token.endswith(b'm'):
                if token == b'\x1b[0m':
                    colour = b'\x1b[37m'
                elif token != b'\x1b[40m':
                    colour = token
        else:
            for char in token.decode():
                cells[y*size[0]+x] = (char.encode(), None if char == ' ' else colour)
                x += 1
    return cells


class DeltaTests(unittest.TestCase):
    def test_exact_random_moving_screens_and_skipped_frames(self):
        import random
        rng = random.Random(9042)
        size = (17, 9)
        rows = [list('M' * size[0]) for _ in range(size[1])]
        colours = [[(50, 90, 130)] * size[0] for _ in range(size[1])]
        encoder = TerminalDelta()
        screen = None
        for frame in range(150):
            for _ in range(rng.randrange(80)):
                x, y = rng.randrange(size[0]), rng.randrange(size[1])
                rows[y][x] = rng.choice(' M8.@é')
                colours[y][x] = tuple(rng.randrange(256) for _ in range(3))
            full = picture(rows, colours)
            if frame % 7 == 3:
                continue  # Producer frames never seen by this client.
            delta = encoder.encode(full, size)
            self.assertLessEqual(len(delta), len(full))
            screen = apply(delta, size, screen)
            self.assertEqual(screen, apply(full, size))

    def test_unchanged_frame_is_constant_size(self):
        encoder = TerminalDelta()
        full = picture(['M' * 60] * 40)
        encoder.encode(full, (60, 40))
        self.assertLess(len(encoder.encode(full, (60, 40))), 40)

    def test_space_tints_are_invisible_but_new_glyph_is_coloured(self):
        encoder = TerminalDelta()
        first = picture([' '], [[(100, 0, 0)]])
        screen = apply(encoder.encode(first, (1, 1)), (1, 1))
        second = picture(['M'], [[(0, 100, 0)]])
        self.assertEqual(apply(encoder.encode(second, (1, 1)), (1, 1), screen),
                         apply(second, (1, 1)))

    def test_resize_reset_and_independent_clients(self):
        a, b = TerminalDelta(), TerminalDelta()
        full = picture(['MMM', '   '])
        a.encode(full, (3, 2))
        self.assertEqual(apply(b.encode(full, (3, 2)), (3, 2)), apply(full, (3, 2)))
        small = picture(['X'])
        self.assertEqual(apply(a.encode(small, (1, 1)), (1, 1)), apply(small, (1, 1)))
        a.reset()
        self.assertEqual(apply(a.encode(full, (3, 2)), (3, 2)), apply(full, (3, 2)))

    def test_unknown_or_incomplete_payload_falls_back_and_forgets(self):
        encoder = TerminalDelta()
        encoder.encode(picture(['M']), (1, 1))
        for bad in (b'\x1b[2J', b'\x1b[1;1H\x1b[41mM', b'\xff'):
            self.assertEqual(encoder.encode(bad, (1, 1)), bad)
            self.assertIsNone(encoder.cells)

    def test_read_picture_rejects_out_of_bounds(self):
        with self.assertRaises(ValueError):
            read_picture(picture(['MM']), (1, 1))


if __name__ == '__main__':
    unittest.main()
