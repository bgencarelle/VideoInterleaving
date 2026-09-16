"""Display-only progressive overlay; decoded values are never modified."""
from PIL import Image
from .imaging import values_image


class PreviewOverlay:
    """Blend each packet against one frozen background at source resolution.

    Opacity follows received coefficient count, not elapsed time or packet
    completion: an erased tail must not force a sparse preview fully opaque.
    Without an older picture, show the first usable reconstruction directly.
    """
    def __init__(self, shapes):
        self.shapes = list(shapes)
        self.count = sum(rows*cols for rows, cols in self.shapes)
        self.image = None
        self._background = None
        self._key = None
        self._complete = False
        self._last = None

    def render(self, result):
        if result.values is None or result is self._last:
            return self.image
        extra = result.extra
        key = (extra.get('packet_id'), result.absolute, result.stamp_ms)
        # GUI queues may coalesce away frame_start. Packet identity handles
        # that; completion/frame_start also handles receiver offset resets.
        if (self._last is None or key != self._key or self._complete
                or extra.get('frame_start', False)):
            self._background = self.image
            self._key = key
        incoming = values_image(result.values, self.shapes)
        received = extra.get('received_values')
        alpha = (1. if received is None else
                 min(1., max(0., received/self.count)))
        if self._background is None or alpha >= 1:
            self.image = incoming
        else:
            # Always blend from the frozen background, never the previous
            # blend. PIL returns a new image and leaves that background intact.
            self.image = Image.blend(self._background, incoming, alpha)
        self._complete = extra.get('complete', True)
        self._last = result
        return self.image
