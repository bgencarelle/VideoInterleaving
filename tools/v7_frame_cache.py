"""One-picture preparation cache for the live sender's capture mailbox."""


class PreparedFrameCache:
    """Reuse only an explicitly identified capture and unchanged live settings.

    Retain the source object as well as its generation: object identity alone
    cannot identify sources that reuse an array, and an id can be recycled.
    No token means the caller has no stable capture identity, so do not reuse.
    """

    def __init__(self):
        self.frame = None
        self.key = None
        self.result = None
        self.gains = ()

    def prepare(self, frame, token, settings, compute, gains_out):
        key = (token, settings)
        if token is None or self.frame is not frame or self.key != key:
            gains = []
            result = compute(gains)
            self.frame, self.key = frame, key
            self.result, self.gains = result, tuple(gains)
        gains_out.extend(self.gains)
        return self.result


def preparation_settings(brightness, gamma, options, rotation=0, mirror=False,
                         preview=False):
    """Small immutable launch/live-control key; a reloaded kernel is distinct."""
    selection = options.get('kernel')
    kernel = (None if selection is None else
              (selection.kernel, selection.values))
    return (brightness, gamma, rotation, mirror, preview, kernel,
            tuple(sorted((name, value) for name, value in options.items()
                         if name != 'kernel')))
