"""Latest-picture selection, separate from GUI and audio timing."""


class LivePicture:
    def __init__(self, on_loss='hold', period=.07):
        self.on_loss = on_loss
        self.period = period
        self.seen_at = None
        self.good = None
        self.candidate = None

    def push(self, result, now):
        self.period = result.extra.get('frame_seconds', self.period)
        self.seen_at = now
        self.candidate = result
        if result.values is not None and result.extra.get('complete', False):
            self.good = result

    def current(self, now):
        if self.on_loss == 'hold':
            return self.good
        if self.seen_at is None or now-self.seen_at > 1.5*self.period+.02:
            return None
        r = self.candidate
        if r is None or r.values is None:
            return None
        if self.on_loss == 'damaged' or r.extra.get('complete', False):
            return r
        return None
