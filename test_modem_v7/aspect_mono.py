"""Aspect-matched mono colour Fold 500 (experimental ``aspect-mono-500``).

The mono wire is mono-colour-500's: every packet carries the same 1,264
M-only slots (no tail rotation, no cross-packet memory), ranked with chroma
variance weighted x4, and the weakest 500 luma slots of that fresh set fold
the next 500 luma coefficients (16 of them signature slots; companded
guests with step D = 1.0, as mono-colour-500). The
packet framing, pilots, metadata and EOF marker are unchanged.

What changes is which coefficients those ranks refer to: the 2,880 positions
of an aspect layout from ``aspect_fold`` (the same frozen, hash-pinned
tables, with V7's 1,920 / 480 / 480 plane split) instead of the fixed 5:6
48x40 / 24x20 corners. A 16:9 source therefore spends its luma on more
horizontal and fewer vertical frequencies.

Signalling: coded status and pulse preamble ID 3 (MONO_OFF; the rotating
mono wire that first used it is retired). The layout
is signalled: ``auto`` follows the aspect code in each packet's metadata,
and a fixed sender layout is sent as its code with the metadata screen bit.
"""
from dataclasses import replace

import numpy as np

from common import Grids, v7
from folding import FoldCodec
import tone_code
from aspect_fold import (LAYOUT_CHOICES, TABLES_SHA256, AspectFoldWire,
                         layout_for_aspect_code)
from mono_video import (CHROMA_RANK_WEIGHT, FOLD_SLOTS, FRESH_SLOTS,
                        MONO_COLOUR_COMPAND, MONO_COLOUR_FOLD_D, SIGNATURE_SLOTS, MonoFreshFoldWire,
                        colour_fold_sets, colour_order, fresh_rank_tables)

PROFILE = 'aspect-mono-500'
STATUS_MODE = tone_code.MONO_OFF
TABLE_FORMAT = 'v7-aspect-mono-colour-fold-1'
# The mono wire has no tail; use the tables with V7's plane split.
BASE_TAIL = 'chroma'


class AspectMonoCodec(FoldCodec):
    """mono-colour-500's luma-only fold over an aspect layout."""

    def __init__(self, model, layout, D=MONO_COLOUR_FOLD_D):
        self.model, self.M, self.layout = model, FOLD_SLOTS, layout
        self.filter, self.conf_min = 'box', .9
        self.design_db, self.fitted_on = 40.0, f'aspect layout {layout} (analytic)'
        self.signature, self.noise_max = SIGNATURE_SLOTS, .3
        self.grid = Grids(v7.V7_GRIDS)
        self.kept = np.asarray(model.coder.positions, dtype=np.int64)
        order, hosts, guests = colour_fold_sets(model)
        self.sent_model_indices = np.asarray(order[:FRESH_SLOTS], dtype=int)
        self.hosts = np.asarray(hosts, dtype=int)
        self.guest_model_indices = np.asarray(guests, dtype=int)
        self.guests = self.kept[self.guest_model_indices]
        self.sd_host = np.sqrt(model.lam[self.hosts])
        self.sd_guest = np.sqrt(model.lam[self.guest_model_indices])
        self.train = []
        self.set_step(D)
        if MONO_COLOUR_COMPAND is not None:
            self.use_compand(MONO_COLOUR_COMPAND['limit'],
                             MONO_COLOUR_COMPAND['mu'], D,
                             MONO_COLOUR_COMPAND['guest_noise_max'])
        self._set_identity()

    def table(self):
        table = FoldCodec.table(self)
        table.update({
            'format': TABLE_FORMAT, 'layout': self.layout,
            'fresh_slots': FRESH_SLOTS,
            'chroma_rank_weight': CHROMA_RANK_WEIGHT,
            'status_mode': STATUS_MODE,
            'aspect_tables_sha256': TABLES_SHA256,
        })
        return table


class AspectMonoWire(MonoFreshFoldWire):
    """Mono video glue: mono-colour-500's wire over aspect layouts."""

    wire_profile = PROFILE
    status_mode = STATUS_MODE

    def __init__(self, model, train_frames=None, side='both', layout='auto'):
        if layout not in LAYOUT_CHOICES:
            raise ValueError(f'unknown aspect layout {layout!r}; choose {LAYOUT_CHOICES}')
        self.layout = layout
        self._aspect = AspectFoldWire(layout, BASE_TAIL)
        super().__init__(model, train_frames=train_frames, side=side)
        # Build every layout this end may use before any audio runs.
        layouts = ((layout,) if layout != 'auto' else tuple(dict.fromkeys(
            layout_for_aspect_code(code) for code in range(8))))
        for name in layouts:
            self.model_for(model, name)

    def layout_for(self, aspect_code):
        return self._aspect.layout_for(aspect_code)

    def coefficient_order(self, model):
        return colour_order(model)

    def model_for(self, model, layout=None):
        """Mono model for ``layout`` (default: the aspect code 0 layout)."""
        if int(model.encoding_type) != v7.ENCODING_FILTER_CODES['box']:
            raise ValueError('aspect-mono-500 requires the box model')
        if id(model) in self._codec_by_model_id:
            return model                                # already a mono model
        layout = layout or self.layout_for(0)
        key = (id(model), layout)
        if key not in self._models:
            aspect = self._aspect.model_for(model, layout)
            ranks = fresh_rank_tables(aspect, self.coefficient_order(aspect))
            priors = tuple(v7.block_priors(aspect.gain, aspect.lam, rank)
                           for rank in ranks)
            mono_model = replace(
                aspect, rank_tables=ranks, block_prior_tables=priors,
                block_prior_tables32=tuple(
                    np.asarray(table, dtype=np.float32) for table in priors),
                scale=float(aspect.scale*np.sqrt(2.0)))
            mono_model._mono_wire_profile = self.wire_profile
            codec = AspectMonoCodec(aspect, layout)
            self._models[key] = mono_model
            self._base_models[key] = model
            self._codecs[key] = codec
            self._codec_by_model_id[id(mono_model)] = codec
        return self._models[key]

    def _packet_model(self, model, aspect_code):
        layout = self.layout_for(aspect_code)
        if layout is None:
            raise ValueError(f'no aspect layout for aspect code {aspect_code!r}')
        return self.model_for(model, layout)
