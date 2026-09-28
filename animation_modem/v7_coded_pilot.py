"""Sender-side coded profile status and pilot overlay for V7 packets."""
import math
from functools import lru_cache

import numpy as np

from animation_modem import v7


FOLD_500_STATUS = (0, 1, 0, 1)*3
PILOT_CODE = (1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1)
CHIP_SHAPE_SAMPLES = 8
_CHIP_RAMP = .5-.5*np.cos(
    np.pi*np.arange(CHIP_SHAPE_SAMPLES, dtype=float)/
    (CHIP_SHAPE_SAMPLES-1))
_CHIP_RAMP.setflags(write=False)


@lru_cache(maxsize=64)
def _tone_template(phase_offset):
    n = np.arange(v7.PULSE_FRAME, dtype=float)
    absolute = (int(phase_offset)+n) % v7.N
    phase1 = 2*np.pi*v7.PILOT_TONE_BINS[0]*absolute/v7.N
    phase3 = (2*np.pi*v7.PILOT_TONE_BINS[1]*absolute/v7.N +
              v7.PILOT_TONE_PHASES[v7.PILOT_TONE_BINS[1]])
    chips3 = np.ones(v7.PULSE_FRAME, dtype=float)
    body_start = v7.PULSE.SYNC_LEN
    symbols = tuple(
        PILOT_CODE[symbol//2] if symbol % 2 == 0 else
        (1 if FOLD_500_STATUS[symbol//2] == 0 else -1)
        for symbol in range(v7.F))
    shape = min(CHIP_SHAPE_SAMPLES, v7.CP)
    for symbol, sign in enumerate(symbols):
        start = body_start + symbol*v7.SYM
        chips3[start:start+v7.SYM] = sign
        previous = 1 if symbol == 0 else symbols[symbol-1]
        if shape > 1:
            chips3[start:start+shape] = (
                previous+(sign-previous)*_CHIP_RAMP[:shape])
    template = np.cos(phase1)+chips3*np.cos(phase3)
    template.setflags(write=False)
    return template


def add_fold500_coded_pilot(packet, counter):
    """Mark one packet as stereo Fold 500 using the receiver's coded pilot."""
    packet = np.asarray(packet)
    if packet.shape != (v7.PULSE_FRAME, 2):
        raise ValueError(f'packet must have shape ({v7.PULSE_FRAME}, 2)')
    packet_counter = int(counter)
    if packet_counter < 1 or packet_counter != counter:
        raise ValueError('packet counter must be one-based')
    body = packet[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
    body_rms = float(np.sqrt(np.mean(np.asarray(body, float)**2)))
    amplitude = math.sqrt(2.0)*body_rms*10**(v7.PILOT_TONE_REL_DB/20)
    phase_offset = ((packet_counter-1)*v7.PILOT_TONE_DURATION) % v7.N
    template = _tone_template(phase_offset)
    output = np.empty(packet.shape, dtype=np.float32)
    np.add(packet, (amplitude*template)[:, None], out=output,
           casting='unsafe')
    return output
