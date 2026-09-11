"""Live, one-pass symbol reception with bounded-rate partial image snapshots.

Reuse transport2 acquisition/wire format. No trial image decoding or EQ replay.
Only new symbols are transformed. A new packet owns a fresh coefficient buffer.
"""
import struct
import time
import zlib
import numpy as np
from scipy.fft import rfft, idctn
from . import transport2 as v


class _Packet:
    def __init__(self, layout, coder):
        self.layout, self.coder = layout, coder
        self.training = []
        self.inverse = None
        self.fields = None
        self.headers = []
        self.sent = np.zeros(layout.capacity)
        self.known = np.zeros(layout.capacity, bool)
        self.data, self.pilots, self.header, _, self.phase = v._decode_tables(layout)
        self.slots = v.coefficient_slots(layout, tuple(coder.shapes))
        self.previous = [None, None]
        self.symbol = 0
        self.dirty = False
        self.emitted = False
        self.pilot_error = 0.
        self.fft_symbols = 0

    def push(self, body):
        spectra = rfft(body, n=v.N, axis=1, workers=1)
        self.fft_symbols += len(body)
        for spectrum in spectra:
            s = self.symbol
            self.symbol += 1
            if s < 2:
                self.training.append(spectrum)
                if s == 1:
                    self.inverse, self.weights, self.variance, self.coherence = v._channel_equalizer(
                        np.asarray(self.training), self.layout)
                    shape = (self.layout.image_symbols, len(self.data), 2, 2)
                    self.reliability = np.broadcast_to(self.weights[self.data][None,:,:,None], shape).ravel()[self.slots]
                    self.noise = np.broadcast_to(self.variance[self.data][None,:,:,None], shape).ravel()[self.slots]
                    gain = self.coder.gains*self.reliability
                    denominator = gain**2*self.coder.variance+self.noise
                    self.factor = np.divide(gain*self.coder.variance,denominator,
                        out=np.zeros(self.coder.count),where=denominator>1e-20)
                    self.fits = []
                    for c in range(2):
                        keep = self.weights[self.pilots,c] > .6
                        bins = self.layout.pilots[keep].astype(float)
                        w = self.weights[self.pilots[keep],c]**2
                        det = w.sum()*np.sum(w*bins*bins)-np.sum(w*bins)**2
                        # Store the two weighted regression rows once.
                        if len(bins)>=2 and det>0:
                            self.fits.append((keep, (w.sum()*bins-np.sum(w*bins))*w/det,
                                              (np.sum(w*bins*bins)-np.sum(w*bins)*bins)*w/det))
                        else:self.fits.append(None)
                continue
            equal = np.einsum('kij,kj->ki', self.inverse,
                              spectrum[self.layout.carriers], optimize=False)*self.phase[s]
            for c, fit in enumerate(self.fits):
                if fit is None:continue
                keep, slope_row, offset_row = fit
                pilot = equal[self.pilots[keep],c]
                angles = np.angle(pilot)
                if np.all(np.abs(pilot)>.25):
                    if self.previous[c] is None:angles = np.unwrap(angles)
                    else:angles = self.previous[c]+np.angle(np.exp(1j*(angles-self.previous[c])))
                    self.previous[c] = angles
                else:
                    self.previous[c] = None
                    angles = np.unwrap(angles)
                slope = np.sum(slope_row*angles)
                offset = np.sum(offset_row*angles)
                equal[:,c] *= np.exp(-1j*(slope*self.layout.carriers+offset))
            self.pilot_error = float(np.sqrt(np.mean(abs(equal[self.pilots]-1)**2)))
            end = 2+self.layout.header_symbols
            if s < end:
                self.headers.append(equal[self.header].copy())
                if s == end-1:self._header()
                continue
            # Accept any supported image slots; identity is not a prerequisite.
            # Reference checks distinguish weak image information from noise.
            if not (self.fields or self.coherence > (.65 if self.layout.top_bin<=13 else .4)):
                continue
            channel_ok = (np.mean(abs(equal[self.pilots]-1)**2, axis=0) < 2.25)
            channel_ok &= np.max(abs(equal[self.pilots]),axis=0) > .15
            block = equal[self.data]/v.IMAGE_GAIN
            values = np.stack((block.real,block.imag),axis=-1).ravel()
            valid = np.repeat(((self.weights[self.data]>.05)&channel_ok).ravel(),2)
            valid &= np.isfinite(values)
            at = (s-end)*len(values)
            self.sent[at:at+len(values)][valid] = values[valid]
            self.known[at:at+len(values)][valid] = True
            self.dirty |= bool(np.any(valid))

    def _header(self):
        h = np.asarray(self.headers)
        for pick in (h.mean(axis=-1), h[:,:,0], h[:,:,1]):
            flat = pick.ravel()[:v.HEADER_SLOTS]
            if len(flat)<v.HEADER_SLOTS:continue
            raw = np.packbits(np.stack((flat.real>0,flat.imag>0),axis=-1).ravel()).tobytes()
            if zlib.crc32(raw[:v.HEADER_BYTES]) != struct.unpack('>I',raw[v.HEADER_BYTES:])[0]:continue
            magic,flags,top,absolute,index,count,stamp = struct.unpack(v.HEADER_FORMAT,raw[:v.HEADER_BYTES])
            if magic == (b'V3' if self.layout.progressive else b'V2') and top==self.layout.top_bin and 1<=index<=count:
                self.fields = flags,absolute,index,count,stamp
                break

    def snapshot(self, complete):
        known = self.known[self.slots]
        if not np.any(known):return None
        # Unknown coefficients remain zero. No previous-frame data is copied.
        coefficients = self.sent[self.slots]*self.factor
        values = np.concatenate([idctn(plane,norm='ortho').ravel()
                                 for plane in self.coder._split(coefficients)])
        coverage = float(np.mean(known & (self.reliability>=.55)))
        result = v.Decoded('received' if complete else 'partial',values=values,
                          pilot_error=self.pilot_error,coverage=coverage,
                          tier='best' if coverage>=.95 else 'better' if coverage>=.7 else 'good' if coverage>=.35 else 'poor')
        if self.fields:
            result.flags,result.absolute,result.index,result.count,result.stamp_ms = self.fields
            result.identity = 'verified_header'
        else:result.status = 'picture_only' if complete else 'partial'
        result.extra.update(complete=complete, frame_start=not self.emitted,
                            symbols_received=self.symbol, fft_symbols=self.fft_symbols,
                            received_values=int(np.count_nonzero(known)))
        self.emitted = True
        self.dirty = False
        return result


class Receiver(v.Receiver):
    """Each symbol is decoded once; previews are paced by received samples."""
    def __init__(self, layout, coder, preview_hz=30):
        if not np.isfinite(preview_hz) or preview_hz<=0:raise ValueError('Positive preview_hz required')
        self.preview_samples = v.RATE/preview_hz
        super().__init__(layout,coder,recovery=False,fast=True)

    def reset(self, preserve_timing=False):
        super().reset(preserve_timing)
        self.packet = None
        self.last_preview = -float('inf')
        self.last_snapshot = None
        self.cpu_pending = 0.

    def _drain(self, final=False):
        out=[]
        while True:
            if self.pending is None:
                if len(self.buffer)<max(128,self.search_after) and not final:break
                if len(self.buffer)<128:break
                started=time.perf_counter()
                acquired=self._acquire()
                self.acquire_ms+=(time.perf_counter()-started)*1000
                if acquired is None:
                    self.search_after=len(self.buffer)+256
                    if len(self.buffer)>self.keep:self._drop(len(self.buffer)-self.keep)
                    break
                at,scale,score=acquired
                self.pending=(at-16*scale,scale,score)
                self.packet=_Packet(self.layout,self.coder)
                self.last_preview=-float('inf')
                self.last_snapshot=None
            begin,scale,score=self.pending
            started=time.perf_counter()
            first=self.packet.symbol
            last=first
            # Wait only for new FFT windows and their interpolation support.
            while last<self.layout.symbols:
                end=begin+(v.SYNC_LEN+last*v.SYMBOL+v.CP-4+v.N-1)*scale+1
                extra=0 if final or (scale==1 and begin==int(begin)) else 8
                if len(self.buffer)<end+extra:break
                last+=1
            if last==first:break
            walk=v._body_walk(self.layout).reshape(self.layout.symbols,v.N)[first:last].ravel()
            if scale==1 and begin==int(begin) and begin>=0:
                body=self.buffer[(int(begin)+walk).astype(int)]
            else:body=v._sample_at(self.buffer,begin+walk*scale)
            self.packet.push(body.reshape(last-first,v.N,2))
            complete=last==self.layout.symbols
            now=self.offset+len(self.buffer)
            result=None
            if self.packet.dirty and (not self.packet.emitted or complete or now-self.last_preview>=self.preview_samples):
                result=self.packet.snapshot(complete)
                self.last_preview=now
                self.last_snapshot=result
            elif complete and self.last_snapshot is not None:
                # Completion of a damaged tail must not clear a usable preview.
                from dataclasses import replace
                result=replace(self.last_snapshot,extra=dict(self.last_snapshot.extra))
                result.extra.update(complete=True,frame_start=False,symbols_received=last,
                                    fft_symbols=self.packet.fft_symbols)
            if result is not None:
                result.rate_error=scale-1
                result.rate_confidence=score
                result.extra.update(packet_id=self.offset+begin,at=self.offset+begin,
                    playback_speed=1/scale, input_path='streaming', sync_score=score,
                    packet_duration_samples=float(self.layout.packet*scale),
                    packet_age_samples=float(max(0, now-self.offset-begin)),
                    decode_ms=self.cpu_pending+(time.perf_counter()-started)*1000,acquire_ms=self.acquire_ms)
                result.extra['receive_cpu_ms']=result.extra['decode_ms']+self.acquire_ms
                out.append(result)
                self.acquire_ms=0.
                self.cpu_pending=0.
            else:
                self.cpu_pending+=(time.perf_counter()-started)*1000
            if not complete:break
            if self.packet.emitted:self.rate,self.confidence=scale-1,score
            else:self.confidence=0.
            self.pending=None
            self.packet=None
            self._drop(max(1,int(np.floor(begin+self.layout.packet*scale))))
        return out
