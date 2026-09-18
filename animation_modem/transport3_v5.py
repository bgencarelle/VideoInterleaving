"""v5 Receiver: mono, LDPC + CDF 9/7, HD wire layout."""

import numpy as np
from scipy.signal import correlate
from . import transport3 as V3
from .core import (REFERENCE_RATE, N, CP, SYNC_LEN, HEADER_BYTES_V5, HEADER_SLOTS_V5,
                   HEADER_GAIN, IMAGE_GAIN, Layout, Decoded, coefficient_slots,
                   phases, unpack_header_v5, _recovery_quality)
from .fec import ldpc_decode_belief_prop, pack_coefficients_to_llrs, unpack_llrs_to_coefficients


class ReceiverV5:
    """v5 Receiver: mono input, LDPC decode, CDF 9/7 inverse."""
    
    def __init__(self, layout, coder, threshold=.4, rate_window=None,
                 min_speed=.25, max_speed=2.0, recovery=False, fast=True,
                 pulse_only=True, input_rate=None, header_tolerance=0):
        if not (0 < min_speed <= 1 <= max_speed and min_speed >= .25 and max_speed <= 2):
            raise ValueError('Supported speed range: .25 <= min_speed <= 1 <= max_speed <= 2')
        if input_rate is not None and not (np.isfinite(input_rate) and input_rate > 0):
            raise ValueError('Input rate must be a positive number of hertz, or None')
        if coder.count > layout.capacity:
            raise ValueError('Source coder exceeds layout capacity')
        self.layout = layout
        self.coder = coder
        self.threshold = threshold
        self.rate_window = rate_window
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.recovery = bool(recovery)
        self.fast = bool(fast)
        self.pulse_only = bool(pulse_only)
        self.input_rate = input_rate
        self.header_tolerance = header_tolerance
        self.detected = None
        self._buf = np.empty((0,), np.float32)
        self._position = 0.0
        self._scale = 1.0
        self._confidence = 0.0
        self._in_packet = False
        self._packet_buf = None
        self._packet_pos = 0
        self._last_absolute = None
        self._speed_history = []

    def feed(self, audio):
        """Feed mono audio, return list of Decoded frames."""
        audio = np.asarray(audio, np.float32)
        if audio.ndim == 2 and audio.shape[1] == 2:
            # Stereo input -> take left channel (or average)
            audio = audio[:, 0]
        elif audio.ndim == 2:
            audio = audio.ravel()
        
        out = []
        self._buf = np.concatenate([self._buf, audio])
        
        while len(self._buf) >= self.layout.frame:
            frame = self._buf[:self.layout.frame]
            self._buf = self._buf[self.layout.frame:]
            
            decoded = self._process_frame(frame)
            if decoded:
                out.append(decoded)
        
        return out

    def flush(self):
        """Flush any remaining buffered data."""
        out = []
        while len(self._buf) >= self.layout.frame:
            frame = self._buf[:self.layout.frame]
            self._buf = self._buf[self.layout.frame:]
            decoded = self._process_frame(frame)
            if decoded:
                out.append(decoded)
        return out

    def _process_frame(self, frame):
        """Process one frame: acquire preamble -> demod -> LDPC -> inverse DWT."""
        # Step 1: Preamble detection (if not locked) - use standard preamble
        if self.detected is None:
            pos, scale, conf = V3.measure_pulses(frame, self.min_speed, self.max_speed)
            if pos is not None:
                self._position = pos
                self._scale = scale
                self._confidence = conf
                self.detected = self.layout.name
            return None
        
        # Step 2: Extract OFDM symbols
        # Preamble is at 16:16+len(PREAMBLE_V5), packet starts at SYNC_LEN
        packet_start = V3.SYNC_LEN
        packet_end = self.layout.packet
        if packet_end > len(frame):
            return None
        
        wave = frame[packet_start:packet_end]
        if len(wave) != self.layout.packet - V3.SYNC_LEN:
            return None
        
        # Step 3: OFDM demod (mono)
        symbols = self._demod_ofdm_mono(wave)
        if symbols is None:
            return None
        
        # Step 4: Extract header
        header_llrs = self._extract_header_llrs(symbols)
        if header_llrs is None:
            return None
        
        # Step 5: LDPC decode header
        header_bits = ldpc_decode_belief_prop(header_llrs, max_iter=10)
        header_bytes = np.packbits(header_bits).tobytes()
        header = unpack_header_v5(header_bytes)
        if header is None:
            return None
        
        # Step 6: Extract image data LLRs
        data_llrs = self._extract_data_llrs(symbols)
        if data_llrs is None:
            return None
        
        # Step 7: LDPC decode image data
        data_bits = ldpc_decode_belief_prop(data_llrs, max_iter=15)
        
        # Step 8: Convert bits back to coefficients
        coeffs = unpack_llrs_to_coefficients(data_bits)
        
        # Step 9: Inverse CDF 9/7 via coder
        values = self.coder.inverse(coeffs)
        
        return Decoded(
            status='received',
            identity='verified_header',
            values=values,
            absolute=header['absolute'],
            index=header['index'],
            extra={'count': header['count'], 'stamp_ms': header['stamp_ms'],
                   'profile': 'hd-dwt', 'shapes': self.coder.shapes,
                   'scale': self._scale, 'confidence': self._confidence}
        )

    def _demod_ofdm_mono(self, wave):
        """Demodulate mono OFDM symbols to complex constellation."""
        # Reshape to symbols x (N+CP)
        wave = wave.reshape(self.layout.symbols, N + CP)
        # Remove CP
        wave = wave[:, CP:]
        # FFT
        spectrum = np.fft.rfft(wave, n=N, axis=1)
        # Extract carriers
        carriers = self.layout.carriers
        data = np.searchsorted(carriers, self.layout.data_bins)
        pilots = np.searchsorted(carriers, self.layout.pilots)
        header = np.searchsorted(carriers, self.layout.header_bins)
        
        # Channel estimation from pilots
        pilot_syms = spectrum[:, pilots]
        h_est = pilot_syms.mean(axis=1)  # average over pilots
        
        # Equalize
        eq = spectrum[:, carriers] / h_est[:, None]
        
        # Remove phase rotation
        eq = eq * np.conj(phases(self.layout))[:, carriers]
        
        return eq  # shape: (symbols, num_carriers)

    def _extract_header_llrs(self, symbols):
        """Extract LLRs for header from header symbols."""
        carriers = self.layout.carriers
        header = np.searchsorted(carriers, self.layout.header_bins)
        header_symbols = self.layout.header_symbols
        
        # Header is in symbols 2 to 2+header_symbols
        h_syms = symbols[2:2+header_symbols, :]
        h_data = h_syms[:, header]
        
        # QPSK demod -> LLRs
        # Real and imag parts are independent BPSK
        llr_real = 2 * h_data.real / IMAGE_GAIN  # scale by gain
        llr_imag = 2 * h_data.imag / IMAGE_GAIN
        
        # Interleave: real0, imag0, real1, imag1...
        llrs = np.empty((header_symbols, len(header) * 2))
        llrs[:, ::2] = llr_real
        llrs[:, 1::2] = llr_imag
        
        return llrs.ravel()

    def _extract_data_llrs(self, symbols):
        """Extract LLRs for image data from data symbols."""
        carriers = self.layout.carriers
        data = np.searchsorted(carriers, self.layout.data_bins)
        header = np.searchsorted(carriers, self.layout.header_bins)
        spare = np.searchsorted(carriers, self.layout.spare_bins)
        
        # Data symbols start at 2 + header_symbols
        data_syms = symbols[2 + self.layout.header_symbols:, :]
        
        # Main data carriers
        main_data = data_syms[:, data]
        
        # Spare carriers in header symbols (dense_header)
        head = self.layout.header_capacity
        if head > 0:
            spare_data = symbols[2:2+self.layout.header_symbols, :][:, spare]
            # Combine
            all_data = np.concatenate([
                spare_data.reshape(-1, len(spare), 2),
                main_data.reshape(-1, len(data), 2)
            ], axis=1).reshape(-1)
        else:
            all_data = main_data.ravel()
        
        # QPSK -> LLRs
        llr_real = 2 * all_data.real / IMAGE_GAIN
        llr_imag = 2 * all_data.imag / IMAGE_GAIN
        
        llrs = np.empty(len(all_data) * 2)
        llrs[::2] = llr_real
        llrs[1::2] = llr_imag
        
        return llrs