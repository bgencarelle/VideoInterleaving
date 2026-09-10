import unittest
import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from animation_modem.transport2 import PRESETS, SourceCoder, decode_packet as _decode
from animation_modem.imaging import DEFAULT_PROFILE, PROFILES, fit_shapes, image_values, plane_shapes
from animation_modem import transport2 as _v2

LAYOUT = PRESETS['wide']
FRAME = LAYOUT.frame


def _coder(profile=DEFAULT_PROFILE):
    return SourceCoder(fit_shapes(plane_shapes(profile), LAYOUT.capacity))


def encode(image, absolute, index, count, numbered=False, profile=DEFAULT_PROFILE,
           target_time_ns=None):
    """v1's call shape over the v2 encoder, so these timing tests stay readable."""
    coder = _coder(profile)
    stamp = 0 if target_time_ns is None else (int(target_time_ns)//1_000_000) & 0xffffffff
    audio = _v2.encode(image_values(image, coder.shapes), LAYOUT, coder,
                       absolute & 0xffffffff, min(index, 0xffff), min(count, 0xffff),
                       stamp_ms=stamp)
    return audio, None, None


def decode_packet(audio, profile=DEFAULT_PROFILE):
    return _decode(audio[:LAYOUT.packet], LAYOUT, _coder(profile))
from animation_modem.timing import expand_timestamp, PresentationBuffer
import index_calculator


class TimingTests(unittest.TestCase):
    def test_timestamp_roundtrip_all_profiles_with_unchanged_packet_size(self):
        stamp=1_700_000_000_123_000_000
        for profile in PROFILES:
            for numbered in (False,True):
                with self.subTest(profile=profile,numbered=numbered):
                    audio,_,_=encode(Image.new('RGB',(40,48)),0xffffffff,2221,2221,
                                     numbered,profile,target_time_ns=stamp)
                    result=decode_packet(audio,profile)
                    self.assertEqual(audio.shape,(FRAME,2))
                    self.assertEqual((result.absolute,result.index,result.count),(0xffffffff,2221,2221))
                    # v2 carries milliseconds; the epoch resolves the same way.
                    self.assertEqual(expand_timestamp(result.stamp_ms,stamp+5_000_000),
                                     (stamp//1_000_000)*1_000_000)
    def test_legacy_packets_have_no_timestamp(self):
        audio,_,_=encode(Image.new('RGB',(40,48)),1,1,1)
        self.assertFalse(decode_packet(audio).stamp_ms)
    def test_erased_header_never_supplies_trusted_time(self):
        audio,_,_=encode(Image.new('RGB',(40,48)),1,1,1,target_time_ns=1_700_000_000_000_000_000)
        # Erase the header symbols. v2 places them after two training symbols,
        # on the lowest carriers, so wipe that span rather than v1's offsets.
        start=_v2.SYNC_LEN+2*_v2.SYMBOL
        audio[start:start+LAYOUT.header_symbols*_v2.SYMBOL]=0
        result=decode_packet(audio)
        self.assertIsNotNone(result.values)
        self.assertIsNone(result.stamp_ms)
        self.assertNotEqual(result.identity,'verified_header')
    def test_millisecond_wrap_in_both_directions(self):
        wrap=(1<<32)*1_000_000
        for target in (wrap-2_000_000,wrap+3_000_000,wrap*400+1_000_000):
            for delta in (-10_000_000,10_000_000):
                self.assertEqual(expand_timestamp((target//1_000_000)&0xffffffff,target+delta),target)
    def test_future_frame_survives_arrival_of_more_future_frames(self):
        q=PresentationBuffer()
        q.put('first',100);q.put('second',200)
        self.assertIsNone(q.pop_due(99))
        self.assertEqual(q.pop_due(100),'first')
        self.assertIsNone(q.pop_due(199))
        self.assertEqual(q.pop_due(200),'second')
    def test_late_frames_skip_to_newest_due_without_losing_future(self):
        q=PresentationBuffer()
        for stamp in (100,200,300):q.put(stamp,stamp)
        self.assertEqual(q.pop_due(250),200)
        self.assertEqual(q.dropped,1)
        self.assertEqual(q.pop_due(300),300)
    def test_live_receiver_reports_shared_time_error(self):
        from animation_modem import decoder
        base=1_700_000_000_000_000_000
        target=base+78_000_000
        audio=encode(Image.new('RGB',(40,48)),1,1,1,target_time_ns=target)[0]
        class Input:
            latency=.005
            position=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,count):
                if self.position>=len(audio):raise KeyboardInterrupt()
                block=audio[self.position:self.position+count]
                self.position+=len(block)
                return block,False
        stream=Input()
        fake=SimpleNamespace(InputStream=lambda **kwargs:stream)
        output=io.StringIO()
        with patch.object(decoder,'sounddevice',return_value=fake), \
             patch.object(decoder.time,'time_ns',side_effect=lambda:base+round(stream.position/48000*1e9)), \
             contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):decoder.main(['--headless'])
        rows=[json.loads(line) for line in output.getvalue().splitlines()]
        received=next(row for row in rows if row['absolute']==1)
        self.assertEqual(received['target_time_ns'],target)
        self.assertLess(received['decode_error_ms'],0)
        self.assertEqual(received['decode_error_window']['median_ms'],received['decode_error_ms'])

    def test_recorded_timestamp_does_not_schedule_against_current_time(self):
        from animation_modem import decoder
        audio=encode(Image.new('RGB',(40,48)),1,1,1,target_time_ns=1_700_000_000_000_000_000)[0]
        output=io.StringIO()
        with patch.object(decoder,'wav_blocks',return_value=iter([audio])), \
             contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
            decoder.main(['--headless','--wav','unused.wav','--fast'])
        row=json.loads(output.getvalue().splitlines()[0])
        self.assertIsNone(row['target_time_ns'])
        self.assertIsNone(row['decode_error_ms'])
        # A replayed stamp is not a deadline: nothing is scheduled against it.

    def test_future_index_matches_existing_clock_and_does_not_publish(self):
        target=1_700_000_000_123_000_000
        with patch.object(index_calculator.time,'time_ns',return_value=target), \
             patch.object(index_calculator,'midi_mode',False):
            expected=index_calculator.update_index(2221)
        before=index_calculator.control_data_dictionary.copy()
        with patch.object(index_calculator.time,'time_ns',return_value=target-150_000_000):
            actual=index_calculator.calculate_free_clock_index(2221,at_time_ns=target,publish=False)
        self.assertEqual(actual,expected)
        self.assertEqual(index_calculator.control_data_dictionary,before)

if __name__=='__main__':unittest.main()
