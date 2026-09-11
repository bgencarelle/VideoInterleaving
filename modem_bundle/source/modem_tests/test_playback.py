import contextlib
import io
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from animation_modem.playback import PacketOutput
from animation_modem.transport2 import PRESETS, Receiver, SourceCoder, encode
from animation_modem.imaging import (DEFAULT_PROFILE, fit_shapes, image_values,
                                     plane_shapes)

LAYOUT = PRESETS['wide']
FRAME = LAYOUT.frame


def _coder():
    return SourceCoder(fit_shapes(plane_shapes(DEFAULT_PROFILE), LAYOUT.capacity))
from PIL import Image


class Stop(Exception):pass


class Stream:
    def __init__(self, **kw):
        self.kw=kw
        self.active=False
        self.latency=.005
        self.time=10.0
    def start(self):self.active=True
    def stop(self):self.active=False
    def close(self):pass


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        fake=SimpleNamespace(OutputStream=Stream,CallbackStop=Stop,CallbackAbort=Stop)
        self.mock=patch('animation_modem.playback.sounddevice',return_value=fake)
        self.mock.start()
        self.stderr=contextlib.redirect_stderr(io.StringIO());self.stderr.__enter__()
        self.output=PacketOutput(channels=(3,1)).__enter__()
    def tearDown(self):
        self.output.__exit__(None,None,None)
        self.stderr.__exit__(None,None,None)
        self.mock.stop()
    def consume(self,count):
        out=np.empty((count,4),np.float32)
        self.output._callback(out,count,None,SimpleNamespace(output_underflow=False))
        self.assertTrue(np.all(out[:,[0,2]]==0))
        return out[:,[3,1]]

    def test_only_one_pending_and_no_mid_packet_replacement(self):
        first=np.ones((FRAME,2),np.float32)
        second=first*2
        self.assertFalse(self.output.stream.active)
        self.output.submit(first)
        self.assertTrue(self.output.stream.active)
        self.assertFalse(self.output.ready())
        with self.assertRaises(RuntimeError):self.output.submit(second)
        np.testing.assert_array_equal(self.consume(71),first[:71])
        self.assertTrue(self.output.ready())
        self.output.submit(second)
        actual=self.consume(FRAME)
        np.testing.assert_array_equal(actual,np.concatenate([first[71:],second[:71]]))
        self.assertTrue(self.output.ready())
        np.testing.assert_array_equal(self.consume(FRAME-71),second[71:])
        self.assertEqual(self.output.completed,2)

    def test_starvation_silence_and_recovery(self):
        self.assertTrue(np.all(self.consume(256)==0))
        self.assertEqual(self.output.starvations,1)
        self.output.submit(np.ones((FRAME,2),np.float32))
        self.assertTrue(np.all(self.consume(FRAME)==1))

    def test_drain_does_not_truncate_last_packet(self):
        self.output.submit(np.ones((FRAME,2),np.float32))
        self.output.finishing=True
        out=np.empty((FRAME+256,4),np.float32)
        with self.assertRaises(Stop):
            self.output._callback(out,len(out),None,SimpleNamespace(output_underflow=False))
        self.assertTrue(np.all(out[:FRAME][:,[3,1]]==1))
        self.assertTrue(np.all(out[FRAME:]==0))
        self.assertEqual(self.output.completed,1)

    def test_waveforms_decode_across_non_frame_callback_sizes(self):
        coder=_coder()
        blocks=[encode(image_values(Image.new('RGB',(40,48),(n*40,90,140)),coder),
                       LAYOUT,coder,n,n,3) for n in range(1,4)]
        captures=[]
        for block in blocks:
            self.assertTrue(self.output.ready())
            self.output.submit(block)
            # Drain exactly a frame via callback boundaries that split symbols.
           # Split symbols on odd boundaries, but drain exactly one frame.
            for count in (257,31,2048,FRAME-257-31-2048):captures.append(self.consume(count))
        audio=np.concatenate(captures)
        np.testing.assert_array_equal(audio,np.concatenate(blocks).astype(np.float32))
        rx=Receiver(LAYOUT,_coder());results=[]
        for start in range(0,len(audio),173):results.extend(rx.feed(audio[start:start+173]))
        self.assertEqual([r.absolute for r in results if r.values is not None],[1,2,3])

    def test_finish_waits_for_callback_drain(self):
        class DrainStream(Stream):
            def start(inner):
                inner.active=True
                def run():
                    try:
                        while inner.active:
                            out=np.empty((256,inner.kw['channels']),np.float32)
                            inner.kw['callback'](out,256,None,SimpleNamespace(output_underflow=False))
                            time.sleep(.001)
                    except Stop:
                        inner.kw['finished_callback']()
                    finally:inner.active=False
                inner.thread=threading.Thread(target=run)
                inner.thread.start()
            def close(inner):inner.thread.join(2)
        fake=SimpleNamespace(OutputStream=DrainStream,CallbackStop=Stop,CallbackAbort=Stop)
        with patch('animation_modem.playback.sounddevice',return_value=fake):
            with PacketOutput() as output:
                output.submit(np.ones((FRAME,2),np.float32))
                output.finish()
                self.assertTrue(output.done.is_set())
                self.assertEqual(output.completed,1)
                self.assertEqual(output.starvations,0)

    def test_scheduled_packet_starts_at_dac_deadline(self):
        wall=1_700_000_000_000_000_000
        with patch('animation_modem.playback.time.time_ns',return_value=wall):
            slot=self.output.reserve(prepare_ms=10,receive_margin_ms=15)
        coder=_coder()
        packet=encode(image_values(Image.new('RGB',(40,48)),coder),LAYOUT,coder,1,1,1,
                      stamp_ms=(slot.target_time_ns//1_000_000)&0xffffffff)
        self.assertTrue(self.output.submit(packet,slot))
        chunks=[];position=0;dac=10.005
        for count in (71,257,512,4096):
            out=np.empty((count,4),np.float32)
            self.output._callback(out,count,SimpleNamespace(outputBufferDacTime=dac+position/48000),
                                  SimpleNamespace(output_underflow=False))
            chunks.append(out[:,[3,1]].copy());position+=count
        actual=np.concatenate(chunks)
        gap=round((slot.start_time-dac)*48000)
        self.assertTrue(np.all(actual[:gap]==0))
        np.testing.assert_array_equal(actual[gap:gap+FRAME],packet)
        self.assertEqual(self.output.deadline_misses,0)
        self.assertEqual(self.output.starvations,0)
        # Packet duration is the layout's, not a constant: wide is 69 ms.
        expected=wall+round((slot.start_time-10+LAYOUT.packet/48000+0.015)*1e9)
        self.assertLess(abs(slot.target_time_ns-expected),1_000_000)

    def test_late_encode_is_rejected_and_retargeted(self):
        with patch('animation_modem.playback.time.time_ns',return_value=1_700_000_000_000_000_000):
            slot=self.output.reserve()
            self.output.stream.time=slot.start_time
            self.assertFalse(self.output.submit(np.ones((FRAME,2),np.float32),slot))
            later=self.output.reserve()
        self.assertGreater(later.start_time,slot.start_time)
        self.assertIsNone(self.output.pending)
        self.assertEqual(self.output.deadline_misses,1)

    def test_missed_dac_deadline_drops_whole_packet(self):
        with patch('animation_modem.playback.time.time_ns',return_value=1_700_000_000_000_000_000):
            slot=self.output.reserve()
        self.output.submit(np.ones((FRAME,2),np.float32),slot)
        out=np.empty((256,4),np.float32)
        self.output._callback(out,256,SimpleNamespace(outputBufferDacTime=slot.start_time+.010),
                              SimpleNamespace(output_underflow=True))
        self.assertTrue(np.all(out==0))
        self.assertTrue(self.output.ready())
        self.assertEqual(self.output.deadline_misses,1)
        self.assertEqual(self.output.completed,0)

    def test_invalid_packet_rejected_before_start(self):
        for bad in [np.zeros((10,2)),np.full((FRAME,2),np.nan)]:
            with self.assertRaises(ValueError):self.output.submit(bad)
        self.assertFalse(self.output.stream.active)


if __name__=='__main__':unittest.main()
