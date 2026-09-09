"""Exercise the shared clock, including offset across ping-pong pivots."""
import unittest
from unittest.mock import patch
import index_calculator as clock


class ClockTests(unittest.TestCase):
    def test_zero_offset_preserves_existing_epoch_and_index(self):
        epoch=1_500_000_000_000_000_000
        with patch.object(clock,'launch_time',epoch), patch.object(clock,'IPS',30), \
             patch.object(clock.time,'time_ns',return_value=epoch+123_456_789), \
             patch.object(clock,'midi_mode',False):
            self.assertEqual(clock.update_index(20),(3,None))
            self.assertEqual(clock.update_index(20,time_offset_ns=0),(3,None))
            self.assertEqual(clock.launch_time,epoch)

    def test_offset_matches_evaluating_clock_at_future_time(self):
        with patch.object(clock,'launch_time',0),patch.object(clock,'IPS',30), \
             patch.object(clock,'midi_mode',False):
            for pingpong in (False,True):
                for now in (0,90_000_000,130_000_000,260_000_000,399_000_000):
                    for offset in (-75_000_000,0,75_000_000,200_000_000):
                        with self.subTest(now=now,offset=offset,pingpong=pingpong):
                            with patch.object(clock.time,'time_ns',return_value=now+offset):
                                expected=clock.update_index(4,pingpong)
                            with patch.object(clock.time,'time_ns',return_value=now):
                                actual=clock.update_index(4,pingpong,time_offset_ns=offset)
                            self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
