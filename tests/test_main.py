import unittest

from main import seconds_to_srt


class SecondsToSrtTests(unittest.TestCase):
    def test_formats_regular_timestamps(self):
        cases = {
            0.0: "00:00:00,000",
            1.2346: "00:00:01,235",
            3661.789: "01:01:01,789",
        }

        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(seconds_to_srt(seconds), expected)

    def test_carries_rounded_milliseconds_to_next_unit(self):
        cases = {
            59.9996: "00:01:00,000",
            3599.9996: "01:00:00,000",
        }

        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(seconds_to_srt(seconds), expected)

    def test_clamps_negative_timestamp(self):
        self.assertEqual(seconds_to_srt(-1.0), "00:00:00,000")
