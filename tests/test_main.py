from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from main import seconds_to_srt, transcribe_file


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


class TemporaryFileCleanupTests(unittest.TestCase):
    def test_removes_converted_wav_when_transcription_fails(self):
        with TemporaryDirectory() as directory:
            temp_wav = Path(directory) / "converted.wav"
            temp_wav.touch()
            model = Mock()
            model.transcribe.side_effect = RuntimeError("model failed")

            with patch("main.ensure_wav_16k", return_value=(temp_wav, temp_wav)), \
                 patch("main.probe_duration_seconds", return_value=0.0), \
                 patch("main.tqdm", return_value=nullcontext()):
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    transcribe_file(model, Path(directory) / "input.m4a", Path(directory) / "out")

            self.assertFalse(temp_wav.exists())
