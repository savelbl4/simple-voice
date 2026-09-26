from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from main import ensure_wav_16k, find_media, main, output_dir_for, seconds_to_srt, transcribe_file, unique_output_stem


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
    def test_creates_unique_wav_for_same_input_name(self):
        def fake_ffmpeg_run(command, **_kwargs):
            Path(command[-1]).touch()
            return SimpleNamespace(returncode=0, stderr=b"")

        with patch("main.sf.info", side_effect=RuntimeError), \
             patch("main.shutil.which", return_value="ffmpeg"), \
             patch("main.subprocess.run", side_effect=fake_ffmpeg_run):
            first_wav, first_temp = ensure_wav_16k(Path("recording.m4a"))
            second_wav, second_temp = ensure_wav_16k(Path("recording.m4a"))

        try:
            self.assertNotEqual(first_wav, second_wav)
            self.assertTrue(first_wav.exists())
            self.assertTrue(second_wav.exists())
        finally:
            first_temp.unlink(missing_ok=True)
            second_temp.unlink(missing_ok=True)

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

    def test_does_not_leave_partial_transcripts_when_transcription_fails(self):
        def segments():
            yield SimpleNamespace(start=0.0, end=1.0, text="First segment")
            raise RuntimeError("model failed")

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model = Mock()
            model.transcribe.return_value = (segments(), SimpleNamespace())

            with patch("main.ensure_wav_16k", return_value=(directory_path / "input.wav", None)), \
                 patch("main.probe_duration_seconds", return_value=0.0), \
                 patch("main.tqdm", return_value=nullcontext()):
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    transcribe_file(model, directory_path / "input.m4a", directory_path / "out")

            self.assertEqual(list((directory_path / "out").iterdir()), [])


class OutputDirectoryTests(unittest.TestCase):
    def test_preserves_source_subdirectories(self):
        with TemporaryDirectory() as directory:
            source_root = Path(directory) / "records"
            media_path = source_root / "a" / "meeting.mp4"
            media_path.parent.mkdir(parents=True)
            media_path.touch()

            result = output_dir_for(media_path, source_root, Path(directory) / "transcripts")

            self.assertEqual(result, Path(directory) / "transcripts" / "a")

    def test_uses_output_root_for_single_file(self):
        with TemporaryDirectory() as directory:
            media_path = Path(directory) / "meeting.mp4"
            media_path.touch()
            out_dir = Path(directory) / "transcripts"

            self.assertEqual(output_dir_for(media_path, media_path, out_dir), out_dir)

    def test_numbers_conflicting_stems_in_same_output_directory(self):
        used_stems = set()
        out_dir = Path("transcripts")

        self.assertEqual(unique_output_stem(Path("meeting.mp3"), out_dir, used_stems), "meeting")
        self.assertEqual(unique_output_stem(Path("meeting.mp4"), out_dir, used_stems), "meeting_2")
        self.assertEqual(unique_output_stem(Path("meeting.wav"), out_dir, used_stems), "meeting_3")

    def test_allows_same_stem_in_different_output_directories(self):
        used_stems = set()

        self.assertEqual(unique_output_stem(Path("meeting.mp3"), Path("transcripts/a"), used_stems), "meeting")
        self.assertEqual(unique_output_stem(Path("meeting.mp4"), Path("transcripts/b"), used_stems), "meeting")


class ExitCodeTests(unittest.TestCase):
    def test_returns_nonzero_when_no_media_files_are_found(self):
        with patch("sys.argv", ["main.py", "records"]), \
             patch("main.WhisperModel"), \
             patch("main.find_media", return_value=[]):
            self.assertEqual(main(), 1)

    def test_returns_nonzero_when_file_processing_fails(self):
        with patch("sys.argv", ["main.py", "records"]), \
             patch("main.WhisperModel"), \
             patch("main.find_media", return_value=[Path("records/broken.m4a")]), \
             patch("main.transcribe_file", side_effect=RuntimeError("failed")), \
             patch("main.tqdm", side_effect=lambda iterable, **_kwargs: iterable):
            self.assertEqual(main(), 1)

    def test_rejects_unsupported_single_file_before_loading_model(self):
        with TemporaryDirectory() as directory:
            unsupported_file = Path(directory) / "notes.txt"
            unsupported_file.touch()

            with patch("sys.argv", ["main.py", str(unsupported_file)]), \
                 patch("main.WhisperModel") as whisper_model, \
                 patch("builtins.print") as print_mock:
                self.assertEqual(main(), 1)

            whisper_model.assert_not_called()
            print_mock.assert_called_once_with("Неподдерживаемый формат файла: .txt")

    def test_find_media_skips_unsupported_single_file(self):
        with TemporaryDirectory() as directory:
            unsupported_file = Path(directory) / "notes.txt"
            unsupported_file.touch()

            self.assertEqual(list(find_media(unsupported_file)), [])
