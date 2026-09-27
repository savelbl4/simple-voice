from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import os
import sys

from main import (UNKNOWN_SPEAKER, assign_speaker, ensure_wav_16k, find_media, load_diarization_pipeline,
                  load_env_file, main, output_dir_for, seconds_to_srt, transcribe_file, unique_output_stem)


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


class AssignSpeakerTests(unittest.TestCase):
    TURNS = [(0.0, 5.0, "SPEAKER_00"), (4.0, 10.0, "SPEAKER_01"), (12.0, 13.0, "SPEAKER_00")]

    def test_picks_speaker_with_largest_overlap(self):
        self.assertEqual(assign_speaker(3.0, 8.0, self.TURNS), "SPEAKER_01")
        self.assertEqual(assign_speaker(0.5, 4.5, self.TURNS), "SPEAKER_00")

    def test_sums_overlap_of_several_turns_of_same_speaker(self):
        turns = [(0.0, 1.0, "A"), (1.0, 2.5, "B"), (2.5, 3.5, "A"), (3.5, 4.0, "A")]
        self.assertEqual(assign_speaker(0.0, 4.0, turns), "A")

    def test_returns_unknown_without_overlap(self):
        self.assertEqual(assign_speaker(10.5, 11.5, self.TURNS), UNKNOWN_SPEAKER)
        self.assertEqual(assign_speaker(1.0, 2.0, []), UNKNOWN_SPEAKER)

    def test_zero_length_segment_uses_containing_turn(self):
        self.assertEqual(assign_speaker(12.5, 12.5, self.TURNS), "SPEAKER_00")
        self.assertEqual(assign_speaker(11.0, 11.0, self.TURNS), UNKNOWN_SPEAKER)


def fake_pyannote_modules(pipeline_factory):
    pyannote_audio = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=pipeline_factory))
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=lambda name: name)
    return patch.dict(sys.modules, {
        "torch": torch,
        "pyannote": SimpleNamespace(audio=pyannote_audio),
        "pyannote.audio": pyannote_audio,
    })


class LoadDiarizationPipelineTests(unittest.TestCase):
    def test_requires_token_with_model_link(self):
        with self.assertRaisesRegex(RuntimeError, "speaker-diarization-community-1"):
            load_diarization_pipeline(None, "cpu")

    def test_reports_missing_optional_dependency(self):
        with patch.dict(sys.modules, {"pyannote.audio": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-diarization.txt"):
                load_diarization_pipeline("hf_secret", "cpu")

    def test_hides_token_in_load_errors(self):
        factory = Mock(side_effect=RuntimeError("401 for token hf_secret"))
        with fake_pyannote_modules(factory):
            with self.assertRaises(RuntimeError) as ctx:
                load_diarization_pipeline("hf_secret", "cpu")
        self.assertNotIn("hf_secret", str(ctx.exception))
        self.assertIn("huggingface.co/pyannote", str(ctx.exception))

    def test_moves_pipeline_to_cuda(self):
        pipeline = Mock()
        with fake_pyannote_modules(Mock(return_value=pipeline)):
            self.assertIs(load_diarization_pipeline("hf_secret", "cuda"), pipeline)
        pipeline.to.assert_called_once_with("cuda")


class LoadEnvFileTests(unittest.TestCase):
    def test_reads_values_without_overriding_environment(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text('# comment\nVOICE_TEST_A="quoted"\nVOICE_TEST_B=keep\n', encoding="utf-8")
            with patch.dict(os.environ, {"VOICE_TEST_B": "existing"}):
                load_env_file(env_file)
                self.assertEqual(os.environ["VOICE_TEST_A"], "quoted")
                self.assertEqual(os.environ["VOICE_TEST_B"], "existing")
            os.environ.pop("VOICE_TEST_A", None)


class DiarizedTranscriptionTests(unittest.TestCase):
    def run_transcription(self, directory_path, diarize_mock, segments):
        model = Mock()
        model.transcribe.return_value = (iter(segments), SimpleNamespace(language="ru"))
        with patch("main.ensure_wav_16k", return_value=(directory_path / "input.wav", None)), \
             patch("main.probe_duration_seconds", return_value=0.0), \
             patch("main.tqdm", return_value=nullcontext()), \
             patch("main.diarize", diarize_mock):
            return transcribe_file(model, directory_path / "input.m4a", directory_path / "out",
                                   diarization_pipeline="pipeline",
                                   diarization_options={"num_speakers": None, "min_speakers": 2,
                                                        "max_speakers": 4})

    def test_writes_speaker_labels_to_txt_and_srt(self):
        segments = [SimpleNamespace(start=0.0, end=2.0, text=" Добрый день."),
                    SimpleNamespace(start=2.0, end=4.0, text=" Здравствуйте."),
                    SimpleNamespace(start=9.0, end=9.5, text=" Шум.")]
        diarize_mock = Mock(return_value=[(0.0, 2.1, "SPEAKER_00"), (2.1, 4.0, "SPEAKER_01")])
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            meta = self.run_transcription(directory_path, diarize_mock, segments)

            self.assertEqual(Path(meta["txt"]).read_text(encoding="utf-8"),
                             "SPEAKER_00: Добрый день.\nSPEAKER_01: Здравствуйте.\nSPEAKER_UNKNOWN: Шум.\n")
            srt = Path(meta["srt"]).read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:02,000\nSPEAKER_00: Добрый день.\n", srt)
            self.assertIn("00:00:02,000 --> 00:00:04,000\nSPEAKER_01: Здравствуйте.\n", srt)

        diarize_mock.assert_called_once_with("pipeline", directory_path / "input.wav",
                                             num_speakers=None, min_speakers=2, max_speakers=4)

    def test_does_not_leave_outputs_when_diarization_fails(self):
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            temp_wav = directory_path / "converted.wav"
            temp_wav.touch()
            model = Mock()
            with patch("main.ensure_wav_16k", return_value=(temp_wav, temp_wav)), \
                 patch("main.diarize", side_effect=RuntimeError("diarization failed")):
                with self.assertRaisesRegex(RuntimeError, "diarization failed"):
                    transcribe_file(model, directory_path / "input.m4a", directory_path / "out",
                                    diarization_pipeline="pipeline")

            self.assertFalse(temp_wav.exists())
            self.assertEqual(list((directory_path / "out").iterdir()), [])
            model.transcribe.assert_not_called()


class DiarizationCliTests(unittest.TestCase):
    def test_passes_speaker_options_and_loads_pipeline_once(self):
        files = [Path("records/a.m4a"), Path("records/b.m4a")]
        with patch("sys.argv", ["main.py", "records", "--device", "cpu", "--diarize", "--min-speakers", "2",
                                "--max-speakers", "5"]), \
             patch.dict(os.environ, {"HF_TOKEN": "hf_secret"}), \
             patch("main.load_env_file"), \
             patch("main.WhisperModel"), \
             patch("main.find_media", return_value=files), \
             patch("main.load_diarization_pipeline", return_value="pipeline") as load_mock, \
             patch("main.transcribe_file", return_value={"txt": "a", "srt": "b", "language": "ru"}) as tr_mock, \
             patch("main.tqdm", side_effect=lambda iterable, **_kwargs: iterable), \
             patch("builtins.print"):
            self.assertEqual(main(), 0)

        load_mock.assert_called_once_with("hf_secret", "cpu", strict_device=False)
        self.assertEqual(tr_mock.call_count, 2)
        self.assertEqual(tr_mock.call_args.kwargs["diarization_pipeline"], "pipeline")
        self.assertEqual(tr_mock.call_args.kwargs["diarization_options"],
                         {"num_speakers": None, "min_speakers": 2, "max_speakers": 5})

    def test_fails_without_token_before_loading_whisper(self):
        with patch("sys.argv", ["main.py", "records", "--device", "cpu", "--diarize"]), \
             patch.dict(os.environ, {}, clear=True), \
             patch("main.load_env_file"), \
             patch("main.WhisperModel") as whisper_model, \
             patch("main.find_media", return_value=[Path("records/a.m4a")]), \
             patch("builtins.print") as print_mock:
            self.assertEqual(main(), 1)

        whisper_model.assert_not_called()
        self.assertIn("HF_TOKEN", print_mock.call_args.args[0])

    def test_rejects_num_speakers_with_bounds(self):
        with patch("sys.argv", ["main.py", "records", "--diarize", "--num-speakers", "2", "--max-speakers", "3"]), \
             patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main()

    def test_rejects_speaker_options_without_diarize(self):
        with patch("sys.argv", ["main.py", "records", "--num-speakers", "2"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main()

    def test_does_not_import_pyannote_without_diarize(self):
        with patch("sys.argv", ["main.py", "records"]), \
             patch("main.WhisperModel"), \
             patch("main.find_media", return_value=[]), \
             patch.dict(sys.modules, {"pyannote.audio": None, "torch": None}):
            self.assertEqual(main(), 1)
