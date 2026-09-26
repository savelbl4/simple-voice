#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple, Optional
import subprocess
import shutil
import json
import tempfile
from contextlib import nullcontext
import soundfile as sf
from tqdm import tqdm
import ctranslate2
from faster_whisper import WhisperModel

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".webm", ".aac"}


def select_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def find_media(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        for p in path.rglob("*"):
            if p.suffix.lower() in AUDIO_EXTS and p.is_file():
                yield p


def output_dir_for(media_path: Path, input_path: Path, out_dir: Path) -> Path:
    if input_path.is_dir():
        return out_dir / media_path.relative_to(input_path).parent
    return out_dir


def unique_output_stem(media_path: Path, out_dir: Path, used_stems: set[tuple[str, str]]) -> str:
    stem = media_path.stem
    candidate = stem
    number = 2
    output_dir_key = str(out_dir).casefold()

    while (output_dir_key, candidate.casefold()) in used_stems:
        candidate = f"{stem}_{number}"
        number += 1

    used_stems.add((output_dir_key, candidate.casefold()))
    return candidate


def seconds_to_srt(ts: float) -> str:
    total_ms = max(0, round((ts or 0.0) * 1000))
    total_seconds, ms = divmod(total_ms, 1000)
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt(seg.start)
            end = seconds_to_srt(seg.end)
            text = seg.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def ensure_wav_16k(input_path: Path) -> Tuple[Path, Optional[Path]]:
    """
    Если вход уже WAV 16k mono — возвращаем как есть.
    Иначе конвертируем через ffmpeg в временный WAV 16k mono.
    """
    try:
        info = sf.info(str(input_path))
        if (input_path.suffix.lower() == ".wav"
            and info.samplerate == 16000
            and info.channels == 1):
            return input_path, None
        # даже если sf прочитал, но не 16k/mono — конвертируем
        need_convert = True
    except Exception:
        # soundfile не смог прочитать (например .m4a) — точно конвертируем
        need_convert = True

    if need_convert:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "Не найден ffmpeg. Установите его (например, brew install ffmpeg) и повторите."
            )
        fd, tmp_name = tempfile.mkstemp(prefix=f"{input_path.stem}_", suffix=".wav")
        os.close(fd)
        tmp_wav = Path(tmp_name)
        # Конвертация в 16 kHz, mono WAV
        cmd = [ffmpeg, "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(tmp_wav)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not tmp_wav.exists():
            err = res.stderr.decode(errors="ignore")
            tmp_wav.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg не смог декодировать {input_path}:\n{err[:2000]}")
        return tmp_wav, tmp_wav


def probe_duration_seconds(path: Path) -> float:
    """Возвращает длительность файла в секундах.
    Сначала пытается ffprobe (любит .m4a/.mp4), иначе через soundfile."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            cmd = [
                ffprobe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "format=duration", "-of", "json", str(path)
            ]
            out = subprocess.check_output(cmd)
            dur = float(json.loads(out)["format"]["duration"])
            if dur > 0:
                return dur
        except Exception:
            pass
    # fallback: soundfile (не читает m4a, но вдруг это wav/flac и т.п.)
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0  # неизвестно, сделаем прогресс без total


def transcribe_file(model: WhisperModel, media_path: Path, out_dir: Path, beam_size=5, vad_filter=True,
                    output_stem: Optional[str] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = output_stem or media_path.stem

    input_for_model, tmp_to_delete = ensure_wav_16k(media_path)
    txt_tmp: Optional[Path] = None
    srt_tmp: Optional[Path] = None
    try:
        total_sec = probe_duration_seconds(input_for_model)
        pbar_cm = tqdm(total=total_sec if total_sec > 0 else None,
                       unit="s", desc=f"{media_path.name}", leave=False)

        # Стримингово пишем SRT/TXT
        txt_path = out_dir / f"{base}.txt"
        srt_path = out_dir / f"{base}.srt"
        txt_fd, txt_tmp_name = tempfile.mkstemp(dir=out_dir, prefix=f".{base}_", suffix=".txt.tmp")
        os.close(txt_fd)
        txt_tmp = Path(txt_tmp_name)
        srt_fd, srt_tmp_name = tempfile.mkstemp(dir=out_dir, prefix=f".{base}_", suffix=".srt.tmp")
        os.close(srt_fd)
        srt_tmp = Path(srt_tmp_name)
        idx = 0
        last_shown = 0.0

        with open(txt_tmp, "w", encoding="utf-8") as f_txt, \
             open(srt_tmp, "w", encoding="utf-8") as f_srt, \
             (pbar_cm if hasattr(pbar_cm, "__enter__") else nullcontext()) as pbar:

            segments, info = model.transcribe(
                str(input_for_model),
                beam_size=beam_size,
                vad_filter=vad_filter,
                language=None,
                task="transcribe",
                word_timestamps=False
            )

            for seg in segments:
                idx += 1
                text = seg.text.strip()
                f_txt.write(text + "\n")

                start = seconds_to_srt(seg.start or 0.0)
                end = seconds_to_srt(seg.end or (seg.start or 0.0))
                f_srt.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

                # прогресс: обновляем на приращение по времени
                if total_sec > 0 and pbar is not None:
                    cur = float(seg.end or 0.0)
                    inc = max(0.0, cur - last_shown)
                    if inc > 0:
                        pbar.update(inc)
                        last_shown = cur

            # если total неизвестен — аккуратно добьём прогресс
            if total_sec > 0 and last_shown < total_sec and pbar is not None:
                pbar.update(total_sec - last_shown)

        txt_tmp.replace(txt_path)
        srt_tmp.replace(srt_path)

        return {
            "language": info.language,
            "language_probability": getattr(info, "language_probability", None),
            "duration": total_sec,
            "txt": str(txt_path),
            "srt": str(srt_path),
        }
    finally:
        for output_tmp in (txt_tmp, srt_tmp):
            if output_tmp:
                output_tmp.unlink(missing_ok=True)
        if tmp_to_delete and tmp_to_delete.exists():
            try:
                tmp_to_delete.unlink()
            except OSError:
                pass

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch speech recognition with faster-whisper")
    parser.add_argument("path", type=str, help="Путь к файлу или папке с аудио/видео")
    parser.add_argument("--model", type=str, default="medium",
                        help="Размер модели: tiny, base, small, medium, large-v3 (качество↑=скорость↓)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Где считать (auto: CUDA при наличии, иначе CPU)")
    parser.add_argument("--compute-type", type=str, default="auto",
                        help="auto / int8 / int8_float16 / float16 / float32 (для GPU обычно float16)")
    parser.add_argument("--out", type=str, default="transcripts", help="Папка для результатов")
    parser.add_argument("--beam", type=int, default=5, help="beam size (качество vs скорость)")
    parser.add_argument("--no-vad", action="store_true", help="Отключить VAD-фильтр")
    args = parser.parse_args()

    device = select_device(args.device)
    model = WhisperModel(
        args.model,
        device=device,
        compute_type="default" if args.compute_type == "auto" else args.compute_type  # type: ignore
    )

    target = Path(args.path)
    out_dir = Path(args.out)
    files = sorted(find_media(target), key=lambda path: str(path).casefold())

    if not files:
        print("Не нашёл аудиофайлов по указанному пути.")
        return 1

    print(f"Файлов к распознаванию: {len(files)}; модель: {args.model}; устройство: {device}")
    used_stems: set[tuple[str, str]] = set()
    failed = False
    for p in tqdm(files, desc="Распознаю"):
        try:
            file_out_dir = output_dir_for(p, target, out_dir)
            meta = transcribe_file(
                model,
                p,
                file_out_dir,
                beam_size=args.beam,
                vad_filter=not args.no_vad,
                output_stem=unique_output_stem(p, file_out_dir, used_stems)
            )
            print(f"[OK] {p.name} → {meta['txt']} ; {meta['srt']} (язык: {meta['language']})")
        except Exception as e:
            failed = True
            print(f"[ERR] {p}: {e}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
