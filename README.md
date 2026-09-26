# Voice Transcriber

CLI для пакетной расшифровки аудио и видео с помощью `faster-whisper`.
Скрипт принимает один файл или папку, рекурсивно находит поддерживаемые медиафайлы и создаёт для каждого расшифровку в форматах TXT и SRT.

## Возможности

- Расшифровка аудио: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`.
- Расшифровка видео: `.mp4`, `.mkv`, `.webm`.
- При подаче видео FFmpeg самостоятельно извлекает и декодирует первую аудиодорожку перед распознаванием.
- Автоматическое определение языка, VAD-фильтрация пауз и прогресс обработки.
- `--device auto` использует CUDA при наличии совместимой GPU, иначе CPU.

## Требования

- Python 3.9 или новее.
- FFmpeg в системной переменной `PATH`. `ffprobe`, устанавливаемый вместе с FFmpeg, используется для точного прогресса.
- Для обработки на GPU требуется совместимая NVIDIA GPU и CUDA-поддержка в CTranslate2.

## Установка

```powershell
python -m pip install -r requirements.txt
```

Установите FFmpeg:

```bash
brew install ffmpeg
```

В Windows добавьте папку `bin` из установленного FFmpeg в `PATH`, затем откройте новое окно PowerShell. Например, если FFmpeg находится в `C:\ffmpeg`:

```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine) + ";C:\ffmpeg\bin",
    [EnvironmentVariableTarget]::Machine
)
```

## Использование

Расшифровать один аудиофайл:

```powershell
python main.py "records\interview.m4a" --model medium --device auto --compute-type auto
```

Расшифровать видео, извлекая аудиодорожку автоматически:

```powershell
python main.py "records\meeting.mp4" --model large-v3 --device auto --compute-type int8
```

Расшифровать все поддерживаемые файлы в папке и её вложенных каталогах:

```powershell
python main.py "records" --out transcripts
```

## Результаты

По умолчанию результаты записываются в папку `transcripts`:

- `<имя-файла>.txt` - текст расшифровки.
- `<имя-файла>.srt` - субтитры с временными метками.

При обработке папки сохраняется её внутренняя структура, поэтому файлы с одинаковым именем из разных каталогов не перезаписывают друг друга.

Папку результатов можно изменить параметром `--out <папка>`.

## Основные параметры

- `--model`: размер модели, например `tiny`, `base`, `small`, `medium`, `large-v3`.
- `--device`: `auto`, `cpu` или `cuda`.
- `--compute-type`: например `int8` для CPU или `float16` для GPU.
- `--beam`: размер beam search, по умолчанию `5`.
- `--no-vad`: отключает VAD-фильтрацию.

## Проверка

Запустить unit-тесты:

```powershell
python -m unittest discover -s tests -v
```
