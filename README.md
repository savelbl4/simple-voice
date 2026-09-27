# Voice Transcriber

CLI-инструмент для пакетной расшифровки аудио и видео с помощью faster-whisper и опциональной диаризацией спикеров.

---

## 🚀 Быстрый старт

```powershell
# 1. Установить зависимости
python -m pip install -r requirements.txt

# 2. Положить аудио/видео в папку records/
# (или указать путь к файлу)

# 3. Запустить транскрипцию
python main.py "records\meeting.m4a"

# 4. Результаты будут в папке transcripts/
```

---

## ✨ Возможности

- ✅ **Поддержка аудио:** `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`
- ✅ **Поддержка видео:** `.mp4`, `.mkv`, `.webm` (автоизвлечение аудиодорожки)
- ✅ **Автоопределение языка** (русский, английский и другие)
- ✅ **VAD-фильтрация** пауз для повышения точности
- ✅ **Прогресс обработки** с отображением времени
- ✅ **Диаризация спикеров** (опционально) — кто и когда говорил
- ✅ **Вывод в двух форматах:** обычный текст (`.txt`) и субтитры (`.srt`)
- ✅ **GPU-ускорение** при наличии NVIDIA CUDA

---

## 💻 Системные требования

### Минимальные
- Python 3.9 или новее
- 4 GB RAM (для моделей `tiny`, `base`, `small`)
- FFmpeg в системной переменной `PATH`
- ~500 MB свободного места (модель + зависимости)

### Рекомендуемые
- 8+ GB RAM (для модели `large-v3`)
- NVIDIA GPU с 4+ GB VRAM (для `--device cuda`)
- SSD для ускорения загрузки моделей

### Для диаризации
- Дополнительно: +2 GB RAM (CPU) или +2 GB VRAM (GPU)
- Дополнительно: ~300 MB для модели pyannote

---

## 📦 Установка

### 1. Установить Python-зависимости

```powershell
python -m pip install -r requirements.txt
```

Это установит:
- `faster-whisper` — модель транскрипции
- `ctranslate2` — ускоритель для CPU/GPU
- `soundfile` — работа с аудио
- `tqdm` — прогресс-бары

### 2. Установить FFmpeg

#### Windows (PowerShell)

**Вариант 1: Chocolatey (рекомендуется)**
```powershell
choco install ffmpeg
```

**Вариант 2: Вручную**
1. Скачать FFmpeg: https://www.gyan.dev/ffmpeg/builds/
2. Распаковать в `C:\ffmpeg`
3. Добавить в PATH:
```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine) + ";C:\ffmpeg\bin",
    [EnvironmentVariableTarget]::Machine
)
```
4. Открыть новое окно PowerShell

**Проверка установки:**
```powershell
ffmpeg -version
```

Должна отобразиться версия FFmpeg (например, `ffmpeg version 6.0`).

#### macOS / Linux

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Проверка
ffmpeg -version
```

---

## 📖 Использование

### Базовые примеры

```powershell
# Расшифровать один файл (модель по умолчанию: medium)
python main.py "records\interview.m4a"

# Указать размер модели
python main.py "records\meeting.mp4" --model large-v3

# Обработать всю папку рекурсивно
python main.py "records"

# Указать папку для результатов
python main.py "records" --out "my_transcripts"

# Использовать GPU (NVIDIA CUDA)
python main.py "records" --model large-v3 --device cuda --compute-type float16

# Быстрая обработка на CPU (низкое качество)
python main.py "records" --model tiny --device cpu
```

### С диаризацией спикеров

```powershell
# Включить диаризацию
python main.py "records\meeting.m4a" --diarize

# Указать точное число спикеров (повышает точность)
python main.py "records\interview.m4a" --diarize --num-speakers 2

# Указать диапазон числа спикеров
python main.py "records\discussion.mp4" --diarize --min-speakers 3 --max-speakers 6
```

**Примечание:** Для диаризации требуются дополнительные зависимости (см. раздел "Диаризация спикеров").

---

## 📁 Результаты

По умолчанию результаты записываются в папку `transcripts/`:

```
transcripts/
├── interview.txt     # Текстовая расшифровка
├── interview.srt     # Субтитры с временными метками
├── meeting.txt
└── meeting.srt
```

### Формат .txt (обычная транскрипция)
```
Добрый день, коллеги.
Начинаем встречу по проекту.
Кто хочет выступить первым?
```

### Формат .srt (субтитры)
```
1
00:00:00,000 --> 00:00:02,340
Добрый день, коллеги.

2
00:00:02,340 --> 00:00:05,120
Начинаем встречу по проекту.

3
00:00:05,120 --> 00:00:07,890
Кто хочет выступить первым?
```

### С диаризацией (--diarize)
```
SPEAKER_00: Добрый день, коллеги.
SPEAKER_00: Начинаем встречу по проекту.
SPEAKER_01: Привет! Я могу начать.
SPEAKER_00: Отлично, слушаем.
```

**Примечание:** При обработке папки сохраняется её структура, поэтому файлы с одинаковым именем из разных каталогов не перезапишут друг друга. Если в одной папке несколько файлов с одинаковым именем но разными расширениями, к результатам добавляется суффикс `_2`, `_3` и т.д.

---

## 🎛️ Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `path` | Путь к файлу или папке с аудио/видео | — |
| `--model` | Размер модели: `tiny`, `base`, `small`, `medium`, `large-v3` | `medium` |
| `--device` | Устройство: `auto`, `cpu`, `cuda` | `auto` |
| `--compute-type` | Тип вычислений: `auto`, `int8`, `int8_float16`, `float16`, `float32` | `auto` |
| `--out` | Папка для результатов | `transcripts` |
| `--beam` | Beam size (↑ качество = ↓ скорость) | `5` |
| `--no-vad` | Отключить VAD-фильтрацию пауз | `False` |
| `--diarize` | Включить диаризацию спикеров | `False` |
| `--hf-token` | Токен Hugging Face для диаризации | `HF_TOKEN` (переменная) |
| `--num-speakers` | Точное число спикеров | — |
| `--min-speakers` | Минимальное число спикеров | — |
| `--max-speakers` | Максимальное число спикеров | — |

### Выбор модели

| Модель | Качество | Скорость | RAM (CPU) | VRAM (GPU) |
|--------|----------|----------|-----------|------------|
| `tiny` | ⭐ | ⚡⚡⚡⚡⚡ | ~1 GB | ~1 GB |
| `base` | ⭐⭐ | ⚡⚡⚡⚡ | ~1 GB | ~1 GB |
| `small` | ⭐⭐⭐ | ⚡⚡⚡ | ~2 GB | ~2 GB |
| `medium` | ⭐⭐⭐⭐ | ⚡⚡ | ~5 GB | ~5 GB |
| `large-v3` | ⭐⭐⭐⭐⭐ | ⚡ | ~10 GB | ~10 GB |

**Рекомендации:**
- Для быстрого тестирования: `tiny` или `base`
- Для качественной расшифровки: `medium` или `large-v3`
- Для GPU: `--compute-type float16` (быстрее и меньше памяти)
- Для CPU: `--compute-type int8` (компромисс скорости и качества)

---

## 🎤 Диаризация спикеров (опционально)

Диаризация позволяет определить **кто и когда говорил** в записи. Каждая строка в результатах получает метку говорящего (`SPEAKER_00`, `SPEAKER_01`, и т.д.).

### Установка дополнительных зависимостей

**⚠️ Рекомендуется использовать отдельное виртуальное окружение** (особенно на Windows с GPU).

```powershell
# Создать виртуальное окружение
py -3.13 -m venv .venv-diar
.venv-diar\Scripts\activate

# Установить PyTorch (выберите вашу конфигурацию)

# NVIDIA GPU (CUDA 12.8, подходит для RTX 40xx/50xx):
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Только CPU (без GPU):
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Установить pyannote и основные зависимости
python -m pip install -r requirements-diarization.txt
```

**Размер зависимостей:**
- PyTorch (CPU): ~200 MB
- PyTorch (CUDA): ~2.5 GB
- pyannote.audio: ~100 MB
- Модель диаризации: ~300 MB (скачивается при первом запуске)

### Получение доступа к модели

Диаризация использует модель [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) (лицензия CC-BY-4.0).

**Шаги:**

1. **Зарегистрироваться на Hugging Face:** https://huggingface.co/join
2. **Принять условия модели:** https://huggingface.co/pyannote/speaker-diarization-community-1
   - Модель требует передачи контактной информации для редких уведомлений
3. **Создать токен с правом Read:** https://huggingface.co/settings/tokens
   - New token → Name: `voice-transcriber` → Type: **Read**
   - Скопировать токен (начинается с `hf_...`)
4. **Задать токен:**

   **Вариант A: Файл `.env` (рекомендуется)**
   
   Создать файл `.env` в корне проекта:
   ```
   HF_TOKEN=hf_ваш_токен_здесь
   ```

   **Вариант B: Переменная окружения**
   ```powershell
   $env:HF_TOKEN = "hf_ваш_токен_здесь"
   ```

   **Вариант C: Аргумент командной строки**
   ```powershell
   python main.py "records" --diarize --hf-token "hf_ваш_токен_здесь"
   ```

**🔒 Конфиденциальность:**
- Модель скачивается **один раз** и кэшируется локально (`%USERPROFILE%\.cache\huggingface\`)
- После первой загрузки работает **оффлайн**
- Аудио **не отправляется** в интернет, обрабатывается только локально
- Токен **не выводится** в логи

### Использование

```powershell
# Базовый запуск с диаризацией
python main.py "records\meeting.m4a" --diarize

# Если известно точное число спикеров (повышает точность)
python main.py "records\interview.m4a" --diarize --num-speakers 2

# Если примерное число спикеров
python main.py "records\discussion.mp4" --diarize --min-speakers 3 --max-speakers 6

# С GPU для ускорения
python main.py "records" --diarize --device cuda --compute-type float16
```

### Ограничения диаризации

- ⚠️ Метки `SPEAKER_XX` **обезличены** и **нумеруются заново** для каждого файла
- ⚠️ Спикер выбирается по **наибольшему пересечению** сегмента Whisper с репликами диаризации
- ⚠️ Короткие реплики на стыке могут попасть к соседнему говорящему
- ⚠️ Сегменты без речи помечаются как `SPEAKER_UNKNOWN`
- ⚠️ Качество зависит от чёткости речи, отсутствия фонового шума и наложений голосов

---

## ❓ Частые вопросы (FAQ)

### Q: Ошибка "ffmpeg not found"
**A:** Установите FFmpeg и убедитесь что он в PATH. Проверка: `ffmpeg -version`. Если версия не отображается, перезапустите терминал после установки.

### Q: Out of memory на GPU
**A:** Используйте меньшую модель (`--model medium` вместо `large-v3`) или переключитесь на CPU (`--device cpu`).

### Q: Диаризация не работает / ошибка токена
**A:** 
1. Проверьте что токен задан в `.env` файле или переменной `HF_TOKEN`
2. Убедитесь что вы приняли условия модели на https://huggingface.co/pyannote/speaker-diarization-community-1
3. Токен должен иметь право **Read**

### Q: Медленная обработка
**A:** 
- Используйте GPU: `--device cuda --compute-type float16`
- Выберите меньшую модель: `--model small` или `--model tiny`
- Отключите диаризацию если не нужна

### Q: Неправильный язык распознавания
**A:** Whisper автоматически определяет язык в начале аудио. Если нужна принудительная установка языка, можно модифицировать код (см. `main.py:297`, параметр `language`).

### Q: Как узнать какой язык был распознан?
**A:** Язык выводится в консоль после обработки каждого файла: `[OK] meeting.m4a → ... (язык: ru)`

### Q: Можно ли обработать файлы без расширения?
**A:** Нет, скрипт определяет формат по расширению файла.

### Q: Как обработать аудио на удалённом сервере?
**A:** Скопируйте проект на сервер, установите зависимости и запустите с `--device cpu` (если нет GPU).

---

## 🧪 Проверка

Запустить unit-тесты:

```powershell
python -m unittest discover -s tests -v
```

Все тесты должны пройти успешно. Тесты проверяют:
- Форматирование временных меток SRT
- Корректную очистку временных файлов
- Атомарность записи результатов
- Обработку ошибок
- Диаризацию (с моками, без загрузки моделей)

---

## 📁 Структура проекта

```
voice/
├── .dev/                        # Документация для разработчиков (AI-агенты)
│   ├── AGENTS.md                # Инструкции для AI
│   └── DIARIZATION_PLAN.md      # План интеграции диаризации
├── .venv*/                      # Виртуальные окружения (игнорируются)
├── records/                     # Ваши исходные аудио/видео (создайте вручную)
├── transcripts/                 # Результаты транскрипции (создаётся автоматически)
├── tests/                       # Unit-тесты
│   └── test_main.py
├── .env                         # Секретные данные (HF_TOKEN) — НЕ коммитить!
├── .gitignore                   # Игнорируемые файлы Git
├── LICENSE                      # MIT License
├── main.py                      # Основной скрипт
├── opencode.json                # Конфиг для OpenCode AI
├── README.md                    # Эта документация
├── requirements-diarization.txt # Доп. зависимости для --diarize
├── requirements.txt             # Основные зависимости
└── THIRD_PARTY_NOTICES.md       # Лицензии сторонних моделей и библиотек
```

---

## 📜 Лицензия

Проект распространяется под лицензией **MIT**. См. файл [LICENSE](LICENSE).

**Сторонние компоненты:**
- `faster-whisper` / `ctranslate2` — MIT License
- `pyannote.audio` — MIT License
- PyTorch — BSD-3-Clause License
- Модель `pyannote/speaker-diarization-community-1` — CC-BY-4.0

Подробности в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 🤝 Вклад в проект

Проект создан для личного использования. Если нашли ошибку или хотите предложить улучшение — welcome to fork & PR!

---

## 📧 Контакты

Автор: **savelbl4**

---

**Приятной расшифровки!** 🎧→📝
