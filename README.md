# Video subtitles: API mode and Offline mode

Project contains two scripts so you can compare quality directly:

- `transcribe_translate.py` (OpenAI API pipeline)
- `transcribe_translate_offline.py` (no API calls, fully local)

Both scripts create:

- `[video]_english*.srt`
- `[video]_english*_subtitles.mp4`

## Requirements

- Python 3.9+
- FFmpeg with `subtitles` filter (libass)

macOS/Homebrew example:

```bash
brew install ffmpeg-full
brew unlink ffmpeg
brew link --overwrite ffmpeg-full
```

## Install

API mode dependencies:

```bash
pip install -r requirements.txt
```

Offline mode dependencies:

```bash
pip install -r requirements-offline.txt
```

## Run API mode

```bash
set -a
source .env
set +a
python3 transcribe_translate.py IMG_4331.mov
```

Output example:

- `IMG_4331_english.srt`
- `IMG_4331_english_subtitles.mp4`

## Run Offline mode (no API)

```bash
python3 transcribe_translate_offline.py IMG_4331.mov
```

Output example:

- `IMG_4331_english_offline.srt`
- `IMG_4331_english_offline_subtitles.mp4`

## Offline env vars

- `WHISPER_MODEL` default: `large-v3`
- `WHISPER_DEVICE` default: `auto` (`mps` on Apple Silicon if available, else `cpu`)
- `WHISPER_COMPUTE_TYPE` default: `float16` (cpu fallback uses float32 in openai-whisper)
- `WHISPER_DOWNLOAD_ROOT` optional local cache path for Whisper model files
- `WHISPER_ALLOW_INSECURE_DOWNLOAD` default: `0` (set to `1` for one-time `curl -k` model download fallback)
- `MARIAN_MODEL` default: `Helsinki-NLP/opus-mt-da-en`
- `TRANSLATION_BATCH_SIZE` default: `12`
- `GLOSSARY_TERMS` default: `H.P Therkelsen,Zeek`
- `SUBTITLE_FONTSIZE` default: `14`
- `SUBTITLE_MAX_WIDTH` default: `32`
- `SUBTITLE_MAX_LINES` default: `2`
- `VIDEO_ENCODER` default: `libx264`
- `VIDEO_PRESET` default: `fast`
- `VIDEO_CRF` default: `21`
- `VIDEO_BITRATE` optional (used mainly with hardware encoders)

## Compare quality (manual)

1. Run API script on a video.
2. Run offline script on same video.
3. Compare:
- subtitle translation quality
- term handling (`H.P Therkelsen`, `Zeek`)
- timing drift
- readability/line breaks

## Notes

- Offline first run can be slow because models may download and cache locally.
- Offline mode uses no inference API calls.
- If your company network uses SSL inspection, first model download can fail with certificate errors.
- In that case, install company CA in Python trust store or predownload models and run from local cache.

Fallback command for SSL-inspected networks (only for first download):

```bash
WHISPER_ALLOW_INSECURE_DOWNLOAD=1 python3 transcribe_translate_offline.py IMG_4331.mov
```
