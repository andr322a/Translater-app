# Subtitles App

Transcribes Danish speech and translates it to English subtitles using OpenAI API, then burns them into the video with FFmpeg.

Output files:

- `[video]_english.srt`
- `[video]_english_subtitles.mp4`

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

```bash
pip install -r requirements.txt
```

## Run

```bash
set -a
source .env
set +a
python3 transcribe_translate.py IMG_4331.mov
```

Output example:

- `IMG_4331_english.srt`
- `IMG_4331_english_subtitles.mp4`
