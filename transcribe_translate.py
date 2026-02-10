#!/usr/bin/env python3
"""
Fuldt API-baseret video-transskription, oversættelse og undertekster.

- Transskriberer dansk tale med OpenAI (gpt-4o-transcribe)
- Henter tidsstempler via OpenAI (whisper-1) til SRT
- Justerer segmenttekst og oversætter til engelsk via OpenAI
- Brænder engelske undertekster ind i videoen med FFmpeg
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
except ModuleNotFoundError:
    print("Fejl: OpenAI SDK er ikke installeret. Kør: pip install -r requirements.txt")
    sys.exit(1)
except Exception:  # Fallback hvis SDK-eksporter ændrer sig
    from openai import OpenAI  # type: ignore

    APIError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi"}
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
TIMESTAMP_MODEL = os.getenv("TIMESTAMP_MODEL", "whisper-1")
ALIGN_MODEL = os.getenv("ALIGN_MODEL", "gpt-4o-mini")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
TRANSCRIBE_LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "da")
BASE_TRANSCRIBE_PROMPT = "Firma- og produktnavne: H.P Therkelsen, Zeek."
TRANSCRIBE_PROMPT = os.getenv("TRANSCRIBE_PROMPT", "").strip()
if TRANSCRIBE_PROMPT:
    TRANSCRIBE_PROMPT = f"{TRANSCRIBE_PROMPT}\n{BASE_TRANSCRIBE_PROMPT}"
else:
    TRANSCRIBE_PROMPT = BASE_TRANSCRIBE_PROMPT

ALIGN_SEGMENTS = os.getenv("ALIGN_SEGMENTS", "1") == "1"
ALIGNMENT_MAX_CHARS = int(os.getenv("ALIGNMENT_MAX_CHARS", "12000"))
TRANSLATION_BATCH_CHARS = int(os.getenv("TRANSLATION_BATCH_CHARS", "2500"))
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "3")))

SUBTITLE_FONTSIZE = int(os.getenv("SUBTITLE_FONTSIZE", "14"))
SUBTITLE_MAX_WIDTH = int(os.getenv("SUBTITLE_MAX_WIDTH", "32"))
SUBTITLE_MAX_LINES = int(os.getenv("SUBTITLE_MAX_LINES", "2"))
VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "libx264")
VIDEO_PRESET = os.getenv("VIDEO_PRESET", "fast")
VIDEO_CRF = os.getenv("VIDEO_CRF", "21")
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "").strip()


class UserFacingError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def ensure_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise UserFacingError(
            "FFmpeg er ikke installeret eller kan ikke findes i PATH."
        ) from exc
    try:
        filters = subprocess.run(
            ["ffmpeg", "-filters"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
        if not re.search(r"\bsubtitles\b", filters):
            raise UserFacingError(
                "Din FFmpeg mangler 'subtitles'-filteret (libass). "
                "Installer en FFmpeg-build med libass. På macOS/Homebrew: "
                "brew install ffmpeg-full og derefter brew link --overwrite ffmpeg-full."
            )
    except UserFacingError:
        raise
    except Exception:
        # Hvis vi ikke kan læse filterlisten, fortsæt og lad FFmpeg fejlere med detaljer.
        pass


def ensure_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise UserFacingError("OPENAI_API_KEY mangler. Sæt miljøvariablen og prøv igen.")


def prompt_for_video_path() -> Path:
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = input("Indtast stien til videofilen (mp4, mov, avi): ").strip()

    if not raw:
        raise UserFacingError("Du skal angive en filsti.")

    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise UserFacingError(f"Filen findes ikke: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UserFacingError(
            f"Ugyldigt filformat ({path.suffix}). Brug mp4, mov eller avi."
        )
    return path


def run_ffmpeg(args: List[str], friendly_step: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise UserFacingError(
            f"FFmpeg-fejl under '{friendly_step}'.\n\nDetaljer:\n{result.stderr.strip()}"
        )


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "aac",
            "-b:a",
            AUDIO_BITRATE,
            str(audio_path),
        ],
        "udtrækning af lyd",
    )


def get_text_from_transcription(response: Any) -> str:
    if hasattr(response, "text"):
        return str(response.text).strip()
    if isinstance(response, dict) and "text" in response:
        return str(response["text"]).strip()
    return str(response).strip()


def get_segments_from_transcription(response: Any) -> List[Dict[str, Any]]:
    if hasattr(response, "segments"):
        segments = getattr(response, "segments")
    elif isinstance(response, dict):
        segments = response.get("segments")
    else:
        segments = None

    if not segments:
        return []

    normalized: List[Dict[str, Any]] = []
    for seg in segments:
        if isinstance(seg, dict):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")
        else:
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
            text = getattr(seg, "text", "")
        normalized.append({"start": start, "end": end, "text": text})
    return normalized


def transcribe_text_openai(client: OpenAI, audio_path: Path) -> str:
    with audio_path.open("rb") as audio_file:
        params: Dict[str, Any] = {"model": TRANSCRIBE_MODEL, "file": audio_file}
        if TRANSCRIBE_LANGUAGE:
            params["language"] = TRANSCRIBE_LANGUAGE
        if TRANSCRIBE_PROMPT:
            params["prompt"] = TRANSCRIBE_PROMPT
        response = client.audio.transcriptions.create(**params)
    return get_text_from_transcription(response)


def transcribe_segments_openai(client: OpenAI, audio_path: Path) -> List[Dict[str, Any]]:
    with audio_path.open("rb") as audio_file:
        params: Dict[str, Any] = {
            "model": TIMESTAMP_MODEL,
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if TRANSCRIBE_LANGUAGE:
            params["language"] = TRANSCRIBE_LANGUAGE
        if TRANSCRIBE_PROMPT:
            params["prompt"] = TRANSCRIBE_PROMPT
        response = client.audio.transcriptions.create(**params)
    return get_segments_from_transcription(response)


def batch_segments(
    segments: List[Dict[str, Any]],
    max_chars: int,
) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_len = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        seg_item = {"id": seg["id"], "text": text}
        text_len = len(text)

        if current and current_len + text_len > max_chars:
            batches.append(current)
            current = []
            current_len = 0

        current.append(seg_item)
        current_len += text_len

    if current:
        batches.append(current)

    return batches


def build_translation_prompt(batch: List[Dict[str, Any]]) -> str:
    payload = {"segments": batch}
    instructions = (
        "You are a professional translator. Translate from Danish to English. "
        "Return ONLY valid JSON with the schema: "
        "{\"translations\": [{\"id\": <int>, \"text_en\": \"<string>\"}, ...]}. "
        "Keep IDs unchanged and translate each segment faithfully."
    )
    return f"{instructions}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False)}"


def build_alignment_prompt(segments: List[Dict[str, Any]], transcript: str) -> str:
    payload = {
        "segments": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in segments
        ],
        "transcript": transcript,
    }
    instructions = (
        "You are aligning a clean Danish transcript to timestamped segments. "
        "Return ONLY valid JSON with the schema: "
        "{\"segments\": [{\"id\": <int>, \"text_da\": \"<string>\"}, ...]}. "
        "Keep the exact number and order of segments. "
        "Use the transcript to correct errors in each segment while keeping timing context. "
        "If unsure, keep the original segment text."
    )
    return f"{instructions}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False)}"


def extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text).strip()

    output = None
    if isinstance(response, dict):
        output = response.get("output")
    else:
        output = getattr(response, "output", None)

    if output:
        parts: List[str] = []
        for item in output:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type != "message":
                continue
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            for block in content or []:
                block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if block_type not in ("output_text", "text"):
                    continue
                text_value = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                if isinstance(text_value, dict) and "value" in text_value:
                    text_value = text_value["value"]
                if text_value:
                    parts.append(str(text_value))
        if parts:
            return "".join(parts).strip()

    return str(response).strip()


def parse_json_from_text(raw_text: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("Kunne ikke parse JSON fra OpenAI-svar.")


def call_openai_response(client: OpenAI, model: str, prompt: str):
    try:
        return client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
            text={"format": {"type": "json_object"}},
        )
    except TypeError:
        return client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )


def is_retriable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True

    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    return status in {429, 500, 502, 503, 504}


def is_fatal_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status in {401, 403}


def call_json_with_retries(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int,
) -> Dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = call_openai_response(client, model, prompt)
            raw_text = extract_response_text(response)
            return parse_json_from_text(raw_text)
        except Exception as exc:
            last_exc = exc
            if is_fatal_auth_error(exc):
                raise
            if attempt >= max_retries or not is_retriable(exc):
                raise

            backoff = 2 ** (attempt - 1)
            log(
                f"  OpenAI-fejl (forsøg {attempt}/{max_retries}). "
                f"Prøver igen om {backoff}s..."
            )
            time.sleep(backoff)

    if last_exc:
        raise last_exc

    return {}


def distribute_transcript_by_duration(
    transcript: str,
    segments: List[Dict[str, Any]],
) -> Dict[int, str]:
    words = transcript.split()
    if not words:
        return {}

    total_duration = 0.0
    for seg in segments:
        total_duration += max(0.01, float(seg["end"]) - float(seg["start"]))

    mapping: Dict[int, str] = {}
    index = 0
    total_words = len(words)

    for i, seg in enumerate(segments):
        if i == len(segments) - 1:
            mapping[seg["id"]] = " ".join(words[index:]).strip()
            break
        duration = max(0.01, float(seg["end"]) - float(seg["start"]))
        share = max(1, round((duration / total_duration) * total_words))
        slice_words = words[index : index + share]
        mapping[seg["id"]] = " ".join(slice_words).strip()
        index += share

    return mapping


def align_segments_with_transcript(
    client: OpenAI,
    segments: List[Dict[str, Any]],
    transcript: str,
) -> Dict[int, str]:
    if not ALIGN_SEGMENTS:
        return {}

    size_hint = len(transcript) + sum(len(seg.get("text", "")) for seg in segments)
    if size_hint > ALIGNMENT_MAX_CHARS:
        log("  Alignment springes over (input for stort). Bruger heuristik.")
        return distribute_transcript_by_duration(transcript, segments)

    prompt = build_alignment_prompt(segments, transcript)
    data = call_json_with_retries(client, ALIGN_MODEL, prompt, MAX_RETRIES)

    items = data.get("segments", [])
    if not isinstance(items, list):
        raise ValueError("Alignment JSON mangler 'segments' liste.")

    mapping: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "text_da" not in item:
            continue
        mapping[int(item["id"])] = str(item["text_da"]).strip()

    return mapping


def translate_segments_openai(
    client: OpenAI,
    segments: List[Dict[str, Any]],
) -> Dict[int, str]:
    batches = batch_segments(segments, max(1, TRANSLATION_BATCH_CHARS))
    translations: Dict[int, str] = {}

    for idx, batch in enumerate(batches, start=1):
        log(f"  Oversætter batch {idx}/{len(batches)}...")
        prompt = build_translation_prompt(batch)
        try:
            data = call_json_with_retries(client, TRANSLATION_MODEL, prompt, MAX_RETRIES)
            items = data.get("translations", [])
            if not isinstance(items, list):
                raise ValueError("Oversættelses JSON mangler 'translations' liste.")
            for item in items:
                if not isinstance(item, dict):
                    continue
                if "id" not in item or "text_en" not in item:
                    continue
                translations[int(item["id"])] = str(item["text_en"]).strip()
        except Exception as exc:
            if is_fatal_auth_error(exc):
                raise UserFacingError(
                    "OpenAI API fejl (autorisering). Tjek OPENAI_API_KEY."
                ) from exc
            log(
                "  Advarsel: Kunne ikke oversætte dette batch. "
                "Bruger dansk tekst for segmenterne i batch."
            )
            continue

    return translations


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def wrap_subtitle(text: str, width: int = SUBTITLE_MAX_WIDTH, max_lines: int = SUBTITLE_MAX_LINES) -> List[str]:
    words = text.replace("\n", " ").strip()
    if not words:
        return [""]

    lines = textwrap.wrap(words, width=width)
    if len(lines) <= max_lines:
        return lines

    combined = lines[: max_lines - 1]
    combined.append(" ".join(lines[max_lines - 1 :]))
    return combined


def build_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = segment["start"]
        end = segment["end"]
        if end <= start:
            end = start + 0.5
        lines.append(str(idx))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.extend(wrap_subtitle(segment["text"]))
        lines.append("")
    return "\n".join(lines)


def escape_ffmpeg_filter_value(value: str) -> str:
    value = value.replace("\\", "\\\\")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    value = value.replace("[", "\\[")
    value = value.replace("]", "\\]")
    value = value.replace(",", "\\,")
    value = value.replace(" ", "\\ ")
    return value


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    style = (
        f"Fontsize={SUBTITLE_FONTSIZE},PrimaryColour=&HFFFFFF&,"
        "OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0"
    )
    escaped_path = escape_ffmpeg_filter_value(str(srt_path))
    escaped_style = escape_ffmpeg_filter_value(style)
    filter_arg = f"subtitles=filename={escaped_path}:force_style={escaped_style}"
    encoder = VIDEO_ENCODER
    if encoder:
        try:
            encoders = subprocess.run(
                ["ffmpeg", "-encoders"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout
            if encoder not in encoders:
                log(f"  Advarsel: encoder '{encoder}' ikke fundet. Bruger libx264.")
                encoder = "libx264"
        except Exception:
            pass

    encode_args = ["-c:v", encoder]
    if encoder in {"libx264", "libx265"}:
        if VIDEO_PRESET:
            encode_args += ["-preset", VIDEO_PRESET]
        if VIDEO_CRF:
            encode_args += ["-crf", VIDEO_CRF]
    elif VIDEO_BITRATE:
        encode_args += ["-b:v", VIDEO_BITRATE]

    run_ffmpeg(
        [
            "-analyzeduration",
            "100M",
            "-probesize",
            "100M",
            "-ignore_unknown",
            "-i",
            str(video_path),
            "-vf",
            filter_arg,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            *encode_args,
            "-c:a",
            "copy",
            str(output_path),
        ],
        "brænding af undertekster",
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    log("Velkommen! Dette script kører fuldt på OpenAI API.")

    try:
        ensure_ffmpeg()
        ensure_openai_key()
        video_path = prompt_for_video_path()

        output_dir = video_path.parent
        base_name = video_path.stem
        audio_path = output_dir / f"{base_name}_audio.m4a"
        srt_path = output_dir / f"{base_name}_english.srt"
        output_video = output_dir / f"{base_name}_english_subtitles.mp4"

        log("Trin 1/4: Udtrækker lyd fra video...")
        extract_audio(video_path, audio_path)

        client = OpenAI()

        log("Trin 2/4: Transskriberer (OpenAI)...")
        transcript_text = transcribe_text_openai(client, audio_path)
        if not transcript_text:
            raise UserFacingError("Transskriptionen blev tom. Tjek lydindholdet.")

        log("Trin 2b/4: Henter tidsstempler (OpenAI whisper-1)...")
        segments = transcribe_segments_openai(client, audio_path)
        if not segments:
            segments = [{"start": 0.0, "end": 0.0, "text": transcript_text}]

        for idx, seg in enumerate(segments):
            seg["id"] = idx + 1

        log("Trin 2c/4: Justerer segmenttekst...")
        try:
            aligned = align_segments_with_transcript(client, segments, transcript_text)
        except Exception:
            aligned = distribute_transcript_by_duration(transcript_text, segments)

        if aligned:
            for seg in segments:
                new_text = aligned.get(seg["id"], "").strip()
                if new_text:
                    seg["text"] = new_text

        log("Trin 3/4: Oversætter segmenter til engelsk (OpenAI)...")
        translation_map = translate_segments_openai(client, segments)

        english_segments = []
        for seg in segments:
            translated_text = translation_map.get(seg["id"], "").strip()
            english_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": translated_text or seg["text"],
                }
            )

        log("Trin 4/4: Opretter undertekster og brænder dem ind i videoen...")
        srt_content = build_srt(english_segments)
        write_text(srt_path, srt_content)
        burn_subtitles(video_path, srt_path, output_video)

        log("Færdig! Filerne er gemt i samme mappe som videoen.")
        log(f"Ny video: {output_video}")
    finally:
        try:
            if "audio_path" in locals() and audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except UserFacingError as exc:
        print(f"Fejl: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAfbrudt af bruger.")
        sys.exit(130)
    except Exception as exc:
        print(f"Uventet fejl: {exc}")
        sys.exit(1)
