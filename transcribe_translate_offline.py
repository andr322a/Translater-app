#!/usr/bin/env python3
"""
Offline video-transskription, oversaettelse og undertekster (ingen API-kald).

- Transskriberer dansk tale lokalt med Whisper
- Oversaetter dansk -> engelsk lokalt med MarianMT
- Genererer SRT og braender undertekster ind i video med FFmpeg
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

try:
    import whisper
except ModuleNotFoundError:
    print(
        "Fejl: openai-whisper er ikke installeret. "
        "Koer: pip install -r requirements-offline.txt"
    )
    sys.exit(1)

try:
    import torch
except ModuleNotFoundError:
    print(
        "Fejl: torch er ikke installeret. "
        "Koer: pip install -r requirements-offline.txt"
    )
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ModuleNotFoundError:
    print(
        "Fejl: transformers er ikke installeret. "
        "Koer: pip install -r requirements-offline.txt"
    )
    sys.exit(1)


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi"}
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "64k")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip().lower()
WHISPER_DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT", "").strip()
WHISPER_ALLOW_INSECURE_DOWNLOAD = os.getenv("WHISPER_ALLOW_INSECURE_DOWNLOAD", "0") == "1"

MARIAN_MODEL = os.getenv("MARIAN_MODEL", "Helsinki-NLP/opus-mt-da-en")
TRANSLATION_BATCH_SIZE = max(1, int(os.getenv("TRANSLATION_BATCH_SIZE", "12")))
GLOSSARY_TERMS_RAW = os.getenv("GLOSSARY_TERMS", "H.P Therkelsen,Zeek")
GLOSSARY_TERMS = [term.strip() for term in GLOSSARY_TERMS_RAW.split(",") if term.strip()]

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
                "Installer en FFmpeg-build med libass."
            )
    except UserFacingError:
        raise
    except Exception:
        pass


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
        "udtraekning af lyd",
    )


def resolve_torch_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "mps"}:
        raise UserFacingError("WHISPER_DEVICE skal vaere auto, cpu eller mps.")

    if requested == "cpu":
        return "cpu"

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if requested == "mps":
        if mps_available:
            return "mps"
        log("  Advarsel: mps er ikke tilgaengelig. Falder tilbage til cpu.")
        return "cpu"

    if mps_available:
        return "mps"
    return "cpu"


def resolve_whisper_fp16(device: str, compute_type: str) -> bool:
    if device == "cpu":
        if compute_type in {"float16", "int8"}:
            log(
                "  Advarsel: WHISPER_COMPUTE_TYPE bruges ikke fuldt paa cpu "
                "(openai-whisper). Bruger float32."
            )
        return False
    return compute_type == "float16"


def is_network_or_ssl_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "certificate_verify_failed",
        "self-signed certificate",
        "ssl:",
        "urlopen error",
        "nodename nor servname provided",
        "name or service not known",
        "temporary failure in name resolution",
        "connection refused",
        "connection reset",
    ]
    return any(marker in text for marker in markers)


def get_whisper_download_root() -> Path:
    if WHISPER_DOWNLOAD_ROOT:
        root = Path(WHISPER_DOWNLOAD_ROOT).expanduser()
    else:
        cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        root = cache_home / "whisper"
    root.mkdir(parents=True, exist_ok=True)
    return root


def insecure_download_whisper_checkpoint(model_name: str) -> Path:
    model_url = getattr(whisper, "_MODELS", {}).get(model_name)
    if not model_url:
        raise UserFacingError(
            f"Kunne ikke finde download-url for Whisper-model '{model_name}'. "
            "Saet WHISPER_MODEL til en officiel model eller lokal checkpoint-sti."
        )

    file_name = Path(urlparse(model_url).path).name or f"{model_name}.pt"
    target = get_whisper_download_root() / file_name
    temp_target = target.with_suffix(target.suffix + ".tmp")

    if target.exists() and target.stat().st_size > 0:
        return target

    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--insecure",
            "--output",
            str(temp_target),
            model_url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise UserFacingError(
            "Insecure Whisper-download fejlede. "
            "Tjek netforbindelse eller installer virksomhedens CA-certifikat.\n"
            f"Detaljer: {result.stderr.strip()}"
        )

    temp_target.replace(target)
    return target


def get_transcription_local(audio_path: Path) -> Dict[str, Any]:
    device = resolve_torch_device(WHISPER_DEVICE)
    fp16 = resolve_whisper_fp16(device, WHISPER_COMPUTE_TYPE)
    log(f"  Whisper device: {device}, fp16={fp16}")

    whisper_kwargs: Dict[str, Any] = {"device": device}
    if WHISPER_DOWNLOAD_ROOT:
        whisper_kwargs["download_root"] = WHISPER_DOWNLOAD_ROOT

    try:
        model = whisper.load_model(WHISPER_MODEL, **whisper_kwargs)
    except Exception as exc:
        if is_network_or_ssl_error(exc):
            if WHISPER_ALLOW_INSECURE_DOWNLOAD:
                log(
                    "  SSL-fejl ved Whisper-download. Proever insecure fallback "
                    "(WHISPER_ALLOW_INSECURE_DOWNLOAD=1)."
                )
                checkpoint = insecure_download_whisper_checkpoint(WHISPER_MODEL)
                model = whisper.load_model(str(checkpoint), **whisper_kwargs)
            else:
                raise UserFacingError(
                    "Whisper-model kunne ikke hentes pga. net/SSL. "
                    "Loesning: 1) installer virksomhedens CA-certifikat i Python trust store, "
                    "2) saet WHISPER_DOWNLOAD_ROOT til lokal cache, eller 3) koer en engangs-"
                    "download med WHISPER_ALLOW_INSECURE_DOWNLOAD=1."
                ) from exc
        else:
            raise

    result = model.transcribe(
        str(audio_path),
        language="da",
        task="transcribe",
        fp16=fp16,
        verbose=False,
    )
    return result


def normalize_segments(transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
    text_da = str(transcription.get("text", "")).strip()
    segments_raw = transcription.get("segments", [])

    if not segments_raw:
        return [{"id": 1, "start": 0.0, "end": 0.5, "text": text_da}]

    normalized: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments_raw, start=1):
        if isinstance(seg, dict):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = str(seg.get("text", "")).strip()
        else:
            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", 0.0))
            text = str(getattr(seg, "text", "")).strip()
        if end <= start:
            end = start + 0.5
        normalized.append({"id": idx, "start": start, "end": end, "text": text})

    return normalized


def protect_glossary(text: str, terms: List[str]) -> Tuple[str, Dict[str, str]]:
    protected = text
    mapping: Dict[str, str] = {}
    for idx, term in enumerate(terms):
        token = f"ZXTERM{idx}ZX"
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        protected, count = pattern.subn(token, protected)
        if count:
            mapping[token] = term
    return protected, mapping


def restore_glossary(text: str, mapping: Dict[str, str]) -> str:
    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    return restored


def build_translation_batches(
    segments: List[Dict[str, Any]],
    batch_size: int,
) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(segments), batch_size):
        batches.append(segments[i : i + batch_size])
    return batches


def init_marian_model() -> Tuple[Any, Any, str]:
    device = resolve_torch_device(WHISPER_DEVICE)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MARIAN_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(MARIAN_MODEL)
    except Exception as exc:
        if is_network_or_ssl_error(exc):
            raise UserFacingError(
                "Marian-model kunne ikke hentes pga. net/SSL. "
                "Loesning: 1) installer virksomhedens CA-certifikat i Python trust store, "
                "eller 2) praehent modellen og peg MARIAN_MODEL paa en lokal modelmappe."
            ) from exc
        raise

    if device in {"mps"}:
        model = model.to(device)
    else:
        device = "cpu"

    model.eval()
    log(f"  Marian device: {device}")
    return tokenizer, model, device


def translate_batch_texts(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
) -> List[str]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    if device != "cpu":
        encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=256,
            num_beams=4,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def translate_segments_local(
    segments: List[Dict[str, Any]],
    glossary_terms: List[str],
) -> Dict[int, str]:
    tokenizer, model, device = init_marian_model()
    translations: Dict[int, str] = {}
    batches = build_translation_batches(segments, TRANSLATION_BATCH_SIZE)

    for idx, batch in enumerate(batches, start=1):
        log(f"  Oversaetter batch {idx}/{len(batches)}...")
        ids = [int(seg["id"]) for seg in batch]
        originals = [str(seg["text"]) for seg in batch]

        protected_texts: List[str] = []
        token_maps: List[Dict[str, str]] = []
        for original in originals:
            protected, token_map = protect_glossary(original, glossary_terms)
            protected_texts.append(protected)
            token_maps.append(token_map)

        try:
            translated = translate_batch_texts(protected_texts, tokenizer, model, device)
        except Exception:
            log("  Advarsel: Batch-oversaettelse fejlede. Bruger dansk tekst for batch.")
            for seg_id, fallback_text in zip(ids, originals):
                translations[seg_id] = fallback_text
            continue

        if len(translated) != len(originals):
            log("  Advarsel: Ugyldigt batch-svar. Bruger dansk tekst for batch.")
            for seg_id, fallback_text in zip(ids, originals):
                translations[seg_id] = fallback_text
            continue

        for seg_id, fallback_text, trans_text, token_map in zip(ids, originals, translated, token_maps):
            restored = restore_glossary(trans_text, token_map).strip()
            translations[seg_id] = restored or fallback_text

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
        "braending af undertekster",
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    log("Velkommen! Dette script koerer lokalt uden API-kald.")

    try:
        ensure_ffmpeg()
        video_path = prompt_for_video_path()

        output_dir = video_path.parent
        base_name = video_path.stem
        audio_path = output_dir / f"{base_name}_audio.m4a"
        srt_path = output_dir / f"{base_name}_english_offline.srt"
        output_video = output_dir / f"{base_name}_english_offline_subtitles.mp4"

        log("Trin 1/4: Udtraekker lyd fra video...")
        extract_audio(video_path, audio_path)

        log("Trin 2/4: Transskriberer lokalt med Whisper...")
        transcription = get_transcription_local(audio_path)
        text_da = str(transcription.get("text", "")).strip()
        if not text_da:
            raise UserFacingError("Transskriptionen blev tom. Tjek lydindholdet.")
        segments = normalize_segments(transcription)

        log("Trin 3/4: Oversaetter segmenter lokalt med MarianMT...")
        translation_map = translate_segments_local(segments, GLOSSARY_TERMS)

        english_segments: List[Dict[str, Any]] = []
        for seg in segments:
            translated_text = translation_map.get(seg["id"], "").strip()
            english_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": translated_text or seg["text"],
                }
            )

        log("Trin 4/4: Opretter undertekster og braender dem ind i videoen...")
        srt_content = build_srt(english_segments)
        write_text(srt_path, srt_content)
        burn_subtitles(video_path, srt_path, output_video)

        log("Faerdig! Filerne er gemt i samme mappe som videoen.")
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
