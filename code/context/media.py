"""Failure-tolerant media resolution, transcription, and image preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps

from agent.client import OpenAIProvider
from config import CACHE_DIR, MAX_IMAGE_DIMENSION, TRANSCRIBE_MODEL
from data.schema import MediaRef


TranscriptStatus = Literal["ok", "transcript_unavailable"]


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str | None
    status: TranscriptStatus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(media_id: str, kind: Literal["image", "voice"] | None, path: Path | None) -> MediaRef:
    if path is None:
        return MediaRef(media_id=media_id, kind=kind, path=None, exists=False, sha256=None)
    try:
        absolute = path.resolve()
        if not absolute.is_file():
            return MediaRef(media_id=media_id, kind=kind, path=absolute, exists=False, sha256=None)
        digest = _sha256(absolute)
    except OSError:
        try:
            absolute = path.absolute()
        except OSError:
            absolute = None
        return MediaRef(media_id=media_id, kind=kind, path=absolute, exists=False, sha256=None)
    return MediaRef(media_id=media_id, kind=kind, path=absolute, exists=True, sha256=digest)


def describe(
    media_id: str, kind: Literal["image", "voice"] | None, path: Path | None
) -> MediaRef:
    """Describe a media file whose path the caller already resolved.

    The agent loop reaches here rather than through ``resolve`` because the dossier has
    already resolved the path; this only adds the existence check and the content digest
    that ``prepare_image`` and ``transcribe`` key their caches on. Non-raising, like
    everything else in this module.
    """
    return _describe(media_id, kind, path)


def _path_from_loaded_ref(ref: object) -> Path | None:
    raw_path = getattr(ref, "path", None)
    if raw_path is None:
        raw_path = getattr(ref, "file_path", None)
    return None if raw_path is None else Path(raw_path)


def _resolve_loaded_dataset(dataset: object, media_id: str) -> MediaRef | None:
    for kind, attribute in (("image", "images_by_id"), ("voice", "voice_notes_by_id")):
        index = getattr(dataset, attribute, None)
        if isinstance(index, dict) and media_id in index:
            return _describe(media_id, kind, _path_from_loaded_ref(index[media_id]))
    return None


def _resolve_csv(dataset_dir: Path, media_id: str) -> MediaRef | None:
    sources = (
        ("image", dataset_dir / "images.csv", "image_id"),
        ("voice", dataset_dir / "voice_notes.csv", "voice_note_id"),
    )
    for kind, csv_path, id_column in sources:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if (row.get(id_column) or "").strip() != media_id:
                        continue
                    raw_path = (row.get("file_path") or "").strip()
                    path = None
                    if raw_path:
                        candidate = (dataset_dir / raw_path).resolve()
                        try:
                            candidate.relative_to(dataset_dir)
                        except ValueError:
                            return MediaRef(
                                media_id=media_id,
                                kind=kind,
                                path=candidate,
                                exists=False,
                                sha256=None,
                            )
                        path = candidate
                    return _describe(media_id, kind, path)
        except (OSError, csv.Error):
            continue
    return None


def resolve(dataset: object, media_id: str) -> MediaRef:
    """Resolve an ID from a loaded Dataset or a dataset directory without raising."""

    try:
        loaded = _resolve_loaded_dataset(dataset, media_id)
        if loaded is not None:
            return loaded
        if isinstance(dataset, (str, os.PathLike)):
            root = Path(dataset).resolve()
            from_csv = _resolve_csv(root, media_id)
            if from_csv is not None:
                return from_csv
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return _describe(media_id, None, None)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _cached_transcript(path: Path) -> Transcript | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    text = payload.get("text") if isinstance(payload, dict) else None
    status = payload.get("status") if isinstance(payload, dict) else None
    if status == "ok" and isinstance(text, str):
        return Transcript(text=text, status="ok")
    return None


def transcribe(media_ref: MediaRef, *, provider: object | None = None) -> Transcript:
    """Return a cached transcript or a non-raising unavailable result."""

    unavailable = Transcript(text=None, status="transcript_unavailable")
    if not media_ref.exists or media_ref.path is None or media_ref.sha256 is None:
        return unavailable

    cache_path = CACHE_DIR / "transcripts" / f"{media_ref.sha256}.json"
    cached = _cached_transcript(cache_path)
    if cached is not None:
        return cached

    try:
        active_provider = provider if provider is not None else OpenAIProvider()
        result = active_provider.transcribe(media_ref.path, TRANSCRIBE_MODEL)
        response = getattr(result, "response", result)
        text = response if isinstance(response, str) else getattr(response, "text", None)
        if not isinstance(text, str):
            return unavailable
        transcript = Transcript(text=text, status="ok")
        payload = json.dumps(
            {"text": transcript.text, "status": transcript.status},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write(cache_path, payload)
        return transcript
    except Exception:
        return unavailable


def _jpeg_bytes(path: Path) -> bytes:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        # 1024px preserves small poster text and placeholder contact details used
        # by the consistency check.
        if max(image.size) > MAX_IMAGE_DIMENSION:
            image.thumbnail(
                (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
                resample=Image.Resampling.LANCZOS,
            )
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()


def prepare_image(media_ref: MediaRef) -> bytes:
    """Prepare and cache a bounded JPEG; invalid or missing media returns empty bytes."""

    if not media_ref.exists or media_ref.path is None or media_ref.sha256 is None:
        return b""
    cache_path = CACHE_DIR / "images" / f"{media_ref.sha256}.jpg"
    try:
        cached = cache_path.read_bytes()
        if cached:
            return cached
    except OSError:
        pass
    try:
        prepared = _jpeg_bytes(media_ref.path)
        _atomic_write(cache_path, prepared)
        return prepared
    except Exception:
        return b""
