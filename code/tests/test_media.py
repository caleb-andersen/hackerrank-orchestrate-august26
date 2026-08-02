"""Network-free tests for media resolution and transcription caching."""

from __future__ import annotations

import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from context.media import prepare_image, resolve, transcribe  # noqa: E402


class CountingProvider:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls = 0
        self.failure = failure

    def transcribe(self, audio_path: Path, model: str) -> object:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(response=SimpleNamespace(text="meeting moved to three"))


class MediaTests(unittest.TestCase):
    def _dataset(self, root: Path, *, create_audio: bool) -> Path:
        (root / "media" / "audio").mkdir(parents=True)
        (root / "images.csv").write_text("image_id,file_path\n", encoding="utf-8")
        (root / "voice_notes.csv").write_text(
            "voice_note_id,file_path\nvoice_alpha,media/audio/voice.mp3\n",
            encoding="utf-8",
        )
        if create_audio:
            (root / "media" / "audio" / "voice.mp3").write_bytes(b"fake audio")
        return root

    def test_missing_file_returns_exists_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(Path(temporary), create_audio=False)
            media_ref = resolve(root, "voice_alpha")

        self.assertEqual(media_ref.kind, "voice")
        self.assertFalse(media_ref.exists)
        self.assertIsNone(media_ref.sha256)
        self.assertIsNotNone(media_ref.path)
        self.assertTrue(media_ref.path.is_absolute())

    def test_cache_hit_does_not_recall_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(Path(temporary) / "dataset", create_audio=True)
            cache_dir = Path(temporary) / "cache"
            media_ref = resolve(root, "voice_alpha")
            provider = CountingProvider()
            with patch("context.media.CACHE_DIR", cache_dir):
                first = transcribe(media_ref, provider=provider)
                second = transcribe(media_ref, provider=provider)
                cache_path = cache_dir / "transcripts" / f"{media_ref.sha256}.json"
                cache_exists = cache_path.is_file()

        self.assertEqual(provider.calls, 1)
        self.assertTrue(cache_exists)
        self.assertEqual(first, second)
        self.assertEqual(second.text, "meeting moved to three")
        self.assertEqual(second.status, "ok")

    def test_transcription_failure_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(Path(temporary) / "dataset", create_audio=True)
            cache_dir = Path(temporary) / "cache"
            media_ref = resolve(root, "voice_alpha")
            provider = CountingProvider(failure=RuntimeError("provider unavailable"))
            with patch("context.media.CACHE_DIR", cache_dir):
                transcript = transcribe(media_ref, provider=provider)

        self.assertEqual(provider.calls, 1)
        self.assertIsNone(transcript.text)
        self.assertEqual(transcript.status, "transcript_unavailable")

    def test_image_is_downscaled_and_cached_as_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            image_path = root / "media" / "images" / "poster.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2048, 1024), "navy").save(image_path)
            (root / "images.csv").write_text(
                "image_id,file_path\nimage_alpha,media/images/poster.png\n",
                encoding="utf-8",
            )
            (root / "voice_notes.csv").write_text(
                "voice_note_id,file_path\n",
                encoding="utf-8",
            )
            media_ref = resolve(root, "image_alpha")
            with patch("context.media.CACHE_DIR", Path(temporary) / "cache"):
                first = prepare_image(media_ref)
                image_path.unlink()
                second = prepare_image(media_ref)

        self.assertEqual(first, second)
        with Image.open(BytesIO(first)) as prepared:
            self.assertEqual(prepared.format, "JPEG")
            self.assertEqual(prepared.size, (1024, 512))


if __name__ == "__main__":
    unittest.main()
