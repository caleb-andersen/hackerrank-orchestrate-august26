"""Probe exact provider capabilities from this Python 3.14 machine.

Last edition a stale API-version default silently reported the newer models as
unavailable, which would have collapsed a model bake-off while I believed I was
being rigorous. "I have credits" and "this exact model id answers with a valid
structured response from this machine" are different claims, and only the
second one is worth building on.
"""

import base64
import io
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from anthropic import Anthropic
from openai import OpenAI
from PIL import Image

from config import (
    ANTHROPIC_API_KEY,
    DATASET_DIR,
    DECISION_MODEL_DEV,
    DECISION_MODEL_PRIMARY,
    FALLBACK_CHAIN,
    JUDGE_MODEL,
    MAX_RETRY_ATTEMPTS,
    OPENAI_API_KEY,
    PER_ROW_TIMEOUT_SECONDS,
    RETRY_BASE_SECONDS,
    RETRY_CAP_SECONDS,
    TRANSCRIBE_MODEL,
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model: str
    provider: str
    reachable: str
    latency_seconds: float
    strict_schema: str
    vision: str
    detail: str


class RetryExhaustedError(RuntimeError):
    pass


class PermanentProviderError(RuntimeError):
    pass


_Result = TypeVar("_Result")


def _is_transient(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429 or isinstance(status_code, int) and status_code >= 500:
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _call_with_retry(call: Callable[[], _Result]) -> _Result:
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return call()
        except Exception as error:
            if not _is_transient(error):
                raise PermanentProviderError(f"{type(error).__name__}: {error}") from error
            if attempt + 1 == MAX_RETRY_ATTEMPTS:
                raise RetryExhaustedError(f"{type(error).__name__}: {error}") from error
            retry_ceiling = min(RETRY_CAP_SECONDS, RETRY_BASE_SECONDS * (2**attempt))
            time.sleep(random.uniform(0.0, retry_ceiling))
    raise AssertionError("Retry loop ended without a result")


def _tiny_png_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }


def _valid_structured_text(text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and set(value) == {"ok"} and value["ok"] is True


def _anthropic_text(response: object) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str):
            return block.text
    return ""


def _response_refused(response: object) -> bool:
    if getattr(response, "stop_reason", None) == "refusal":
        return True
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) == "refusal":
                return True
    return False


def _probe_anthropic(model: str, image_base64: str) -> ProbeResult:
    if not ANTHROPIC_API_KEY:
        return ProbeResult(model, "anthropic", "no", 0.0, "no", "no", "missing ANTHROPIC_API_KEY")
    client = Anthropic(
        api_key=ANTHROPIC_API_KEY,
        max_retries=0,
        timeout=PER_ROW_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    try:
        response = _call_with_retry(
            lambda: client.messages.create(
                model=model,
                max_tokens=32,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": "Inspect the image and return ok=true."},
                        ],
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _schema(),
                    }
                },
            )
        )
    except RetryExhaustedError as error:
        return ProbeResult(model, "anthropic", "no", time.perf_counter() - started, "no", "no", f"retry exhausted: {error}")
    except PermanentProviderError as error:
        return ProbeResult(model, "anthropic", "no", time.perf_counter() - started, "no", "no", f"permanent failure: {error}")

    if _response_refused(response):
        return ProbeResult(
            model,
            "anthropic",
            "yes",
            time.perf_counter() - started,
            "no",
            "yes",
            "provider refusal (not retried)",
        )
    text = _anthropic_text(response)
    valid = _valid_structured_text(text)
    detail = "" if valid else "response did not match the strict probe schema"
    return ProbeResult(model, "anthropic", "yes", time.perf_counter() - started, "yes" if valid else "no", "yes", detail)


def _probe_openai_vision(model: str, image_base64: str) -> ProbeResult:
    if not OPENAI_API_KEY:
        return ProbeResult(model, "openai", "no", 0.0, "no", "no", "missing OPENAI_API_KEY")
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=0,
        timeout=PER_ROW_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    try:
        response = _call_with_retry(
            lambda: client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                            {"type": "input_text", "text": "Inspect the image and return ok=true."},
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "capability_probe",
                        "schema": _schema(),
                        "strict": True,
                    }
                },
            )
        )
    except RetryExhaustedError as error:
        return ProbeResult(model, "openai", "no", time.perf_counter() - started, "no", "no", f"retry exhausted: {error}")
    except PermanentProviderError as error:
        return ProbeResult(model, "openai", "no", time.perf_counter() - started, "no", "no", f"permanent failure: {error}")

    if _response_refused(response):
        return ProbeResult(
            model,
            "openai",
            "yes",
            time.perf_counter() - started,
            "no",
            "yes",
            "provider refusal (not retried)",
        )
    output_text = getattr(response, "output_text", "")
    valid = isinstance(output_text, str) and _valid_structured_text(output_text)
    detail = "" if valid else "response did not match the strict probe schema"
    return ProbeResult(model, "openai", "yes", time.perf_counter() - started, "yes" if valid else "no", "yes", detail)


def _first_audio_file() -> Path:
    audio_dir = DATASET_DIR / "media" / "audio"
    candidates = sorted(path for path in audio_dir.iterdir() if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No transcription fixture found in {audio_dir}")
    return candidates[0]


def _probe_transcription(model: str) -> ProbeResult:
    if not OPENAI_API_KEY:
        return ProbeResult(model, "openai", "no", 0.0, "n/a", "n/a", "missing OPENAI_API_KEY")
    try:
        audio_path = _first_audio_file()
    except (FileNotFoundError, OSError) as error:
        return ProbeResult(model, "openai", "no", 0.0, "n/a", "n/a", str(error))

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=0,
        timeout=PER_ROW_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()

    def transcribe() -> object:
        with audio_path.open("rb") as audio:
            return client.audio.transcriptions.create(model=model, file=audio)

    try:
        response = _call_with_retry(transcribe)
    except RetryExhaustedError as error:
        return ProbeResult(model, "openai", "no", time.perf_counter() - started, "n/a", "n/a", f"retry exhausted: {error}")
    except PermanentProviderError as error:
        return ProbeResult(model, "openai", "no", time.perf_counter() - started, "n/a", "n/a", f"permanent failure: {error}")

    transcript = getattr(response, "text", None)
    if not isinstance(transcript, str):
        return ProbeResult(model, "openai", "yes", time.perf_counter() - started, "n/a", "n/a", "response had no text field")
    return ProbeResult(model, "openai", "yes", time.perf_counter() - started, "n/a", "n/a", "")


def _models() -> list[str]:
    return list(
        dict.fromkeys(
            [
                DECISION_MODEL_PRIMARY,
                DECISION_MODEL_DEV,
                *FALLBACK_CHAIN,
                TRANSCRIBE_MODEL,
                JUDGE_MODEL,
            ]
        )
    )


def _print_table(results: list[ProbeResult]) -> None:
    headers = ("model", "provider", "reachable", "latency_s", "strict_schema", "vision", "detail")
    rows = [
        (
            result.model,
            result.provider,
            result.reachable,
            f"{result.latency_seconds:.3f}",
            result.strict_schema,
            result.vision,
            " ".join(result.detail.splitlines()),
        )
        for result in results
    ]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    image_base64 = _tiny_png_base64()
    results: list[ProbeResult] = []
    for model in _models():
        if model == TRANSCRIBE_MODEL:
            results.append(_probe_transcription(model))
        elif model.startswith("claude-"):
            results.append(_probe_anthropic(model, image_base64))
        else:
            results.append(_probe_openai_vision(model, image_base64))

    _print_table(results)
    failed_primary = [
        result.model
        for result in results
        if (
            result.model == DECISION_MODEL_PRIMARY
            and (result.reachable != "yes" or result.strict_schema != "yes" or result.vision != "yes")
        )
        or (
            result.model == TRANSCRIBE_MODEL
            and (result.reachable != "yes" or bool(result.detail))
        )
    ]
    if failed_primary:
        print(f"Primary path failed: {', '.join(failed_primary)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
