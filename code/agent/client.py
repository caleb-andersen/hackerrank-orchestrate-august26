"""Provider-neutral completion clients with explicit failure semantics.

The public surface is intentionally structural: orchestration code only needs an
object with ``complete``, ``supports_vision``, ``supports_audio``, and
``batch_tool_results`` methods.  Raw SDK responses remain available in the result,
while retry counts and the model that answered are represented as typed metadata.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from config import (
    ANTHROPIC_API_KEY,
    FALLBACK_CHAIN,
    MAX_RETRY_ATTEMPTS,
    OPENAI_API_KEY,
    PER_ROW_TIMEOUT_SECONDS,
    RETRY_BASE_SECONDS,
    RETRY_CAP_SECONDS,
)


_LOGGER = logging.getLogger(__name__)
_STREAM_TOKEN_THRESHOLD = 16_000
_MAX_PAUSE_TURN_RESTARTS = 5

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RetryResult(Generic[_T]):
    """A successful operation and the number of calls needed to obtain it."""

    value: _T
    attempts: int

    @property
    def retry_count(self) -> int:
        return self.attempts - 1


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A provider completion, including retry and pause-turn accounting."""

    response: object
    attempts: int
    retry_count: int
    pause_restarts: int = 0


@dataclass(frozen=True, slots=True)
class FallbackResult:
    """The row-scoped fallback result returned to the orchestration layer."""

    response: object
    model: str
    attempts: int
    retry_count: int
    models_tried: tuple[str, ...]
    pause_restarts: int = 0


class ProviderClientError(RuntimeError):
    """Base class for failures whose downstream outcome must remain explicit."""

    outcome = "provider_failure"

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        retry_count: int = 0,
        category: str = "provider",
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retry_count = retry_count
        self.category = category

    def add_prior_attempts(self, attempts: int, retry_count: int) -> None:
        """Include work done before this error, such as a paused request."""

        self.attempts += attempts
        self.retry_count += retry_count


class PermanentProviderError(ProviderClientError):
    """A non-retryable provider or request failure."""

    outcome = "permanent_failure"


class AuthenticationProviderError(PermanentProviderError):
    outcome = "authentication_failure"


class BillingProviderError(PermanentProviderError):
    outcome = "billing_failure"


class RequestTooLargeError(PermanentProviderError):
    outcome = "request_too_large"


class SchemaValidationError(PermanentProviderError):
    outcome = "schema_validation_failure"


class ProviderRefusalError(PermanentProviderError):
    outcome = "refusal"


class UnsupportedCapabilityError(PermanentProviderError):
    outcome = "unsupported_capability"


class ModelFailureError(ProviderClientError):
    """A failure attributable to the selected model, eligible for fallback."""

    outcome = "model_failure"


class PauseTurnLimitError(ModelFailureError):
    outcome = "pause_turn_limit"


class RetryExhaustedError(ProviderClientError):
    """A retryable error that remained transient through the final attempt."""

    outcome = "retry_exhausted"


class FallbackExhaustedError(ModelFailureError):
    outcome = "fallback_exhausted"

    def __init__(
        self,
        failures: Sequence[tuple[str, ModelFailureError]],
        *,
        attempts: int,
        retry_count: int,
    ) -> None:
        self.failures = tuple(failures)
        models = ", ".join(model for model, _ in failures)
        super().__init__(
            f"Every fallback model failed: {models}",
            attempts=attempts,
            retry_count=retry_count,
            category="model",
        )


@runtime_checkable
class Provider(Protocol):
    """The duck-typed provider contract consumed by the agent loop."""

    def complete(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        model: str,
        **kw: object,
    ) -> CompletionResult: ...

    def supports_vision(self) -> bool: ...

    def supports_audio(self) -> bool: ...

    def batch_tool_results(
        self, results: Sequence[Mapping[str, object] | object]
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class _ErrorClassification:
    retryable: bool
    category: str
    error_type: type[PermanentProviderError] | None = None


_BILLING_CODES = frozenset(
    {
        "billing_error",
        "credit_balance_too_low",
        "insufficient_credits",
        "insufficient_quota",
    }
)
_REQUEST_SIZE_CODES = frozenset(
    {
        "context_length_exceeded",
        "max_tokens_exceeded",
        "request_too_large",
        "request_too_large_error",
    }
)
_SCHEMA_CODES = frozenset(
    {
        "invalid_json_schema",
        "invalid_response_format",
        "schema_validation_error",
    }
)
_MODEL_FAILURE_CODES = frozenset(
    {
        "model_error",
        "model_failure",
        "model_not_found",
        "model_unavailable",
    }
)
_CONNECTION_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ReadError",
        "ReadTimeout",
        "TimeoutException",
    }
)
_AUTH_ERROR_NAMES = frozenset(
    {"AuthenticationError", "AuthenticationException", "PermissionDeniedError"}
)
_VALIDATION_ERROR_NAMES = frozenset(
    {"APIResponseValidationError", "SchemaValidationError", "ValidationError"}
)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _status_code(error: BaseException) -> int | None:
    direct = _as_int(getattr(error, "status_code", None))
    if direct is not None:
        return direct
    response = getattr(error, "response", None)
    return _as_int(getattr(response, "status_code", None))


def _mapping_value(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _error_code(error: BaseException) -> str:
    candidates: list[object] = [error]
    for attribute in ("body", "error"):
        nested = getattr(error, attribute, None)
        if nested is not None:
            candidates.append(nested)
            deeper = _mapping_value(nested, "error")
            if deeper is not None:
                candidates.append(deeper)
    for candidate in candidates:
        for key in ("code", "type"):
            value = _mapping_value(candidate, key)
            if isinstance(value, str) and value:
                return value.lower()
    return ""


def _classify_error(error: BaseException) -> _ErrorClassification:
    """Classify an exception before any retry or fallback decision is made."""

    if isinstance(error, ProviderClientError):
        return _ErrorClassification(False, error.category, None)

    status = _status_code(error)
    code = _error_code(error)
    name = type(error).__name__
    message = str(error).lower()

    if status in {401, 403} or name in _AUTH_ERROR_NAMES:
        return _ErrorClassification(False, "authentication", AuthenticationProviderError)
    if status == 402 or code in _BILLING_CODES:
        return _ErrorClassification(False, "billing", BillingProviderError)
    if (
        status == 413
        or code in _REQUEST_SIZE_CODES
        or "maximum context length" in message
        or "request too large" in message
    ):
        return _ErrorClassification(False, "request_size", RequestTooLargeError)
    if code in _SCHEMA_CODES or name in _VALIDATION_ERROR_NAMES:
        return _ErrorClassification(False, "schema", SchemaValidationError)
    if status == 429:
        return _ErrorClassification(True, "rate_limit")
    if status is not None and 500 <= status <= 599:
        return _ErrorClassification(True, "server")
    if isinstance(error, (ConnectionError, TimeoutError)) or name in _CONNECTION_ERROR_NAMES:
        return _ErrorClassification(True, "transport")
    if code in _MODEL_FAILURE_CODES:
        return _ErrorClassification(False, "model", None)
    return _ErrorClassification(False, "unclassified", PermanentProviderError)


def _response_value(response: object, key: str) -> object | None:
    if isinstance(response, Mapping):
        return response.get(key)
    return getattr(response, key, None)


def _stop_reason(response: object) -> str | None:
    for key in ("stop_reason", "finish_reason"):
        value = _response_value(response, key)
        if isinstance(value, str):
            return value.lower()
    return None


def _response_contains_refusal(response: object) -> bool:
    """Inspect content only after the response stop reason has been handled."""

    output = _response_value(response, "output")
    if not isinstance(output, Iterable) or isinstance(output, (str, bytes, Mapping)):
        return False
    for item in output:
        content = _mapping_value(item, "content")
        if not isinstance(content, Iterable) or isinstance(content, (str, bytes, Mapping)):
            continue
        for block in content:
            if _mapping_value(block, "type") == "refusal":
                return True
    return False


def _raise_for_response_outcome(response: object, attempts: int) -> None:
    # The stop reason is intentionally inspected before any response content.
    reason = _stop_reason(response)
    if reason == "refusal":
        raise ProviderRefusalError(
            "The provider refused the request",
            attempts=attempts,
            retry_count=attempts - 1,
            category="refusal",
        )
    if reason in _MODEL_FAILURE_CODES:
        raise ModelFailureError(
            f"The model stopped with {reason!r}",
            attempts=attempts,
            retry_count=attempts - 1,
            category="model",
        )

    status = _response_value(response, "status")
    if isinstance(status, str) and status.lower() == "failed":
        raise ModelFailureError(
            "The model returned a failed response",
            attempts=attempts,
            retry_count=attempts - 1,
            category="model",
        )
    if _response_contains_refusal(response):
        raise ProviderRefusalError(
            "The provider returned a refusal content block",
            attempts=attempts,
            retry_count=attempts - 1,
            category="refusal",
        )


def _headers(error: BaseException) -> object | None:
    response = getattr(error, "response", None)
    response_headers = getattr(response, "headers", None)
    return response_headers if response_headers is not None else getattr(error, "headers", None)


def _header(headers: object | None, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is None:
            value = getter(name.lower())
        if value is not None:
            return str(value)
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return str(value)
    return None


def _retry_after_seconds(error: BaseException) -> float | None:
    raw = _header(_headers(error), "retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


def _typed_permanent_error(
    error: BaseException,
    classification: _ErrorClassification,
    attempts: int,
) -> ProviderClientError:
    if isinstance(error, ProviderClientError):
        error.attempts = attempts
        error.retry_count = attempts - 1
        return error
    if classification.category == "model":
        return ModelFailureError(
            f"{type(error).__name__}: {error}",
            attempts=attempts,
            retry_count=attempts - 1,
            category="model",
        )
    error_type = classification.error_type or PermanentProviderError
    return error_type(
        f"{type(error).__name__}: {error}",
        attempts=attempts,
        retry_count=attempts - 1,
        category=classification.category,
    )


def retry_with_backoff(
    fn: Callable[[], _T],
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    base: float = RETRY_BASE_SECONDS,
    cap: float = RETRY_CAP_SECONDS,
    jitter: Literal["full", "none"] = "full",
    *,
    _sleep: Callable[[float], None] = time.sleep,
    _random: Callable[[], float] = random.random,
) -> RetryResult[_T]:
    """Run ``fn`` with classified retries and return exact attempt metadata.

    Permanent failures are raised immediately as typed errors.  A transient error
    on the last attempt raises :class:`RetryExhaustedError`; it never becomes a
    refusal or a conservative routing verdict.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if base < 0 or cap < 0:
        raise ValueError("retry delays cannot be negative")
    if jitter not in {"full", "none"}:
        raise ValueError("jitter must be 'full' or 'none'")

    for attempt in range(1, max_attempts + 1):
        try:
            response = fn()
        except Exception as error:
            classification = _classify_error(error)
            if not classification.retryable:
                typed_error = _typed_permanent_error(error, classification, attempt)
                if typed_error is error:
                    raise
                raise typed_error from error
            if attempt == max_attempts:
                raise RetryExhaustedError(
                    f"{type(error).__name__}: {error}",
                    attempts=attempt,
                    retry_count=attempt - 1,
                    category=classification.category,
                ) from error

            retry_after = _retry_after_seconds(error) if _status_code(error) == 429 else None
            if retry_after is not None:
                delay = retry_after
            else:
                ceiling = min(cap, base * (2 ** (attempt - 1)))
                fraction = min(1.0, max(0.0, _random()))
                delay = ceiling * fraction if jitter == "full" else ceiling
            _sleep(delay)
            continue

        _raise_for_response_outcome(response, attempt)
        return RetryResult(response, attempt)

    raise AssertionError("retry loop ended without returning or raising")


def _plain(value: object) -> object:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _json_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_plain(value), ensure_ascii=False, separators=(",", ":"))


def _tool_result_parts(result: Mapping[str, object] | object) -> tuple[str, object, bool]:
    identifier: object | None = None
    for key in ("tool_use_id", "call_id", "tool_call_id", "id"):
        identifier = _mapping_value(result, key)
        if isinstance(identifier, str) and identifier:
            break
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("tool result is missing a call identifier")

    sentinel = object()
    content: object = sentinel
    for key in ("content", "output", "result"):
        candidate = _mapping_value(result, key)
        if candidate is not None:
            content = candidate
            break
    if content is sentinel:
        content = ""
    return identifier, content, bool(_mapping_value(result, "is_error"))


def _anthropic_tools(tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for tool in tools:
        source: Mapping[str, object] = tool
        nested = tool.get("function")
        if isinstance(nested, Mapping):
            source = nested
        name = source.get("name")
        schema = source.get("input_schema", source.get("parameters"))
        if not isinstance(name, str) or not isinstance(schema, Mapping):
            raise SchemaValidationError(
                "Each tool needs a name and an input_schema/parameters object",
                category="schema",
            )
        item: dict[str, object] = {"name": name, "input_schema": dict(schema)}
        description = source.get("description")
        if isinstance(description, str):
            item["description"] = description
        # Carried through rather than rebuilt away: this converter reconstructs each tool
        # field by field, so anything not named here is silently dropped. A dropped
        # cache_control is the worst shape of that bug — the request still succeeds, the
        # caller still believes it asked for caching, and the only visible symptom is a
        # cache_read counter that stays at zero.
        cache_control = source.get("cache_control")
        if isinstance(cache_control, Mapping):
            item["cache_control"] = dict(cache_control)
        converted.append(item)
    return converted


def _openai_tools(tools: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("name"), str):
            converted.append(dict(tool))
            continue
        source: Mapping[str, object] = tool
        nested = tool.get("function")
        if isinstance(nested, Mapping):
            source = nested
        name = source.get("name")
        schema = source.get("parameters", source.get("input_schema"))
        if not isinstance(name, str) or not isinstance(schema, Mapping):
            raise SchemaValidationError(
                "Each tool needs a name and an input_schema/parameters object",
                category="schema",
            )
        item: dict[str, object] = {
            "type": "function",
            "name": name,
            "parameters": dict(schema),
            "strict": bool(source.get("strict", True)),
        }
        description = source.get("description")
        if isinstance(description, str):
            item["description"] = description
        converted.append(item)
    return converted


def _anthropic_block(block: object) -> object:
    plain = _plain(block)
    if not isinstance(plain, Mapping):
        return plain
    block_type = plain.get("type")
    if block_type in {"input_text", "output_text"}:
        return {"type": "text", "text": plain.get("text", "")}
    if block_type == "input_image":
        image_url = plain.get("image_url")
        if not isinstance(image_url, str):
            raise SchemaValidationError("input_image is missing image_url", category="schema")
        if image_url.startswith("data:") and ";base64," in image_url:
            header, data = image_url.split(";base64,", 1)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": header.removeprefix("data:"),
                    "data": data,
                },
            }
        return {"type": "image", "source": {"type": "url", "url": image_url}}
    if block_type in {"input_audio", "audio"}:
        raise UnsupportedCapabilityError(
            "Anthropic messages do not accept audio input",
            category="capability",
        )
    return dict(plain)


def _system_blocks(content: object) -> list[dict[str, object]]:
    """Normalise system content to text blocks, preserving any cache breakpoint.

    ``cache_control`` survives this conversion for the same reason it survives
    ``_anthropic_tools``: the block is rebuilt field by field, so a breakpoint the caller
    set would otherwise vanish here and the request would run uncached while looking
    exactly like one that had asked for caching.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        blocks: list[dict[str, object]] = []
        for block in content:
            converted = _anthropic_block(block)
            if not isinstance(converted, Mapping) or converted.get("type") != "text":
                raise SchemaValidationError(
                    "system content must contain only text", category="schema"
                )
            text_block: dict[str, object] = {
                "type": "text",
                "text": str(converted.get("text", "")),
            }
            cache_control = converted.get("cache_control")
            if isinstance(cache_control, Mapping):
                text_block["cache_control"] = dict(cache_control)
            blocks.append(text_block)
        return blocks
    raise SchemaValidationError("system content must be text or text blocks", category="schema")


def _anthropic_input(
    messages: Sequence[Mapping[str, object]], explicit_system: object | None
) -> tuple[list[dict[str, object]], object | None]:
    converted: list[dict[str, object]] = []
    system: list[dict[str, str]] = []
    if explicit_system is not None:
        system.extend(_system_blocks(explicit_system))
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            system.extend(_system_blocks(content))
            continue
        if role not in {"user", "assistant"}:
            converted.append(dict(message))
            continue
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            normalised_content: object = [_anthropic_block(block) for block in content]
        else:
            normalised_content = content
        converted.append({"role": role, "content": normalised_content})
    return converted, system or None


def _openai_block(block: object, role: object) -> object:
    plain = _plain(block)
    if not isinstance(plain, Mapping):
        return plain
    block_type = plain.get("type")
    if block_type == "text":
        target_type = "output_text" if role == "assistant" else "input_text"
        return {"type": target_type, "text": plain.get("text", "")}
    if block_type == "image":
        source = plain.get("source")
        if not isinstance(source, Mapping):
            raise SchemaValidationError("image is missing source", category="schema")
        if source.get("type") == "base64":
            url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
        else:
            url = source.get("url")
        if not isinstance(url, str):
            raise SchemaValidationError("image source is missing data/url", category="schema")
        return {"type": "input_image", "image_url": url}
    return dict(plain)


def _openai_input(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for message in messages:
        if "role" not in message:
            converted.append(dict(message))
            continue
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            content = [_openai_block(block, role) for block in content]
        converted.append({"role": role, "content": content})
    return converted


def _token_limit(kw: dict[str, object], *, openai: bool) -> int:
    primary = "max_output_tokens" if openai else "max_tokens"
    secondary = "max_tokens" if openai else "max_output_tokens"
    value = kw.pop(primary, kw.pop(secondary, 4_096))
    limit = _as_int(value)
    if limit is None or limit < 1:
        raise ValueError(f"{primary} must be a positive integer")
    return limit


class AnthropicProvider:
    """Anthropic Messages API adapter with grouped tool results."""

    def __init__(
        self,
        client: object | None = None,
        *,
        max_attempts: int = MAX_RETRY_ATTEMPTS,
        retry_base: float = RETRY_BASE_SECONDS,
        retry_cap: float = RETRY_CAP_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        if client is None:
            if not ANTHROPIC_API_KEY:
                raise AuthenticationProviderError(
                    "ANTHROPIC_API_KEY is not configured",
                    attempts=0,
                    category="authentication",
                )
            from anthropic import Anthropic

            client = Anthropic(
                api_key=ANTHROPIC_API_KEY,
                max_retries=0,
                timeout=PER_ROW_TIMEOUT_SECONDS,
            )
        self._client = client
        self._max_attempts = max_attempts
        self._retry_base = retry_base
        self._retry_cap = retry_cap
        self._sleep = sleep
        self._random = random_source

    def supports_vision(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return False

    def batch_tool_results(
        self, results: Sequence[Mapping[str, object] | object]
    ) -> list[dict[str, object]]:
        if not results:
            return []
        blocks: list[dict[str, object]] = []
        for result in results:
            identifier, content, is_error = _tool_result_parts(result)
            block: dict[str, object] = {
                "type": "tool_result",
                "tool_use_id": identifier,
                "content": content if isinstance(content, (str, list)) else _json_text(content),
            }
            if is_error:
                block["is_error"] = True
            blocks.append(block)
        return [{"role": "user", "content": blocks}]

    def _request(self, params: dict[str, object], stream: bool) -> object:
        messages_api = getattr(self._client, "messages")
        if not stream:
            return messages_api.create(**params)
        with messages_api.stream(**params) as response_stream:
            return response_stream.get_final_message()

    def complete(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        model: str,
        **kw: object,
    ) -> CompletionResult:
        options = dict(kw)
        max_tokens = _token_limit(options, openai=False)
        stream = bool(options.pop("stream", False)) or max_tokens > _STREAM_TOKEN_THRESHOLD
        explicit_system = options.pop("system", None)
        request_messages, system = _anthropic_input(messages, explicit_system)
        params: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": request_messages,
            **options,
        }
        if tools:
            params["tools"] = _anthropic_tools(tools)
        if system is not None:
            params["system"] = system

        total_attempts = 0
        total_retries = 0
        pause_restarts = 0
        while True:
            try:
                result = retry_with_backoff(
                    lambda: self._request(params, stream),
                    self._max_attempts,
                    self._retry_base,
                    self._retry_cap,
                    _sleep=self._sleep,
                    _random=self._random,
                )
            except ProviderClientError as error:
                error.add_prior_attempts(total_attempts, total_retries)
                raise
            total_attempts += result.attempts
            total_retries += result.retry_count

            if _stop_reason(result.value) != "pause_turn":
                return CompletionResult(
                    result.value,
                    total_attempts,
                    total_retries,
                    pause_restarts,
                )
            if pause_restarts == _MAX_PAUSE_TURN_RESTARTS:
                raise PauseTurnLimitError(
                    f"pause_turn exceeded {_MAX_PAUSE_TURN_RESTARTS} restarts",
                    attempts=total_attempts,
                    retry_count=total_retries,
                    category="model",
                )

            # Anthropic requires the paused assistant turn itself, not a synthetic
            # user message such as "Continue".
            content = _response_value(result.value, "content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
                raise ModelFailureError(
                    "pause_turn response did not contain an assistant turn",
                    attempts=total_attempts,
                    retry_count=total_retries,
                    category="model",
                )
            request_messages.append(
                {"role": "assistant", "content": [_plain(block) for block in content]}
            )
            pause_restarts += 1


class OpenAIProvider:
    """OpenAI Responses API adapter with one item per tool result."""

    def __init__(
        self,
        client: object | None = None,
        *,
        max_attempts: int = MAX_RETRY_ATTEMPTS,
        retry_base: float = RETRY_BASE_SECONDS,
        retry_cap: float = RETRY_CAP_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        if client is None:
            if not OPENAI_API_KEY:
                raise AuthenticationProviderError(
                    "OPENAI_API_KEY is not configured",
                    attempts=0,
                    category="authentication",
                )
            from openai import OpenAI

            client = OpenAI(
                api_key=OPENAI_API_KEY,
                max_retries=0,
                timeout=PER_ROW_TIMEOUT_SECONDS,
            )
        self._client = client
        self._max_attempts = max_attempts
        self._retry_base = retry_base
        self._retry_cap = retry_cap
        self._sleep = sleep
        self._random = random_source

    def supports_vision(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return True

    def transcribe(self, audio_path: str | Path, model: str) -> CompletionResult:
        """Transcribe a local audio file with the provider retry policy."""

        path = Path(audio_path)

        def request() -> object:
            with path.open("rb") as audio:
                return self._client.audio.transcriptions.create(model=model, file=audio)

        result = retry_with_backoff(
            request,
            self._max_attempts,
            self._retry_base,
            self._retry_cap,
            _sleep=self._sleep,
            _random=self._random,
        )
        return CompletionResult(result.value, result.attempts, result.retry_count)

    def batch_tool_results(
        self, results: Sequence[Mapping[str, object] | object]
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for result in results:
            identifier, content, _ = _tool_result_parts(result)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": identifier,
                    "output": _json_text(content),
                }
            )
        return items

    def _request(self, params: dict[str, object], stream: bool) -> object:
        responses_api = getattr(self._client, "responses")
        if not stream:
            return responses_api.create(**params)
        with responses_api.stream(**params) as response_stream:
            return response_stream.get_final_response()

    def complete(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        model: str,
        **kw: object,
    ) -> CompletionResult:
        options = dict(kw)
        max_tokens = _token_limit(options, openai=True)
        stream = bool(options.pop("stream", False)) or max_tokens > _STREAM_TOKEN_THRESHOLD
        params: dict[str, object] = {
            "model": model,
            "max_output_tokens": max_tokens,
            "input": _openai_input(messages),
            **options,
        }
        if tools:
            params["tools"] = _openai_tools(tools)
        result = retry_with_backoff(
            lambda: self._request(params, stream),
            self._max_attempts,
            self._retry_base,
            self._retry_cap,
            _sleep=self._sleep,
            _random=self._random,
        )
        return CompletionResult(result.value, result.attempts, result.retry_count)


ProviderResolver = Callable[[str], Provider]


def default_provider_resolver() -> ProviderResolver:
    instances: dict[str, Provider] = {}

    def resolve(model: str) -> Provider:
        provider_name = "anthropic" if model.lower().startswith("claude") else "openai"
        if provider_name not in instances:
            instances[provider_name] = (
                AnthropicProvider() if provider_name == "anthropic" else OpenAIProvider()
            )
        return instances[provider_name]

    return resolve


def call_with_fallback(
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    chain: Sequence[str] = FALLBACK_CHAIN,
    *,
    provider_resolver: ProviderResolver | None = None,
    **kw: object,
) -> FallbackResult:
    """Try at most three distinct models for this row.

    Only :class:`ModelFailureError` advances the chain. Authentication, billing,
    rate-limit exhaustion, oversized requests, and transport exhaustion propagate
    immediately, so fallback never hides an account or request-level problem.
    """

    models = tuple(dict.fromkeys(chain))[:3]
    if not models:
        raise ValueError("fallback chain must contain at least one model")
    resolve = provider_resolver or default_provider_resolver()
    failures: list[tuple[str, ModelFailureError]] = []
    models_tried: list[str] = []
    total_attempts = 0
    total_retries = 0

    for model in models:
        models_tried.append(model)
        provider = resolve(model)
        try:
            completion = provider.complete(messages, tools, model, **kw)
        except ModelFailureError as error:
            failures.append((model, error))
            total_attempts += error.attempts
            total_retries += error.retry_count
            _LOGGER.warning(
                "model_failed model=%s fallback_position=%d outcome=%s",
                model,
                len(models_tried),
                error.outcome,
            )
            continue

        total_attempts += completion.attempts
        total_retries += completion.retry_count
        _LOGGER.info(
            "model_answered model=%s fallback_position=%d attempts=%d retries=%d",
            model,
            len(models_tried),
            total_attempts,
            total_retries,
        )
        return FallbackResult(
            response=completion.response,
            model=model,
            attempts=total_attempts,
            retry_count=total_retries,
            models_tried=tuple(models_tried),
            pause_restarts=completion.pause_restarts,
        )

    raise FallbackExhaustedError(
        failures,
        attempts=total_attempts,
        retry_count=total_retries,
    )
