"""Network-free tests for provider retries, fallback, and turn semantics."""

from __future__ import annotations

import copy
import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent.client import (  # noqa: E402
    AnthropicProvider,
    AuthenticationProviderError,
    BillingProviderError,
    CompletionResult,
    ModelFailureError,
    OpenAIProvider,
    PauseTurnLimitError,
    ProviderRefusalError,
    RequestTooLargeError,
    RetryExhaustedError,
    _anthropic_input,
    _anthropic_tools,
    _system_blocks,
    call_with_fallback,
    retry_with_backoff,
)
from agent.loop import _usage, fallback_chain_for  # noqa: E402
from config import DECISION_MODEL_DEV, FALLBACK_CHAIN  # noqa: E402


class FakeHTTPError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class RefusalResponse:
    stop_reason = "refusal"

    @property
    def output(self) -> object:
        raise AssertionError("response content was read before stop_reason")


class ScriptedProvider:
    """A duck-typed provider whose outcomes are supplied by the test."""

    def __init__(self, scripts: Mapping[str, Sequence[object]]) -> None:
        self._scripts = {model: list(outcomes) for model, outcomes in scripts.items()}
        self.calls: list[str] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        model: str,
        **kw: object,
    ) -> CompletionResult:
        del messages, tools, kw
        self.calls.append(model)
        outcome = self._scripts[model].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, CompletionResult):
            return outcome
        return CompletionResult(outcome, attempts=1, retry_count=0)

    def supports_vision(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return False

    def batch_tool_results(
        self, results: Sequence[Mapping[str, object] | object]
    ) -> list[dict[str, object]]:
        return [dict(result) for result in results if isinstance(result, Mapping)]


class ScriptedMessagesAPI:
    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.methods: list[str] = []

    def create(self, **params: object) -> object:
        self.methods.append("create")
        self.calls.append(copy.deepcopy(params))
        return self._responses.pop(0)

    def stream(self, **params: object) -> object:
        self.methods.append("stream")
        self.calls.append(copy.deepcopy(params))
        return AnthropicStream(self._responses.pop(0))


class AnthropicStream:
    def __init__(self, response: object) -> None:
        self._response = response

    def __enter__(self) -> AnthropicStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_final_message(self) -> object:
        return self._response


class ScriptedResponsesAPI:
    def __init__(self, response: object) -> None:
        self._response = response
        self.create_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []

    def create(self, **params: object) -> object:
        self.create_calls.append(copy.deepcopy(params))
        return self._response

    def stream(self, **params: object) -> object:
        self.stream_calls.append(copy.deepcopy(params))
        return OpenAIStream(self._response)


class OpenAIStream:
    def __init__(self, response: object) -> None:
        self._response = response

    def __enter__(self) -> OpenAIStream:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_final_response(self) -> object:
        return self._response


class RetryTest(unittest.TestCase):
    def test_429_retries_and_honours_retry_after(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def operation() -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FakeHTTPError(429, "slow down", {"Retry-After": "0.25"})
            return SimpleNamespace(stop_reason="end_turn")

        result = retry_with_backoff(
            operation,
            max_attempts=2,
            base=99,
            cap=99,
            _sleep=sleeps.append,
        )

        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [0.25])

    def test_5xx_retries(self) -> None:
        outcomes: list[object] = [
            FakeHTTPError(503, "temporarily unavailable"),
            SimpleNamespace(stop_reason="end_turn"),
        ]

        def operation() -> object:
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        result = retry_with_backoff(
            operation,
            max_attempts=2,
            base=0,
            cap=0,
            _sleep=lambda _: None,
        )

        self.assertEqual(result.attempts, 2)

    def test_permanent_http_failures_are_not_retried(self) -> None:
        cases = (
            (401, AuthenticationProviderError),
            (402, BillingProviderError),
            (413, RequestTooLargeError),
        )

        for status, expected in cases:
            with self.subTest(status=status):
                calls = 0

                def operation() -> object:
                    nonlocal calls
                    calls += 1
                    raise FakeHTTPError(status, "permanent")

                with self.assertRaises(expected):
                    retry_with_backoff(operation, max_attempts=4)
                self.assertEqual(calls, 1)

    def test_refusal_is_not_retried_or_content_read(self) -> None:
        calls = 0

        def operation() -> object:
            nonlocal calls
            calls += 1
            return RefusalResponse()

        with self.assertRaises(ProviderRefusalError) as caught:
            retry_with_backoff(operation, max_attempts=4, _sleep=lambda _: None)

        self.assertEqual(calls, 1)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(caught.exception.outcome, "refusal")

    def test_retry_result_counts_attempts_and_retries(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("temporary disconnect")
            return "ok"

        result = retry_with_backoff(
            operation,
            max_attempts=3,
            base=0,
            cap=0,
            _sleep=lambda _: None,
        )

        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.retry_count, 2)

    def test_exhaustion_and_refusal_are_distinct_outcomes(self) -> None:
        with self.assertRaises(RetryExhaustedError) as exhausted:
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(ConnectionError("offline")),
                max_attempts=2,
                base=0,
                cap=0,
                _sleep=lambda _: None,
            )
        with self.assertRaises(ProviderRefusalError) as refused:
            retry_with_backoff(lambda: RefusalResponse(), max_attempts=2)

        self.assertEqual(exhausted.exception.outcome, "retry_exhausted")
        self.assertEqual(exhausted.exception.attempts, 2)
        self.assertEqual(refused.exception.outcome, "refusal")
        self.assertEqual(refused.exception.attempts, 1)
        self.assertNotIsInstance(exhausted.exception, ProviderRefusalError)


class FallbackTest(unittest.TestCase):
    def test_chain_advances_only_after_model_failure(self) -> None:
        provider = ScriptedProvider(
            {
                "first": [ModelFailureError("model failed", category="model")],
                "second": [SimpleNamespace(stop_reason="end_turn")],
            }
        )

        result = call_with_fallback(
            [],
            [],
            chain=["first", "second"],
            provider_resolver=lambda _: provider,
        )

        self.assertEqual(provider.calls, ["first", "second"])
        self.assertEqual(result.model, "second")
        self.assertEqual(result.models_tried, ("first", "second"))
        self.assertEqual(result.attempts, 2)

    def test_chain_is_deduplicated_then_capped_at_three(self) -> None:
        provider = ScriptedProvider(
            {
                "first": [ModelFailureError("failed", category="model")],
                "second": [ModelFailureError("failed", category="model")],
                "third": [SimpleNamespace(stop_reason="end_turn")],
                "fourth": [SimpleNamespace(stop_reason="end_turn")],
            }
        )

        result = call_with_fallback(
            [],
            [],
            chain=["first", "first", "second", "third", "fourth"],
            provider_resolver=lambda _: provider,
        )

        self.assertEqual(provider.calls, ["first", "second", "third"])
        self.assertEqual(result.model, "third")
        self.assertEqual(result.models_tried, ("first", "second", "third"))

    def test_authentication_error_does_not_advance_chain(self) -> None:
        provider = ScriptedProvider(
            {
                "first": [
                    AuthenticationProviderError(
                        "bad key", category="authentication"
                    )
                ],
                "second": [SimpleNamespace(stop_reason="end_turn")],
            }
        )

        with self.assertRaises(AuthenticationProviderError):
            call_with_fallback(
                [],
                [],
                chain=["first", "second"],
                provider_resolver=lambda _: provider,
            )

        self.assertEqual(provider.calls, ["first"])

    def test_other_non_model_failures_do_not_advance_chain(self) -> None:
        failures = (
            BillingProviderError("no credit", category="billing"),
            RequestTooLargeError("too large", category="request_size"),
            RetryExhaustedError("rate limited", category="rate_limit"),
            RetryExhaustedError("connection failed", category="transport"),
        )

        for failure in failures:
            with self.subTest(outcome=failure.outcome, category=failure.category):
                provider = ScriptedProvider(
                    {
                        "first": [failure],
                        "second": [SimpleNamespace(stop_reason="end_turn")],
                    }
                )
                with self.assertRaises(type(failure)):
                    call_with_fallback(
                        [],
                        [],
                        chain=["first", "second"],
                        provider_resolver=lambda _: provider,
                    )
                self.assertEqual(provider.calls, ["first"])


class ProviderShapeTest(unittest.TestCase):
    def test_tool_results_follow_each_provider_batching_rule(self) -> None:
        anthropic = AnthropicProvider(client=object())
        openai = OpenAIProvider(client=object())
        results = [
            {"call_id": "call-a", "output": {"value": 1}},
            {"call_id": "call-b", "output": "done", "is_error": True},
        ]

        anthropic_messages = anthropic.batch_tool_results(results)
        openai_messages = openai.batch_tool_results(results)

        self.assertEqual(len(anthropic_messages), 1)
        self.assertEqual(len(anthropic_messages[0]["content"]), 2)
        self.assertEqual(len(openai_messages), 2)
        self.assertTrue(
            all(item["type"] == "function_call_output" for item in openai_messages)
        )

    def test_pause_turn_resends_the_assistant_turn(self) -> None:
        api = ScriptedMessagesAPI(
            [
                SimpleNamespace(
                    stop_reason="pause_turn",
                    content=[{"type": "text", "text": "Still working"}],
                ),
                SimpleNamespace(stop_reason="end_turn", content=[]),
            ]
        )
        provider = AnthropicProvider(client=SimpleNamespace(messages=api))

        result = provider.complete(
            [{"role": "user", "content": "Route this"}],
            [],
            "claude-test",
            max_tokens=128,
        )

        self.assertEqual(result.pause_restarts, 1)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.retry_count, 0)
        second_messages = api.calls[1]["messages"]
        self.assertEqual(
            second_messages[-1],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Still working"}],
            },
        )
        self.assertNotIn("Continue", repr(second_messages))

    def test_pause_turn_is_capped_at_five_restarts(self) -> None:
        paused = [
            SimpleNamespace(
                stop_reason="pause_turn",
                content=[{"type": "text", "text": "Working"}],
            )
            for _ in range(6)
        ]
        api = ScriptedMessagesAPI(paused)
        provider = AnthropicProvider(client=SimpleNamespace(messages=api))

        with self.assertRaises(PauseTurnLimitError) as caught:
            provider.complete([], [], "claude-test", max_tokens=128)

        self.assertEqual(caught.exception.attempts, 6)
        self.assertEqual(len(api.calls), 6)

    def test_large_anthropic_request_streams(self) -> None:
        response = SimpleNamespace(stop_reason="end_turn", content=[])
        api = ScriptedMessagesAPI([response])
        provider = AnthropicProvider(client=SimpleNamespace(messages=api))

        result = provider.complete([], [], "claude-test", max_tokens=16_001)

        self.assertIs(result.response, response)
        self.assertEqual(api.methods, ["stream"])

    def test_large_openai_request_streams(self) -> None:
        response = SimpleNamespace(status="completed", output=[])
        api = ScriptedResponsesAPI(response)
        provider = OpenAIProvider(client=SimpleNamespace(responses=api))

        result = provider.complete([], [], "gpt-test", max_tokens=16_001)

        self.assertIs(result.response, response)
        self.assertEqual(api.create_calls, [])
        self.assertEqual(len(api.stream_calls), 1)


class PromptCacheWiringTest(unittest.TestCase):
    """A dropped cache breakpoint is silent, so the wiring is asserted rather than assumed.

    Both converters rebuild their output field by field. That is what makes them safe
    against unknown keys and what makes a dropped ``cache_control`` invisible: the request
    still succeeds and the only symptom is a cache-read counter that never leaves zero.
    """

    def test_system_blocks_preserve_the_cache_breakpoint(self) -> None:
        blocks = _system_blocks(
            [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]
        )
        self.assertEqual(
            blocks, [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]
        )

    def test_plain_system_string_carries_no_breakpoint(self) -> None:
        self.assertEqual(_system_blocks("rules"), [{"type": "text", "text": "rules"}])

    def test_tools_preserve_the_cache_breakpoint(self) -> None:
        converted = _anthropic_tools(
            [
                {
                    "name": "t",
                    "description": "d",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        )
        self.assertEqual(converted[0]["cache_control"], {"type": "ephemeral"})

    def test_the_request_carries_the_breakpoint_into_system(self) -> None:
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "user", "content": "facts"},
        ]
        request_messages, system = _anthropic_input(messages, None)
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        # Hoisted out of messages[], or it would be cached at the wrong prefix position.
        self.assertEqual([m["role"] for m in request_messages], ["user"])


class UsageAccountingTest(unittest.TestCase):
    def test_cache_fields_are_read(self) -> None:
        usage = _usage(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_input_tokens=700,
                    cache_creation_input_tokens=900,
                )
            )
        )
        self.assertEqual(
            (usage.input_tokens, usage.output_tokens, usage.cache_read, usage.cache_write),
            (10, 5, 700, 900),
        )

    def test_absent_cache_fields_report_zero_rather_than_failing(self) -> None:
        usage = _usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5)))
        self.assertEqual((usage.cache_read, usage.cache_write), (0, 0))


class ModelSelectionTest(unittest.TestCase):
    def test_selected_model_heads_the_chain_and_keeps_the_fallbacks(self) -> None:
        chain = fallback_chain_for(DECISION_MODEL_DEV)
        self.assertEqual(chain[0], DECISION_MODEL_DEV)
        self.assertEqual(set(chain), {DECISION_MODEL_DEV, *FALLBACK_CHAIN})

    def test_a_model_already_in_the_chain_is_reordered_not_duplicated(self) -> None:
        chain = fallback_chain_for(FALLBACK_CHAIN[-1])
        self.assertEqual(chain[0], FALLBACK_CHAIN[-1])
        self.assertEqual(len(chain), len(set(chain)))

    def test_an_unlisted_model_is_prepended(self) -> None:
        chain = fallback_chain_for("some-other-model")
        self.assertEqual(chain, ("some-other-model", *FALLBACK_CHAIN))


if __name__ == "__main__":
    unittest.main()
