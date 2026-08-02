"""Route every message in the dataset and write the graded CSV.

    python code/main.py                 # full run over dataset/messages.csv
    python code/main.py --dry-run       # build every prompt, call nothing
    python code/main.py --limit 5       # first five rows, to a subset file
    python code/main.py --resume        # skip rows already checkpointed
    python code/main.py --no-dnd        # ablate the gate's quiet-hours modifier

Rows are iterated in **file order** and addressed by ``message_id`` throughout. Nothing
here indexes a parsed row number: the CSV is the ordering authority, and a row number is
a property of one particular parse of one particular file.

Output order is a separate question with a separate answer. ``dataset/output.csv`` ships
pre-keyed in a shuffled order that the submission must preserve (§9.12), so the writer is
opened against that order and buffers completed rows until their turn arrives — which is
also what makes concurrency safe to add and safe to remove.

A partial run never overwrites the deliverable. ``--limit`` writes to
``output.subset.csv`` beside it, because a full-order file containing 5 of 110 rows is a
broken submission that looks like a finished one. Labelled samples are loaded and
routed exclusively by ``evaluation/main.py`` so gold labels are unreachable here.
"""

# code/ is deliberately not a Python package: "code" is a stdlib module name, so making
# it one would shadow Python's own. Imports are therefore rooted at code/, and this line
# puts code/ on the path regardless of the working directory the script was launched from.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Iterator, Sequence  # noqa: E402

from agent import loop as agent_loop  # noqa: E402
from agent.prompts import NO_MEDIA, PROMPT_VERSION, MediaPayload, build_messages  # noqa: E402
from agent.tools import build_tools  # noqa: E402
from config import (  # noqa: E402
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    CONCURRENCY_RAMP_START,
    DATASET_DIR,
    DECISION_MODEL_PRIMARY,
    MAX_CONCURRENCY,
    MODEL_PRICE_PER_MTOK,
    OUTPUT_PATH,
    TRACE_DIR,
)
from context import media as media_module  # noqa: E402
from context.features import Dossier, build_dossier  # noqa: E402
from context.index import FeatureIndex, build_feature_index  # noqa: E402
from data.loader import Dataset, load_dataset  # noqa: E402
from data.schema import Message  # noqa: E402
from guards.decision import FinalDecision  # noqa: E402
from guards.safety_gate import apply_gate  # noqa: E402
from output import checkpoint, writer  # noqa: E402


LOGGER = logging.getLogger("router")


@dataclass(frozen=True, slots=True)
class Selection:
    """The rows to route, the order to write them in, and where to write them."""

    messages: list[Message]
    order: list[str]
    destination: pathlib.Path
    is_full_run: bool


def select(dataset: Dataset, args: argparse.Namespace) -> Selection:
    """Choose which rows to route, in the order messages.csv lists them.

    A partial run never writes over the deliverable: ``dataset/output.csv`` is keyed to
    all 110 rows in a fixed order, and a file containing five of them in that order is a
    broken submission that looks like a finished one.
    """
    messages = list(dataset.messages)
    if args.limit is not None:
        messages = messages[: args.limit]
    chosen = {message.message_id for message in messages}
    order = [
        message_id for message_id in dataset.output_row_order if message_id in chosen
    ]
    missing = chosen - set(order)
    if missing:
        raise ValueError(
            f"Selected message ids are absent from output.csv: {sorted(missing)}"
        )
    full = len(messages) == len(dataset.messages)
    return Selection(
        messages=messages,
        order=order,
        destination=OUTPUT_PATH if full else OUTPUT_PATH.with_name("output.subset.csv"),
        is_full_run=full,
    )


def assert_full_coverage(dataset: Dataset) -> None:
    """§9.12: the submitted file must key exactly the rows messages.csv contains."""
    expected = {message.message_id for message in dataset.messages}
    template = set(dataset.output_row_order)
    if expected != template:
        raise ValueError(
            "output.csv and messages.csv disagree: "
            f"missing={sorted(expected - template)} extra={sorted(template - expected)}"
        )


# --------------------------------------------------------------------------------------
# Per-row inputs and their fingerprint
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowPlan:
    """Everything a row needs, resolved before any model is contacted."""

    message: Message
    dossier: Dossier
    media: MediaPayload
    messages: list[dict[str, object]]
    tools: list[dict[str, object]]
    fingerprint: str


def _media_payload(dossier: Dossier) -> MediaPayload:
    """Transcribe a voice note deterministically; images are the model's to open."""
    attachment = dossier.media
    if attachment.media_type != "voice" or attachment.media_id is None:
        return NO_MEDIA
    reference = media_module.describe(
        attachment.media_id, "voice", attachment.file_path
    )
    transcript = media_module.transcribe(reference)
    return MediaPayload(transcript=transcript.text, transcript_status=transcript.status)


def _fingerprint(
    messages: Sequence[dict[str, object]],
    tools: Sequence[dict[str, object]],
    model: str,
) -> str:
    """Key a completed row by its inputs, its prompt version and its model (§9.10.4).

    The rendered prompt *is* the input digest — it already contains every dossier fact,
    every evidence candidate and the message text — so editing a feature, a retrieval
    weight or a single word of a prompt busts every cached row without anyone having to
    remember to bump a version.
    """
    payload = json.dumps(
        {
            "messages": messages,
            "tools": tools,
            "model": model,
            "prompt_version": PROMPT_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plan_row(
    dataset: Dataset, index: FeatureIndex, message: Message, model: str
) -> RowPlan:
    dossier = build_dossier(dataset, index, message)
    media = _media_payload(dossier)
    # Rendered here for the fingerprint and again inside the loop for the request. The
    # renderer is pure and its inputs are identical, so the two are the same bytes —
    # which is what makes the fingerprint a description of what was actually sent.
    prompt = build_messages(dossier, dossier.evidence_candidates, media)
    tools = build_tools(dossier)
    return RowPlan(
        message=message,
        dossier=dossier,
        media=media,
        messages=prompt,
        tools=tools,
        fingerprint=_fingerprint(prompt, tools, model),
    )


# --------------------------------------------------------------------------------------
# Bounded, ramping concurrency
# --------------------------------------------------------------------------------------


class RampingGate:
    """Admission control that opens from ``start`` to ``cap`` as rows come back clean.

    A run that opens at full width sends a burst of identical-shaped requests into a
    rate limiter that has never seen this workload. Starting narrow and widening one slot
    per successful row costs a few seconds at the head of the run and keeps the 429 storm
    from happening at all. A failing row does not widen the gate — a run that is going
    badly should not accelerate into it.
    """

    def __init__(self, start: int, cap: int) -> None:
        self._cap = max(1, cap)
        self._permits = max(1, min(start, self._cap))
        self._in_flight = 0
        self._condition = threading.Condition()

    @property
    def width(self) -> int:
        with self._condition:
            return self._permits

    @contextlib.contextmanager
    def slot(self) -> Iterator["_Slot"]:
        with self._condition:
            while self._in_flight >= self._permits:
                self._condition.wait()
            self._in_flight += 1
        slot = _Slot()
        try:
            yield slot
        finally:
            with self._condition:
                self._in_flight -= 1
                if slot.widen and self._permits < self._cap:
                    self._permits += 1
                self._condition.notify_all()


@dataclass
class _Slot:
    widen: bool = False


# --------------------------------------------------------------------------------------
# Run accounting
# --------------------------------------------------------------------------------------


@dataclass
class RunReport:
    """What the run cost and how it behaved, accumulated under one lock."""

    rows: int = 0
    resumed: int = 0
    fallbacks: int = 0
    crashes: int = 0
    refusals: int = 0
    retries: int = 0
    coercions: int = 0
    tool_calls: int = 0
    inspect_calls: int = 0
    model_calls: int = 0
    actions: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)
    gate_rules: Counter[str] = field(default_factory=Counter)
    # model -> [input, output, cache_read, cache_write]
    tokens: dict[str, list[int]] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        raw: agent_loop.RawDecision,
        final: FinalDecision,
        fired: Sequence[str],
    ) -> None:
        metrics = raw.metrics
        with self.lock:
            self.rows += 1
            self.actions[final.action] += 1
            self.outcomes[raw.outcome] += 1
            self.retries += metrics.retries
            self.coercions += raw.decision.coercion_count
            self.tool_calls += metrics.tool_calls
            self.inspect_calls += metrics.inspect_calls
            self.model_calls += metrics.model_calls
            self.latencies.append(metrics.wall_seconds)
            if raw.is_fallback:
                self.fallbacks += 1
            if raw.outcome == "refusal":
                self.refusals += 1
            for rule in fired:
                self.gate_rules[rule] += 1
            bucket = self.tokens.setdefault(metrics.model or "unknown", [0, 0, 0, 0])
            bucket[0] += metrics.input_tokens
            bucket[1] += metrics.output_tokens
            bucket[2] += metrics.cache_read_tokens
            bucket[3] += metrics.cache_write_tokens

    def note_resumed(self) -> None:
        with self.lock:
            self.resumed += 1

    def note_crash(self) -> None:
        with self.lock:
            self.crashes += 1

    def cost(self) -> tuple[float, list[str]]:
        """Price what can be priced and name what cannot, rather than guessing a rate.

        Cached tokens are priced at their own rates rather than folded into the input
        count. On a cached request the provider reports ``input_tokens`` as the *uncached
        remainder*, so pricing that field alone would quietly under-report a well-cached
        run and make the cache look free instead of cheap.
        """
        total = 0.0
        unpriced: list[str] = []
        for model, (inputs, outputs, cache_read, cache_write) in sorted(self.tokens.items()):
            rates = MODEL_PRICE_PER_MTOK.get(model)
            if rates is None:
                if inputs or outputs or cache_read or cache_write:
                    unpriced.append(f"{model} ({inputs:,} in / {outputs:,} out)")
                continue
            billable_input = (
                inputs
                + cache_read * CACHE_READ_MULTIPLIER
                + cache_write * CACHE_WRITE_MULTIPLIER
            )
            total += (billable_input * rates[0] + outputs * rates[1]) / 1_000_000
        return total, unpriced

    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        rank = max(0, -(-95 * len(ordered) // 100) - 1)
        return ordered[rank]


def print_report(report: RunReport, elapsed: float, destination: object) -> None:
    cost, unpriced = report.cost()
    inputs = sum(bucket[0] for bucket in report.tokens.values())
    outputs = sum(bucket[1] for bucket in report.tokens.values())
    cache_read = sum(bucket[2] for bucket in report.tokens.values())
    cache_write = sum(bucket[3] for bucket in report.tokens.values())
    # The denominator is every prompt token the run sent, cached or not — the share is
    # meaningless against the uncached remainder alone.
    prompt_tokens = inputs + cache_read + cache_write
    hit_rate = cache_read / prompt_tokens if prompt_tokens else 0.0
    mean_tools = report.tool_calls / report.rows if report.rows else 0.0

    lines = [
        "",
        "=" * 72,
        f"rows written        {report.rows}   (resumed from checkpoint: {report.resumed})",
        f"crashes             {report.crashes}",
        f"fallback rows       {report.fallbacks}",
        f"refusals            {report.refusals}",
        f"retries             {report.retries}",
        f"coercions           {report.coercions}",
        f"model calls         {report.model_calls}",
        f"mean tool calls     {mean_tools:.2f}   (image inspections: {report.inspect_calls})",
        f"p95 row latency     {report.p95_latency():.1f}s   (wall clock: {elapsed:.1f}s)",
        f"tokens              {inputs:,} in (uncached) / {outputs:,} out",
        f"cache_read          {cache_read:,}   ({hit_rate:.1%} of {prompt_tokens:,} prompt tokens)",
        f"cache_write         {cache_write:,}",
        f"total cost          ${cost:.4f}",
    ]
    if prompt_tokens and cache_read == 0:
        # A zero here means the breakpoint is set and nothing ever read it. Said out loud
        # rather than left as a zero to be read past, because the failure is silent: the
        # request succeeds, the write premium is paid, and no hit is ever taken.
        lines.append(
            "  NOTE              cache_read is 0 — every request wrote a fresh prefix and "
            "none re-read one"
        )
    if unpriced:
        lines.append(f"  unpriced models   {', '.join(unpriced)}")
    lines.append(
        "actions             "
        + ", ".join(f"{action}={count}" for action, count in sorted(report.actions.items()))
    )
    lines.append(
        "outcomes            "
        + ", ".join(f"{name}={count}" for name, count in sorted(report.outcomes.items()))
    )
    if report.gate_rules:
        lines.append("gate rules fired")
        for rule, count in sorted(report.gate_rules.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {rule:<28} {count}")
    else:
        lines.append("gate rules fired    none")
    lines.append(f"output              {destination}")
    lines.append("=" * 72)
    print("\n".join(lines))


# --------------------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------------------


def write_trace(
    plan: RowPlan,
    raw: agent_loop.RawDecision,
    final: FinalDecision,
    fired: Sequence[str],
) -> None:
    """One JSON file per row: why this row came out the way it did."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = raw.metrics
    payload = {
        "message_id": plan.message.message_id,
        "user_id": plan.message.user_id,
        "conversation_type": plan.message.conversation_type,
        "media_type": plan.dossier.media.media_type,
        "transcript_status": plan.media.transcript_status,
        "prompt_version": PROMPT_VERSION,
        "fingerprint": plan.fingerprint,
        "outcome": raw.outcome,
        "failure_reason": raw.failure_reason,
        "last_model_text": raw.last_text,
        "evidence_candidates": [
            candidate.history_message_id
            for candidate in plan.dossier.evidence_candidates
        ],
        "metrics": {
            "model": metrics.model,
            "models_tried": list(metrics.models_tried),
            "model_calls": metrics.model_calls,
            "iterations": metrics.iterations,
            "tool_calls": metrics.tool_calls,
            "inspect_calls": metrics.inspect_calls,
            "retries": metrics.retries,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "wall_seconds": metrics.wall_seconds,
            "validation_failures": list(metrics.validation_failures),
        },
        "gate_rules_fired": list(fired),
        "gate_trace": {key: _jsonable(value) for key, value in final.trace.items()},
        "row": final.csv_row(),
    }
    path = TRACE_DIR / f"{plan.message.message_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------------------
# Routing one row
# --------------------------------------------------------------------------------------


def route_row(plan: RowPlan, args: argparse.Namespace, report: RunReport) -> dict[str, str]:
    """Decide one row and return its CSV row. Never raises."""
    try:
        client = agent_loop.RowClient(agent_loop.fallback_chain_for(args.model))
        raw = agent_loop.run(plan.dossier, plan.dossier.evidence_candidates, plan.media, client)
        final, fired = apply_gate(raw.decision, plan.dossier, dnd_modifier=not args.no_dnd)
        report.record(raw, final, fired)
        write_trace(plan, raw, final, fired)
        return final.csv_row()
    except Exception as error:  # noqa: BLE001 — no row may crash the run (§9.10.2)
        LOGGER.exception("row_failed message_id=%s", plan.message.message_id)
        report.note_crash()
        fallback = agent_loop.fallback_decision(plan.dossier, type(error).__name__)
        return FinalDecision(
            message_id=plan.message.message_id,
            action=fallback.action,
            message_type=fallback.message_type,
            reason=fallback.reason,
            confidence=fallback.confidence,
            evidence_message_ids=fallback.evidence_message_ids,
        ).csv_row()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Route WhatsApp messages into notify / digest / mute."
    )
    parser.add_argument(
        "--limit", type=int, metavar="N", help="route only the first N rows of messages.csv"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse checkpointed rows whose inputs, prompt and model are unchanged",
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="K",
        default=MAX_CONCURRENCY,
        help=f"maximum rows in flight (default {MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every dossier, prompt and tool set, but contact no model",
    )
    parser.add_argument(
        "--no-dnd",
        action="store_true",
        help="disable the gate's do-not-disturb interruption-cost modifier (ablation)",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        default=DECISION_MODEL_PRIMARY,
        help=(
            "decision model for this run; heads the fallback chain and is part of the "
            f"checkpoint fingerprint (default {DECISION_MODEL_PRIMARY})"
        ),
    )
    parser.add_argument(
        "--dataset", type=pathlib.Path, default=DATASET_DIR, help="dataset directory"
    )
    parser.add_argument("--verbose", action="store_true", help="log every row")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def _report_dry_run(plans: Sequence[RowPlan], destination: object) -> int:
    media = Counter(plan.dossier.media.media_type or "text" for plan in plans)
    inspectable = sum(1 for plan in plans if len(plan.tools) > 1)
    characters = sum(
        len(str(message.get("content", ""))) for plan in plans for message in plan.messages
    )
    no_candidates = sum(1 for plan in plans if not plan.dossier.evidence_candidates)
    # Tools render ahead of the system prompt in the cache prefix, so rows can only share
    # a cached span when their tool schemas are byte-identical. This counts how many
    # distinct prefixes the run will actually produce: one means the cache is shared
    # across every row, len(plans) means it is shared across none of them.
    distinct_tools = len(
        {
            json.dumps(plan.tools, ensure_ascii=False, sort_keys=True, default=str)
            for plan in plans
        }
    )
    attachments = ", ".join(f"{kind}={count}" for kind, count in sorted(media.items()))
    print(
        "\n".join(
            [
                "",
                "=" * 72,
                f"dry run             {len(plans)} rows planned, no model contacted",
                f"attachments         {attachments}",
                f"inspectable images  {inspectable}",
                f"no evidence cands   {no_candidates}",
                f"rendered prompt     {characters:,} characters across all rows",
                f"prompt version      {PROMPT_VERSION}",
                f"distinct tool sets  {distinct_tools} across {len(plans)} rows",
                f"would write         {destination}",
                "=" * 72,
            ]
        )
    )
    return 0


def run_selection(
    dataset: Dataset, selection: Selection, args: argparse.Namespace
) -> int:
    """Run an already-selected set of unlabelled messages through the router."""
    index = build_feature_index(dataset)
    if not selection.messages:
        print("No messages selected.")
        return 1
    if selection.is_full_run:
        assert_full_coverage(dataset)

    # The fingerprint is keyed on the model that will actually answer, so a run under a
    # different --model cannot resume rows produced by the previous one.
    plans = [plan_row(dataset, index, message, args.model) for message in selection.messages]
    if args.dry_run:
        return _report_dry_run(plans, selection.destination)

    completed = checkpoint.load_completed() if args.resume else {}
    report = RunReport()
    gate = RampingGate(CONCURRENCY_RAMP_START, args.workers)
    started = time.perf_counter()

    def handle(plan: RowPlan) -> None:
        cached = completed.get(plan.message.message_id)
        if cached is not None and cached[0] == plan.fingerprint:
            writer.append_row(cached[1])
            report.note_resumed()
            return
        with gate.slot() as slot:
            row = route_row(plan, args, report)
            slot.widen = True
        writer.append_row(row)
        checkpoint.save(plan.message.message_id, plan.fingerprint, row)
        if args.verbose:
            LOGGER.info("row_done message_id=%s action=%s", plan.message.message_id, row["action"])

    with writer.open_writer(selection.destination, selection.order):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Consumed eagerly so a worker exception surfaces here rather than being
            # swallowed when the pool shuts down.
            for _ in pool.map(handle, plans):
                pass

    print_report(report, time.perf_counter() - started, selection.destination)
    return 1 if report.crashes else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    dataset = load_dataset(args.dataset)
    selection = select(dataset, args)
    return run_selection(dataset, selection, args)


if __name__ == "__main__":
    raise SystemExit(main())
