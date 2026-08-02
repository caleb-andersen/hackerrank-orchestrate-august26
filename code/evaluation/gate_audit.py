import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

"""Report what the deterministic safety gate actually did, per rule, from the run traces.

The gate is the one place where deterministic code is allowed to overrule the model
(Decision 3), so "it is wired in" is not the interesting claim — "it fired on N rows and
changed the action on M of them, and here is which rule did it" is. Everything here is read
back out of ``traces/`` after a run, so it costs nothing and cannot influence a decision.

A rule that fires often but never changes an action is doing nothing except adding a sentence
to a reason; a rule that never fires at all on the full set is unexercised. Both are worth
seeing separately, so the per-rule table reports fired and changed as distinct columns.
"""

import argparse  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from typing import Sequence  # noqa: E402

from config import TRACE_DIR  # noqa: E402
from data.schema import ACTIONS  # noqa: E402


def load_traces(trace_dir: pathlib.Path) -> list[dict[str, object]]:
    """Read every trace, sorted by filename so the report is stable across runs."""
    traces: list[dict[str, object]] = []
    for path in sorted(trace_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                traces.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as error:
            print(f"skipped unreadable trace {path.name}: {error}")
    return traces


def _gate(trace: dict[str, object]) -> dict[str, object]:
    value = trace.get("gate_trace")
    return value if isinstance(value, dict) else {}


def _rules(trace: dict[str, object]) -> tuple[str, ...]:
    fired = trace.get("gate_rules_fired") or _gate(trace).get("_gate_rules_fired") or ()
    if isinstance(fired, str):
        return (fired,)
    return tuple(str(rule) for rule in fired) if isinstance(fired, (list, tuple)) else ()


def report(traces: Sequence[dict[str, object]], prompt_version_filter: str | None = None) -> int:
    if prompt_version_filter:
        traces = [t for t in traces if t.get("prompt_version") == prompt_version_filter]

    fired_rows = 0
    changed_rows = 0
    hard_blocked = 0
    per_rule_fired: Counter[str] = Counter()
    per_rule_changed: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    pre_actions: Counter[str] = Counter()
    post_actions: Counter[str] = Counter()
    changed_detail: list[tuple[str, str, str, str]] = []

    for trace in traces:
        gate = _gate(trace)
        pre = str(gate.get("_pre_gate_action") or "")
        post = str(gate.get("_post_gate_action") or "")
        if pre:
            pre_actions[pre] += 1
        if post:
            post_actions[post] += 1
        if gate.get("_gate_hard_blocked"):
            hard_blocked += 1

        rules = _rules(trace)
        if rules:
            fired_rows += 1
            for rule in rules:
                per_rule_fired[rule] += 1

        changed = bool(gate.get("_gate_action_changed")) or (bool(pre) and bool(post) and pre != post)
        if changed:
            changed_rows += 1
            transitions[(pre, post)] += 1
            changed_detail.append((str(trace.get("message_id", "")), pre, post, ", ".join(rules) or "-"))
            for rule in rules:
                per_rule_changed[rule] += 1

    lines = [
        "",
        "=" * 100,
        f"SAFETY GATE AUDIT ({len(traces)} traces)",
        f"rows where at least one rule fired      {fired_rows}",
        f"rows where the gate CHANGED the action  {changed_rows}",
        f"rows hard-blocked by the gate           {hard_blocked}",
        "",
        f"{'rule':<28} {'fired':>7} {'changed action':>15}",
    ]
    for rule in sorted(set(per_rule_fired) | set(per_rule_changed)):
        lines.append(f"{rule:<28} {per_rule_fired[rule]:>7} {per_rule_changed[rule]:>15}")
    if not per_rule_fired:
        lines.append("(no rule fired on any row)")

    lines.extend(["", "ACTION DISTRIBUTION", f"{'action':<12} {'pre-gate':>9} {'post-gate':>10} {'delta':>7}"])
    for action in ACTIONS:
        delta = post_actions[action] - pre_actions[action]
        lines.append(f"{action:<12} {pre_actions[action]:>9} {post_actions[action]:>10} {delta:>+7}")

    if transitions:
        lines.extend(["", "ACTION TRANSITIONS (only rows the gate changed)"])
        for (pre, post), count in sorted(transitions.items(), key=lambda item: -item[1]):
            lines.append(f"  {pre} -> {post}: {count}")

    if changed_detail:
        lines.extend(["", "CHANGED ROWS"])
        for message_id, pre, post, rules in sorted(changed_detail):
            lines.append(f"  {message_id:<14} {pre:>7} -> {post:<7} {rules}")

    lines.append("=" * 100)
    print("\n".join(lines))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=pathlib.Path, default=TRACE_DIR)
    parser.add_argument(
        "--prompt-version",
        help="only audit traces carrying this prompt_version, so a stale trace cannot be counted",
    )
    args = parser.parse_args(argv)
    traces = load_traces(args.traces)
    if not traces:
        print(f"No traces found in {args.traces}")
        return 1
    return report(traces, args.prompt_version)


if __name__ == "__main__":
    raise SystemExit(main())
