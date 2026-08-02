"""Preview the repaired reason on real dossiers, with no model calls.

Dossier construction is deterministic, so the sentence ``reason_repair`` would author for
a given row can be shown without contacting a provider. The decision passed in is the one
currently in ``dataset/output.csv`` — on the ten affected rows that is the fallback the
old policy produced, not the model's discarded decision, so the *action* shown here is not
what a re-run would ship. What this checks is the sentence: that the repair finds a true,
contract-clean fact to name on every one of these rows.
"""

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from context.features import build_dossier  # noqa: E402
from context.index import build_feature_index  # noqa: E402
from data.loader import load_dataset  # noqa: E402
from guards import reason_repair  # noqa: E402
from guards.decision import ValidatedDecision  # noqa: E402
from guards.validate import reason_issues  # noqa: E402

AFFECTED = (
    "msg_003", "msg_019", "msg_020", "msg_025", "msg_030",
    "msg_041", "msg_064", "msg_072", "msg_085", "msg_093",
)


def main() -> None:
    dataset = load_dataset(Path(__file__).resolve().parents[2] / "dataset")
    index = build_feature_index(dataset)
    by_id = {m.message_id: m for m in dataset.messages}

    for message_id in AFFECTED:
        dossier = build_dossier(dataset, index, by_id[message_id])
        stand_in = ValidatedDecision(
            action="digest",
            message_type="unknown",
            reason="placeholder",
            confidence=0.55,
            evidence_message_ids=(),
            risk="clean",
            relevance="medium",
            urgency="none",
        )
        sentence = reason_repair.repair(dossier, stand_in)
        issues = reason_issues(sentence)
        flag = "OK " if not issues else "BAD"
        print(f"{flag} {message_id} ({len(sentence):3d} ch)  {sentence}")
        if issues:
            print(f"      issues: {[i.code for i in issues]}")


if __name__ == "__main__":
    main()
