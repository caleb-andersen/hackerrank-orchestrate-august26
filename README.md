# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon.

## Message Notification Router

Build an AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

The system must reason over multimodal messages, including text messages, image posters/screenshots, and voice notes.

WhatsApp is noisy. A user can receive family chats, society notices, school updates, co-worker messages, business account promotions, image posters, voice notes, and scams in the same message stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted or risky messages interrupt the user.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, and submission format.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
├── requirements.txt                  # Pinned runtime dependencies
├── .env.example                      # Names of the two required environment variables
├── code/                             # The solution — see code/README.md to run it
│   ├── README.md                     # Setup, run commands, architecture, limitations
│   ├── SPEC_features.md              # Feature-by-feature specification of the dossier
│   ├── main.py                       # Router entry point
│   ├── config.py                     # Every tunable constant, each read on a live path
│   ├── data/                         # CSV loading and frozen row schemas
│   ├── context/                      # Dossier construction: features, retrieval, media
│   ├── agent/                        # Prompt, tool schemas, provider client, tool loop
│   ├── guards/                       # Validation, injection scan, deterministic gate
│   ├── output/                       # Order-preserving CSV writer and checkpoints
│   ├── evaluation/                   # Scoring harness — the only reader of gold labels
│   └── tests/                        # Unit and property tests
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # Submission CSV, pre-keyed in a shuffled order
    ├── sample_messages.csv           # Solved examples (evaluation-only, see §9.8)
    ├── users.csv                     # User notification behavior
    ├── groups.csv                    # Group metadata
    ├── group_members.csv             # User-group relationships
    ├── business_accounts.csv         # Business sender metadata
    ├── user_business_history.csv     # User-business history
    ├── message_history.csv           # Historical messages
    ├── message_events.csv            # User reactions to historical messages
    ├── images.csv                    # Image IDs and media file paths
    ├── voice_notes.csv               # Voice note IDs and media file paths
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## What You Need to Build

For every row in `dataset/messages.csv`, produce one row in `output.csv` with:

| Column | Meaning |
|---|---|
| `message_id` | Incoming message ID |
| `action` | One of `notify`, `digest`, or `mute` |
| `message_type` | Best-fit message category |
| `reason` | Short human-readable explanation |
| `confidence` | Number from `0` to `1` |
| `evidence_message_ids` | Historical message IDs used as evidence; write `none` if there is no useful evidence |

Your system should make personalized decisions using the provided message, user, group, business, media, and historical interaction data.
For image and voice-note messages, `images.csv` and `voice_notes.csv` only provide file paths; your system should inspect the media files themselves.

---

## Suggested Workflow

1. Inspect `dataset/sample_messages.csv` to understand the expected output format.
2. Load `dataset/messages.csv` and all relevant context files.
3. Build your routing system using any approach: LLMs, retrieval, rules, classifiers, agents, or hybrids.
4. Write predictions to `output.csv`.
5. Evaluate your approach on the solved sample rows before submitting.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv`
- include one prediction for every `message_id` in `dataset/messages.csv`
- not use organizer-only files or hardcoded labels

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth labels.

The scoring will consider:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

Strong systems will combine retrieval, structured metadata, behavioral history, safety checks, OCR/ASR handling, and contextual reasoning.

Run the complete local evaluation workflow directly from the repository root:

```text
python code/evaluation/main.py
```

It routes the labelled samples without exposing their gold fields to the production
runner, prints field-level metrics and the composite tuning score, validates the full
`dataset/output.csv`, and audits near-duplicate consistency across all 110 predictions.
To rescore an existing sample run without re-running the router, pass
`--predictions dataset/output.samples.csv`; add `--no-judge` to skip the reason-quality
judge as well, which is the only remaining source of model calls on that path. The two
full-set checks can also be run independently:

```text
python code/evaluation/validate_output.py
python code/evaluation/consistency.py
```

Both default to `dataset/output.csv`. To check a file elsewhere, pass it with the
`--output` flag — **these scripts take named flags, not a positional path**, so
`validate_output.py output.csv` exits 2 with an argparse error rather than validating
anything:

```text
python code/evaluation/validate_output.py --output dataset/output.csv
```

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

Upload this log as your chat transcript at submission time. Do not paste secrets into the chat.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: full runnable solution, prompts/configs, README, and any evaluation files.
2. **Predictions CSV**: final `output.csv` for all rows in `dataset/messages.csv`.
3. **Chat transcript**: the `log.txt` described above.

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/messages.csv`.
- `output.csv` has the exact required columns in the exact required order.
- Your runnable code and setup instructions are included in `code.zip`.
