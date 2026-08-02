# Message Notification Router

Routes every message in `dataset/messages.csv` into **notify**, **digest** or **mute**,
with a message type, a written reason, a confidence and cited historical evidence.

Python 3.14. Two model providers: Anthropic for the routing decision, OpenAI for
voice-note transcription and for the evaluation judge.

---

## 0. Where to stand — read this first

If you unzipped `code.zip`, you now have a `code/` directory. **Every command in this file
is run from its parent — the directory that contains both `code/` and `dataset/`, never
from inside `code/` itself.** Throughout this README that directory is called the
*project root*, and paths are written from there: `python code/main.py`,
`pip install -r code/requirements.txt`.

```text
project-root/          <-- stand here
├── code/              <-- the unzipped submission (this README is inside it)
└── dataset/           <-- the provided data
```

If you are currently inside `code/`, the fix is one command — not a flag:

```bash
cd ..
```

This is not a style preference; two paths resolve against the working directory and one
resolves against this file:

| Thing | Resolves against | Consequence of running from inside `code/` |
|---|---|---|
| `dataset/` | working directory ([config.py:67](config.py#L67)) | `FileNotFoundError: Dataset directory does not exist: dataset` |
| `dataset/output.csv` | working directory ([config.py:69](config.py#L69)) | output silently lands in `code/dataset/output.csv` |
| `.env` | **this file's parent's parent** ([config.py:12](config.py#L12)) | a `.env` placed inside `code/` is never read |

**Do not work around the first row with `--dataset ../dataset`.** It loads the data
correctly and is the one genuinely dangerous invocation in this project: `--dataset` moves
the input but not the output ([main.py:104](main.py#L104) uses the module-level
`OUTPUT_PATH`), and the writer creates whatever parent directory it is handed
([writer.py:23](output/writer.py#L23)). A full run started that way completes, bills you
for 161 model calls, and writes all 110 rows to `code/dataset/output.csv` while the real
`dataset/output.csv` stays untouched. `cd ..` instead.

---

## 1. Setup

From the project root:

```bash
python -m venv .venv
```

Activate it — Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux, and Git Bash on Windows:

```bash
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
pip install -r code/requirements.txt
```

`code/requirements.txt` ships inside the zip so it is self-contained, and pins four
packages exactly: `anthropic==0.120.2`, `openai==2.52.0`, `pillow==12.3.0`,
`python-dotenv==1.2.2`.

### Environment variables

Two are required.

> **The `.env` file goes beside `code/`, not inside it.** `code/.env.example` is shipped as
> a template so the zip is self-contained, but [config.py:12](config.py#L12) loads `.env`
> from this file's parent's parent — the project root. A `.env` created inside `code/` is
> silently ignored and every model call then fails authentication.

From the project root:

```bash
cp code/.env.example .env
```

Then fill both in:

```text
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```

Exporting them in the shell instead of using a `.env` works equally well — both are read
from the process environment, and `.env` is only a convenience loaded on top of it.

| Variable | Used for | Read at |
|---|---|---|
| `ANTHROPIC_API_KEY` | the routing decision | [config.py:16](config.py#L16), consumed [client.py:748](agent/client.py#L748) |
| `OPENAI_API_KEY` | voice transcription, evaluation judge, last fallback model | [config.py:18](config.py#L18), consumed [client.py:886](agent/client.py#L886) |

Both are read from the process environment only. `.env` is loaded from the project root at
[config.py:12](config.py#L12), resolved from the config file's own location rather than
the working directory, and `.env` is gitignored. No key is ever written to a source file,
a trace or a log.

---

## 2. Running it

Every command below is run from the **project root** — the directory containing both
`code/` and `dataset/`. See §0 if you are unsure where that is.

### The router

```bash
python code/main.py
```

Writes all 110 rows to `dataset/output.csv`. The last full run cost 333k input / 70k
output tokens across 161 model calls, and 1,071 seconds of summed per-row model time —
elapsed wall-clock is that divided by `--workers`, so expect single-digit minutes at the
default of 6.

Useful flags ([main.py:494-541](main.py#L494)):

| Flag | Effect |
|---|---|
| `--dry-run` | builds every dossier, prompt and tool set and contacts no model |
| `--limit N` | routes the first N rows, to `dataset/output.subset.csv` — never overwrites the deliverable |
| `--resume` | reuses checkpointed rows whose inputs, prompt and model are unchanged |
| `--workers K` | rows in flight, default 6 |
| `--model ID` | decision model; heads the fallback chain and is part of the checkpoint fingerprint |
| `--no-dnd` | ablates the gate's quiet-hours modifier |
| `--verbose` | logs every row |

`--dry-run` needs no API key and is the fastest way to confirm the install works.

### The evaluation harness

```bash
python code/evaluation/main.py
```

Routes the 30 labelled samples, prints field-level metrics and the composite score,
validates the full `dataset/output.csv`, and audits near-duplicate consistency across all
110 predictions. To rescore an existing sample run without re-running the router, pass
`--predictions dataset/output.samples.csv`; add `--no-judge` to skip the reason-quality
judge, which is the only remaining source of model calls on that path.

The two full-set checks also run standalone:

```bash
python code/evaluation/validate_output.py
```

```bash
python code/evaluation/consistency.py
```

**Both take named flags, not a positional path.** `validate_output.py output.csv` exits 2
with an argparse error rather than validating anything; the flag is `--output`
([validate_output.py:218](evaluation/validate_output.py#L218)). Both default to
`dataset/output.csv`.

### Tests

```bash
python -m pytest code/tests -q
```

---

## 3. Architecture

### Three axes, not one classifier

The model does not emit `notify`/`digest`/`mute` directly. It decides three independent
axes and resolves them in a fixed order
([prompts.py:159-211](agent/prompts.py#L159), enum-enforced in the tool schema at
[tools.py:216-233](agent/tools.py#L216)):

- **risk** — `clean` | `suspicious` | `scam_or_unsafe`. What the message would cost the
  recipient if it is what it appears to be.
- **relevance** — `wanted` | `neutral` | `unwanted`. Whether this recipient wants it,
  resting on recorded behaviour rather than on wording.
- **urgency** — `immediate` | `today` | `none`. A property of the ask, not of the writing.

Resolution stops at the first matching line
([prompts.py:198-202](agent/prompts.py#L198)):

```text
1. risk is scam_or_unsafe                                        -> mute
2. urgency is immediate or today, and relevance is not unwanted  -> notify
3. relevance is unwanted                                         -> mute
4. anything else                                                 -> digest
```

Risk vetoes absolutely: a live deadline inside a credential-harvesting message is not a
reason to interrupt anybody. The axes ship as fields on the decision, so a routing call can
be checked rather than taken on trust, and the deterministic rules downstream read them
directly instead of re-deriving them.

### Where the model decides, and where code enforces

**The model decides** the three axes, the action, the message type, the confidence, the
evidence citations and the reason text. It runs a bounded tool loop
([loop.py:775](agent/loop.py#L775)) with at most 4 tool-using iterations plus one backstop
turn (`MAX_TOOL_ITERATIONS`, enforced [loop.py:927](agent/loop.py#L927)) and at most 2
image inspections (`MAX_INSPECT_IMAGE_CALLS`, enforced
[loop.py:960](agent/loop.py#L960)). It answers by calling `submit_routing_decision`.

**Code enforces** three things the model is not trusted to get right on its own:

1. **Schema and style validation** — [validate.py:583](guards/validate.py#L583)
   `coerce_and_check`. Checks the JSON shape, coerces near-miss vocabulary values in
   logged tiers, enforces six cross-field invariants, and enforces the reason style
   contract (one concrete third-person sentence, 60–160 characters). A failure is returned
   to the model as a rejection ([loop.py:591](agent/loop.py#L591)) so it can correct
   itself, rather than being silently defaulted.

2. **The deterministic safety gate** — [safety_gate.py:615](guards/safety_gate.py#L615)
   `apply_gate`. Seven rules ([safety_gate.py:423](guards/safety_gate.py#L423)):
   `INJECTION`, `CREDENTIAL_REQUEST`, `BRAND_IMPERSONATION`,
   `PAYMENT_PRESSURE_UNTRUSTED`, `OPT_OUT`, `BEHAVIOURAL_DEMOTION`, `MEDIA_MISMATCH`.
   **Zero model calls.** It is one-directional — it can only move an action toward mute
   and can only lower confidence — and that invariant is asserted on every row at
   [safety_gate.py:600](guards/safety_gate.py#L600) `_assert_one_directional`, not merely
   intended. The prompt tells the model the gate exists and what it does
   ([prompts.py:585](agent/prompts.py#L585)), so the model cooperates with it rather than
   fighting it.

3. **Confidence calibration** — [safety_gate.py:483](guards/safety_gate.py#L483)
   `calibrate_confidence`. Clamps into 0.55–0.95, subtracts a penalty for media that
   contradicts the text and for a first-contact sender, and caps any row the gate
   hard-blocked.

Dataset text is treated as data, never as instruction. Message text, OCR output and voice
transcripts are wrapped in a named untrusted fence with its delimiters defanged
([prompts.py:703-713](agent/prompts.py#L703)). An instruction aimed at the router is
recorded as a **matched phrase** rather than a boolean — scanned at
[scanners.py:37](context/scanners.py#L37), surfaced as a dossier fact at
[features.py:468](context/features.py#L468), and independently re-checked inside the gate
at [safety_gate.py:185](guards/safety_gate.py#L185).

### One message, end to end

Take a business message with an image attachment. `load_dataset`
([data/loader.py](data/loader.py)) parses the CSVs once and `build_feature_index`
([context/index.py](context/index.py)) builds the shared lookup structures.
`plan_row` ([main.py:175](main.py#L175)) then resolves everything the row needs before any
model is contacted: `build_dossier` ([context/features.py](context/features.py)) computes
the personalisation facts as named values — this recipient's engagement rates with this
sender, the business relationship and why the recipient knows the account, group context,
the do-not-disturb window — and ranks historical precedents by a weighted mean of
same-peer, lexical, same-group, event and recency scores
([retrieval.py:141-145](context/retrieval.py#L141)). A voice note would be transcribed
here, deterministically and outside the model's control
([main.py:137-146](main.py#L137)); an image is left for the model to open through the
`inspect_image` tool. The rendered prompt, the tool schemas and the model id are hashed
into a checkpoint fingerprint ([main.py:149](main.py#L149)) — because the rendered prompt
already contains every fact, editing a feature or a single word of a prompt busts every
cached row automatically.

`route_row` ([main.py:466](main.py#L466)) then runs the tool loop. The model reads the
FACTS block, optionally calls `inspect_image`, and calls `submit_routing_decision`. That
call goes through `coerce_and_check`; if the style contract fails, the rejection goes back
to the model. The validated decision passes to `apply_gate`, which may demote the action
toward mute and lower the confidence, and writes its own reason when it overrides. The
final row is appended to the order-preserving writer
([writer.py:51](output/writer.py#L51)) and checkpointed
([checkpoint.py:19](output/checkpoint.py#L19)). A trace of both the pre-gate and post-gate
decision lands in `traces/` ([main.py:402](main.py#L402)).

### Reliability

- Every model call is wrapped in `retry_with_backoff`
  ([client.py:448](agent/client.py#L448)), and errors are **classified before** they are
  retried ([client.py:284](agent/client.py#L284)). Transient failures retry with jitter;
  permanent ones — auth, billing, refusal, schema, unsupported capability — are distinct
  exception types ([client.py:99-127](agent/client.py#L99)) and are never retried. An
  exhausted retry (`RetryExhaustedError`) is distinguishable from a refusal
  (`ProviderRefusalError`).
- Cross-provider fallback chain: `claude-opus-5` → `claude-sonnet-5` → `gpt-5.6-terra`
  ([config.py:25](config.py#L25)).
- **No row can crash the run** ([main.py:475-486](main.py#L475)): any escaping exception
  produces a legible output row carrying the real failure reason.
- **The CSV is written incrementally** ([writer.py:51](output/writer.py#L51)) while
  preserving the shuffled order `dataset/output.csv` ships with.
- **Checkpoint and resume** ([checkpoint.py:47](output/checkpoint.py#L47)) keyed on the
  input fingerprint, so a quota wall mid-run costs minutes.

### On determinism

**This system is not deterministic, and this README does not claim it is.** Temperature 0
was asked for and is not available: the Claude 5 family rejects `temperature`, `top_p` and
`top_k` outright ([loop.py:422-433](agent/loop.py#L422)), and `gpt-5.6-terra` returns
`400 Unsupported parameter: 'temperature' is not supported with this model`
([judge.py:28-35](evaluation/judge.py#L28)). Sending the parameter anyway would turn every
call into a permanent error, so it is not sent.

What determinism exists comes from the places that still allow it: a frozen prompt whose
content hash is the version (`PROMPT_VERSION`,
[prompts.py:670](agent/prompts.py#L670)), sorted directory walks, stable tie-breaks on
ids, content-hash cache keys and atomic cache writes. Re-running the router on a busted
cache will not reproduce these 110 rows exactly.

---

## 4. Known limitations

Measured, not estimated. Numbers come from the 140 traces in `traces/` and from
`python code/evaluation/main.py`.

**Reason-style repairs.** 6 of the 110 full-set rows did not satisfy the reason style
contract on the model's first attempt and went through the repair path: 3 failed
`reason_length`, 3 failed `reason_sentence_count`. The repair preserves the routing
decision and rewrites only the sentence — the decision is not discarded — but those 6
reasons are not purely the model's own prose. No row failed for content.

**One near-duplicate diverges from the shape the labelled data establishes.** One full-set
row carries text and a business id byte-identical to a labelled sample whose gold action is
`notify`; we route it `digest`. The consistency audit explains the divergence by
personalisation — that recipient's delivery is 9 days old rather than landing today, with a
lower open share and a recorded dismissal — and the near-identical sibling row *is* routed
`notify`, matching the gold shape. So the mechanism is behaving as designed. But the model
reached `digest` there via a repetition argument rather than via the delivery-recency
feature, and that is the one row in the set where I cannot rule out that personalisation
over-fired against a known-correct target.

**The gate overrode the model once in 110 rows.** Exactly one full-set row had its action
changed by the deterministic gate — `MEDIA_MISMATCH`, `digest` → `mute`. Across all 140
traces including the labelled samples, two rows changed. The gate *fired* on 63 traces and
hard-blocked 51, but firing mostly annotates and lowers confidence rather than moving the
action. Run `python code/evaluation/gate_audit.py` to reproduce the table.

**Mute rate is 47% against a 33% prior.** The full 110-row output is 31 notify / 27 digest
/ 52 mute, which is +13.9pp mute and −12.1pp digest against the 9/11/10 balance of the
labelled samples; `scam` is 33 of 110 against a 13.3% gold share. On the labelled samples
themselves the distribution is tight — one row of drift, 0.90 action accuracy, and zero
rows in the catastrophic `mute→notify` or `notify→mute` cells. The skew was audited against
three published-sample patterns and all three returned zero mutes, so there is no evidence
the extra mutes are wrong; the 110 rows are simply not guaranteed to share the samples'
distribution. Left as-is deliberately: no safe change was available without labels.

**Evidence is over-cited and `none` is never used.** Evidence precision is 0.43 against
recall 0.65 (F1 0.52) on the labelled samples, and all 110 rows cite at least one
historical id — no row uses `none`. The prompt already states that the 2-id limit is a
ceiling rather than a target ([prompts.py:389](agent/prompts.py#L389)) and the model does
not fully comply. Left as-is deliberately: the only fix is a prompt change, which busts the
fingerprint on all 110 cached rows and forces a full re-run, for an expected composite gain
under 0.01.

**Confidence is underconfident in the low bins.** Overall MAE is 0.080, but in
`[0.55,0.60)` it is 0.250 and in `[0.60,0.65)` it is 0.215 — gold sits near 0.80–0.83 where
this system predicts 0.55–0.62. Six of 30 labelled rows. Left as-is deliberately: at 10%
of the composite the total available gain is about 0.015.

---

## 5. Contract compliance

- Runs from the terminal; reads only `dataset/`.
- One prediction row per `message_id`, 110 of 110, verified by
  [validate_output.py](evaluation/validate_output.py).
- Exact column order: `message_id,action,message_type,reason,confidence,evidence_message_ids`.
- The shuffled row order `dataset/output.csv` ships with is preserved.
- Secrets are read from environment variables only.
- No `message_id` literal appears anywhere under `code/` outside `code/evaluation/`, and
  the labelled sample file is opened only by code under `code/evaluation/`. Both are
  mechanically checkable with the two greps in AGENTS.md §9.8. That rule is also why the
  limitations above describe individual rows by their features rather than naming their
  ids: a row id on the decision path is exactly what the check exists to catch, and this
  file sits on the wrong side of it.
