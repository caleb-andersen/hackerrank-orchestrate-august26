# SPEC — Feature / Dossier Layer (`code/context/`)

Implementation spec for **Decision 2: personalisation is computed in code, not inferred by the model.**

This layer turns one `Message` plus the loaded `Dataset` into one `Dossier`: a flat, typed bundle of named
facts. The agent layer renders the dossier into the prompt. The model reads facts; it never reads a raw CSV
and never derives a rate itself.

**Status:** specification only. No implementation in this file.

---

## 0. Non-negotiable invariants

| # | Invariant | Rationale |
|---|---|---|
| I1 | **Zero model calls.** No network, no API client import, no transcription, no OCR anywhere in this module. | Decision 2. A rate the model computed is a rate the model can hallucinate. |
| I2 | **Pure functions.** Every function is a deterministic map from arguments to return value. No *mutable* module-level state, no mutation of `Dataset`, no clock reads, no `random`, no filesystem writes. See I2a for what "no module-level state" does and does not forbid. | §6.3 determinism; makes every function unit-testable without fixtures. |
| I2a | **Immutable module-level bindings are permitted and expected.** The ban is on state that can change after import, not on names defined at module level. **Permitted:** the §4 constants; `EVENT_RELEVANCE` as a `Mapping` proxy or frozen tuple of pairs; scanner patterns compiled once at import into a module-level tuple (§6.5.2); precomputed frozensets. **Forbidden:** any accumulating cache (`_cache: dict = {}`), memo table, counter, lazily-populated singleton, mutable default argument, any name rebound after import, and any pattern compiled from a value that varies at runtime. | The property that matters is referential transparency: a binding that is never rebound and never mutated cannot make a function's result depend on call order or call history, so it cannot break determinism. Import-time compilation is in fact *required* here — recompiling per call would be both slower and a second place for behaviour to diverge. |
| I3 | **Every rate is `float \| None`. `None` when the denominator is zero — never `0.0`.** | A user who has never received a message from this sender is not a user who dismissed nothing. Collapsing the two is the single most damaging error this layer can make. |
| I4 | **Every count that is a denominator is carried alongside its rate.** | A 1.0 open rate over 1 row and over 23 rows are different claims. The model must be able to tell them apart. |
| I5 | **All frozen dataclasses**, `slots=True`, tuples not lists for collections. | Hashable, safe to cache, safe to fingerprint for §9.10.4 checkpointing. |
| I6 | **No `message_id` literal anywhere in this module or this spec** (§9.8). Degenerate cases are described by structure and count, never by row id. | The §9.8.5 grep must stay clean. |
| I7 | **Dossier text fields are data, never instruction** (§9.7). Every field sourced from `message_text` is rendered inside the untrusted fence by the caller. This module never builds a prompt. | Injection containment. |
| I8 | **Field budget.** Every `Dossier` field is either rendered into the prompt or read by deterministic code downstream. Any field that is neither is deleted before packaging (§9.9). | Unused fields are indefensible in the interview. |

---

## 1. Module layout and public signatures

```
code/context/
├── __init__.py
├── text.py          # normalise_text / trigrams / jaccard
├── scanners.py      # scan_injection / scan_credential_request / scan_payment_pressure
├── timewindow.py    # parse_dnd_window / dnd_state
├── index.py         # FeatureIndex + build_feature_index
├── retrieval.py     # EvidenceScore / EvidenceCandidate / NearDuplicate / Repetition
│                    # + score_evidence / select_evidence / build_repetition
└── features.py      # Dossier + the section dataclasses + build_dossier
```

**Import direction is strictly one-way and acyclic**, in the order listed:
`text` → `scanners`, `timewindow`, `index` → `retrieval` → `features`.
`retrieval` never imports `features`; `features` is the only module at the top. This is why `FeatureIndex`
lives in its own `index.py` rather than in `features.py` — both `retrieval` and `features` need it, and
defining it in `features` would force `retrieval` to import downward and create a cycle.

```python
# index.py
def build_feature_index(dataset: Dataset) -> FeatureIndex: ...

# retrieval.py
def score_evidence(
    subject_peer_id: str | None,
    subject_group_id: str | None,
    subject_trigrams: frozenset[str],
    subject_created_at: datetime,
    row: HistoryMessage,
    row_peer_id: str | None,
    row_trigrams: frozenset[str],
    event: MessageEvent | None,
) -> EvidenceScore: ...

def select_evidence(
    message: Message,
    index: FeatureIndex,
    dataset: Dataset,
    k: int = EVIDENCE_TOP_K,
) -> tuple[EvidenceCandidate, ...]: ...

def build_repetition(
    message: Message,
    index: FeatureIndex,
    dataset: Dataset,
    k: int = NEAR_DUPLICATE_TOP_K,
) -> Repetition: ...

# features.py
def build_dossier(dataset: Dataset, index: FeatureIndex, message: Message) -> Dossier: ...
```

**`retrieval.py` owns scoring and selection; `features.py` calls it.** Steps 5 and 6 of §8 are
`build_repetition(...)` and `select_evidence(...)` respectively; `features.py` assigns the results into the
`Dossier` and does no ranking of its own. The split is not cosmetic:

- Ranking over history is one cohesive concern, and it is the largest algorithmic block in the layer.
- It owns nine of the sixteen §4 constants, including all five evidence weights. Decision 9 puts the eval
  harness in charge of tuning exactly those, and a harness that ablates one weight should not need to open
  the module that assembles the dossier.
- `score_evidence` takes **scalars, not a `Dossier`**, so a weight can be unit-tested against a
  hand-built row with no dataset, no index, and no fixtures.

`score_evidence` returns `EvidenceScore` — the five per-term values plus the total — rather than a bare
`float`, so `select_evidence` can populate `EvidenceCandidate`'s per-term fields (§6.7) without recomputing
them, and so an ablation reads a term directly instead of inferring it from the total.

```python
@dataclass(frozen=True, slots=True)
class EvidenceScore:
    total: float
    same_peer: float
    same_group: float
    lexical: float
    event_relevance: float
    recency: float
```

`build_feature_index` is called **once per run**; `build_dossier` is called once per message. The index is a
pure derived value (frozen dataclass) holding the joins and the 412 precomputed history trigram sets. It
exists so `build_dossier` stays O(history for this user) instead of O(all history), and so trigram sets are
computed 412 times per run rather than 110 × 412 times.

`FeatureIndex` fields:

| Field | Type | Contents |
|---|---|---|
| `history_by_user_peer` | `dict[tuple[str, str], tuple[HistoryMessage, ...]]` | key `(user_id, peer_id)`, sorted by `(created_at, message_id)` |
| `history_by_peer` | `dict[str, tuple[HistoryMessage, ...]]` | key `peer_id` across all users, same sort |
| `trigrams_by_history_id` | `dict[str, frozenset[str]]` | normalised character trigrams per history row |
| `daily_totals_by_user` | `dict[str, tuple[int, int, int]]` | `(notifications_sent, notifications_dismissed, n_days)` summed from `daily_notification_summary.csv` |

`peer_id` resolution (§2) is applied identically when building the index and when querying it.

---

## 2. Peer resolution

One rule, used by `peer_engagement`, `peer_global`, `near_duplicate_history`, and evidence scoring.

```
peer_kind, peer_id =
    ("business", message.business_id)      if conversation_type == "business"
    ("user",     message.sender_user_id)   otherwise   # personal and group
```

Grounding: all 30 business rows in `messages.csv` have an empty `sender_user_id`, and all 155 business rows
in `message_history.csv` do too. `business_id` is the only sender identity a business message carries, so
keying on `sender_user_id` alone would give every business message a null peer.

The same rule is applied to each history row using **that row's own** `conversation_type`, so a
personal-conversation history row and a business history row are never conflated under one peer key.

If the resolved `peer_id` is empty or `None`, `peer_id` is `None`, all peer-scoped rate blocks report
`n = 0` with every rate `None`, and `evidence_state` is `"none"` (§9.1). No dataset row currently hits this.

---

## 3. Rate convention

```python
Rate = float | None
```

Every rate in this layer is produced by exactly one helper:

```python
def _rate(numerator: int, denominator: int) -> Rate:
    return None if denominator == 0 else numerator / denominator
```

There is no other division on the decision path. Rates are **not** rounded at computation time; rounding to
3 decimals happens only in the renderer, so the stored value stays exact for the eval harness.

---

## 4. Constants

All constants live in `code/config.py`, per the §9.9.5 convention that every constant there be traceable to
a reader. Each is imported by the module named in the **Read by** column; nothing here is decorative (§9.9).

| Constant | Value | Read by | Justification |
|---|---|---|---|
| `RECENCY_HALF_LIFE_DAYS` | `30.0` | `retrieval._recency_term` | History spans 2026-03-01 → 2026-07-16 (~138 days). A 30-day half-life leaves the oldest rows at ~4% weight — present but not competitive. |
| `EVIDENCE_TOP_K` | `6` | `retrieval.select_evidence` | Requested. Enough precedent for the model to compare, few enough to keep the prompt bounded. |
| `EVIDENCE_MIN_SCORE` | `0.12` | `retrieval.select_evidence` | Just above the score of a row that matches on nothing but is ~1 half-life old (`0.10 × 0.5 = 0.05`) and below a row matching same-peer alone (`0.35`). Lets `evidence_message_ids` legitimately emit `none` per §6.2. |
| `W_SAME_PEER` | `0.35` | `retrieval.score_evidence` | §7.3 |
| `W_LEXICAL` | `0.25` | `retrieval.score_evidence` | §7.3 |
| `W_SAME_GROUP` | `0.15` | `retrieval.score_evidence` | §7.3 |
| `W_EVENT` | `0.15` | `retrieval.score_evidence` | §7.3 |
| `W_RECENCY` | `0.10` | `retrieval.score_evidence` | §7.3 |
| `EVENT_RELEVANCE` | mapping, §7.3 | `retrieval._event_relevance_term` | §7.3 |
| `NEAR_DUPLICATE_TOP_K` | `3` | `retrieval.build_repetition` | Repetition needs the strongest few, not a ranked list. |
| `NEAR_DUPLICATE_MIN_JACCARD` | `0.45` | `retrieval.build_repetition` | Character-trigram Jaccard on independent promotional text sits well under this; reworded resends of the same template sit above it. Calibrated by the harness (Decision 9), not by inspecting a row. |
| `BURST_WINDOW_HOURS` | `24` | `retrieval._sender_burst` | One day is the natural unit for "this sender is flooding me". |
| `TRIGRAM_N` | `3` | `text.trigrams` | §7.2 |
| `BRAND_MIN_AGE_DAYS` | `365` | `features.brand_integrity` | §6.2. **Pre-existing name in `config.py`, reused.** Condition 3 is `account_age_days < BRAND_MIN_AGE_DAYS` — "an account younger than this is not trusted on its own". |
| `BRAND_MIN_DOMAIN_AGE_DAYS` | `180` | `features.brand_integrity` | §6.2. **New constant** — `config.py` reserves no domain-age threshold, and condition 4 needs one distinct from the account-age threshold. |
| `BRAND_MAX_REPORTS` | `29` | `features.brand_integrity` | §6.2. **Pre-existing name in `config.py`, reused.** Condition 5 is `user_reports_30d > BRAND_MAX_REPORTS` — "more reports than this is adverse". `29` is the midpoint of the measured `20 → 38` gap; this is the only one of the three brand thresholds that actually separates the cohorts, so it is the one that had to be placed carefully. |

**Reconciliation with the existing `config.py`.** Three of these names already exist there as bare
annotations with `TODO: Set after calibration` comments. A bare annotation at module level creates **no
binding**, so reading one today raises `NameError` — they must be given values, not just left annotated.
This spec adopts the existing names rather than introducing parallel ones:

| `config.py` today | This spec | Resolution |
|---|---|---|
| `BRAND_MIN_AGE_DAYS: int` | condition 3 | Reused as-is. The name is better than the `BRAND_ACCOUNT_AGE_MAX_DAYS` this spec first proposed — it reads from the trust side, matching how the conjunction is written. |
| `BRAND_MAX_REPORTS: int` | condition 5 | Reused as-is; the name already reads in the `>` direction the condition needs. Value `29` is derived in §6.2 from the widest gap in the candidate set, not carried over from any earlier draft. |
| — | `BRAND_MIN_DOMAIN_AGE_DAYS` | Added. Nothing in `config.py` covers domain age. |
| `MIN_PEER_HISTORY: int` | — | **Not read by this layer.** `peer_engagement` reports `n` unthresholded and lets `evidence_state` (§6.3) express sufficiency, so no minimum is applied here. If nothing downstream reads it either, delete it under §9.9.1. |
| `DISMISS_MUTE_THRESHOLD: float` | — | **Not read by this layer.** It belongs to the routing rules, not to feature computation. |
| `MAX_EVIDENCE_IDS: int = 2` | — | Not read here, but must not be confused with `EVIDENCE_TOP_K = 6`. This layer **offers** up to 6 ranked candidates; the model **cites** at most 2 in `evidence_message_ids`. Offering more than can be cited is deliberate: the ranking guarantees the best precedent is present, and the model chooses which 2 actually justify its decision. |

---

## 5. `Dossier` — top level

```python
@dataclass(frozen=True, slots=True)
class Dossier:
    message_id: str
    user_id: str
    conversation_type: Literal["personal", "group", "business"]
    created_at: datetime
    sender_identity: SenderIdentity
    relationship: Relationship
    content_signals: ContentSignals
    repetition: Repetition
    evidence_candidates: tuple[EvidenceCandidate, ...]   # length 0..6
    media: Media
    timing: TimingContext
```

> **Note on section count.** Six section names were specified as "five sections"; all six are built.
> `timing` is a **seventh** top-level block rather than a member of `relationship`, because `in_dnd` is a
> property of the recipient and the arrival time, not of the sender or of the user↔sender relationship.
> Say the word and it folds into `relationship` — nothing else in the spec changes.

---

## 6. Section specifications

### 6.1 `sender_identity`

```python
@dataclass(frozen=True, slots=True)
class SenderIdentity:
    peer_kind: Literal["user", "business"]
    peer_id: str | None
    display_name: str | None
    brand_name: str | None
    category: str | None
    brand_integrity: BrandIntegrity | None
```

| Field | Type | Source | Derivation |
|---|---|---|---|
| `peer_kind` | `Literal["user","business"]` | `messages.conversation_type` | §2 |
| `peer_id` | `str \| None` | `messages.business_id` or `messages.sender_user_id` | §2 |
| `display_name` | `str \| None` | `business_accounts.display_name` | `None` unless business. The dataset carries no personal-user names. |
| `brand_name` | `str \| None` | `business_accounts.brand_name` | `None` unless business |
| `category` | `str \| None` | `business_accounts.category` | `None` unless business |
| `brand_integrity` | `BrandIntegrity \| None` | `business_accounts` | `None` unless `conversation_type == "business"` |

`display_name` and `brand_name` are operator-controlled dataset strings, not sender-controlled — but they
are still rendered inside the untrusted fence (I7), because a display name is exactly where an
impersonation attempt would place text.

### 6.2 `brand_integrity`

```python
@dataclass(frozen=True, slots=True)
class BrandIntegrity:
    verified: bool
    official_domain: str | None
    domain_used_by_sender: str | None
    domain_mismatch: bool | None          # None == not computable
    account_age_days: int
    domain_used_by_sender_age_days: int
    user_reports_30d: int
    verdict: Literal["clean", "suspect", "impersonation"]
    verdict_basis: tuple[str, ...]        # named conditions that fired, in fixed order
```

All eight input fields are copied verbatim from the `business_accounts.csv` row for `message.business_id`.

**`domain_mismatch`** — three-valued, never a silent `False`:

```
normalise(d) = d.strip().casefold() with a single leading "www." removed

domain_mismatch = None   if official_domain is None or domain_used_by_sender is None
                = False  if normalise(official) == normalise(used)
                       or normalise(used).endswith("." + normalise(official))   # subdomain of official
                = True   otherwise
```

The subdomain clause is defensive: no current row exercises it (0 of 23 mismatches are subdomains of their
official domain), but treating `mail.example.com` as an impersonation of `example.com` would be a
false positive the moment the data changed.

**`verdict` — `impersonation` is a five-way conjunction (Decision 4).** All five must hold:

| # | Condition | `verdict_basis` token |
|---|---|---|
| 1 | `domain_mismatch is True` | `domain_mismatch` |
| 2 | `verified is False` | `unverified` |
| 3 | `account_age_days < BRAND_MIN_AGE_DAYS` | `account_new` |
| 4 | `domain_used_by_sender_age_days < BRAND_MIN_DOMAIN_AGE_DAYS` | `sender_domain_new` |
| 5 | `user_reports_30d > BRAND_MAX_REPORTS` | `reported_by_users` |

**Why a conjunction and not a disjunction.** Across the 110 business accounts, **23 have a domain
mismatch**. The conjunction fires on **18** of them and spares **5**. Those 5 are the precision traps, and
they come in two distinct kinds:

| Kind | n | Profile | Spared by |
|---|---|---|---|
| Legitimate brand on a partner/shortener domain | 2 | verified, account ~4300 days old, sender domain ~3400 days old, 4 and 7 reports | all four of conditions 2–5 |
| Young unverified account that is *not* being reported | 3 | unverified, account 34–35 days, sender domain 9–14 days, **10, 13 and 20 reports** | condition 5 only |

A disjunctive rule keyed on mismatch alone mutes all 5. The first kind is obvious. **The second kind is the
one that makes the conjunction necessary**: those 3 accounts match 4 of the 5 conditions and are separated
from the 18 impersonators by nothing except report volume. Any rule that drops condition 5, or that treats
"mismatch + unverified + new" as sufficient, mutes 3 legitimate businesses.

**Which conditions actually discriminate, stated honestly.** Among the 23 mismatched accounts the work is
done by conditions 2 and 5:

| Axis | Impersonators (18) | Mismatched but spared (5) | Separating? |
|---|---|---|---|
| `verified` | all unverified | `1, 1, 0, 0, 0` | partly — excludes the 2 legitimate brands |
| `account_age_days` | `20 … 33` | `34, 34, 35, 4304, 4415` | **no** — all 21 unverified candidates are under any sane threshold |
| `domain_used_by_sender_age_days` | `2 … 19` | `9, 13, 14, 3368, 3455` | **no** — the spared values overlap the impersonator range |
| `user_reports_30d` | `38 … 77` | `4, 7, 10, 13, 20` | **yes** — a clean gap, `20 → 38` |

Conditions 3 and 4 are **not discriminative on this dataset** and are retained deliberately as
defence-in-depth, not because they are earning their keep here. Claiming otherwise would not survive an
interview: the honest statement is that `verified` plus `user_reports_30d` carry the partition today, and
the age conditions exist so that a future account which is unverified and heavily reported but *aged* is
not automatically condemned.

**Why the reports threshold is not overfit.** Restricting to the 21 accounts where conditions 1–4 already
hold, the sorted `user_reports_30d` values are:

```
10, 13, 20,   38, 41, 44, 47, 50, 53, 55, 56, 58, 59, 61, 62, 64, 65, 68, 71, 74, 77
          ^^^^^^^ widest gap in the series: 20 → 38, width 18
```

`BRAND_MAX_REPORTS = 29` is the **midpoint of that gap**, and the condition is `>`. Any value from 21 to 37
produces the identical partition, so no single row can move the outcome — which is what §9.8.4 requires.
Verified: ±25% on each of the three thresholds independently (`22`/`36` reports, `274`/`456` account days,
`135`/`225` domain days) yields the same 18 accounts every time.

> **Correction against an earlier draft of this spec.** Earlier revisions stated 21 impersonations, 2 spared
> accounts, and a reports gap of `10 → 38`, with `BRAND_MAX_REPORTS` at the bottom edge of that range. Those
> figures came from a threshold that was never the specified one, and the `10 → 38` gap in particular was
> not a measured minimum. The correct figures are 18 / 5 / `20 → 38`. Any test written to the old numbers
> will fail, correctly.

**The verdict is a three-branch decision, evaluated strictly in this order. `clean` is the residual, and it
is reached only by falling through both tests above it — it is never a default.**

```
if all five conditions of the conjunction hold:      verdict = "impersonation"
elif  domain_mismatch is True                        # a mismatch that is not impersonation
   or (domain_mismatch is None and not verified)     # unverifiable domain on an unverified account
   or user_reports_30d > BRAND_MAX_REPORTS:          # report pressure regardless of domain
                                                     verdict = "suspect"
else:                                                verdict = "clean"
```

**`clean` does not mean "not impersonation".** It means *no adverse signal was found at all*: the sender's
domain matches its official domain (or is a subdomain of it), and there is no report pressure. A domain
mismatch that fails the conjunction is **`suspect`, never `clean`** — that is the first disjunct of branch
two, and it is the branch the 2 verified mismatched accounts land in. Reading `clean` as "not
impersonation" would collapse the middle branch and hand those 2 accounts a clean bill of health, which is
precisely the outcome the three-valued verdict exists to prevent. `impersonation` and `suspect` are not
opposites of one another; they are two different strengths of adverse finding, and `clean` is the absence
of both.

**`suspect` is a disjunction, and that asymmetry against the conjunctive `impersonation` is deliberate.**
`impersonation` feeds the deterministic safety gate (Decision 3) and can mute a message on its own, so it
must be conjunctive or it produces false mutes. `suspect` only enters the prompt as context and cannot mute
anything, so a looser rule costs nothing but a sentence of context. The rule is: **the branch that can act
alone is narrow; the branch that only informs is wide.**

`verdict_basis` carries the fired tokens in the fixed order of the table above, so the reason string can
name *why* rather than assert a verdict — and so the ordering is deterministic.

### 6.3 `relationship`

```python
@dataclass(frozen=True, slots=True)
class Relationship:
    peer_engagement: EngagementRates          # scope="user_peer"
    peer_global: EngagementRates              # scope="global_peer", is_fallback=True
    evidence_state: Literal["peer", "global_fallback", "none"]
    user_baseline: UserBaseline
    group_context: GroupContext | None
    business_relationship: BusinessRelationship | None
```

#### `EngagementRates`

```python
@dataclass(frozen=True, slots=True)
class EngagementRates:
    scope: Literal["user_peer", "global_peer"]
    is_fallback: bool
    basis_note: str
    n: int
    open_rate: Rate
    reply_rate: Rate
    dismiss_rate: Rate
    mute_rate: Rate
    report_rate: Rate
    n_reacted: int
    median_reaction_minutes: float | None
```

**Row set.** For `peer_engagement`: every `message_history` row where `user_id == message.user_id`, the
row's resolved peer (§2) equals `message`'s peer, and `row.created_at < message.created_at`; inner-joined to
`message_events` on `(user_id, message_id)`. For `peer_global`: identical, except the `user_id ==
message.user_id` restriction is dropped — the join key stays `(row.user_id, row.message_id)`, so each row is
matched to *its own* recipient's event.

`n` = number of rows surviving the join. It is the denominator for all five rates. In the current dataset
all 412 history rows have exactly one matching event, so the join never drops a row — but it is specified as
an inner join so that a future missing event reduces `n` rather than inflating a rate.

The temporal guard `row.created_at < message.created_at` currently excludes nothing (history ends
2026-07-16, the test window opens 2026-07-18) and is specified anyway: a rate that can see the future is a
leak, and it should be impossible by construction rather than by luck.

| Field | Type | Source columns | Formula |
|---|---|---|---|
| `n` | `int` | join result | count of rows |
| `open_rate` | `Rate` | `message_events.message_opened` | `_rate(sum(message_opened), n)` |
| `reply_rate` | `Rate` | `message_events.message_replied` | `_rate(sum(message_replied), n)` |
| `dismiss_rate` | `Rate` | `message_events.notification_dismissed` | `_rate(sum(notification_dismissed), n)` |
| `mute_rate` | `Rate` | `message_events.muted_after_message` | `_rate(sum(muted_after_message), n)` |
| `report_rate` | `Rate` | `message_events.message_reported` | `_rate(sum(message_reported), n)` |
| `n_reacted` | `int` | `message_events.reaction_time_minutes` | count where non-null |
| `median_reaction_minutes` | `float \| None` | `message_events.reaction_time_minutes` | median of non-null values; `None` when `n_reacted == 0` |

`median_reaction_minutes` **has its own denominator**, and that is why `n_reacted` is a separate field: 134
of 412 events carry a null `reaction_time_minutes`. Those 134 are exactly the unopened messages (0 of the
134 have `message_opened == 1`), so dividing by `n` would understate reaction speed by silently treating
"never opened" as "reacted at 0 minutes". Median, not mean — reaction times are long-tailed.

**`basis_note`** is a plain-English sentence rendered verbatim into the prompt so the strength of the
evidence cannot be missed. Exactly one of:

- `peer_engagement`, `n > 0` — `"Computed over {n} earlier messages this user received from this sender."`
- `peer_engagement`, `n == 0` — `"This user has never received a message from this sender. No per-user rates exist; all rates below are null."`
- `peer_global`, `n > 0` — `"FALLBACK — weaker evidence: this sender's behaviour across all {n} messages it sent to any user, not to this user."`
- `peer_global`, `n == 0` — `"FALLBACK unavailable: this sender does not appear anywhere in message history."`

`is_fallback` is `True` for `peer_global` unconditionally — it is structurally weaker evidence whether or not
it is being leaned on. `evidence_state` says whether it is load-bearing.

**`evidence_state`**

```
"peer"            if peer_engagement.n > 0
"global_fallback" if peer_engagement.n == 0 and peer_global.n > 0
"none"            if both are 0
```

#### `UserBaseline`

```python
@dataclass(frozen=True, slots=True)
class UserBaseline:
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int
    notifications_sent_30d: int
    notifications_dismissed_total: int
    n_summary_days: int
    baseline_dismiss_rate: Rate
    mean_daily_notifications: float | None
```

| Field | Source | Formula |
|---|---|---|
| first four | `users.csv` for `message.user_id` | copied verbatim |
| `notifications_sent_30d` | `daily_notification_summary.notifications_sent` | sum over that user's rows |
| `notifications_dismissed_total` | `daily_notification_summary.notifications_dismissed` | sum over that user's rows |
| `n_summary_days` | `daily_notification_summary` | row count for that user |
| `baseline_dismiss_rate` | both summary columns | `_rate(notifications_dismissed_total, notifications_sent_30d)` |
| `mean_daily_notifications` | both | `None` if `n_summary_days == 0` else `notifications_sent_30d / n_summary_days` |

`daily_notification_summary` is used for `baseline_dismiss_rate` rather than the `users.csv` counters
because it is the only source carrying a true **denominator of notifications sent**. Dividing the `users.csv`
dismissed count by its opened count would be a ratio of two numerators. This rate is what lets the router
distinguish "dismisses everything" from "dismisses this sender".

#### `GroupContext` — `None` unless `conversation_type == "group"`

```python
@dataclass(frozen=True, slots=True)
class GroupContext:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    group_messages_30d: int
    user_role: str
    group_muted_by_user: bool
    user_messages_sent_30d: int
    user_messages_read_30d: int
    user_replies_sent_30d: int
    user_notifications_dismissed_30d: int
    group_read_rate: Rate
    group_reply_rate: Rate
    group_dismiss_rate: Rate
```

| Field | Source | Formula |
|---|---|---|
| `group_id`…`group_messages_30d` | `groups.csv` on `group_id` | copied (`messages_30d` → `group_messages_30d`) |
| `user_role`, `group_muted_by_user`, the four `user_*` counters | `group_members.csv` on `(group_id, user_id)` | copied |
| `group_read_rate` | `group_members.messages_read_30d` / `groups.messages_30d` | `_rate(user_messages_read_30d, group_messages_30d)` — share of this group's traffic the user actually reads |
| `group_reply_rate` | `group_members.replies_sent_30d` / `group_members.messages_read_30d` | `_rate(user_replies_sent_30d, user_messages_read_30d)` — how often reading turns into replying |
| `group_dismiss_rate` | `group_members.notifications_dismissed_30d` / `groups.messages_30d` | `_rate(user_notifications_dismissed_30d, group_messages_30d)` |

All 63 group messages have a matching `group_members` row, so `GroupContext` is never `None` for a group
message. If a future row breaks that, the field is `None` and the group rates are absent rather than zero —
it does not raise.

`group_read_rate` may exceed 1.0 if the member counter and the group counter disagree. It is **not clamped**;
a value above 1.0 is real information about inconsistent inputs and clamping would hide it.

#### `BusinessRelationship` — `None` unless business, **and `None` when no history row exists**

```python
@dataclass(frozen=True, slots=True)
class BusinessRelationship:
    why_user_knows_account: str
    last_activity_at: datetime | None
    days_since_last_activity: int | None
    allows_promotions: bool
    promotions_opted_out_at: datetime | None
    opted_out: bool
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: datetime | None
    open_share: Rate
```

All inputs come from `user_business_history.csv` on `(user_id, business_id)`.

| Derived field | Formula |
|---|---|
| `days_since_last_activity` | `None` if `last_activity_at is None` else `(message.created_at - last_activity_at).days` |
| `opted_out` | `promotions_opted_out_at is not None` |
| `open_share` | `_rate(messages_opened_30d, messages_opened_30d + messages_dismissed_30d)` |

`open_share` is named a **share, not a rate**, and the renderer labels it as such: `user_business_history`
carries no "messages received" column, so its denominator is a reconstructed proxy (opened + dismissed) and
is not comparable to `peer_engagement.open_rate`, whose denominator is real receipts. Giving them the same
name would invite the model to compare two different quantities.

**Degenerate case not in the original list:** **11 business message rows (9 distinct `(user, business)`
pairs) have no `user_business_history` row at all.** `business_relationship` is `None` for those, the
renderer emits `"No prior relationship on record between this user and this business."`, and the router
falls back to `brand_integrity` plus `peer_global`. Flagging because it is nearly twice the size of the
zero-peer-history case and was not in the brief.

### 6.4 `content_signals`

```python
@dataclass(frozen=True, slots=True)
class ContentSignals:
    raw_text: str
    normalised_text: str
    text_length: int
    is_empty_text: bool
    text_scanned: bool
    forwarded_count: int
    is_forwarded: bool
    url_domains: tuple[str, ...]
    injection_match: str | None
    credential_request: str | None
    payment_pressure: str | None
```

| Field | Type | Source | Derivation |
|---|---|---|---|
| `raw_text` | `str` | `messages.message_text` | verbatim; rendered only inside the untrusted fence |
| `normalised_text` | `str` | `raw_text` | §7.1 |
| `text_length` | `int` | `raw_text` | `len(raw_text.strip())` |
| `is_empty_text` | `bool` | `raw_text` | `text_length == 0` |
| `text_scanned` | `bool` | — | `len(normalised_text) > 0` |
| `forwarded_count` | `int` | `messages.forwarded_count` | copied |
| `is_forwarded` | `bool` | `messages.forwarded_count` | `forwarded_count > 0` |
| `url_domains` | `tuple[str,...]` | `raw_text` | registrable hosts extracted by regex, normalised as in §6.2, de-duplicated, **sorted** for determinism |
| `injection_match` | `str \| None` | `normalised_text` | §6.5 |
| `credential_request` | `str \| None` | `normalised_text` | §6.5 |
| `payment_pressure` | `str \| None` | `normalised_text` | §6.5 |

**`text_scanned` exists because `None` is overloaded on the three scanner fields.** For a voice note with no
text, all three scanners return `None` — and `None` there means *"nothing was scanned"*, not *"scanned and
clean"*. Reporting "no injection detected" about text that never existed would be a false assurance on
exactly the 8 rows where the content is hidden behind a media file. The renderer reads `text_scanned` and
emits `"message carries no text; text-based scanners did not run"` instead of a clean bill of health.

### 6.5 The three scanners

```python
def scan_injection(text: str) -> str | None: ...
def scan_credential_request(text: str) -> str | None: ...
def scan_payment_pressure(text: str) -> str | None: ...
```

**Each returns the matched phrase, or `None`. Never a boolean** — the matched phrase is what makes the
detection explainable in the `reason` column, and a boolean cannot be defended in an interview. Contract:

1. Input is `normalised_text` (§7.1). Empty input returns `None` without evaluating any pattern.
2. Each scanner owns an **ordered** tuple of `(pattern, label)` compiled once at module import. The
   **first** pattern to match wins; ordering is most-specific-first so the returned phrase is the most
   informative one. First-match-wins makes the result a deterministic function of the input.
3. The return value is **the substring of `normalised_text` that matched** — bounded to 80 characters and
   suffixed `…` if longer, so a long match cannot blow out the prompt or the CSV cell.
4. The phrase is dataset-derived and therefore untrusted: it is rendered inside the fence and written into
   `reason` only via the renderer's escaping path. It never reaches a system instruction (§9.7.4).
5. A match is **an observed fact about the message, not an instruction to act on** (§9.7.2). Detecting an
   injection attempt does not change control flow in this layer at all — it records a field and returns.

Scope of each scanner:

| Scanner | Detects | Feeds |
|---|---|---|
| `scan_injection` | text addressed at the router rather than the recipient: asserting a routing action, claiming system/operator voice, asserting sender metadata it cannot assert, instructing that prior rules be ignored | Decision 7; a flag in the prompt |
| `scan_credential_request` | solicitation of OTP, PIN, password, CVV, KYC re-verification, account re-activation | Decision 6 — the deterministic safety gate, which overrides sender trust |
| `scan_payment_pressure` | demand for payment coupled with urgency, penalty, or expiry | Decision 3 safety gate input |

These three functions are **public and re-usable**. When the media layer produces a voice transcript or
image text via a model call, it re-runs the same three scanners on that string. Keeping them here — pure,
model-free, independently testable — is what lets the same detection logic cover all three modalities
without duplicating patterns.

### 6.6 `repetition`

```python
@dataclass(frozen=True, slots=True)
class NearDuplicate:
    history_message_id: str
    jaccard: float
    created_at: datetime
    days_ago: float
    peer_id: str | None
    same_peer: bool
    opened: bool
    replied: bool
    dismissed: bool
    muted_after: bool
    reported: bool

@dataclass(frozen=True, slots=True)
class Repetition:
    near_duplicate_history: tuple[NearDuplicate, ...]   # length 0..3
    max_jaccard: float | None
    duplicate_count_at_threshold: int
    sender_burst_24h: int
```

**`near_duplicate_history`** — trigram Jaccard (§7.2) between `normalised_text` and the normalised text of
**this user's** history rows (all peers, not just this one — the same promotional template arriving from a
new sender is precisely the signal worth catching), restricted to `created_at < message.created_at`.

Rows with `jaccard >= NEAR_DUPLICATE_MIN_JACCARD` are sorted by `(-jaccard, -created_at, message_id)` and
the top `NEAR_DUPLICATE_TOP_K` are kept.

**The event outcome is attached to each match**, inner-joined from `message_events` on
`(history.user_id, history.message_id)`. A near-duplicate the user muted and a near-duplicate the user
replied to point in opposite directions; the similarity score alone is not actionable. If a match somehow
has no event row, the five boolean fields are all `False` and the row is retained — the similarity is still
real evidence.

| Field | Formula |
|---|---|
| `max_jaccard` | max similarity over all comparable rows, **before** the threshold filter; `None` when no comparison was possible (§9.2) |
| `duplicate_count_at_threshold` | count of all rows at or above `NEAR_DUPLICATE_MIN_JACCARD`, **not truncated to K** — so "3 shown" can honestly read "3 of 11" |
| `sender_burst_24h` | count of this user's history rows from the **same peer** with `created_at` in `[message.created_at - BURST_WINDOW_HOURS, message.created_at)` |

### 6.7 `evidence_candidates`

```python
@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    history_message_id: str
    score: float
    same_peer: bool
    same_group: bool
    jaccard: float
    event_relevance: float
    recency: float
    created_at: datetime
    days_ago: float
    conversation_type: str
    text_excerpt: str        # normalised text, first 160 chars
    opened: bool
    replied: bool
    dismissed: bool
    muted_after: bool
    reported: bool
```

Top `EVIDENCE_TOP_K = 6` rows from this user's history. Scoring in §7.3. The five per-term values are stored
alongside the total so the renderer can show *why* a row was selected, and so the eval harness can ablate a
single term without re-deriving the score.

These are **candidates**. The model selects the subset it actually relies on and emits those ids in
`evidence_message_ids`; the layer's job is to guarantee that if a useful precedent exists it is in the
candidate set. When the set is empty, the output contract's `none` is the correct value (§6.2).

### 6.8 `media`

```python
@dataclass(frozen=True, slots=True)
class Media:
    media_type: Literal["image", "voice"] | None
    media_id: str | None
    file_path: Path | None
    file_exists: bool
    file_size_bytes: int | None
    requires_transcription: bool
```

| Field | Source | Derivation |
|---|---|---|
| `media_type` | `messages.media_type` | `None` for the 87 text-only rows |
| `media_id` | `messages.media_id` | `None` when absent. `media_type` and `media_id` are always both present or both absent (verified across all 110 rows) |
| `file_path` | `images.csv` / `voice_notes.csv` on `media_id` | resolved by the loader; already constrained to stay inside `dataset/` |
| `file_exists` | filesystem | `file_path.is_file()` |
| `file_size_bytes` | filesystem | `None` when `file_exists` is `False` |
| `requires_transcription` | — | `media_type is not None` |

**This section deliberately carries no transcript, no OCR text, and no image description.** Producing any of
those requires a model call, and I1 forbids model calls in this layer. `requires_transcription` is the
explicit hand-off: it tells the agent layer that text signals for this row are unavailable until it spends a
model call, which for the 8 voice notes is the difference between routing on content and routing blind.

`file_exists` and `file_size_bytes` are the only filesystem reads in the module. They are `stat` calls, not
content reads, and are the one tolerated exception to "no I/O" in I2 — a missing media file must degrade to a
legible output row rather than an exception (§9.10.2). If this pair ends up read by nothing downstream, both
fields are deleted under I8.

### 6.9 `timing`

```python
@dataclass(frozen=True, slots=True)
class TimingContext:
    created_at: datetime
    local_time: time
    dnd_window_raw: str | None
    dnd_start: time | None
    dnd_end: time | None
    dnd_wraps_midnight: bool
    in_dnd: bool
    minutes_until_dnd_ends: int | None
```

| Field | Source | Derivation |
|---|---|---|
| `dnd_window_raw` | `users.do_not_disturb_window` | verbatim, e.g. `"22:00-07:00"` |
| `dnd_start`, `dnd_end` | parsed | §7.4 |
| `dnd_wraps_midnight` | derived | `dnd_start > dnd_end` |
| `in_dnd` | derived | §7.4 |
| `minutes_until_dnd_ends` | derived | §7.4; `None` iff `in_dnd is False` |

**Timezone assumption, stated explicitly:** the dataset carries no timezone on `created_at` and none on the
DND window. Both are treated as **naive local wall-clock time for the recipient**, and no conversion is
performed. This is the only assumption available, and it is recorded here so it is a documented choice
rather than an oversight.

---

## 7. Algorithms

### 7.1 `normalise_text`

```
1. NFKC-normalise
2. casefold
3. replace every run of characters that is not a Unicode letter or digit with a single space
4. strip, and collapse remaining runs of spaces to one
```

Deliberately lossy and identical for all consumers (scanners, trigrams, excerpts). Punctuation, emoji, and
casing are the cheapest things for a spam template to vary between resends, so stripping them is what makes
near-duplicate detection survive a reword. Currency amounts and digits survive, because those carry meaning.

### 7.2 Trigram Jaccard

```python
def trigrams(text: str) -> frozenset[str]:
    # text is already normalised; no padding
    return frozenset(text[i:i + TRIGRAM_N] for i in range(len(text) - TRIGRAM_N + 1))

def jaccard(a: frozenset[str], b: frozenset[str]) -> float | None:
    if not a or not b:
        return None
    return len(a & b) / len(a | b)
```

**Character** trigrams, not word trigrams: the texts are short (median 157 chars for messages, 102 for
history), and character trigrams degrade gracefully under the small edits — a changed amount, an inserted
name — that distinguish two sends of one template. Word trigrams on a 15-word message give ~13 features and
are brittle.

**`jaccard` returns `None`, not `0.0`, when either side has no trigrams** (normalised length < 3). This is
I3 applied to similarity: 8 message rows and 4 history rows normalise to the empty string, and reporting
"0% similar to everything" for a voice note is a fabricated measurement.

Where a downstream **weighted sum** needs a number, the `None` is converted to `0.0` at exactly one place —
`_lexical_term` in §7.3 — and that conversion is deliberate and local. `max_jaccard` and
`near_duplicate_history` keep the `None` / empty-tuple form.

### 7.3 Evidence scoring

Candidate set: this user's history rows with `created_at < message.created_at`.

```
score = W_SAME_PEER  * same_peer          # 0.35
      + W_LEXICAL    * lexical            # 0.25
      + W_SAME_GROUP * same_group         # 0.15
      + W_EVENT      * event_relevance    # 0.15
      + W_RECENCY    * recency            # 0.10
```

Every term is normalised to `[0, 1]` and the weights sum to `1.0`, so `score` is a weighted mean and is
directly comparable across rows and interpretable in the renderer.

| Term | Definition |
|---|---|
| `same_peer` | `1.0` if the history row's resolved peer (§2) equals the message's peer, else `0.0` |
| `same_group` | `1.0` if both are group messages with the same `group_id`, else `0.0` |
| `lexical` | `jaccard(...)`, with `None → 0.0` |
| `event_relevance` | `max` of `EVENT_RELEVANCE[flag]` over the event flags set on that row; `0.0` if none set |
| `recency` | `0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)`, in `(0, 1]` |

`EVENT_RELEVANCE = {reported: 1.0, muted_after: 0.9, replied: 0.7, dismissed: 0.6, opened: 0.3}`

**Why each weight:**

- **`same_peer` 0.35 — highest.** The routing question is peer-specific: what this user does with *this*
  sender is the most transferable fact available, and it is the term the `reason` column can cite most
  directly.
- **`lexical` 0.25 — second, deliberately below `same_peer`.** A near-duplicate with a known outcome is
  powerful precedent, but Decision 1 records that identical text received different gold labels for
  different users. Text must therefore never outrank sender identity, or the router regresses toward a
  content classifier and loses the personalisation this whole layer exists to provide.
- **`same_group` 0.15.** Real but weaker: shared social context and a shared mute setting. It is additive
  with `same_peer`, so a prior message from the same person in the same group scores `0.50` before any
  other term — correctly, since that is the ideal precedent.
- **`event_relevance` 0.15.** A row the user reported or muted teaches far more than one they merely
  opened, and a row with no event flags teaches nothing. Weighted at the same level as `same_group` because
  outcome strength qualifies a precedent rather than establishing it.
- **`recency` 0.10 — lowest, but non-zero.** It prefers current behaviour and breaks ties. It is lowest
  because a 90-day-old report is still a report; decaying it hard would discard the rarest and most
  informative outcomes first, since history spans ~138 days.

**Selection.** Rows scoring below `EVIDENCE_MIN_SCORE` are dropped. Survivors are sorted by
`(-score, -created_at, message_id)` and the top `EVIDENCE_TOP_K` kept. The three-part key is fully
deterministic: `message_id` is unique, so no tie can reach the end unresolved (§6.3, §9.10.5).

### 7.4 DND interval — half-open `[start, end)`, wrapping across midnight

```python
def parse_dnd_window(raw: str | None) -> tuple[time, time] | None:
    # "HH:MM-HH:MM" -> (start, end); None on absent or unparseable input
```

```python
def dnd_state(created_at: datetime, window: tuple[time, time] | None) -> tuple[bool, int | None]:
    if window is None:
        return (False, None)
    start, end = window
    t = created_at.time()
    if start == end:
        in_dnd = False                              # zero-length window, see below
    elif start < end:
        in_dnd = start <= t < end                   # same-day window
    else:
        in_dnd = t >= start or t < end              # wraps midnight
    if not in_dnd:
        return (False, None)
    minutes = (_minutes(end) - _minutes(t)) % 1440
    return (True, minutes)
```

**Why the wrap branch is mandatory, not defensive.** Of the 54 users, **49 have a window whose start is
later in the day than its end** (`22:00-07:00`, `21:30-06:30`, `23:30-07:30`, …); only 5 do not
(`00:00-06:30`, `00:00-07:00`). A naive `start <= t <= end` returns `False` for every overnight window at
every hour it is actually active, and `True` for the entire daytime — so it is not merely imprecise, it is
**inverted for 91% of users**. This is the single highest-impact correctness detail in the layer.

**Why half-open.** `[start, end)` makes the window length exactly `(end - start) mod 1440` minutes with no
off-by-one, and makes the boundary unambiguous: a message arriving exactly at the end of DND is **not** in
DND. It also yields a clean invariant — see below.

**`start == end`** is defined as a **zero-length window (never in DND)**, not a 24-hour one. No user
currently has such a window; the choice is recorded so the behaviour is specified rather than accidental,
and the non-punitive reading is the safe one, since treating it as all-day DND would suppress a user's
entire feed on a data-entry error.

**`minutes_until_dnd_ends`** is minutes from arrival to the end of the current DND window, wrapping forward
over midnight. It is `None` **iff** `in_dnd` is `False` — there is no window to wait out, and `0` would
falsely read as "DND ends right now".

*Invariant: when `in_dnd` is `True`, `minutes` is in `[1, 1439]` and can never be `0`.* Proof: `minutes == 0`
requires `t == end`. In the same-day branch `start <= t < end` excludes `t == end`. In the wrap branch
`start > end`, so `t == end` implies `t < start` (making `t >= start` false) and `t < end` false — hence
`in_dnd` is `False`. `t == end` is therefore unreachable while `in_dnd` is `True`. Since `created_at` has
minute granularity, `minutes` is always a whole number of minutes.

**Why the downstream rule needs it.** `in_dnd` alone cannot tell an actionable message from a suppressible
one. A payment due in 2 hours that arrives at 23:00 into a window ending at 07:00 has its deadline *inside*
the window — digesting it means the user misses it, so it escalates. The same message with a 3-day deadline
has its deadline *after* the window, and digesting costs nothing. Decision 10 keeps DND a **bounded,
demote-only modifier**; `minutes_until_dnd_ends` is what bounds it.

---

## 8. `build_dossier` — order of operations

```
1. resolve peer (§2)
2. sender_identity  — business lookup, brand_integrity (business only)
3. relationship     — peer_engagement, peer_global, evidence_state,
                      user_baseline, group_context, business_relationship
4. content_signals  — normalise, url extraction, three scanners
5. repetition       — retrieval.build_repetition(...): near-duplicates over this user's past history
6. evidence_candidates — retrieval.select_evidence(...): score, filter, sort, truncate
7. media            — media refs and stat
8. timing           — parse window, dnd_state
9. assemble frozen Dossier
```

No step depends on a later step. Steps 2–8 are independently testable with a stub `Dataset`.

**`build_dossier` never raises for a data-shaped reason.** Missing joins produce `None` sections; missing
media produces `file_exists=False`; unparseable DND produces `(False, None)`. Only a violated *internal*
invariant raises. This is what makes §9.10.2 achievable — no row can crash the run.

---

## 9. Degenerate cases — required behaviour

### 9.1 User with zero history rows for this peer — **6 rows**

Exactly 6 of the 110 rows: **3 personal, 3 business**. Confirmed under the temporal guard as well.

| Field | Required value |
|---|---|
| `peer_engagement.n` | `0` |
| `peer_engagement.*_rate` | **all `None`** — never `0.0` |
| `peer_engagement.median_reaction_minutes` | `None`, `n_reacted = 0` |
| `peer_engagement.basis_note` | the `n == 0` sentence in §6.3 |
| `evidence_state` | `"global_fallback"` or `"none"` per §6.3 |

**`peer_global` does not rescue all six.** For the 3 personal rows the sender appears in other users'
history (21–22 joined rows each), so `peer_global` is populated and `evidence_state` is `"global_fallback"`.
For the **3 business rows the business id appears nowhere in `message_history.csv` at all** — so
`peer_global.n == 0` too, every global rate is also `None`, and `evidence_state` is `"none"`.

That third state is not decoration. On those 3 rows the router has **no behavioural evidence of any kind**
and must decide from `content_signals`, `brand_integrity`, and `user_baseline` alone. The renderer states
this explicitly so the model does not silently treat absent evidence as benign evidence, and the confidence
it emits should reflect it (Decision 5 — thin evidence penalises confidence).

`evidence_candidates` is **still populated** on all six rows: every user in `messages.csv` has at least 3
history rows (median 8, max 32), so the lexical, recency, and event terms still rank *something*. Losing the
`same_peer` term degrades the ranking; it does not empty it.

### 9.2 Empty `message_text` — **8 rows, all `media_type == "voice"`**

Empty text occurs on exactly the 8 voice notes and nowhere else.

| Field | Required value |
|---|---|
| `raw_text` | `""` |
| `normalised_text` | `""` |
| `text_length` | `0`, `is_empty_text = True` |
| `text_scanned` | **`False`** |
| `injection_match` / `credential_request` / `payment_pressure` | `None` — meaning *not scanned*, surfaced as such by the renderer (§6.4) |
| `url_domains` | `()` |
| `near_duplicate_history` | `()` |
| `max_jaccard` | **`None`**, not `0.0` |
| `duplicate_count_at_threshold` | `0` |
| `media.requires_transcription` | `True` |

The trigram guard in §7.2 handles this without a special case: an empty normalised string yields an empty
trigram set, and `jaccard` returns `None` for every comparison. **The same guard also protects the 4
history rows whose text is empty**, which would otherwise produce a spurious `1.0` similarity against each
other under a naive `len(a & b) / len(a | b)` with `0/0` guarded to 1.

`evidence_candidates` **must still be produced.** The lexical term contributes `0.0` (§7.3) while
`same_peer`, `same_group`, `event_relevance`, and `recency` all still function. A voice note from a known
sender still gets its precedent set — the evidence path must not collapse merely because the text is in the
audio.

### 9.3 Business with empty `official_domain` — **5 of 110 accounts**

5 accounts have an empty `official_domain`; **1 of those 5 also has an empty `domain_used_by_sender`**.

| Field | Required value |
|---|---|
| `official_domain` | `None` (the loader already maps `""` → `None`) |
| `domain_mismatch` | **`None`** — not computable. Never `False`, never `True` |
| `verdict` | **cannot be `impersonation`** — condition 1 of the conjunction requires `domain_mismatch is True`, and `None` is not `True` |
| `verdict` | `"suspect"` if the §6.2 advisory disjunction fires, else `"clean"` |
| `verdict_basis` | includes `official_domain_absent` |

Both failure directions are wrong and both are avoided. Coercing `None → True` would let missing reference
data mute a legitimate business. Coercing `None → False` would silently launder an unverified, brand-new,
heavily-reported account into a clean verdict. The three-valued field plus the explicit `verdict_basis`
token keeps the gap visible in the output instead of resolving it by accident.

The identity comparison must be written to reach `None` **before** any string comparison, so an empty
`official_domain` can never compare equal to an empty `domain_used_by_sender` and score as a match.

### 9.4 Business with no `user_business_history` row — **11 rows, 9 pairs**

Not in the original brief; see §6.3. `business_relationship = None`, no exception, rendered explicitly.

---

## 10. Determinism, purity, and acceptance

### 10.1 Determinism checklist (§6.3, §9.10.5)

- Every sort key ends in `message_id`, which is unique — no tie is ever broken arbitrarily.
- `url_domains` and `verdict_basis` are sorted / fixed-order tuples, never set-iteration order.
- No `dict` iteration order is relied on for output ordering.
- No floating-point value is compared for equality; thresholds use `<` / `>=` only.
- Running `build_dossier` twice on the same inputs yields two equal `Dossier` values (`==` holds on frozen
  dataclasses), which is directly assertable in a test.

### 10.2 Acceptance checks before this layer is considered done

| # | Check |
|---|---|
| A1 | `grep -n "anthropic\|openai\|requests\|httpx\|urllib" code/context/` returns nothing (I1). |
| A2 | The AGENTS.md §9.8.5 no-hardcoded-labels grep, run over `code/context/`, returns nothing (I6). The pattern is referenced rather than inlined here, because writing it out would make this document its own only match. |
| A3 | For all 110 messages, `build_dossier` returns without raising. |
| A4 | Exactly **6** dossiers have `peer_engagement.n == 0`; of those exactly **3** have `evidence_state == "none"` and **3** have `"global_fallback"`. |
| A5 | Exactly **8** dossiers have `text_scanned is False`, and all 8 have `media.media_type == "voice"`. |
| A6 | No dossier anywhere contains a rate equal to `0.0` whose corresponding `n` is `0` (I3, asserted by walking every `Rate` field). |
| A7 | Exactly **49** users evaluate `dnd_wraps_midnight is True`; `dnd_state` unit tests cover: inside a wrapped window before midnight, inside it after midnight, at `t == start` (in), at `t == end` (out), and one minute before `end` (in, `minutes == 1`). |
| A8 | Every dossier with `in_dnd is True` has `minutes_until_dnd_ends` in `[1, 1439]`; every dossier with `in_dnd is False` has it `None`. |
| A9 | Exactly **18** of 110 business accounts yield `verdict == "impersonation"`, **82** yield `"clean"`, **10** yield `"suspect"`. Of the 23 domain-mismatched accounts, exactly **5** are spared — the 2 verified long-lived ones **and the 3 unverified accounts with 10, 13 and 20 reports**, which fail only condition 5 (Decision 4). Assert the counts and the partition, **not** the business ids — an id in a test is the same overfit §9.8 forbids on the decision path. |
| A9a | **Precision-trap assertion form.** The trap test asserts `verdict != "impersonation"`, **not** `verdict == "clean"`. The property under test is *"this account is never condemned"*, and that is what protects it from a false mute. Asserting the literal `"clean"` would test the middle branch's *name* rather than the safety property, and would fail correctly against this spec — a mismatched account is `"suspect"` by §6.2. A requirement phrased as "must come back clean despite a domain mismatch" means *tests negative for impersonation*; it is not a claim that the enum equals `"clean"`. |
| A10 | Business accounts with an absent `official_domain` yield `domain_mismatch is None` and never `"impersonation"`. |
| A11 | Perturbing each of the three brand thresholds by ±25% independently (`22`/`36` reports, `274`/`456` account days, `135`/`225` domain days) leaves the impersonation partition at the identical 18 accounts (§9.8.4 — proves the thresholds are not fitted). This check is what caught the earlier `BRAND_MAX_REPORTS` value sitting on a cohort boundary; keep it running. |
| A12 | Every field on every dataclass in this module is read by either the renderer or a downstream deterministic rule (I8). Enumerate and confirm before packaging. |
| A13 | A domain-mismatched account that fails the conjunction yields `"suspect"`. Assert directly that **no** business account yields `"clean"` while `domain_mismatch is True` — this is the middle branch collapsing, and it is silent if untested. |
| A14 | Import graph is acyclic in the §1 order. `python -c "import context.features"` succeeds, and no module in `code/context/` imports one listed below it. |
| A15 | No mutable module-level state (I2a): every module-level name in `code/context/` is either a frozen dataclass, a `Callable`, a compiled pattern, or an immutable scalar/tuple/frozenset/`Mapping` proxy. No module-level `dict`, `list`, or `set` literal is bound outside a frozen container. |
| A16 | Every constant in §4 resolves at import (`from config import ...` succeeds for all sixteen). This specifically catches the bare-annotation case in `config.py`, which type-checks but raises `NameError` at runtime. |

### 10.3 Hand-off to the renderer (not implemented here)

This module builds no prompt text. The renderer owns §9.7 containment, and these are the fields it must
place **inside** the untrusted fence, because each is dataset-derived and could carry adversarial text:

`content_signals.raw_text`, `content_signals.normalised_text`, the three scanner match phrases,
`content_signals.url_domains`, `sender_identity.display_name`, `sender_identity.brand_name`,
`business_relationship.why_user_knows_account`, `group_context.group_name`, and every
`evidence_candidate.text_excerpt`.

Everything else in the `Dossier` is a number, a boolean, an enum, a timestamp, or a string this layer
authored itself (`basis_note`, `verdict`, `verdict_basis`), and is safe to render as trusted structure.
