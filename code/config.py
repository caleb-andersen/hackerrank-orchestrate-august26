"""Central configuration for the notification router."""

import os
from types import MappingProxyType
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


# Resolve from this file rather than cwd because evaluation may start elsewhere.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


# Anthropic credentials are supplied by the process environment, never source control.
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
# OpenAI credentials are supplied by the process environment, never source control.
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")

# The strongest Claude model is the production decision-maker.
DECISION_MODEL_PRIMARY: str = "claude-opus-5"
# The faster Claude model keeps development iterations affordable.
DECISION_MODEL_DEV: str = "claude-sonnet-5"
# Ordered fallbacks preserve service when the preferred decision model is unavailable.
FALLBACK_CHAIN: list[str] = ["claude-opus-5", "claude-sonnet-5", "gpt-5.6-terra"]
# Voice notes use OpenAI's dedicated transcription model.
TRANSCRIBE_MODEL: str = "gpt-4o-transcribe"
# An independent OpenAI model judges evaluation outputs.
JUDGE_MODEL: str = "gpt-5.6-terra"

# Tool loops stop after four iterations to bound cost and runaway behavior.
MAX_TOOL_ITERATIONS: int = 4
# One routing turn never needs more than this; thinking and text share the budget.
MAX_OUTPUT_TOKENS: int = 8_000
# Routing is a bounded judgement over precomputed facts, not open-ended research.
DECISION_EFFORT: str = "medium"
# Image inspection is capped at two calls so one message cannot dominate the budget.
MAX_INSPECT_IMAGE_CALLS: int = 2
# At most two historical messages may be cited to keep evidence focused.
MAX_EVIDENCE_IDS: int = 2
# Transient provider failures receive up to four attempts before exhaustion is reported.
MAX_RETRY_ATTEMPTS: int = 4
# Retry delays begin at one second to recover quickly from short provider hiccups.
RETRY_BASE_SECONDS: float = 1.0
# Retry delays are capped at twenty seconds to keep a failed row bounded.
RETRY_CAP_SECONDS: float = 20.0
# Each row has three minutes to finish before it becomes a legible timeout result.
PER_ROW_TIMEOUT_SECONDS: int = 180
# Six concurrent rows balance throughput against provider rate limits.
MAX_CONCURRENCY: int = 6
# Concurrency ramps from two workers to avoid an initial request spike.
CONCURRENCY_RAMP_START: int = 2
# Images are resized within 1024 pixels to control latency and token use.
MAX_IMAGE_DIMENSION: int = 1024
# Reported confidence cannot fall below 0.55 so weak decisions remain distinguishable.
CONF_FLOOR: float = 0.55
# Reported confidence cannot exceed 0.95 because routing decisions retain uncertainty.
CONF_CEIL: float = 0.95
# Media that contradicts the message text costs confidence without moving the action.
MEDIA_MISMATCH_CONFIDENCE_PENALTY: float = 0.05
# A sender with no history and no citable precedent is a weaker basis for any routing.
FIRST_CONTACT_CONFIDENCE_PENALTY: float = 0.05
# A row where deterministic code overrode the model is capped here, inside the mute band.
HARD_BLOCK_CONFIDENCE: float = 0.85

# Dataset inputs are resolved from the repository working directory.
DATASET_DIR: Path = Path("dataset")
# Predictions overwrite the provided output template in its repository-relative location.
OUTPUT_PATH: Path = DATASET_DIR / "output.csv"
# Per-row traces stay in a repository-relative ignored directory.
TRACE_DIR: Path = Path("traces")
# Checkpoints stay in a repository-relative ignored cache directory.
CACHE_DIR: Path = Path(".cache")

# History recency decays by half every thirty days.
RECENCY_HALF_LIFE_DAYS: float = 30.0
# The dossier offers at most six ranked historical precedents.
EVIDENCE_TOP_K: int = 6
# Evidence below this relevance score is omitted.
EVIDENCE_MIN_SCORE: float = 0.12
# Evidence scoring weights form an interpretable weighted mean.
W_SAME_PEER: float = 0.35
W_LEXICAL: float = 0.25
W_SAME_GROUP: float = 0.15
W_EVENT: float = 0.15
W_RECENCY: float = 0.10
# Stronger user actions make a history row more relevant evidence.
EVENT_RELEVANCE: Mapping[str, float] = MappingProxyType(
    {
        "reported": 1.0,
        "muted_after": 0.9,
        "replied": 0.7,
        "dismissed": 0.6,
        "opened": 0.3,
    }
)
# Repetition exposes only the strongest three near-duplicates.
NEAR_DUPLICATE_TOP_K: int = 3
# Independent short messages normally fall below this trigram overlap.
NEAR_DUPLICATE_MIN_JACCARD: float = 0.45
# Sender bursts are measured over the preceding day.
BURST_WINDOW_HOURS: int = 24
# Text similarity uses unpadded character trigrams.
TRIGRAM_N: int = 3
# Published per-million-token rates, used only to price the run report. A model absent
# here has its tokens reported under "unpriced" rather than costed at a guessed rate.
MODEL_PRICE_PER_MTOK: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "claude-opus-5": (5.00, 25.00),
        "claude-sonnet-5": (3.00, 15.00),
        "claude-haiku-4-5": (1.00, 5.00),
    }
)
# A cached prefix is re-read at a tenth of the input rate. Without this the run report
# would price a cache hit as if it were a fresh read and the saving would be invisible.
CACHE_READ_MULTIPLIER: float = 0.10
# Writing a prefix into the five-minute cache costs a quarter more than reading it
# uncached. Two reads pay that premium back; one read does not, which is exactly the
# question the cache_read counter in the run report exists to answer.
CACHE_WRITE_MULTIPLIER: float = 1.25

# A business account younger than one year is not trusted on age alone.
BRAND_MIN_AGE_DAYS: int = 365
# A sender domain younger than six months is not trusted on age alone.
BRAND_MIN_DOMAIN_AGE_DAYS: int = 180
# Report pressure becomes adverse above the midpoint of the measured gap.
BRAND_MAX_REPORTS: int = 29
# A sender this user dismisses at least half the time has earned suppression.
DISMISS_MUTE_THRESHOLD: float = 0.5
# One prior message already reads as a dismiss rate; demanding more discards evidence.
MIN_PEER_HISTORY: int = 1
