"""Central configuration for the notification router."""

import os
from pathlib import Path

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

# Dataset inputs are resolved from the repository working directory.
DATASET_DIR: Path = Path("dataset")
# Predictions overwrite the provided output template in its repository-relative location.
OUTPUT_PATH: Path = DATASET_DIR / "output.csv"
# Per-row traces stay in a repository-relative ignored directory.
TRACE_DIR: Path = Path("traces")
# Checkpoints stay in a repository-relative ignored cache directory.
CACHE_DIR: Path = Path(".cache")

# TODO: Set after calibration against the sample file in a later task.
BRAND_MIN_AGE_DAYS: int
# TODO: Set after calibration against the sample file in a later task.
BRAND_MAX_REPORTS: int
# TODO: Set after calibration against the sample file in a later task.
DISMISS_MUTE_THRESHOLD: float
# TODO: Set after calibration against the sample file in a later task.
MIN_PEER_HISTORY: int
