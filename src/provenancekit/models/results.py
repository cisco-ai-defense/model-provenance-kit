# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for comparison and scan results.

Houses the scoring decomposition and top-level output structures that
match the benchmark JSON shape.

Scoring recap::

    Pipeline Score (MFI-gated) — THE decision score:
        Tier 1/2 match  →  pipeline_score = mfi_score
        Tier 3 (soft)   →  pipeline_score = identity_score

    Identity Score  = NaN-aware weighted avg of EAS, NLF, LEP, END, WVC
    Tokenizer Score = 0.25 × TFV + 0.75 × VOA   (reporting only)
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from provenancekit.config.constants import EXTRACTOR_VERSION
from provenancekit.models.signals import ConfigInconsistency

# ── Model loading result ──────────────────────────────────────────


class LoadStrategy(StrEnum):
    """Strategy indicator returned by the model loader."""

    full = "full"
    streaming = "streaming"


class LoadResult(BaseModel):
    """Structured result from :func:`services.model_loader.load_state_dict`.

    * ``full`` — ``state_dict`` is populated; caller uses it directly.
    * ``streaming`` — model is too large; caller must stream tensors.
    * ``failed`` — unrecoverable error; ``source`` has the reason.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_dict: dict[str, Any] | None = None
    config: Any | None = None
    strategy: LoadStrategy
    source: str = ""


class SignalScores(BaseModel):
    """All pairwise signal similarity values.

    ``None`` represents NaN or an incompatible/unavailable comparison
    (e.g. dimension mismatch, model too large, tokenizer load failure).
    """

    eas: float | None
    nlf: float | None
    lep: float | None
    end: float | None
    wvc: float | None
    tfv: float | None
    voa: float | None


class PipelineScore(BaseModel):
    """Decomposed scoring output.

    ``pipeline_score`` normally equals ``identity_score`` whenever weight
    evidence exists, otherwise ``mfi_score``.  It is capped in the Weak Match
    band when only embedding-level weight signals are available and the
    metadata/tokenizer guard does not confirm them.  ``verification`` states
    whether the decision is backed by weights: ``weights_agree``,
    ``weights_weak``, ``weights_conflict``, ``weights_only``,
    ``weights_shallow`` (limited evidence without LEP/WVC), or ``unverified``.
    """

    mfi_score: float
    mfi_tier: int
    mfi_match: str
    identity_score: float | None
    tokenizer_score: float | None
    pipeline_score: float
    provenance_decision: str
    verification: str = "unverified"


class ScoreInterpretation(BaseModel):
    """Human-readable verdict derived from the pipeline score."""

    label: str
    colour: str


class TimingBreakdown(BaseModel):
    """High-level elapsed durations for major compare phases."""

    total_seconds: float
    metadata_extract_seconds: float
    weight_feature_extract_seconds: float
    cache_hit: str


class ModelConfigInconsistency(ConfigInconsistency):
    """A :class:`ConfigInconsistency` attributed to one compared model."""

    model: str


class CompareResult(BaseModel):
    """Full top-level comparison output for a model pair.

    Matches the structure produced by the benchmark pipeline and
    serialised to JSON.

    ``config_inconsistent`` lists ``config.json`` fields that contradict
    the stored tensors of either model.  When non-empty, MFI Tier 1/2
    is not applied for the pair.
    """

    model_a: str
    model_b: str
    family_a: str
    family_b: str
    signals: SignalScores
    scores: PipelineScore
    interpretation: ScoreInterpretation
    time_seconds: float
    timing: TimingBreakdown | None = None
    config_inconsistent: list[ModelConfigInconsistency] = Field(default_factory=list)


class CachedEntry(BaseModel):
    """Typed cache payload for a single model.

    Stored on disk as JSON via ``model_dump`` / ``model_validate``.
    The ``mfi``, ``tfv``, and ``ws`` fields are raw dicts so the cache
    layer stays decoupled from signal-specific Pydantic models; callers
    convert with ``MFIFingerprint.model_validate(entry.mfi)`` or
    ``WeightSignalFeatures.from_cache_dict(entry.ws)`` when they need
    typed objects.

    ``extractor_version`` defaults to the running version for new
    entries; files on disk without it are treated as stale.
    ``extraction_params`` holds the settings used for extraction and
    ``ws_loader`` records how the weights were read (strategy, source,
    dtype).
    """

    model_id: str
    extractor_version: str = EXTRACTOR_VERSION
    extraction_params: dict[str, int] | None = None
    mfi: dict[str, Any] | None = None
    tfv: dict[str, Any] | None = None
    ws: dict[str, Any] | None = None
    ws_loader: dict[str, str | None] | None = None
    vocab: list[str] | None = None


# ── Scan workflow results ─────────────────────────────────────────


class ScanMatchScores(BaseModel):
    """Score breakdown for a single scan match against a DB model.

    All signal values are ``None`` when the signal could not be
    computed (e.g. missing deep-signals parquet).  ``verification`` has
    the same meaning as in :class:`PipelineScore`.
    """

    pipeline_score: float | None
    identity_score: float | None
    mfi_score: float
    mfi_tier: int
    mfi_match_type: str
    tokenizer_score: float | None
    eas: float | None
    nlf: float | None
    lep: float | None
    end: float | None
    wvc: float | None
    tfv: float | None
    verification: str = "unverified"


class ScanMatch(BaseModel):
    """A single ranked match from the scan pipeline."""

    asset_id: str
    model_id: str
    family_id: str
    family_name: str
    param_bucket: str = ""
    match_type: str
    scores: ScanMatchScores
    provenance_decision: str
    elapsed_ms: float = 0.0


class ScanModelInfo(BaseModel):
    """Extracted metadata about the scanned (query) model.

    ``config_inconsistent`` lists ``config.json`` fields that contradict
    the query model's stored tensors.  When non-empty, MFI Tier 1/2 is
    not applied to any database candidate.
    """

    model_path: str
    model_type: str
    architectures: list[str]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None
    head_dim: int | None = None
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool | None
    hidden_act: str | None
    num_parameters: int | None = None
    arch_hash: str
    family_hash: str
    param_bucket: str
    has_weight_signals: bool
    config_inconsistent: list[ConfigInconsistency] = Field(default_factory=list)


class ScanResult(BaseModel):
    """Full scan result: query model info plus ranked DB matches."""

    model_info: ScanModelInfo
    matches: list[ScanMatch]
    match_count: int = 0
    elapsed_ms: float = 0.0
    extract_seconds: float = 0.0
    lookup_seconds: float = 0.0
