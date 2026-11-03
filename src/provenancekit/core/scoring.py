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

"""Provenance scoring: NaN-aware weighted identity and tokenizer scores.

Scoring architecture::

    identity_score  = NaN-aware weighted avg of EAS, NLF, LEP, END, WVC
    tokenizer_score = 0.25 × TFV + 0.75 × VOA  (reporting only)

    Pipeline Score  = identity_score when sufficient weight evidence exists;
                      capped in the Weak Match band for unconfirmed shallow
                      evidence; otherwise mfi_score (marked unverified)

Weights decide, metadata supports: MFI Tier 1/2 alone never yields a
match verdict (see :func:`decide`).

NaN-aware: when a signal returns NaN (incompatible comparison), it is
excluded and the remaining weights are proportionally rescaled.
"""

import math
from dataclasses import dataclass

from provenancekit.config.constants import IDENTITY_WEIGHTS, TOKENIZER_WEIGHTS
from provenancekit.models.results import ScoreInterpretation

# ── NaN-aware weighted average ────────────────────────────────────


def _nan_weighted_avg(
    weights: dict[str, float],
    signals: dict[str, float],
) -> float:
    """NaN-aware weighted average: skip NaN signals, rescale weights."""
    valid = {k: v for k, v in signals.items() if not math.isnan(v)}
    if not valid:
        return float("nan")
    total_w = sum(weights[k] for k in valid)
    if total_w < 1e-10:
        return float("nan")
    return round(sum((weights[k] / total_w) * valid[k] for k in valid), 4)


# ── Component scores ──────────────────────────────────────────────


def compute_identity_score(
    eas: float,
    nlf: float,
    lep: float,
    end: float,
    wvc: float,
) -> float:
    """Primary weight-identity score (NaN-aware).

    Uses all five identity signals: EAS, NLF, LEP, END, WVC.
    Skips NaN signals and proportionally rescales weights.
    """
    return _nan_weighted_avg(
        IDENTITY_WEIGHTS,
        {"eas": eas, "nlf": nlf, "lep": lep, "end": end, "wvc": wvc},
    )


def compute_tokenizer_score(tfv_sim: float, voa_sim: float) -> float:
    """Supplementary tokenizer consensus score."""
    w = TOKENIZER_WEIGHTS
    return round(w["tfv"] * tfv_sim + w["voa"] * voa_sim, 4)


# ── Interpretation ────────────────────────────────────────────────
# Verdict thresholds calibrated on the 111-pair benchmark.

_SCORE_HIGH_CONFIDENCE = 0.75
_SCORE_WEAK_MATCH = 0.65

# Without LEP and WVC (e.g. packed quantized weights) identity rests on
# embedding-level signals, which score many unrelated models above 0.65.
# Such a match only stands when metadata and the tokenizers also agree.
_SHALLOW_TFV_MIN = 0.9

# Unrelated weights give WVC around 0.003 even when architecture and
# tokenizer are shared and lift the other signals; related models
# (including distillation and continued pretraining) stay above 0.1.
_WVC_UNRELATED_MAX = 0.01


def interpret_score(score: float) -> ScoreInterpretation:
    """Return a human-readable verdict and colour for a pipeline score."""
    if math.isnan(score):
        return ScoreInterpretation(label="Insufficient data", colour="#999999")
    if score > _SCORE_HIGH_CONFIDENCE:
        return ScoreInterpretation(label="High-Confidence Match", colour="#2ecc71")
    if score > _SCORE_WEAK_MATCH:
        return ScoreInterpretation(label="Weak Match", colour="#f39c12")
    return ScoreInterpretation(label="Not Matched", colour="#e74c3c")


# ── Verdict ───────────────────────────────────────────────────────

VERIFICATION_WEIGHTS_AGREE = "weights_agree"
VERIFICATION_WEIGHTS_WEAK = "weights_weak"
VERIFICATION_WEIGHTS_CONFLICT = "weights_conflict"
VERIFICATION_WEIGHTS_ONLY = "weights_only"
VERIFICATION_WEIGHTS_SHALLOW = "weights_shallow"
VERIFICATION_UNVERIFIED = "unverified"

DECISION_CONFIRMED = "Confirmed Match"
DECISION_METADATA_WEAK = "Weak Match (metadata match, weak weight evidence)"
DECISION_METADATA_CONFLICT = "Not Matched (metadata/weight conflict)"
DECISION_METADATA_UNVERIFIED = "Metadata Match (unverified)"
DECISION_NOT_MATCHED_UNVERIFIED = "Not Matched (unverified)"
DECISION_SHALLOW_UNVERIFIED = "Weak Match (unverified, no deep weight signals)"

_UNVERIFIED_INTERPRETATION = ScoreInterpretation(
    label="Unverified (no weight evidence)", colour="#999999"
)
_SHALLOW_INTERPRETATION = ScoreInterpretation(
    label="Weak Match (unverified)", colour="#f39c12"
)
_SHALLOW_CONFIRMED_INTERPRETATION = ScoreInterpretation(
    label="Confirmed Match (limited weight evidence)", colour="#f39c12"
)


@dataclass(frozen=True)
class Verdict:
    """Final pipeline score, decision label and verification status."""

    pipeline_score: float
    decision: str
    verification: str
    interpretation: ScoreInterpretation


def decide(
    mfi_score: float,
    mfi_tier: int,
    identity: float,
    *,
    deep_signals: bool = True,
    tfv: float = float("nan"),
    wvc: float = float("nan"),
) -> Verdict:
    """Combine MFI and weight identity into a provenance verdict.

    Only weight evidence can produce a match verdict.  Without it the
    MFI score is kept as ``pipeline_score`` (for ranking) and the
    decision is marked unverified.

    ``deep_signals`` is False when neither LEP nor WVC could be compared.
    A match on the remaining embedding-level signals is confirmed only
    when MFI is Tier 1/2 and ``tfv`` shows tokenizer agreement.  Otherwise
    it is capped at the top of the Weak Match band; the raw identity remains
    available separately in ``identity_score``.

    A ``wvc`` below ``_WVC_UNRELATED_MAX`` caps the score at the Weak Match
    boundary, so the verdict is Not Matched whatever the other signals say.
    """
    if math.isnan(identity):
        metadata_match = mfi_tier <= 2 or mfi_score > _SCORE_WEAK_MATCH
        return Verdict(
            pipeline_score=mfi_score,
            decision=(
                DECISION_METADATA_UNVERIFIED
                if metadata_match
                else DECISION_NOT_MATCHED_UNVERIFIED
            ),
            verification=VERIFICATION_UNVERIFIED,
            interpretation=_UNVERIFIED_INTERPRETATION,
        )

    if wvc < _WVC_UNRELATED_MAX:
        identity = min(identity, _SCORE_WEAK_MATCH)

    if not deep_signals and identity > _SCORE_WEAK_MATCH:
        shallow_agreement = (
            mfi_tier <= 2
            and identity > _SCORE_HIGH_CONFIDENCE
            and tfv >= _SHALLOW_TFV_MIN
        )
        if shallow_agreement:
            return Verdict(
                identity,
                DECISION_CONFIRMED,
                VERIFICATION_WEIGHTS_SHALLOW,
                _SHALLOW_CONFIRMED_INTERPRETATION,
            )
        return Verdict(
            min(identity, _SCORE_HIGH_CONFIDENCE),
            DECISION_SHALLOW_UNVERIFIED,
            VERIFICATION_WEIGHTS_SHALLOW,
            _SHALLOW_INTERPRETATION,
        )

    interp = interpret_score(identity)
    if mfi_tier > 2:
        return Verdict(identity, interp.label, VERIFICATION_WEIGHTS_ONLY, interp)
    if identity > _SCORE_HIGH_CONFIDENCE:
        return Verdict(identity, DECISION_CONFIRMED, VERIFICATION_WEIGHTS_AGREE, interp)
    if identity > _SCORE_WEAK_MATCH:
        return Verdict(
            identity, DECISION_METADATA_WEAK, VERIFICATION_WEIGHTS_WEAK, interp
        )
    return Verdict(
        identity, DECISION_METADATA_CONFLICT, VERIFICATION_WEIGHTS_CONFLICT, interp
    )
