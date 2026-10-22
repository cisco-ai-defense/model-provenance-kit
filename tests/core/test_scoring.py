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

"""Tests for provenancekit.core.scoring — all offline (pure math)."""

import math

import pytest

from provenancekit.core.scoring import (
    compute_identity_score,
    compute_tokenizer_score,
    decide,
    interpret_score,
)
from provenancekit.models.results import ScoreInterpretation

NAN = float("nan")


# ── compute_identity_score ────────────────────────────────────────


class TestComputeIdentityScore:
    def test_all_ones(self) -> None:
        assert compute_identity_score(1.0, 1.0, 1.0, 1.0, 1.0) == 1.0

    def test_all_zeros(self) -> None:
        assert compute_identity_score(0.0, 0.0, 0.0, 0.0, 0.0) == 0.0

    def test_single_nan_excluded(self) -> None:
        score = compute_identity_score(1.0, NAN, 1.0, 1.0, 1.0)
        assert not math.isnan(score)
        assert 0.9 < score <= 1.0

    def test_all_nan_returns_nan(self) -> None:
        assert math.isnan(compute_identity_score(NAN, NAN, NAN, NAN, NAN))

    def test_only_eas_valid_equals_eas(self) -> None:
        score = compute_identity_score(0.5, NAN, NAN, NAN, NAN)
        assert score == 0.5

    def test_two_signals_rescale(self) -> None:
        score = compute_identity_score(1.0, NAN, NAN, NAN, 1.0)
        assert score == 1.0

    def test_mixed_values(self) -> None:
        score = compute_identity_score(0.9, 0.5, 0.7, 0.8, 0.6)
        assert 0.0 < score < 1.0
        assert not math.isnan(score)


# ── compute_tokenizer_score ───────────────────────────────────────


class TestComputeTokenizerScore:
    def test_known_value(self) -> None:
        expected = round(0.25 * 0.8 + 0.75 * 0.6, 4)
        assert compute_tokenizer_score(0.8, 0.6) == expected

    def test_both_zero(self) -> None:
        assert compute_tokenizer_score(0.0, 0.0) == 0.0

    def test_both_one(self) -> None:
        assert compute_tokenizer_score(1.0, 1.0) == 1.0

    def test_tfv_only(self) -> None:
        assert compute_tokenizer_score(1.0, 0.0) == 0.25

    def test_voa_only(self) -> None:
        assert compute_tokenizer_score(0.0, 1.0) == 0.75


# ── interpret_score ───────────────────────────────────────────────


class TestInterpretScore:
    def test_returns_score_interpretation(self) -> None:
        result = interpret_score(0.90)
        assert isinstance(result, ScoreInterpretation)

    def test_nan_insufficient_data(self) -> None:
        result = interpret_score(NAN)
        assert result.label == "Insufficient data"
        assert result.colour == "#999999"

    def test_band_high_confidence(self) -> None:
        assert interpret_score(0.76).label == "High-Confidence Match"
        assert interpret_score(0.85).label == "High-Confidence Match"
        assert interpret_score(0.95).label == "High-Confidence Match"
        assert interpret_score(1.00).label == "High-Confidence Match"

    def test_band_weak_match(self) -> None:
        assert interpret_score(0.66).label == "Weak Match"
        assert interpret_score(0.70).label == "Weak Match"
        assert interpret_score(0.75).label == "Weak Match"

    def test_band_not_matched(self) -> None:
        assert interpret_score(0.00).label == "Not Matched"
        assert interpret_score(0.10).label == "Not Matched"
        assert interpret_score(0.30).label == "Not Matched"
        assert interpret_score(0.50).label == "Not Matched"
        assert interpret_score(0.65).label == "Not Matched"

    def test_boundary_075_is_weak(self) -> None:
        assert interpret_score(0.75).label == "Weak Match"

    def test_boundary_065_is_not_matched(self) -> None:
        assert interpret_score(0.65).label == "Not Matched"

    def test_each_band_has_colour(self) -> None:
        for s in [NAN, 0.90, 0.70, 0.50, 0.10]:
            result = interpret_score(s)
            assert result.colour.startswith("#")
            assert len(result.colour) == 7


# ── decide ────────────────────────────────────────────────────────


class TestDecide:
    @pytest.mark.parametrize("tier", [1, 2])
    def test_metadata_match_weights_agree(self, tier: int) -> None:
        v = decide(1.0, tier, 0.90)
        assert v.pipeline_score == 0.90
        assert v.decision == "Confirmed Match"
        assert v.verification == "weights_agree"

    @pytest.mark.parametrize("tier", [1, 2])
    def test_metadata_match_weights_weak(self, tier: int) -> None:
        v = decide(0.9, tier, 0.70)
        assert v.pipeline_score == 0.70
        assert v.decision == "Weak Match (metadata match, weak weight evidence)"
        assert v.verification == "weights_weak"

    @pytest.mark.parametrize("tier", [1, 2])
    def test_metadata_match_weights_conflict(self, tier: int) -> None:
        v = decide(1.0, tier, 0.5248)
        assert v.pipeline_score == 0.5248
        assert v.decision == "Not Matched (metadata/weight conflict)"
        assert v.verification == "weights_conflict"

    @pytest.mark.parametrize("tier", [1, 2])
    def test_metadata_match_no_weights_unverified(self, tier: int) -> None:
        v = decide(1.0, tier, NAN)
        assert v.pipeline_score == 1.0
        assert v.decision == "Metadata Match (unverified)"
        assert v.verification == "unverified"
        assert v.interpretation.colour == "#999999"

    def test_tier3_with_weights_uses_score_bands(self) -> None:
        assert decide(0.4, 3, 0.80).decision == "High-Confidence Match"
        assert decide(0.4, 3, 0.70).decision == "Weak Match"
        v = decide(0.4, 3, 0.30)
        assert v.decision == "Not Matched"
        assert v.verification == "weights_only"
        assert v.pipeline_score == 0.30

    def test_tier3_no_weights_high_soft_score_unverified(self) -> None:
        v = decide(0.85, 3, NAN)
        assert v.pipeline_score == 0.85
        assert v.decision == "Metadata Match (unverified)"
        assert v.verification == "unverified"

    def test_tier3_no_weights_low_soft_score_not_matched(self) -> None:
        v = decide(0.40, 3, NAN)
        assert v.decision == "Not Matched (unverified)"
        assert v.verification == "unverified"

    def test_boundaries(self) -> None:
        assert decide(1.0, 1, 0.75).verification == "weights_weak"
        assert decide(1.0, 1, 0.65).verification == "weights_conflict"

    def test_metadata_alone_never_confirms(self) -> None:
        for tier in (1, 2, 3):
            assert "Confirmed" not in decide(1.0, tier, NAN).decision
