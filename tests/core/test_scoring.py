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

from provenancekit.core.scoring import (
    compute_identity_score,
    compute_tokenizer_score,
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

    def test_band_high(self) -> None:
        assert interpret_score(0.85).label == "Same family / direct derivative"
        assert interpret_score(0.95).label == "Same family / direct derivative"
        assert interpret_score(1.00).label == "Same family / direct derivative"

    def test_band_likely(self) -> None:
        result = interpret_score(0.75)
        assert result.label == "Likely same family or closely related"
        assert interpret_score(0.72).label == "Likely same family or closely related"

    def test_band_possibly(self) -> None:
        assert interpret_score(0.60).label == "Possibly related"
        assert interpret_score(0.65).label == "Possibly related"

    def test_band_weakly(self) -> None:
        assert interpret_score(0.50).label == "Weakly related"
        assert interpret_score(0.55).label == "Weakly related"

    def test_band_different(self) -> None:
        assert interpret_score(0.10).label == "Different families"
        assert interpret_score(0.00).label == "Different families"
        assert interpret_score(0.30).label == "Different families"

    def test_each_band_has_colour(self) -> None:
        for s in [NAN, 0.90, 0.70, 0.50, 0.30, 0.10]:
            result = interpret_score(s)
            assert result.colour.startswith("#")
            assert len(result.colour) == 7
