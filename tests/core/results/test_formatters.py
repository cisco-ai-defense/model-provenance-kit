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

"""Tests for provenancekit.core.results.formatters."""

import json
from typing import Any

import pytest

from provenancekit.core.results.formatters import (
    format_json,
    format_plain,
    format_table,
)
from provenancekit.models.results import (
    CompareResult,
    PipelineScore,
    ScoreInterpretation,
    SignalScores,
    TimingBreakdown,
)


@pytest.fixture()
def full_result() -> CompareResult:
    return CompareResult(
        model_a="org/model-a",
        model_b="org/model-b",
        family_a="gpt2",
        family_b="gpt2",
        signals=SignalScores(
            eas=0.9912,
            nlf=0.9543,
            lep=0.9801,
            end=0.9654,
            wvc=0.9123,
            tfv=0.9500,
            voa=0.8800,
        ),
        scores=PipelineScore(
            mfi_score=1.0,
            mfi_tier=1,
            mfi_match="exact",
            identity_score=0.9607,
            tokenizer_score=0.8975,
            pipeline_score=1.0,
            provenance_decision="Confirmed Match",
        ),
        interpretation=ScoreInterpretation(
            label="High-Confidence Match",
            colour="#2ecc71",
        ),
        time_seconds=4.2,
    )


@pytest.fixture()
def none_signals_result() -> CompareResult:
    return CompareResult(
        model_a="a",
        model_b="b",
        family_a="unknown",
        family_b="unknown",
        signals=SignalScores(
            eas=None,
            nlf=None,
            lep=None,
            end=None,
            wvc=None,
            tfv=0.5,
            voa=0.3,
        ),
        scores=PipelineScore(
            mfi_score=0.2,
            mfi_tier=3,
            mfi_match="soft_match",
            identity_score=None,
            tokenizer_score=0.35,
            pipeline_score=0.2,
            provenance_decision="Not Matched",
        ),
        interpretation=ScoreInterpretation(
            label="Not Matched",
            colour="#e74c3c",
        ),
        time_seconds=1.0,
    )


class TestFormatJson:
    def test_valid_json(self, full_result: CompareResult) -> None:
        output = format_json(full_result)
        data: dict[str, Any] = json.loads(output)
        assert "scores" in data
        assert data["scores"]["pipeline_score"] == 1.0
        assert "family_a" not in data
        assert "family_b" not in data

    def test_none_signals_serialize(self, none_signals_result: CompareResult) -> None:
        output = format_json(none_signals_result)
        data: dict[str, Any] = json.loads(output)
        assert data["signals"]["eas"] is None
        assert data["signals"]["tfv"] == 0.5


class TestFormatTable:
    def test_contains_verdict(self, full_result: CompareResult) -> None:
        output = format_table(full_result)
        assert "High-Confidence Match" in output

    def test_contains_models(self, full_result: CompareResult) -> None:
        output = format_table(full_result)
        assert "org/model-a" in output
        assert "org/model-b" in output

    def test_none_shows_na(self, none_signals_result: CompareResult) -> None:
        output = format_table(none_signals_result)
        assert "N/A" in output

    def test_renamed_labels(self, full_result: CompareResult) -> None:
        output = format_table(full_result)
        assert "Final Pipeline Score" in output
        assert "Metadata Feature Identifier" in output
        assert "Weight Identity Score" in output
        assert "Weight Feature Scores" in output

    def test_decision_at_top(self, full_result: CompareResult) -> None:
        output = format_table(full_result)
        decision_pos = output.index("Decision")
        pipeline_pos = output.index("Final Pipeline Score")
        assert decision_pos < pipeline_pos

    def test_include_timing_shows_phase_rows(
        self,
        full_result: CompareResult,
    ) -> None:
        full_result.timing = TimingBreakdown(
            total_seconds=4.2,
            metadata_extract_seconds=1.2,
            weight_feature_extract_seconds=2.3,
            cache_hit="ws, mfi",
        )
        output = format_table(full_result, include_timing=True)
        assert "Time Taken Breakdown" in output
        assert "Total Time" in output
        assert "Model Metadata Extract Time" in output
        assert "Model Weight Feature Extract Time" in output
        assert "Cache Hit" in output
        assert "ws, mfi" in output
        assert "1.2s" in output


class TestFormatPlain:
    def test_contains_all_keys(self, full_result: CompareResult) -> None:
        output = format_plain(full_result)
        expected_keys = [
            "model_a:",
            "model_b:",
            "pipeline_score:",
            "verdict:",
            "mfi_score:",
            "mfi_tier:",
            "identity_score:",
            "tokenizer_score:",
            "eas:",
            "nlf:",
            "lep:",
            "end:",
            "wvc:",
            "tfv:",
            "voa:",
            "provenance_decision:",
            "time_seconds:",
        ]
        for key in expected_keys:
            assert key in output, f"Missing key: {key}"
        assert "family_a:" not in output
        assert "family_b:" not in output

    def test_none_shows_na(self, none_signals_result: CompareResult) -> None:
        output = format_plain(none_signals_result)
        lines_with_na = [line for line in output.splitlines() if "N/A" in line]
        assert len(lines_with_na) >= 5

    def test_include_timing_adds_lines(self, full_result: CompareResult) -> None:
        full_result.timing = TimingBreakdown(
            total_seconds=4.2,
            metadata_extract_seconds=1.2,
            weight_feature_extract_seconds=2.3,
            cache_hit="ws, mfi",
        )
        output = format_plain(full_result, include_timing=True)
        assert "total_time_s:" in output
        assert "metadata_extract_time_s:" in output
        assert "weight_feature_extract_time_s:" in output
        assert "cache_hit:" in output
        assert "ws, mfi" in output
