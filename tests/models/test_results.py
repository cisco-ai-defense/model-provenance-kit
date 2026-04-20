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

import pytest
from pydantic import ValidationError

from provenancekit.models.results import (
    CompareResult,
    PipelineScore,
    ScanMatch,
    ScanMatchScores,
    ScanModelInfo,
    ScanResult,
    ScoreInterpretation,
    SignalScores,
)

# ── SignalScores ───────────────────────────────────────────────────


class TestSignalScores:
    def test_all_present(self):
        ss = SignalScores(
            eas=0.9998,
            nlf=0.23,
            lep=1.0,
            end=0.999,
            wvc=0.246,
            tfv=1.0,
            voa=1.0,
        )
        assert ss.eas == 0.9998
        assert ss.wvc == 0.246
        assert ss.tfv == 1.0

    def test_weight_signals_none(self):
        ss = SignalScores(
            eas=0.99,
            nlf=None,
            lep=1.0,
            end=0.98,
            wvc=None,
            tfv=1.0,
            voa=1.0,
        )
        assert ss.nlf is None
        assert ss.wvc is None

    def test_all_weight_signals_none(self):
        ss = SignalScores(
            eas=None,
            nlf=None,
            lep=None,
            end=None,
            wvc=None,
            tfv=0.8,
            voa=0.75,
        )
        assert ss.eas is None
        assert ss.tfv == 0.8

    def test_tokenizer_signals_none(self):
        ss = SignalScores(
            eas=0.99,
            nlf=0.5,
            lep=1.0,
            end=0.98,
            wvc=0.3,
            tfv=None,
            voa=None,
        )
        assert ss.tfv is None
        assert ss.voa is None

    def test_all_signals_none(self):
        ss = SignalScores(
            eas=None,
            nlf=None,
            lep=None,
            end=None,
            wvc=None,
            tfv=None,
            voa=None,
        )
        assert ss.tfv is None
        assert ss.voa is None

    def test_rejects_missing_required(self):
        with pytest.raises(ValidationError):
            SignalScores(eas=0.9)  # type: ignore[call-arg]


# ── PipelineScore ──────────────────────────────────────────────────


class TestPipelineScore:
    def test_tier1_exact(self):
        ps = PipelineScore(
            mfi_score=1.0,
            mfi_tier=1,
            mfi_match="exact",
            identity_score=0.78,
            tokenizer_score=1.0,
            pipeline_score=1.0,
            provenance_decision="Confirmed Match",
        )
        assert ps.pipeline_score == ps.mfi_score
        assert ps.mfi_tier == 1

    def test_tier2_family(self):
        ps = PipelineScore(
            mfi_score=0.9,
            mfi_tier=2,
            mfi_match="family",
            identity_score=0.85,
            tokenizer_score=0.95,
            pipeline_score=0.9,
            provenance_decision="Confirmed Match",
        )
        assert ps.pipeline_score == ps.mfi_score

    def test_tier3_uses_identity(self):
        ps = PipelineScore(
            mfi_score=0.62,
            mfi_tier=3,
            mfi_match="soft_match",
            identity_score=0.71,
            tokenizer_score=0.88,
            pipeline_score=0.71,
            provenance_decision="Weak Match",
        )
        assert ps.pipeline_score == ps.identity_score

    def test_identity_score_none(self):
        ps = PipelineScore(
            mfi_score=0.45,
            mfi_tier=3,
            mfi_match="soft_match",
            identity_score=None,
            tokenizer_score=0.5,
            pipeline_score=0.45,
            provenance_decision="Insufficient data",
        )
        assert ps.identity_score is None

    def test_tokenizer_score_none(self):
        ps = PipelineScore(
            mfi_score=1.0,
            mfi_tier=1,
            mfi_match="exact",
            identity_score=0.9,
            tokenizer_score=None,
            pipeline_score=1.0,
            provenance_decision="Confirmed Match",
        )
        assert ps.tokenizer_score is None


# ── ScoreInterpretation ────────────────────────────────────────────


class TestScoreInterpretation:
    def test_valid(self):
        si = ScoreInterpretation(
            label="High-Confidence Match",
            colour="#2ecc71",
        )
        assert si.label == "High-Confidence Match"
        assert si.colour == "#2ecc71"

    def test_insufficient_data(self):
        si = ScoreInterpretation(
            label="Insufficient data",
            colour="#999999",
        )
        assert si.colour == "#999999"


# ── CompareResult ──────────────────────────────────────────────────


def _make_compare_result() -> CompareResult:
    return CompareResult(
        model_a="bloom-560m",
        model_b="bloomz-560m",
        family_a="bloom",
        family_b="bloom",
        signals=SignalScores(
            eas=0.9998,
            nlf=0.23,
            lep=1.0,
            end=0.999,
            wvc=0.246,
            tfv=1.0,
            voa=1.0,
        ),
        scores=PipelineScore(
            mfi_score=1.0,
            mfi_tier=1,
            mfi_match="exact",
            identity_score=0.78,
            tokenizer_score=1.0,
            pipeline_score=1.0,
            provenance_decision="Confirmed Match",
        ),
        interpretation=ScoreInterpretation(
            label="High-Confidence Match",
            colour="#2ecc71",
        ),
        time_seconds=9.5,
    )


class TestCompareResult:
    def test_full_construction(self):
        cr = _make_compare_result()
        assert cr.model_a == "bloom-560m"
        assert cr.scores.pipeline_score == 1.0
        assert cr.interpretation.colour == "#2ecc71"
        assert cr.time_seconds == 9.5

    def test_nested_access(self):
        cr = _make_compare_result()
        assert cr.signals.eas == 0.9998
        assert cr.scores.mfi_tier == 1

    def test_model_dump_round_trip(self):
        cr = _make_compare_result()
        data = cr.model_dump()
        assert data["model_a"] == "bloom-560m"
        assert data["signals"]["eas"] == 0.9998
        assert data["scores"]["mfi_tier"] == 1
        restored = CompareResult.model_validate(data)
        assert restored == cr

    def test_json_round_trip(self):
        cr = _make_compare_result()
        json_str = cr.model_dump_json()
        restored = CompareResult.model_validate_json(json_str)
        assert restored == cr


# ── ScanResult ─────────────────────────────────────────────────────


class TestScanResult:
    def _make_model_info(self) -> ScanModelInfo:
        return ScanModelInfo(
            model_path="meta-llama/Llama-2-7b",
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            tie_word_embeddings=False,
            hidden_act="silu",
            arch_hash="abc123",
            family_hash="def456",
            param_bucket="1-10B",
            has_weight_signals=True,
        )

    def _make_scan_match(self) -> ScanMatch:
        return ScanMatch(
            asset_id="bloom-560m__hf-safetensors",
            model_id="bloom-560m",
            family_id="bloom",
            family_name="BLOOM",
            match_type="exact_arch",
            scores=ScanMatchScores(
                pipeline_score=0.98,
                identity_score=0.97,
                mfi_score=1.0,
                mfi_tier=1,
                mfi_match_type="exact_arch",
                tokenizer_score=0.85,
                eas=0.99,
                nlf=0.95,
                lep=0.98,
                end=0.96,
                wvc=0.94,
                tfv=0.85,
            ),
            provenance_decision="Confirmed Match",
            elapsed_ms=12.5,
        )

    def test_empty_matches(self):
        sr = ScanResult(model_info=self._make_model_info(), matches=[])
        assert sr.model_info.model_path == "meta-llama/Llama-2-7b"
        assert sr.matches == []
        assert sr.match_count == 0

    def test_with_matches(self):
        match = self._make_scan_match()
        sr = ScanResult(
            model_info=self._make_model_info(),
            matches=[match],
            match_count=1,
            elapsed_ms=150.0,
        )
        assert len(sr.matches) == 1
        assert sr.matches[0].model_id == "bloom-560m"
        assert sr.matches[0].scores.pipeline_score == 0.98
        assert sr.elapsed_ms == 150.0
