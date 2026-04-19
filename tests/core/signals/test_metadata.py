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

from typing import Any

import pytest

from provenancekit.core.signals.metadata import (
    _resolve_intermediate_size,
    _tier3_soft_score,
    classify,
    similarity,
)
from provenancekit.models.signals import MFIFingerprint, MFISimilarity

# ── Reusable fixture builders ──────────────────────────────────────


def _make_fp(**overrides: Any) -> MFIFingerprint:
    defaults: dict[str, Any] = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "rope_theta": 500000.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "gqa_ratio": 4.0,
        "attention_style": "GQA",
        "norm_type": "RMSNorm",
        "attention_bias": False,
        "qk_norm": False,
        "pos_encoding": "RoPE",
        "tokenizer_hash": "abc123",
        "token_id_signature": "bos=1|eos=2",
        "arch_hash": "default_arch_hash",
        "family_hash": "default_family_hash",
    }
    defaults.update(overrides)
    return MFIFingerprint(**defaults)


class _MockConfig:
    """Lightweight mock for transformers config objects."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── _resolve_intermediate_size ─────────────────────────────────────


class TestResolveIntermediateSize:
    def test_standard_field(self) -> None:
        cfg = _MockConfig(intermediate_size=11008)
        assert _resolve_intermediate_size(cfg) == 11008

    def test_t5_d_ff(self) -> None:
        cfg = _MockConfig(d_ff=2048)
        assert _resolve_intermediate_size(cfg) == 2048

    def test_bart_encoder_ffn_dim(self) -> None:
        cfg = _MockConfig(encoder_ffn_dim=3072)
        assert _resolve_intermediate_size(cfg) == 3072

    def test_opt_ffn_dim(self) -> None:
        cfg = _MockConfig(ffn_dim=4096)
        assert _resolve_intermediate_size(cfg) == 4096

    def test_distilbert_hidden_dim(self) -> None:
        cfg = _MockConfig(hidden_size=768, hidden_dim=3072)
        assert _resolve_intermediate_size(cfg) == 3072

    def test_distilbert_hidden_dim_same_as_hidden_size(self) -> None:
        cfg = _MockConfig(hidden_size=768, hidden_dim=768)
        assert _resolve_intermediate_size(cfg) == 0

    def test_no_fields(self) -> None:
        cfg = _MockConfig()
        assert _resolve_intermediate_size(cfg) == 0

    def test_priority_order(self) -> None:
        cfg = _MockConfig(intermediate_size=100, d_ff=200)
        assert _resolve_intermediate_size(cfg) == 100


# ── classify ───────────────────────────────────────────────────────


class TestClassify:
    def test_known_family_llama(self) -> None:
        fp = _make_fp(model_type="llama")
        family, confidence = classify(fp)
        assert family == "llama"
        assert confidence == 1.0

    def test_known_family_bert(self) -> None:
        fp = _make_fp(model_type="bert")
        assert classify(fp) == ("bert", 1.0)

    def test_known_family_qwen_variant(self) -> None:
        fp = _make_fp(model_type="qwen2")
        assert classify(fp) == ("qwen", 1.0)

    def test_unknown_family(self) -> None:
        fp = _make_fp(model_type="totally_new_model")
        assert classify(fp) == ("unknown", 0.0)

    def test_case_insensitive(self) -> None:
        fp = _make_fp(model_type="LLAMA")
        assert classify(fp) == ("llama", 1.0)


# ── similarity — Tier 1 ───────────────────────────────────────────


class TestSimilarityTier1:
    def test_exact_arch_hash_match(self) -> None:
        fp = _make_fp(arch_hash="identical")
        result = similarity(fp, fp)
        assert isinstance(result, MFISimilarity)
        assert result.tier == 1
        assert result.score == 1.0
        assert result.match_type == "exact"

    def test_same_arch_hash_different_objects(self) -> None:
        fp_a = _make_fp(arch_hash="same_hash", family_hash="fam_a")
        fp_b = _make_fp(arch_hash="same_hash", family_hash="fam_b")
        result = similarity(fp_a, fp_b)
        assert result.tier == 1

    def test_empty_arch_hash_skips_tier1(self) -> None:
        fp_a = _make_fp(arch_hash="", family_hash="")
        fp_b = _make_fp(arch_hash="", family_hash="")
        result = similarity(fp_a, fp_b)
        assert result.tier == 3


# ── similarity — Tier 2 ───────────────────────────────────────────


class TestSimilarityTier2:
    def test_family_hash_match_same_dims(self) -> None:
        fp_a = _make_fp(arch_hash="a1", family_hash="SAME")
        fp_b = _make_fp(arch_hash="b2", family_hash="SAME")
        result = similarity(fp_a, fp_b)
        assert result.tier == 2
        assert result.score == 0.9
        assert result.match_type == "family"

    def test_family_hash_match_diff_hidden_size_demotes(self) -> None:
        fp_a = _make_fp(
            arch_hash="a1",
            family_hash="SAME",
            hidden_size=4096,
        )
        fp_b = _make_fp(
            arch_hash="b2",
            family_hash="SAME",
            hidden_size=2048,
        )
        result = similarity(fp_a, fp_b)
        assert result.tier == 3
        assert result.score > 0.5

    def test_family_hash_match_diff_layers_demotes(self) -> None:
        fp_a = _make_fp(
            arch_hash="a1",
            family_hash="SAME",
            num_hidden_layers=32,
        )
        fp_b = _make_fp(
            arch_hash="b2",
            family_hash="SAME",
            num_hidden_layers=24,
        )
        result = similarity(fp_a, fp_b)
        assert result.tier == 3


# ── similarity — Tier 3 ───────────────────────────────────────────


class TestSimilarityTier3:
    def test_identical_features_high_score(self) -> None:
        fp_a = _make_fp(arch_hash="a1", family_hash="fam_a")
        fp_b = _make_fp(arch_hash="b2", family_hash="fam_b")
        result = similarity(fp_a, fp_b)
        assert result.tier == 3
        assert result.score > 0.9

    def test_completely_different_low_score(self) -> None:
        fp_a = _make_fp(
            arch_hash="a",
            family_hash="fa",
            model_type="llama",
            bos_token_id=1,
            eos_token_id=2,
            qk_norm=False,
            vocab_size=32000,
            attention_style="GQA",
            rope_theta=500000.0,
            norm_type="RMSNorm",
            pos_encoding="RoPE",
            attention_bias=False,
        )
        fp_b = _make_fp(
            arch_hash="b",
            family_hash="fb",
            model_type="bert",
            bos_token_id=101,
            eos_token_id=102,
            qk_norm=True,
            vocab_size=30522,
            attention_style="MHA",
            rope_theta=None,
            norm_type="LayerNorm",
            pos_encoding="absolute",
            attention_bias=True,
        )
        result = similarity(fp_a, fp_b)
        assert result.tier == 3
        assert result.score < 0.5

    def test_same_dims_scores_higher_than_different_dims(self) -> None:
        fp_a = _make_fp(
            arch_hash="a1",
            family_hash="fam_a",
            hidden_size=4096,
            num_hidden_layers=32,
        )
        fp_b = _make_fp(
            arch_hash="b2",
            family_hash="fam_b",
            hidden_size=4096,
            num_hidden_layers=32,
        )
        result_same = similarity(fp_a, fp_b)

        fp_c = _make_fp(
            arch_hash="c3",
            family_hash="fam_c",
            hidden_size=4096,
            num_hidden_layers=32,
        )
        fp_d = _make_fp(
            arch_hash="d4",
            family_hash="fam_d",
            hidden_size=2048,
            num_hidden_layers=16,
        )
        result_diff = similarity(fp_c, fp_d)

        assert result_same.score > result_diff.score

    def test_dims_within_threshold_get_points(self) -> None:
        fp_a = _make_fp(
            arch_hash="a1",
            family_hash="fam_a",
            hidden_size=4096,
            num_hidden_layers=32,
        )
        fp_b = _make_fp(
            arch_hash="b2",
            family_hash="fam_b",
            hidden_size=4000,
            num_hidden_layers=30,
        )
        result = similarity(fp_a, fp_b)
        assert result.tier == 3
        assert result.score > 0.9


# ── _tier3_soft_score (unit test for internal helper) ──────────────


class TestTier3SoftScore:
    def test_identical_features_perfect(self) -> None:
        fp = _make_fp()
        score = _tier3_soft_score(fp, fp)
        assert score == 1.0

    def test_zero_vocab_excluded_from_total(self) -> None:
        fp_a = _make_fp(vocab_size=0)
        fp_b = _make_fp(vocab_size=0)
        score = _tier3_soft_score(fp_a, fp_b)
        assert 0.0 <= score <= 1.0

    def test_dimension_features_contribute(self) -> None:
        fp_same = _make_fp(hidden_size=4096, num_hidden_layers=32)
        score_same = _tier3_soft_score(fp_same, fp_same)

        fp_a = _make_fp(hidden_size=4096, num_hidden_layers=32)
        fp_b = _make_fp(hidden_size=2048, num_hidden_layers=16)
        score_diff = _tier3_soft_score(fp_a, fp_b)

        assert score_same > score_diff


# ── extract_fingerprint (online, slow) ─────────────────────────────


@pytest.mark.slow
class TestExtractFingerprintOnline:
    def test_gpt2_golden(self) -> None:
        from provenancekit.core.signals.metadata import extract_fingerprint

        fp, tok = extract_fingerprint("gpt2")
        assert isinstance(fp, MFIFingerprint)
        assert fp.model_type == "gpt2"
        assert fp.architectures == ["GPT2LMHeadModel"]
        assert fp.hidden_size == 768
        assert fp.num_hidden_layers == 12
        assert fp.num_attention_heads == 12
        assert fp.vocab_size == 50257
        assert fp.norm_type == "LayerNorm"
        assert fp.attention_style == "MHA"
        assert fp.pos_encoding == "absolute"
        assert fp.arch_hash is not None
        assert fp.family_hash is not None
        assert len(fp.tokenizer_hash) == 64
        assert tok is not None
