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

import numpy as np
import pytest
from pydantic import ValidationError

from provenancekit.models.signals import (
    MFIFingerprint,
    MFISimilarity,
    TokenizerFeatures,
    VocabOverlap,
    WeightSignalFeatures,
)

# ── Reusable fixture data ──────────────────────────────────────────


def _llama_fp_kwargs() -> dict:
    return {
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
        "tokenizer_hash": "abc123def456",
        "token_id_signature": "bos=1|eos=2",
        "arch_hash": "a1b2c3d4e5f6",
        "family_hash": "f1e2d3c4b5a6",
    }


def _tokenizer_features_kwargs() -> dict:
    return {
        "vocab_size": 32000,
        "tokenizer_class": "LlamaTokenizerFast",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "num_added_tokens": 0,
        "num_special_tokens": 3,
        "num_merges": 31000,
        "first_5_merges": ["▁ t", "▁ a", "e r", "i n", "▁ th"],
        "merge_rule_hash": "deadbeef12345678",
        "all_merges_str": "▁ t\n▁ a\ne r",
        "special_token_ids": {"pad": None, "unk": 0, "mask": None},
        "pct_single_char": 0.0097,
        "avg_token_length": 5.42,
        "max_token_length": 48,
        "pct_whitespace_prefix": 0.4312,
        "pct_byte_tokens": 0.0078,
        "script_distribution": {"Latin": 0.82, "CJK": 0.10, "Other": 0.08},
    }


# ── MFIFingerprint ─────────────────────────────────────────────────


class TestMFIFingerprint:
    def test_all_25_fields(self):
        fp = MFIFingerprint(**_llama_fp_kwargs())
        assert fp.model_type == "llama"
        assert fp.architectures == ["LlamaForCausalLM"]
        assert fp.hidden_size == 4096
        assert fp.num_hidden_layers == 32
        assert fp.num_attention_heads == 32
        assert fp.num_key_value_heads == 8
        assert fp.intermediate_size == 11008
        assert fp.vocab_size == 32000
        assert fp.max_position_embeddings == 4096
        assert fp.hidden_act == "silu"
        assert fp.rope_theta == 500000.0
        assert fp.rope_scaling is None
        assert fp.tie_word_embeddings is False
        assert fp.bos_token_id == 1
        assert fp.eos_token_id == 2
        assert fp.gqa_ratio == 4.0
        assert fp.attention_style == "GQA"
        assert fp.norm_type == "RMSNorm"
        assert fp.attention_bias is False
        assert fp.qk_norm is False
        assert fp.pos_encoding == "RoPE"
        assert fp.tokenizer_hash == "abc123def456"
        assert fp.token_id_signature == "bos=1|eos=2"
        assert fp.arch_hash == "a1b2c3d4e5f6"
        assert fp.family_hash == "f1e2d3c4b5a6"

    def test_rejects_wrong_type(self):
        with pytest.raises(ValidationError):
            MFIFingerprint(model_type=123)  # type: ignore[arg-type]

    def test_optional_fields_accept_none(self):
        kwargs = _llama_fp_kwargs()
        kwargs.update(
            {
                "num_key_value_heads": None,
                "hidden_act": None,
                "rope_theta": None,
                "rope_scaling": None,
                "tie_word_embeddings": None,
                "bos_token_id": None,
                "eos_token_id": None,
                "attention_bias": None,
            }
        )
        fp = MFIFingerprint(**kwargs)
        assert fp.num_key_value_heads is None
        assert fp.hidden_act is None
        assert fp.attention_bias is None

    def test_rope_scaling_dict(self):
        kwargs = _llama_fp_kwargs()
        kwargs["rope_scaling"] = {"type": "linear", "factor": 2.0}
        fp = MFIFingerprint(**kwargs)
        assert fp.rope_scaling == {"type": "linear", "factor": 2.0}

    def test_serialization_round_trip(self):
        fp = MFIFingerprint(**_llama_fp_kwargs())
        data = fp.model_dump()
        restored = MFIFingerprint.model_validate(data)
        assert restored == fp
        assert len(data) == 26


# ── MFISimilarity ──────────────────────────────────────────────────


class TestMFISimilarity:
    def test_tier1_exact(self):
        sim = MFISimilarity(score=1.0, tier=1, match_type="exact")
        assert sim.score == 1.0
        assert sim.tier == 1
        assert sim.match_type == "exact"

    def test_tier2_family(self):
        sim = MFISimilarity(score=0.9, tier=2, match_type="family")
        assert sim.tier == 2

    def test_tier3_soft(self):
        sim = MFISimilarity(score=0.62, tier=3, match_type="soft_match")
        assert sim.match_type == "soft_match"


# ── TokenizerFeatures ─────────────────────────────────────────────


class TestTokenizerFeatures:
    def test_all_18_fields(self):
        tf = TokenizerFeatures(**_tokenizer_features_kwargs())
        assert tf.vocab_size == 32000
        assert tf.tokenizer_class == "LlamaTokenizerFast"
        assert tf.bos_token_id == 1
        assert tf.eos_token_id == 2
        assert tf.pad_token_id is None
        assert tf.num_added_tokens == 0
        assert tf.num_special_tokens == 3
        assert tf.num_merges == 31000
        assert tf.first_5_merges == ["▁ t", "▁ a", "e r", "i n", "▁ th"]
        assert tf.merge_rule_hash == "deadbeef12345678"
        assert "▁ t" in tf.all_merges_str
        assert tf.special_token_ids["unk"] == 0
        assert tf.pct_single_char == 0.0097
        assert tf.avg_token_length == 5.42
        assert tf.max_token_length == 48
        assert tf.pct_whitespace_prefix == 0.4312
        assert tf.pct_byte_tokens == 0.0078
        assert tf.script_distribution["Latin"] == 0.82

    def test_empty_merges(self):
        kwargs = _tokenizer_features_kwargs()
        kwargs.update(
            {
                "num_merges": 0,
                "first_5_merges": [],
                "all_merges_str": "",
            }
        )
        tf = TokenizerFeatures(**kwargs)
        assert tf.num_merges == 0
        assert tf.first_5_merges == []

    def test_serialization_round_trip(self):
        tf = TokenizerFeatures(**_tokenizer_features_kwargs())
        data = tf.model_dump()
        restored = TokenizerFeatures.model_validate(data)
        assert restored == tf
        assert len(data) == 18


# ── VocabOverlap ───────────────────────────────────────────────────


class TestVocabOverlap:
    def test_valid(self):
        vo = VocabOverlap(
            jaccard=0.85,
            vocab_a_size=32000,
            vocab_b_size=32000,
            intersection=27200,
            union=32000,
            only_a=4800,
            only_b=0,
            overlap_pct_a=0.85,
            overlap_pct_b=0.85,
        )
        assert vo.jaccard == 0.85
        assert vo.intersection == 27200

    def test_zero_overlap(self):
        vo = VocabOverlap(
            jaccard=0.0,
            vocab_a_size=100,
            vocab_b_size=200,
            intersection=0,
            union=300,
            only_a=100,
            only_b=200,
            overlap_pct_a=0.0,
            overlap_pct_b=0.0,
        )
        assert vo.jaccard == 0.0
        assert vo.union == 300


# ── WeightSignalFeatures ──────────────────────────────────────────


class TestWeightSignalFeatures:
    def test_with_numpy_arrays(self):
        sim_matrix = np.eye(64, dtype=np.float32)
        nlf = np.random.default_rng(42).random(128)
        lep = np.random.default_rng(42).random(32)
        end_hist = np.random.default_rng(42).random(20)
        wsp = np.random.default_rng(42).random(64)
        wvc = {0: np.ones(4096), 1: np.zeros(4096)}

        wsf = WeightSignalFeatures(
            hidden_size=4096,
            num_layers=32,
            eas_self_sim=sim_matrix,
            eas_anchor_count=64,
            nlf_vector=nlf,
            nlf_mode="direct",
            nlf_num_layers=64,
            lep_profile=lep,
            end_histogram=end_hist,
            wsp_signature=wsp,
            wvc_layer_sigs=wvc,
        )
        assert wsf.hidden_size == 4096
        assert wsf.eas_self_sim is not None
        assert wsf.eas_self_sim.shape == (64, 64)
        assert wsf.nlf_mode == "direct"
        assert wsf.wvc_layer_sigs is not None
        assert 0 in wsf.wvc_layer_sigs

    def test_all_optional_none(self):
        wsf = WeightSignalFeatures(
            hidden_size=768,
            num_layers=12,
            eas_self_sim=None,
            eas_anchor_count=0,
            nlf_vector=None,
            nlf_mode=None,
            nlf_num_layers=0,
            lep_profile=None,
            end_histogram=None,
            wsp_signature=None,
            wvc_layer_sigs=None,
        )
        assert wsf.eas_self_sim is None
        assert wsf.nlf_vector is None
        assert wsf.lep_profile is None
        assert wsf.end_histogram is None
        assert wsf.wsp_signature is None
        assert wsf.wvc_layer_sigs is None

    def test_stats_mode(self):
        stats_vec = np.array([1.0, 0.1, 1.2, 0.9] * 12)
        wsf = WeightSignalFeatures(
            hidden_size=768,
            num_layers=12,
            eas_self_sim=None,
            eas_anchor_count=0,
            nlf_vector=stats_vec,
            nlf_mode="stats",
            nlf_num_layers=12,
            lep_profile=None,
            end_histogram=None,
            wsp_signature=None,
            wvc_layer_sigs=None,
        )
        assert wsf.nlf_mode == "stats"
        assert wsf.nlf_vector is not None
        assert len(wsf.nlf_vector) == 48
