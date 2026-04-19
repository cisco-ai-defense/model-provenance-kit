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

"""Tests for core/signals/tokenizer.py — TFV extraction, similarity, and VOA."""

from typing import Any

import pytest

from provenancekit.core.signals.tokenizer import (
    classify_script,
    compute_script_distribution,
    tfv_similarity,
    vocab_overlap,
)
from provenancekit.models.signals import TokenizerFeatures, VocabOverlap

# ── Reusable fixture builder ───────────────────────────────────────


def _make_tfv(**overrides: Any) -> TokenizerFeatures:
    defaults: dict[str, Any] = {
        "vocab_size": 50257,
        "tokenizer_class": "GPT2TokenizerFast",
        "bos_token_id": None,
        "eos_token_id": 50256,
        "pad_token_id": None,
        "num_added_tokens": 0,
        "num_special_tokens": 1,
        "num_merges": 50000,
        "first_5_merges": ["Ġ t", "Ġ a", "h e", "i n", "r e"],
        "merge_rule_hash": "abcdef1234567890",
        "all_merges_str": "Ġ t\nĠ a\nh e",
        "special_token_ids": {"pad": None, "unk": None, "mask": None},
        "pct_single_char": 0.006,
        "avg_token_length": 6.2,
        "max_token_length": 50,
        "pct_whitespace_prefix": 0.82,
        "pct_byte_tokens": 0.005,
        "script_distribution": {"Latin": 0.95, "Other": 0.05},
    }
    defaults.update(overrides)
    return TokenizerFeatures(**defaults)


# ── classify_script (offline) ──────────────────────────────────────


class TestClassifyScript:
    def test_ascii_latin(self) -> None:
        assert classify_script("A") == "Latin"
        assert classify_script("z") == "Latin"

    def test_extended_latin(self) -> None:
        assert classify_script("é") == "Latin"

    def test_cjk(self) -> None:
        assert classify_script("中") == "CJK"

    def test_hangul(self) -> None:
        assert classify_script("한") == "CJK"

    def test_cyrillic(self) -> None:
        assert classify_script("Д") == "Cyrillic"

    def test_arabic(self) -> None:
        assert classify_script("ع") == "Arabic"

    def test_devanagari(self) -> None:
        assert classify_script("क") == "Devanagari"

    def test_other(self) -> None:
        assert classify_script("→") == "Other"

    def test_digit_ascii(self) -> None:
        assert classify_script("5") == "Latin"


# ── compute_script_distribution (offline) ──────────────────────────


class TestComputeScriptDistribution:
    def test_all_latin(self) -> None:
        dist = compute_script_distribution(["hello", "world"])
        assert dist == {"Latin": 1.0}

    def test_mixed_scripts(self) -> None:
        dist = compute_script_distribution(["hello", "中文"])
        assert "Latin" in dist
        assert "CJK" in dist
        assert abs(sum(dist.values()) - 1.0) < 1e-4

    def test_empty_tokens(self) -> None:
        assert compute_script_distribution([]) == {}

    def test_no_alpha_chars(self) -> None:
        assert compute_script_distribution(["123", "!!!"]) == {}


# ── tfv_similarity (offline, mock TokenizerFeatures) ───────────────


class TestTfvSimilarity:
    def test_identical_self(self) -> None:
        feats = _make_tfv()
        assert tfv_similarity(feats, feats) == 1.0

    def test_different_tokenizer_class(self) -> None:
        fa = _make_tfv(tokenizer_class="GPT2TokenizerFast")
        fb = _make_tfv(tokenizer_class="LlamaTokenizerFast")
        score = tfv_similarity(fa, fb)
        assert score < 1.0
        assert score > 0.5

    def test_different_vocab_size(self) -> None:
        fa = _make_tfv(vocab_size=50257)
        fb = _make_tfv(vocab_size=32000)
        score = tfv_similarity(fa, fb)
        assert score < 1.0

    def test_completely_different(self) -> None:
        fa = _make_tfv(
            vocab_size=50257,
            tokenizer_class="GPT2TokenizerFast",
            bos_token_id=None,
            eos_token_id=50256,
            num_merges=50000,
            merge_rule_hash="aaaa",
            all_merges_str="a b",
            pct_byte_tokens=0.005,
            special_token_ids={"pad": None, "unk": None, "mask": None},
            script_distribution={"Latin": 1.0},
        )
        fb = _make_tfv(
            vocab_size=32000,
            tokenizer_class="LlamaTokenizerFast",
            bos_token_id=1,
            eos_token_id=2,
            num_merges=30000,
            merge_rule_hash="bbbb",
            all_merges_str="c d",
            pct_byte_tokens=0.0,
            special_token_ids={"pad": 0, "unk": 1, "mask": 2},
            script_distribution={"CJK": 1.0},
        )
        score = tfv_similarity(fa, fb)
        assert score < 0.5

    def test_no_merges_both(self) -> None:
        fa = _make_tfv(
            num_merges=0,
            merge_rule_hash="",
            all_merges_str="",
        )
        fb = _make_tfv(
            num_merges=0,
            merge_rule_hash="",
            all_merges_str="",
        )
        score = tfv_similarity(fa, fb)
        assert score > 0.5

    def test_score_in_range(self) -> None:
        fa = _make_tfv()
        fb = _make_tfv(vocab_size=40000)
        score = tfv_similarity(fa, fb)
        assert 0.0 <= score <= 1.0


# ── vocab_overlap (offline, pre-built vocab sets) ──────────────────


class TestVocabOverlap:
    def test_identical_sets(self) -> None:
        v = {"hello", "world", "test"}
        result = vocab_overlap("a", "b", vocab_a=v, vocab_b=v)
        assert isinstance(result, VocabOverlap)
        assert result.jaccard == 1.0
        assert result.intersection == 3
        assert result.only_a == 0
        assert result.only_b == 0
        assert result.overlap_pct_a == 1.0
        assert result.overlap_pct_b == 1.0

    def test_disjoint_sets(self) -> None:
        result = vocab_overlap("a", "b", vocab_a={"x"}, vocab_b={"y"})
        assert result.jaccard == 0.0
        assert result.only_a == 1
        assert result.only_b == 1
        assert result.intersection == 0

    def test_partial_overlap(self) -> None:
        result = vocab_overlap(
            "a",
            "b",
            vocab_a={"a", "b", "c"},
            vocab_b={"b", "c", "d"},
        )
        assert result.intersection == 2
        assert result.union == 4
        assert result.jaccard == 0.5
        assert result.only_a == 1
        assert result.only_b == 1

    def test_empty_sets(self) -> None:
        result = vocab_overlap(
            "a",
            "b",
            vocab_a=set(),
            vocab_b=set(),
        )
        assert result.jaccard == 0.0
        assert result.vocab_a_size == 0

    def test_subset(self) -> None:
        result = vocab_overlap(
            "a",
            "b",
            vocab_a={"a", "b"},
            vocab_b={"a", "b", "c"},
        )
        assert result.overlap_pct_a == 1.0
        assert result.overlap_pct_b == round(2 / 3, 4)


# ── extract_tokenizer_features (online, slow) ─────────────────────


@pytest.mark.slow
class TestExtractTokenizerFeaturesOnline:
    def test_gpt2_golden(self) -> None:
        from provenancekit.core.signals.tokenizer import (
            extract_tokenizer_features,
        )

        feats = extract_tokenizer_features("gpt2")
        assert isinstance(feats, TokenizerFeatures)
        assert feats.vocab_size == 50257
        assert feats.tokenizer_class in ("GPT2Tokenizer", "GPT2TokenizerFast")
        assert feats.num_merges == 50000
        assert len(feats.merge_rule_hash) == 64
        assert feats.eos_token_id == 50256
        assert feats.bos_token_id is not None or feats.bos_token_id is None
        assert feats.pct_single_char > 0
        assert "Latin" in feats.script_distribution

    def test_tfv_self_similarity(self) -> None:
        from provenancekit.core.signals.tokenizer import (
            extract_tokenizer_features,
        )

        feats = extract_tokenizer_features("gpt2")
        assert tfv_similarity(feats, feats) == 1.0


@pytest.mark.slow
class TestVocabOverlapOnline:
    def test_identical_model(self) -> None:
        result = vocab_overlap("gpt2", "gpt2")
        assert isinstance(result, VocabOverlap)
        assert result.jaccard == 1.0
