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

"""Tests for core/signals/anchors.py — anchor token selection."""

from typing import Any

import pytest

from provenancekit.core.signals.anchors import (
    ANCHOR_LATIN,
    ANCHOR_PUNCT,
    SCRIPT_ANCHORS,
    get_anchor_ids,
)

# ── Mock tokenizer for offline tests ──────────────────────────────


class _MockTokenizer:
    """Minimal mock that maps single tokens to deterministic IDs."""

    def __init__(self, vocab_size: int = 50000) -> None:
        self._vocab_size = vocab_size

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,  # noqa: ARG002
    ) -> list[int]:
        return [hash(text) % self._vocab_size]


# ── Constants sanity checks ────────────────────────────────────────


class TestConstants:
    def test_anchor_punct_not_empty(self) -> None:
        assert len(ANCHOR_PUNCT) >= 10

    def test_anchor_latin_not_empty(self) -> None:
        assert len(ANCHOR_LATIN) >= 20

    def test_script_anchors_has_expected_keys(self) -> None:
        assert set(SCRIPT_ANCHORS.keys()) == {
            "Latin",
            "CJK",
            "Cyrillic",
            "Arabic",
            "Devanagari",
        }

    def test_all_script_anchor_lists_non_empty(self) -> None:
        for script, tokens in SCRIPT_ANCHORS.items():
            assert len(tokens) > 0, f"{script} anchor list is empty"


# ── get_anchor_ids (offline) ──────────────────────────────────────


class TestGetAnchorIds:
    def test_returns_correct_count(self) -> None:
        tok = _MockTokenizer()
        ids = get_anchor_ids(tok, ["hello", "world"], 50000, anchor_k=64)
        assert len(ids) == 64

    def test_deterministic(self) -> None:
        tok = _MockTokenizer()
        ids_1 = get_anchor_ids(tok, ["hello"], 50000, anchor_k=64)
        ids_2 = get_anchor_ids(tok, ["hello"], 50000, anchor_k=64)
        assert ids_1 == ids_2

    def test_no_tokenizer_random_fallback(self) -> None:
        ids = get_anchor_ids(None, None, vocab_size=50000, anchor_k=64)
        assert len(ids) == 64

    def test_custom_k(self) -> None:
        tok = _MockTokenizer()
        ids = get_anchor_ids(tok, ["hello"], 50000, anchor_k=32)
        assert len(ids) == 32

    def test_ids_within_vocab_bounds(self) -> None:
        tok = _MockTokenizer(vocab_size=1000)
        ids = get_anchor_ids(tok, ["hello"], vocab_size=1000, anchor_k=64)
        assert all(0 <= i < 1000 for i in ids)

    def test_no_duplicates(self) -> None:
        tok = _MockTokenizer()
        ids = get_anchor_ids(tok, ["hello", "world"], 50000, anchor_k=64)
        assert len(ids) == len(set(ids))

    def test_small_vocab(self) -> None:
        tok = _MockTokenizer(vocab_size=100)
        ids = get_anchor_ids(tok, ["hi"], vocab_size=100, anchor_k=64)
        assert len(ids) <= 64
        assert all(0 <= i < 100 for i in ids)

    def test_empty_vocab_list(self) -> None:
        tok = _MockTokenizer()
        ids = get_anchor_ids(tok, [], 50000, anchor_k=64)
        assert len(ids) == 64

    def test_defaults_to_settings_anchor_k(self) -> None:
        tok = _MockTokenizer()
        ids = get_anchor_ids(tok, ["hello"], 50000)
        assert len(ids) == 64  # Settings().anchor_k default


# ── _vocab_script_distribution (indirectly via get_anchor_ids) ────


class TestScriptDistribution:
    def test_latin_vocab_uses_latin_anchors(self) -> None:
        tok = _MockTokenizer()
        latin_tokens = ["hello", "world", "test", "code", "python"]
        ids = get_anchor_ids(tok, latin_tokens, 50000, anchor_k=64)
        assert len(ids) == 64

    def test_none_vocab_defaults_to_latin(self) -> None:
        tok = _MockTokenizer()
        ids = get_anchor_ids(tok, None, 50000, anchor_k=64)
        assert len(ids) == 64


# ── Online golden test ─────────────────────────────────────────────


@pytest.mark.slow
class TestAnchorIdsOnline:
    def test_gpt2_real_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        tok: Any = AutoTokenizer.from_pretrained("gpt2")
        vocab = set(tok.get_vocab().keys())
        ids = get_anchor_ids(tok, vocab, vocab_size=50257, anchor_k=64)
        assert len(ids) == 64
        assert all(0 <= i < 50257 for i in ids)
        assert len(ids) == len(set(ids))
