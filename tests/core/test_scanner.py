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

"""Tests for provenancekit.core.scanner — orchestrator unit + integration."""

from contextlib import ExitStack
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from provenancekit.config.constants import EXTRACTOR_VERSION
from provenancekit.config.settings import Settings
from provenancekit.core.results.formatters import format_plain
from provenancekit.core.scanner import ModelProvenanceScanner
from provenancekit.exceptions import ExtractionError, ModelLoadError
from provenancekit.models.results import (
    CachedEntry,
    CompareResult,
    LoadResult,
    LoadStrategy,
    SignalScores,
)
from provenancekit.models.signals import (
    ConfigInconsistency,
    MFIFingerprint,
    MFISimilarity,
    TokenizerFeatures,
    VocabOverlap,
    WeightSignalFeatures,
)
from provenancekit.services.cache import CacheService
from provenancekit.services.database import DatabaseError

# ── Fixtures ──────────────────────────────────────────────────────


def _fake_fp(model_type: str = "gpt2") -> MFIFingerprint:
    return MFIFingerprint(
        model_type=model_type,
        architectures=["GPT2LMHeadModel"],
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        intermediate_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_act="gelu_new",
        rope_theta=None,
        rope_scaling=None,
        tie_word_embeddings=True,
        bos_token_id=50256,
        eos_token_id=50256,
        gqa_ratio=1.0,
        attention_style="mha",
        norm_type="layer_norm",
        attention_bias=True,
        qk_norm=False,
        pos_encoding="learned",
        tokenizer_hash="abc",
        token_id_signature="def",
        arch_hash="ghi",
        family_hash="jkl",
    )


def _fake_tfv() -> TokenizerFeatures:
    return TokenizerFeatures(
        vocab_size=50257,
        tokenizer_class="GPT2Tokenizer",
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=None,
        num_added_tokens=0,
        num_special_tokens=3,
        num_merges=50000,
        first_5_merges=["Ġ t", "Ġ a", "h e", "i n", "r e"],
        merge_rule_hash="mrh",
        all_merges_str="merges",
        special_token_ids={"bos": 50256, "eos": 50256},
        pct_single_char=0.05,
        avg_token_length=4.2,
        max_token_length=20,
        pct_whitespace_prefix=0.6,
        pct_byte_tokens=0.01,
        script_distribution={"Latin": 0.9, "Other": 0.1},
    )


def _fake_ws(
    config_inconsistencies: list[ConfigInconsistency] | None = None,
) -> WeightSignalFeatures:
    return WeightSignalFeatures(
        hidden_size=768,
        num_layers=12,
        eas_self_sim=np.eye(64, dtype=np.float64),
        eas_anchor_count=64,
        nlf_vector=np.ones(12, dtype=np.float64),
        nlf_mode="vector",
        nlf_num_layers=12,
        lep_profile=np.ones(12, dtype=np.float64),
        end_histogram=np.ones(50, dtype=np.float64),
        wsp_signature=np.ones(12, dtype=np.float64),
        wvc_layer_sigs={i: np.ones(100, dtype=np.float64) for i in range(12)},
        config_inconsistencies=config_inconsistencies,
    )


_PATCH_BASE = "provenancekit.core.scanner"


# ── Offline unit tests ────────────────────────────────────────────


class TestCompareUnit:
    """Unit tests with all extraction mocked."""

    @pytest.fixture(autouse=True)
    def _patch_all(self, tmp_path: Any) -> Any:  # noqa: ANN401
        fp = _fake_fp()
        tfv = _fake_tfv()
        ws = _fake_ws()

        patches = {
            "extract_fingerprint": patch(
                f"{_PATCH_BASE}.extract_fingerprint",
                return_value=(fp, MagicMock()),
            ),
            "extract_tokenizer_features": patch(
                f"{_PATCH_BASE}.extract_tokenizer_features",
                return_value=tfv,
            ),
            "mfi_similarity": patch(
                f"{_PATCH_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            "classify": patch(
                f"{_PATCH_BASE}.classify",
                return_value=("gpt2", 1.0),
            ),
            "tfv_similarity": patch(
                f"{_PATCH_BASE}.tfv_similarity",
                return_value=1.0,
            ),
            "vocab_overlap": patch(
                f"{_PATCH_BASE}.vocab_overlap",
                return_value=VocabOverlap(
                    jaccard=1.0,
                    vocab_a_size=50257,
                    vocab_b_size=50257,
                    intersection=50257,
                    union=50257,
                    only_a=0,
                    only_b=0,
                    overlap_pct_a=1.0,
                    overlap_pct_b=1.0,
                ),
            ),
            "load_state_dict": patch(
                f"{_PATCH_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            "extract_signals": patch(
                f"{_PATCH_BASE}.extract_signals",
                return_value=ws,
            ),
            "eas_similarity": patch(
                f"{_PATCH_BASE}.eas_similarity",
                return_value=1.0,
            ),
            "nlf_similarity": patch(
                f"{_PATCH_BASE}.nlf_similarity",
                return_value=1.0,
            ),
            "lep_similarity": patch(
                f"{_PATCH_BASE}.lep_similarity",
                return_value=1.0,
            ),
            "end_similarity": patch(
                f"{_PATCH_BASE}.end_similarity",
                return_value=1.0,
            ),
            "wvc_similarity": patch(
                f"{_PATCH_BASE}.wvc_similarity",
                return_value=1.0,
            ),
        }

        self._mocks: dict[str, Any] = {}
        for name, p in patches.items():
            self._mocks[name] = p.start()

        self._cache = CacheService(cache_dir=tmp_path)
        self._scanner = ModelProvenanceScanner(cache=self._cache)

        yield

        for p in patches.values():
            p.stop()

    def test_returns_compare_result(self) -> None:
        result = self._scanner.compare("gpt2", "gpt2")
        assert isinstance(result, CompareResult)

    def test_mfi_tier1_with_agreeing_weights_confirms(self) -> None:
        result = self._scanner.compare("gpt2", "gpt2")
        assert result.scores.pipeline_score == result.scores.identity_score == 1.0
        assert result.scores.provenance_decision == "Confirmed Match"
        assert result.scores.verification == "weights_agree"
        assert result.scores.mfi_tier == 1

    def test_mfi_tier1_with_conflicting_weights_not_matched(self) -> None:
        for name in ("eas", "nlf", "lep", "end", "wvc"):
            self._mocks[f"{name}_similarity"].return_value = 0.5
        result = self._scanner.compare("bert", "biomedbert")
        assert result.scores.mfi_tier == 1
        assert result.scores.pipeline_score == 0.5
        assert result.scores.provenance_decision == (
            "Not Matched (metadata/weight conflict)"
        )
        assert result.scores.verification == "weights_conflict"

    def test_mfi_tier1_without_weights_is_unverified(self) -> None:
        self._mocks["extract_signals"].return_value = None
        for name in ("eas", "nlf", "lep", "end", "wvc"):
            self._mocks[f"{name}_similarity"].return_value = float("nan")
        result = self._scanner.compare("bart", "distilbart")
        assert result.scores.identity_score is None
        assert result.scores.pipeline_score == 1.0
        assert result.scores.provenance_decision == "Metadata Match (unverified)"
        assert result.scores.verification == "unverified"

    def test_mfi_tier3_uses_identity_score(self) -> None:
        self._mocks["mfi_similarity"].return_value = MFISimilarity(
            score=0.3, tier=3, match_type="soft_match"
        )
        result = self._scanner.compare("model_a", "model_b")
        assert result.scores.mfi_tier == 3
        assert result.scores.pipeline_score == result.scores.identity_score

    def test_shallow_guard_caps_pipeline_but_retains_identity(self) -> None:
        self._mocks["mfi_similarity"].return_value = MFISimilarity(
            score=0.3, tier=3, match_type="soft_match"
        )
        self._mocks["tfv_similarity"].return_value = 0.4
        for name in ("eas", "nlf", "end"):
            self._mocks[f"{name}_similarity"].return_value = 0.8
        for name in ("lep", "wvc"):
            self._mocks[f"{name}_similarity"].return_value = float("nan")

        result = self._scanner.compare("model_a", "model_b")

        assert result.scores.identity_score == 0.8
        assert result.scores.pipeline_score == 0.75
        assert result.scores.verification == "weights_shallow"
        assert result.scores.provenance_decision == (
            "Weak Match (unverified, no deep weight signals)"
        )

    def test_assembles_all_signal_fields(self) -> None:
        result = self._scanner.compare("gpt2", "gpt2")
        signals = result.signals
        assert isinstance(signals, SignalScores)
        for field in ("eas", "nlf", "lep", "end", "wvc", "tfv", "voa"):
            val = getattr(signals, field)
            assert val is not None, f"{field} should not be None"

    def test_family_fields_populated(self) -> None:
        result = self._scanner.compare("gpt2", "gpt2")
        assert result.family_a == "gpt2"
        assert result.family_b == "gpt2"

    def test_time_seconds_positive(self) -> None:
        result = self._scanner.compare("gpt2", "gpt2")
        assert result.time_seconds >= 0.0

    def test_interpretation_populated(self) -> None:
        result = self._scanner.compare("gpt2", "gpt2")
        assert result.interpretation.label != ""
        assert result.interpretation.colour.startswith("#")

    def test_consistent_configs_keep_metadata_tiers(self) -> None:
        self._mocks["extract_signals"].return_value = _fake_ws(
            config_inconsistencies=[]
        )
        result = self._scanner.compare("model_a", "model_b")
        _, kwargs = self._mocks["mfi_similarity"].call_args
        assert kwargs["metadata_trusted"] is True
        assert result.config_inconsistent == []

    def test_inconsistent_config_disables_metadata_tiers(self) -> None:
        finding = ConfigInconsistency(
            field="vocab_size", declared=28996, observed=[30522]
        )
        self._mocks["extract_signals"].return_value = _fake_ws(
            config_inconsistencies=[finding]
        )
        result = self._scanner.compare("model_a", "model_b")
        _, kwargs = self._mocks["mfi_similarity"].call_args
        assert kwargs["metadata_trusted"] is False
        assert [(c.model, c.field) for c in result.config_inconsistent] == [
            ("model_a", "vocab_size"),
            ("model_b", "vocab_size"),
        ]
        assert "config_inconsistent: model_a: vocab_size" in format_plain(result)


class TestCompareWeightFailure:
    """Weight extraction failure produces NaN signals but doesn't crash."""

    def test_weight_failure_yields_nan_signals(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        tfv = _fake_tfv()

        with (
            patch(
                f"{_PATCH_BASE}.extract_fingerprint",
                return_value=(fp, MagicMock()),
            ),
            patch(
                f"{_PATCH_BASE}.extract_tokenizer_features",
                return_value=tfv,
            ),
            patch(
                f"{_PATCH_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=0.3, tier=3, match_type="soft_match"),
            ),
            patch(
                f"{_PATCH_BASE}.classify",
                return_value=("gpt2", 1.0),
            ),
            patch(
                f"{_PATCH_BASE}.tfv_similarity",
                return_value=0.9,
            ),
            patch(
                f"{_PATCH_BASE}.vocab_overlap",
                return_value=VocabOverlap(
                    jaccard=0.8,
                    vocab_a_size=100,
                    vocab_b_size=100,
                    intersection=80,
                    union=120,
                    only_a=20,
                    only_b=20,
                    overlap_pct_a=0.8,
                    overlap_pct_b=0.8,
                ),
            ),
            patch(
                f"{_PATCH_BASE}.load_state_dict",
                side_effect=ModelLoadError("download failed"),
            ),
        ):
            scanner = ModelProvenanceScanner(cache=CacheService(cache_dir=tmp_path))
            result = scanner.compare("bad/model", "bad/model")

            assert result.signals.eas is None
            assert result.signals.nlf is None
            assert result.signals.lep is None
            assert result.signals.end is None
            assert result.signals.wvc is None

            assert result.scores.identity_score is None


class TestScanIntegration:
    @pytest.mark.xfail(
        raises=DatabaseError,
        strict=True,
        reason="bundled database not yet rebuilt for the current extractor version",
    )
    def test_scan_returns_scan_result(self, tmp_path: Any) -> None:
        scanner = ModelProvenanceScanner(cache=CacheService(cache_dir=tmp_path))
        result = scanner.scan("gpt2", top_k=1, threshold=0.5)
        assert result.model_info.model_type == "gpt2"
        assert result.model_info.param_bucket == "<=1B"
        assert result.match_count >= 0
        assert result.elapsed_ms > 0


class TestCacheIntegration:
    """Verify the scanner uses cache on second call."""

    def test_second_call_uses_cache(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        tfv = _fake_tfv()

        mock_extract_fp = MagicMock(return_value=(fp, MagicMock()))
        mock_extract_tfv = MagicMock(return_value=tfv)

        with (
            patch(f"{_PATCH_BASE}.extract_fingerprint", mock_extract_fp),
            patch(f"{_PATCH_BASE}.extract_tokenizer_features", mock_extract_tfv),
            patch(
                f"{_PATCH_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_PATCH_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_PATCH_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_PATCH_BASE}.vocab_overlap",
                return_value=VocabOverlap(
                    jaccard=1.0,
                    vocab_a_size=100,
                    vocab_b_size=100,
                    intersection=100,
                    union=100,
                    only_a=0,
                    only_b=0,
                    overlap_pct_a=1.0,
                    overlap_pct_b=1.0,
                ),
            ),
            patch(
                f"{_PATCH_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(f"{_PATCH_BASE}.extract_signals", return_value=_fake_ws()),
            patch(f"{_PATCH_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.end_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.wvc_similarity", return_value=1.0),
        ):
            cache = CacheService(cache_dir=tmp_path)
            scanner = ModelProvenanceScanner(cache=cache)

            scanner.compare("model_a", "model_b")
            assert mock_extract_fp.call_count == 2

            scanner.compare("model_a", "model_b")
            assert mock_extract_fp.call_count == 2


# ── Streaming strategy path ───────────────────────────────────────


class TestCompareStreamingStrategy:
    def test_streaming_calls_extract_signals_streaming(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        tfv = _fake_tfv()
        ws = _fake_ws()

        mock_streaming = MagicMock(return_value=ws)

        with (
            patch(
                f"{_PATCH_BASE}.extract_fingerprint",
                return_value=(fp, MagicMock()),
            ),
            patch(
                f"{_PATCH_BASE}.extract_tokenizer_features",
                return_value=tfv,
            ),
            patch(
                f"{_PATCH_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_PATCH_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_PATCH_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_PATCH_BASE}.vocab_overlap",
                return_value=VocabOverlap(
                    jaccard=1.0,
                    vocab_a_size=100,
                    vocab_b_size=100,
                    intersection=100,
                    union=100,
                    only_a=0,
                    only_b=0,
                    overlap_pct_a=1.0,
                    overlap_pct_b=1.0,
                ),
            ),
            patch(
                f"{_PATCH_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict=None,
                    config=None,
                    strategy=LoadStrategy.streaming,
                    source="too large",
                ),
            ),
            patch(f"{_PATCH_BASE}.extract_signals_streaming", mock_streaming),
            patch(f"{_PATCH_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.end_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.wvc_similarity", return_value=1.0),
        ):
            scanner = ModelProvenanceScanner(cache=CacheService(cache_dir=tmp_path))
            result = scanner.compare("big/model", "big/model")
            assert mock_streaming.call_count >= 1
            assert isinstance(result, CompareResult)


# ── Signal extraction failure (not load failure) ─────────────────


class TestSignalExtractionFailure:
    def test_extract_signals_crash_returns_nan(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        tfv = _fake_tfv()

        with (
            patch(
                f"{_PATCH_BASE}.extract_fingerprint",
                return_value=(fp, MagicMock()),
            ),
            patch(
                f"{_PATCH_BASE}.extract_tokenizer_features",
                return_value=tfv,
            ),
            patch(
                f"{_PATCH_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=0.3, tier=3, match_type="soft_match"),
            ),
            patch(f"{_PATCH_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_PATCH_BASE}.tfv_similarity", return_value=0.9),
            patch(
                f"{_PATCH_BASE}.vocab_overlap",
                return_value=VocabOverlap(
                    jaccard=0.8,
                    vocab_a_size=100,
                    vocab_b_size=100,
                    intersection=80,
                    union=120,
                    only_a=20,
                    only_b=20,
                    overlap_pct_a=0.8,
                    overlap_pct_b=0.8,
                ),
            ),
            patch(
                f"{_PATCH_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(
                f"{_PATCH_BASE}.extract_signals",
                side_effect=ExtractionError("OOM"),
            ),
        ):
            scanner = ModelProvenanceScanner(cache=CacheService(cache_dir=tmp_path))
            result = scanner.compare("oom/model", "oom/model")
            assert result.signals.eas is None
            assert result.scores.identity_score is None


class TestGetVocab:
    def test_bytes_vocab_keys_are_decoded(self) -> None:
        tok = MagicMock()
        tok.get_vocab.return_value = {b"the": 0, "of": 1, b"\xe7\x9a\x84": 2}
        assert ModelProvenanceScanner._get_vocab(tok) == ["the", "of", "\u7684"]


# ── Partial cache (mfi set, tfv missing) ─────────────────────────


class TestPartialCache:
    def test_partial_cache_re_extracts(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        tfv = _fake_tfv()

        cache = CacheService(cache_dir=tmp_path)
        cache.put(
            "gpt2",
            CachedEntry(model_id="gpt2", mfi=fp.model_dump(), tfv=None),
        )

        mock_extract_fp = MagicMock(return_value=(fp, MagicMock()))

        with (
            patch(f"{_PATCH_BASE}.extract_fingerprint", mock_extract_fp),
            patch(
                f"{_PATCH_BASE}.extract_tokenizer_features",
                return_value=tfv,
            ),
            patch(
                f"{_PATCH_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_PATCH_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_PATCH_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_PATCH_BASE}.vocab_overlap",
                return_value=VocabOverlap(
                    jaccard=1.0,
                    vocab_a_size=100,
                    vocab_b_size=100,
                    intersection=100,
                    union=100,
                    only_a=0,
                    only_b=0,
                    overlap_pct_a=1.0,
                    overlap_pct_b=1.0,
                ),
            ),
            patch(
                f"{_PATCH_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(f"{_PATCH_BASE}.extract_signals", return_value=_fake_ws()),
            patch(f"{_PATCH_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.end_similarity", return_value=1.0),
            patch(f"{_PATCH_BASE}.wvc_similarity", return_value=1.0),
        ):
            scanner = ModelProvenanceScanner(cache=cache)
            result = scanner.compare("gpt2", "gpt2")
            assert mock_extract_fp.call_count == 1
            assert isinstance(result, CompareResult)


# ── Extractor version and extraction settings ────────────────────


def _patch_extraction(stack: ExitStack, fp: MFIFingerprint) -> MagicMock:
    mock_extract_fp = MagicMock(return_value=(fp, MagicMock()))
    stack.enter_context(patch(f"{_PATCH_BASE}.extract_fingerprint", mock_extract_fp))
    stack.enter_context(
        patch(f"{_PATCH_BASE}.extract_tokenizer_features", return_value=_fake_tfv())
    )
    stack.enter_context(
        patch(
            f"{_PATCH_BASE}.load_state_dict",
            return_value=LoadResult(
                state_dict={"w": torch.zeros(2, dtype=torch.bfloat16)},
                config=None,
                strategy=LoadStrategy.full,
                source="safetensors_single",
            ),
        )
    )
    stack.enter_context(
        patch(f"{_PATCH_BASE}.extract_signals", return_value=_fake_ws())
    )
    stack.enter_context(patch(f"{_PATCH_BASE}.AutoTokenizer", MagicMock()))
    return mock_extract_fp


class TestExtractionStamp:
    def test_new_entry_records_params_and_loader(self, tmp_path: Any) -> None:
        cache = CacheService(cache_dir=tmp_path)
        with ExitStack() as stack:
            _patch_extraction(stack, _fake_fp())
            ModelProvenanceScanner(cache=cache)._extract_model("gpt2")

        entry = CacheService(cache_dir=tmp_path).get("gpt2")
        assert entry is not None
        assert entry.extractor_version == EXTRACTOR_VERSION
        assert entry.extraction_params == Settings().extraction_params()
        assert entry.ws_loader == {
            "strategy": "full",
            "source": "safetensors_single",
            "dtype": "torch.bfloat16",
        }

    def test_changed_settings_force_re_extraction(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        cache = CacheService(cache_dir=tmp_path)
        cache.put(
            "gpt2",
            CachedEntry(
                model_id="gpt2",
                extraction_params={"anchor_k": 32, "wvc_subsample": 4096},
                mfi=fp.model_dump(),
                tfv=_fake_tfv().model_dump(),
                ws=_fake_ws().to_cache_dict(),
            ),
        )
        with ExitStack() as stack:
            mock_extract_fp = _patch_extraction(stack, fp)
            data = ModelProvenanceScanner(cache=cache)._extract_model("gpt2")

        assert mock_extract_fp.call_count == 1
        assert not data.ws_cache_hit
        entry = cache.get("gpt2")
        assert entry is not None
        assert entry.extraction_params == Settings().extraction_params()

    def test_matching_settings_use_cache(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        cache = CacheService(cache_dir=tmp_path)
        cache.put(
            "gpt2",
            CachedEntry(
                model_id="gpt2",
                extraction_params=Settings().extraction_params(),
                mfi=fp.model_dump(),
                tfv=_fake_tfv().model_dump(),
                ws=_fake_ws().to_cache_dict(),
            ),
        )
        with ExitStack() as stack:
            mock_extract_fp = _patch_extraction(stack, fp)
            data = ModelProvenanceScanner(cache=cache)._extract_model("gpt2")

        assert mock_extract_fp.call_count == 0
        assert data.ws_cache_hit

    def test_scan_rejects_stale_database_before_extraction(self, tmp_path: Any) -> None:
        scanner = ModelProvenanceScanner(cache=CacheService(cache_dir=tmp_path))
        db = MagicMock()
        db.check_compatibility.side_effect = DatabaseError("stale")
        scanner._db_service = db
        with ExitStack() as stack:
            mock_extract_fp = _patch_extraction(stack, _fake_fp())
            with pytest.raises(DatabaseError, match="stale"):
                scanner.scan("gpt2")
        assert mock_extract_fp.call_count == 0


# ── Online integration tests ──────────────────────────────────────


@pytest.mark.slow
class TestScannerOnline:
    def test_compare_gpt2_self(self, tmp_path: Any) -> None:
        scanner = ModelProvenanceScanner(cache=CacheService(cache_dir=tmp_path))
        result = scanner.compare("gpt2", "gpt2")

        assert isinstance(result, CompareResult)
        assert result.scores.mfi_tier == 1
        assert result.scores.pipeline_score == 1.0
        assert result.scores.provenance_decision == "Confirmed Match"
        assert result.family_a == "gpt2"
        assert result.family_b == "gpt2"
        assert result.time_seconds > 0
