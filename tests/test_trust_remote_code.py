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

"""Tests for the trust_remote_code flag across Settings, CLI, scanner, and loaders.

Verifies that trust_remote_code defaults to False for security and can be
explicitly enabled via CLI flags, environment variables, or Settings.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from provenancekit.config.settings import Settings
from provenancekit.models.results import (
    CachedEntry,
    LoadResult,
    LoadStrategy,
)
from provenancekit.models.signals import (
    MFIFingerprint,
    MFISimilarity,
    TokenizerFeatures,
    VocabOverlap,
    WeightSignalFeatures,
)

# ── Helpers ───────────────────────────────────────────────────────


def _fake_fp() -> MFIFingerprint:
    return MFIFingerprint(
        model_type="gpt2",
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


def _fake_ws() -> WeightSignalFeatures:
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
    )


# ── 1. Settings defaults and overrides ────────────────────────────


class TestSettingsTrustRemoteCode:
    def test_default_is_false(self) -> None:
        s = Settings()
        assert s.trust_remote_code is False

    def test_explicit_true(self) -> None:
        s = Settings(trust_remote_code=True)
        assert s.trust_remote_code is True

    def test_explicit_false(self) -> None:
        s = Settings(trust_remote_code=False)
        assert s.trust_remote_code is False

    def test_env_override_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROVENANCEKIT_TRUST_REMOTE_CODE", "true")
        s = Settings()
        assert s.trust_remote_code is True

    def test_env_override_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROVENANCEKIT_TRUST_REMOTE_CODE", "false")
        s = Settings()
        assert s.trust_remote_code is False


# ── 2. CLI flag parsing ───────────────────────────────────────────


_SCANNER_PATH = "provenancekit.core.scanner.ModelProvenanceScanner"
_CACHE_PATH = "provenancekit.services.cache.CacheService"
_SETTINGS_PATH = "provenancekit.cli.Settings"


def _run_cli(argv: list[str]) -> tuple[str, str, int]:
    """Invoke ``main()`` with *argv* and capture stdout, stderr, exit code."""
    import sys
    from io import StringIO

    from provenancekit.cli import main

    old_argv = sys.argv
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buf, stderr_buf = StringIO(), StringIO()
    exit_code = 0
    try:
        sys.argv = ["provenancekit", *argv]
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        main()
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return stdout_buf.getvalue(), stderr_buf.getvalue(), exit_code


class TestCliTrustRemoteCode:
    def test_compare_without_flag_defaults_false(self) -> None:
        mock_scanner = MagicMock()
        mock_settings_cls = MagicMock()

        from provenancekit.models.results import (
            CompareResult,
            PipelineScore,
            ScoreInterpretation,
            SignalScores,
            TimingBreakdown,
        )

        mock_scanner.return_value.compare.return_value = CompareResult(
            model_a="a",
            model_b="b",
            family_a="gpt2",
            family_b="gpt2",
            signals=SignalScores(
                eas=1.0, nlf=1.0, lep=1.0, end=1.0, wvc=1.0, tfv=1.0, voa=1.0
            ),
            scores=PipelineScore(
                mfi_score=1.0,
                mfi_tier=1,
                mfi_match="exact",
                identity_score=1.0,
                tokenizer_score=1.0,
                pipeline_score=1.0,
                provenance_decision="Confirmed Match",
            ),
            interpretation=ScoreInterpretation(label="High-Confidence Match", colour="#2ecc71"),
            time_seconds=0.1,
            timing=TimingBreakdown(
                total_seconds=0.1,
                metadata_extract_seconds=0.05,
                weight_feature_extract_seconds=0.05,
                cache_hit="False",
            ),
        )

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH, mock_settings_cls),
        ):
            _, _, code = _run_cli(["compare", "a", "b", "--json"])

        assert code == 0
        settings_kwargs = mock_settings_cls.call_args
        if settings_kwargs[1]:
            assert settings_kwargs[1].get("trust_remote_code") is not True

    def test_compare_with_flag_sets_true(self) -> None:
        mock_scanner = MagicMock()
        mock_settings_cls = MagicMock()

        from provenancekit.models.results import (
            CompareResult,
            PipelineScore,
            ScoreInterpretation,
            SignalScores,
            TimingBreakdown,
        )

        mock_scanner.return_value.compare.return_value = CompareResult(
            model_a="a",
            model_b="b",
            family_a="gpt2",
            family_b="gpt2",
            signals=SignalScores(
                eas=1.0, nlf=1.0, lep=1.0, end=1.0, wvc=1.0, tfv=1.0, voa=1.0
            ),
            scores=PipelineScore(
                mfi_score=1.0,
                mfi_tier=1,
                mfi_match="exact",
                identity_score=1.0,
                tokenizer_score=1.0,
                pipeline_score=1.0,
                provenance_decision="Confirmed Match",
            ),
            interpretation=ScoreInterpretation(label="High-Confidence Match", colour="#2ecc71"),
            time_seconds=0.1,
            timing=TimingBreakdown(
                total_seconds=0.1,
                metadata_extract_seconds=0.05,
                weight_feature_extract_seconds=0.05,
                cache_hit="False",
            ),
        )

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH, mock_settings_cls),
        ):
            _, _, code = _run_cli(
                ["compare", "a", "b", "--json", "--trust-remote-code"]
            )

        assert code == 0
        settings_kwargs = mock_settings_cls.call_args[1]
        assert settings_kwargs.get("trust_remote_code") is True

    def test_scan_without_flag_defaults_false(self) -> None:
        mock_scanner = MagicMock()
        mock_settings_cls = MagicMock()

        from provenancekit.models.results import (
            ScanModelInfo,
            ScanResult,
        )

        mock_scanner.return_value.scan.return_value = ScanResult(
            model_info=ScanModelInfo(
                model_path="gpt2",
                model_type="gpt2",
                architectures=["GPT2LMHeadModel"],
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_key_value_heads=12,
                intermediate_size=3072,
                vocab_size=50257,
                tie_word_embeddings=True,
                hidden_act="gelu_new",
                arch_hash="abc",
                family_hash="def",
                param_bucket="<=1B",
                has_weight_signals=True,
            ),
            matches=[],
            match_count=0,
            elapsed_ms=50.0,
        )

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH, mock_settings_cls),
            patch("provenancekit.cli.has_deep_signals", return_value=True),
        ):
            _, _, code = _run_cli(["scan", "gpt2", "--json"])

        assert code == 0
        settings_kwargs = mock_settings_cls.call_args
        if settings_kwargs[1]:
            assert settings_kwargs[1].get("trust_remote_code") is not True

    def test_scan_with_flag_sets_true(self) -> None:
        mock_scanner = MagicMock()
        mock_settings_cls = MagicMock()

        from provenancekit.models.results import (
            ScanModelInfo,
            ScanResult,
        )

        mock_scanner.return_value.scan.return_value = ScanResult(
            model_info=ScanModelInfo(
                model_path="gpt2",
                model_type="gpt2",
                architectures=["GPT2LMHeadModel"],
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_key_value_heads=12,
                intermediate_size=3072,
                vocab_size=50257,
                tie_word_embeddings=True,
                hidden_act="gelu_new",
                arch_hash="abc",
                family_hash="def",
                param_bucket="<=1B",
                has_weight_signals=True,
            ),
            matches=[],
            match_count=0,
            elapsed_ms=50.0,
        )

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH, mock_settings_cls),
            patch("provenancekit.cli.has_deep_signals", return_value=True),
        ):
            _, _, code = _run_cli(["scan", "gpt2", "--json", "--trust-remote-code"])

        assert code == 0
        settings_kwargs = mock_settings_cls.call_args[1]
        assert settings_kwargs.get("trust_remote_code") is True


# ── 3. Scanner threads trust_remote_code to extraction ────────────

_SCAN_BASE = "provenancekit.core.scanner"


class TestScannerTrustRemoteCode:
    """Verify the scanner forwards settings.trust_remote_code to callees."""

    def _build_scanner(self, tmp_path: Any, trust: bool) -> Any:
        from provenancekit.core.scanner import ModelProvenanceScanner
        from provenancekit.services.cache import CacheService

        settings = Settings(trust_remote_code=trust)
        return ModelProvenanceScanner(
            settings=settings, cache=CacheService(cache_dir=tmp_path)
        )

    def test_extract_fingerprint_receives_flag_true(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        mock_extract_fp = MagicMock(return_value=(fp, MagicMock()))

        with (
            patch(f"{_SCAN_BASE}.extract_fingerprint", mock_extract_fp),
            patch(f"{_SCAN_BASE}.extract_tokenizer_features", return_value=_fake_tfv()),
            patch(
                f"{_SCAN_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_SCAN_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_SCAN_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_SCAN_BASE}.vocab_overlap",
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
                f"{_SCAN_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(f"{_SCAN_BASE}.extract_signals", return_value=_fake_ws()),
            patch(f"{_SCAN_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.end_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.wvc_similarity", return_value=1.0),
        ):
            scanner = self._build_scanner(tmp_path, trust=True)
            scanner.compare("a", "b")

            for c in mock_extract_fp.call_args_list:
                assert c[1].get("trust_remote_code") is True

    def test_extract_fingerprint_receives_flag_false(self, tmp_path: Any) -> None:
        fp = _fake_fp()
        mock_extract_fp = MagicMock(return_value=(fp, MagicMock()))

        with (
            patch(f"{_SCAN_BASE}.extract_fingerprint", mock_extract_fp),
            patch(f"{_SCAN_BASE}.extract_tokenizer_features", return_value=_fake_tfv()),
            patch(
                f"{_SCAN_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_SCAN_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_SCAN_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_SCAN_BASE}.vocab_overlap",
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
                f"{_SCAN_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(f"{_SCAN_BASE}.extract_signals", return_value=_fake_ws()),
            patch(f"{_SCAN_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.end_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.wvc_similarity", return_value=1.0),
        ):
            scanner = self._build_scanner(tmp_path, trust=False)
            scanner.compare("a", "b")

            for c in mock_extract_fp.call_args_list:
                assert c[1].get("trust_remote_code") is False

    def test_tokenizer_features_receives_flag(self, tmp_path: Any) -> None:
        mock_extract_tfv = MagicMock(return_value=_fake_tfv())

        with (
            patch(
                f"{_SCAN_BASE}.extract_fingerprint",
                return_value=(_fake_fp(), MagicMock()),
            ),
            patch(f"{_SCAN_BASE}.extract_tokenizer_features", mock_extract_tfv),
            patch(
                f"{_SCAN_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_SCAN_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_SCAN_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_SCAN_BASE}.vocab_overlap",
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
                f"{_SCAN_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(f"{_SCAN_BASE}.extract_signals", return_value=_fake_ws()),
            patch(f"{_SCAN_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.end_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.wvc_similarity", return_value=1.0),
        ):
            scanner = self._build_scanner(tmp_path, trust=True)
            scanner.compare("a", "b")

            for c in mock_extract_tfv.call_args_list:
                assert c[1].get("trust_remote_code") is True

    def test_cache_hit_tokenizer_receives_flag(self, tmp_path: Any) -> None:
        from provenancekit.core.scanner import ModelProvenanceScanner
        from provenancekit.services.cache import CacheService

        fp = _fake_fp()
        tfv = _fake_tfv()
        cache = CacheService(cache_dir=tmp_path)
        cache.put(
            "model_x",
            CachedEntry(
                model_id="model_x",
                mfi=fp.model_dump(),
                tfv=tfv.model_dump(),
            ),
        )

        mock_auto_tok = MagicMock()
        mock_auto_tok.from_pretrained.return_value = MagicMock()

        with (
            patch(f"{_SCAN_BASE}.AutoTokenizer", mock_auto_tok),
            patch(
                f"{_SCAN_BASE}.mfi_similarity",
                return_value=MFISimilarity(score=1.0, tier=1, match_type="exact"),
            ),
            patch(f"{_SCAN_BASE}.classify", return_value=("gpt2", 1.0)),
            patch(f"{_SCAN_BASE}.tfv_similarity", return_value=1.0),
            patch(
                f"{_SCAN_BASE}.vocab_overlap",
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
                f"{_SCAN_BASE}.load_state_dict",
                return_value=LoadResult(
                    state_dict={"w": "fake"},
                    config=None,
                    strategy=LoadStrategy.full,
                    source="mock",
                ),
            ),
            patch(f"{_SCAN_BASE}.extract_signals", return_value=_fake_ws()),
            patch(f"{_SCAN_BASE}.eas_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.nlf_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.lep_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.end_similarity", return_value=1.0),
            patch(f"{_SCAN_BASE}.wvc_similarity", return_value=1.0),
        ):
            settings = Settings(trust_remote_code=True)
            scanner = ModelProvenanceScanner(settings=settings, cache=cache)
            scanner.compare("model_x", "model_x")

            for c in mock_auto_tok.from_pretrained.call_args_list:
                assert c[1].get("trust_remote_code") is True


# ── 4. extract_fingerprint threads flag to AutoConfig/AutoTokenizer ─


class TestExtractFingerprintTrustRemoteCode:
    _MOD = "provenancekit.core.signals.metadata"

    @staticmethod
    def _make_mock_config() -> MagicMock:
        mock_config = MagicMock(
            spec=[
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
                "hidden_act",
                "head_dim",
                "rope_theta",
                "rope_scaling",
                "tie_word_embeddings",
                "position_embedding_type",
            ]
        )
        mock_config.model_type = "gpt2"
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.num_key_value_heads = None
        mock_config.intermediate_size = 3072
        mock_config.vocab_size = 50257
        mock_config.max_position_embeddings = 1024
        mock_config.hidden_act = "gelu_new"
        mock_config.head_dim = None
        mock_config.rope_theta = None
        mock_config.rope_scaling = None
        mock_config.tie_word_embeddings = True
        mock_config.position_embedding_type = "absolute"
        return mock_config

    @staticmethod
    def _make_mock_tokenizer() -> MagicMock:
        mock_tok = MagicMock()
        mock_tok.bos_token_id = 50256
        mock_tok.eos_token_id = 50256
        mock_tok.backend_tokenizer.to_str.return_value = "{}"
        mock_tok.get_vocab.return_value = {}
        return mock_tok

    def test_defaults_to_false(self) -> None:
        mock_config = self._make_mock_config()
        mock_tok = self._make_mock_tokenizer()

        mock_auto_cfg = MagicMock()
        mock_auto_cfg.from_pretrained.return_value = mock_config
        mock_auto_tok = MagicMock()
        mock_auto_tok.from_pretrained.return_value = mock_tok

        with (
            patch(f"{self._MOD}.AutoConfig", mock_auto_cfg),
            patch(f"{self._MOD}.AutoTokenizer", mock_auto_tok),
        ):
            from provenancekit.core.signals.metadata import extract_fingerprint

            extract_fingerprint("test/model")

            mock_auto_cfg.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=False
            )
            mock_auto_tok.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=False
            )

    def test_passes_true_when_set(self) -> None:
        mock_config = self._make_mock_config()
        mock_tok = self._make_mock_tokenizer()

        mock_auto_cfg = MagicMock()
        mock_auto_cfg.from_pretrained.return_value = mock_config
        mock_auto_tok = MagicMock()
        mock_auto_tok.from_pretrained.return_value = mock_tok

        with (
            patch(f"{self._MOD}.AutoConfig", mock_auto_cfg),
            patch(f"{self._MOD}.AutoTokenizer", mock_auto_tok),
        ):
            from provenancekit.core.signals.metadata import extract_fingerprint

            extract_fingerprint("test/model", trust_remote_code=True)

            mock_auto_cfg.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=True
            )
            mock_auto_tok.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=True
            )

    def test_skips_autoload_when_config_and_tokenizer_provided(self) -> None:
        mock_auto_cfg = MagicMock()
        mock_auto_tok = MagicMock()

        mock_config = self._make_mock_config()
        mock_tok = self._make_mock_tokenizer()

        with (
            patch(f"{self._MOD}.AutoConfig", mock_auto_cfg),
            patch(f"{self._MOD}.AutoTokenizer", mock_auto_tok),
        ):
            from provenancekit.core.signals.metadata import extract_fingerprint

            extract_fingerprint("test/model", config=mock_config, tokenizer=mock_tok)

            mock_auto_cfg.from_pretrained.assert_not_called()
            mock_auto_tok.from_pretrained.assert_not_called()


# ── 5. model_loader threads flag to AutoConfig / AutoModel ────────


class TestModelLoaderTrustRemoteCode:
    _MOD = "provenancekit.services.model_loader"

    def test_estimate_model_params_default_false(self) -> None:
        mock_auto_cfg = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.hidden_size = 768
        mock_cfg.num_hidden_layers = 12
        mock_cfg.num_attention_heads = 12
        mock_cfg.num_key_value_heads = None
        mock_cfg.head_dim = None
        mock_cfg.vocab_size = 50257
        mock_cfg.intermediate_size = 3072
        mock_cfg.tie_word_embeddings = None
        mock_cfg.hidden_act = "gelu_new"
        mock_cfg.model_type = "gpt2"
        mock_auto_cfg.from_pretrained.return_value = mock_cfg

        with patch(f"{self._MOD}.AutoConfig", mock_auto_cfg):
            from provenancekit.services.model_loader import estimate_model_params

            estimate_model_params("test/model")

            mock_auto_cfg.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=False
            )

    def test_estimate_model_params_with_true(self) -> None:
        mock_auto_cfg = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.hidden_size = 768
        mock_cfg.num_hidden_layers = 12
        mock_cfg.num_attention_heads = 12
        mock_cfg.num_key_value_heads = None
        mock_cfg.head_dim = None
        mock_cfg.vocab_size = 50257
        mock_cfg.intermediate_size = 3072
        mock_cfg.tie_word_embeddings = None
        mock_cfg.hidden_act = "gelu_new"
        mock_cfg.model_type = "gpt2"
        mock_auto_cfg.from_pretrained.return_value = mock_cfg

        with patch(f"{self._MOD}.AutoConfig", mock_auto_cfg):
            from provenancekit.services.model_loader import estimate_model_params

            estimate_model_params("test/model", trust_remote_code=True)

            mock_auto_cfg.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=True
            )

    def test_load_state_dict_threads_trust_to_config(self) -> None:
        mock_auto_cfg = MagicMock()
        mock_auto_cfg.from_pretrained.return_value = MagicMock()

        settings = Settings(trust_remote_code=True)

        with (
            patch(
                f"{self._MOD}.estimate_model_params",
                return_value=int(1e8),
            ),
            patch(f"{self._MOD}.AutoConfig", mock_auto_cfg),
            patch(
                f"{self._MOD}._try_safetensors",
                return_value=LoadResult(
                    state_dict={"w": MagicMock()},
                    config=MagicMock(),
                    strategy=LoadStrategy.full,
                    source="safetensors",
                ),
            ),
        ):
            from provenancekit.services.model_loader import load_state_dict

            load_state_dict("test/model", settings=settings)

            mock_auto_cfg.from_pretrained.assert_called_once_with(
                "test/model", trust_remote_code=True
            )

    def test_load_state_dict_threads_trust_to_automodel(self) -> None:
        mock_auto_cfg = MagicMock()
        mock_auto_cfg.from_pretrained.return_value = MagicMock()

        mock_auto_causal = MagicMock()
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"w": MagicMock()}
        mock_model.config = MagicMock()
        mock_auto_causal.from_pretrained.return_value = mock_model

        settings = Settings(trust_remote_code=True)

        with (
            patch(
                f"{self._MOD}.estimate_model_params",
                return_value=int(1e8),
            ),
            patch(f"{self._MOD}.AutoConfig", mock_auto_cfg),
            patch(f"{self._MOD}._try_safetensors", return_value=None),
            patch(f"{self._MOD}.AutoModelForCausalLM", mock_auto_causal),
        ):
            from provenancekit.services.model_loader import load_state_dict

            load_state_dict("test/model", settings=settings)

            _, kwargs = mock_auto_causal.from_pretrained.call_args
            assert kwargs["trust_remote_code"] is True


# ── 6. No hardcoded True remains in src/ ──────────────────────────


class TestNoHardcodedTrue:
    """Static check: no trust_remote_code=True in source code."""

    def test_no_hardcoded_true_in_src(self) -> None:
        from pathlib import Path

        src_root = Path(__file__).resolve().parents[1] / "src"
        violations: list[str] = []

        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                needle = "trust_remote_code=True"
                if needle in line and not line.strip().startswith("#"):
                    rel = py_file.relative_to(src_root)
                    violations.append(f"{rel}:{i}: {line.strip()}")

        assert violations == [], (
            "Found hardcoded trust_remote_code=True in src/:\n" + "\n".join(violations)
        )
