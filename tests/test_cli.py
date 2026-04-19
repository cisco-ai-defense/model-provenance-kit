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

"""Tests for provenancekit.cli — offline (scanner is mocked)."""

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from provenancekit.cli import main
from provenancekit.models.results import (
    CompareResult,
    PipelineScore,
    ScanMatch,
    ScanMatchScores,
    ScanModelInfo,
    ScanResult,
    ScoreInterpretation,
    SignalScores,
    TimingBreakdown,
)


def _fake_result() -> CompareResult:
    return CompareResult(
        model_a="gpt2",
        model_b="gpt2",
        family_a="gpt2",
        family_b="gpt2",
        signals=SignalScores(
            eas=0.99,
            nlf=0.95,
            lep=0.98,
            end=0.96,
            wvc=0.91,
            tfv=0.95,
            voa=0.88,
        ),
        scores=PipelineScore(
            mfi_score=1.0,
            mfi_tier=1,
            mfi_match="exact",
            identity_score=0.96,
            tokenizer_score=0.90,
            pipeline_score=1.0,
            provenance_decision="same_family",
        ),
        interpretation=ScoreInterpretation(
            label="Same family / direct derivative",
            colour="#2ecc71",
        ),
        time_seconds=2.5,
        timing=TimingBreakdown(
            total_seconds=2.5,
            metadata_extract_seconds=1.2,
            weight_feature_extract_seconds=1.1,
            cache_hit="ws, mfi, tfv",
        ),
    )


_SCANNER_PATH = "provenancekit.core.scanner.ModelProvenanceScanner"
_CACHE_PATH = "provenancekit.services.cache.CacheService"
_SETTINGS_PATH = "provenancekit.cli.Settings"
_DL_PATH = "provenancekit.cli.download_deep_signals"
_DL_STATUS_PATH = "provenancekit.cli.show_deep_signals_status"


def _run_cli(argv: list[str]) -> tuple[str, str, int]:
    """Invoke ``main()`` with *argv* and capture stdout, stderr, exit code."""
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


class TestCompareCommand:
    def test_json_flag(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.compare.return_value = _fake_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(["compare", "gpt2", "gpt2", "--json"])

        assert code == 0
        data: dict[str, Any] = json.loads(stdout)
        assert data["scores"]["pipeline_score"] == 1.0

    def test_default_table(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.compare.return_value = _fake_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(["compare", "gpt2", "gpt2"])

        assert code == 0
        assert "Pipeline Score" in stdout
        assert "Same family / direct derivative" in stdout

    def test_plain_flag(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.compare.return_value = _fake_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(["compare", "gpt2", "gpt2", "--plain"])

        assert code == 0
        assert "pipeline_score:" in stdout
        assert "verdict:" in stdout

    def test_plain_timing_flag(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.compare.return_value = _fake_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(
                ["compare", "gpt2", "gpt2", "--plain", "--timing"]
            )

        assert code == 0
        assert "total_time_s:" in stdout
        assert "metadata_extract_time_s:" in stdout
        assert "weight_feature_extract_time_s:" in stdout
        assert "cache_hit:" in stdout
        assert "ws, mfi, tfv" in stdout

    def test_scanner_error(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.compare.side_effect = RuntimeError("boom")

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            _, stderr, code = _run_cli(["compare", "gpt2", "gpt2"])

        assert code == 1
        assert "boom" in stderr


def _fake_scan_result() -> ScanResult:
    return ScanResult(
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
            arch_hash="abc123",
            family_hash="def456",
            param_bucket="<=1B",
            has_weight_signals=True,
        ),
        matches=[
            ScanMatch(
                asset_id="asset-1",
                model_id="openai-community/gpt2",
                family_id="fam-gpt2",
                family_name="GPT-2",
                match_type="exact_hash",
                scores=ScanMatchScores(
                    pipeline_score=0.98,
                    identity_score=0.96,
                    mfi_score=1.0,
                    mfi_tier=1,
                    mfi_match_type="exact",
                    tokenizer_score=0.95,
                    eas=0.99,
                    nlf=0.97,
                    lep=0.96,
                    end=0.94,
                    wvc=0.92,
                    tfv=0.91,
                ),
                provenance_decision="same_family",
                elapsed_ms=50.0,
            ),
        ],
        match_count=1,
        elapsed_ms=120.5,
    )


class TestScanCommand:
    def test_json_flag(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.scan.return_value = _fake_scan_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(["scan", "gpt2", "--json"])

        assert code == 0
        data: dict[str, Any] = json.loads(stdout)
        assert data["model_info"]["model_path"] == "gpt2"
        assert len(data["matches"]) == 1
        assert data["matches"][0]["scores"]["pipeline_score"] == 0.98

    def test_default_table(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.scan.return_value = _fake_scan_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(["scan", "gpt2"])

        assert code == 0
        assert "Provenance Scan" in stdout
        assert "Scanned Model" in stdout
        assert "Parameters" in stdout
        assert "Layers" in stdout
        assert "gpt2" in stdout
        assert "same_family" in stdout
        assert "Pipeline Score" in stdout
        assert "MFI Score" in stdout
        assert "Weight Score" in stdout
        assert "Tokenizer Score" in stdout

    def test_plain_flag(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.scan.return_value = _fake_scan_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            stdout, _, code = _run_cli(["scan", "gpt2", "--plain"])

        assert code == 0
        assert "model:" in stdout and "gpt2" in stdout
        assert "match_1_pipeline_score:" in stdout
        assert "match_1_provenance_decision:" in stdout and "same_family" in stdout

    def test_scanner_error(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.scan.side_effect = RuntimeError("db error")

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            _, stderr, code = _run_cli(["scan", "gpt2"])

        assert code == 1
        assert "db error" in stderr

    def test_top_k_and_threshold_forwarded(self) -> None:
        mock_scanner = MagicMock()
        mock_scanner.return_value.scan.return_value = _fake_scan_result()

        with (
            patch(_SCANNER_PATH, mock_scanner),
            patch(_CACHE_PATH),
            patch(_SETTINGS_PATH),
        ):
            _, _, code = _run_cli(
                ["scan", "gpt2", "--top-k", "5", "--threshold", "0.7"]
            )

        assert code == 0
        call_kwargs = mock_scanner.return_value.scan.call_args
        assert call_kwargs[1]["top_k"] == 5
        assert call_kwargs[1]["threshold"] == 0.7


class TestNoCommand:
    def test_no_command_shows_help(self) -> None:
        stdout, _, code = _run_cli([])
        assert code == 0
        assert "provenancekit" in stdout.lower() or "usage" in stdout.lower()


class TestDownloadCommand:
    def test_no_verify_rejected_outside_dev_mode(self) -> None:
        settings = MagicMock()
        settings.db_root = Path("/tmp/db")
        settings.dev_mode = False

        with (
            patch(_SETTINGS_PATH, return_value=settings),
            patch(_DL_PATH) as mock_download,
            patch(_DL_STATUS_PATH),
        ):
            _, stderr, code = _run_cli(
                ["download-deepsignals-fingerprint", "--no-verify"]
            )

        assert code == 1
        assert "--no-verify is disabled outside dev mode" in stderr
        mock_download.assert_not_called()

    def test_no_verify_allowed_in_dev_mode_with_warning(self) -> None:
        settings = MagicMock()
        settings.db_root = Path("/tmp/db")
        settings.dev_mode = True

        with (
            patch(_SETTINGS_PATH, return_value=settings),
            patch(_DL_PATH, return_value=0) as mock_download,
            patch(_DL_STATUS_PATH),
        ):
            _, stderr, code = _run_cli(
                ["download-deepsignals-fingerprint", "--no-verify"]
            )

        assert code == 0
        assert "WARNING: --no-verify skips SHA-256 integrity validation" in stderr
        mock_download.assert_called_once_with(
            settings.db_root,
            update=False,
            verify=False,
            settings=settings,
        )
