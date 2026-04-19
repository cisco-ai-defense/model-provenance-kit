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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from provenancekit.core.signals.weight_signals import extract_signals_streaming
from provenancekit.exceptions import (
    CacheError,
    ExtractionError,
    ModelLoadError,
    ProvenanceError,
)
from provenancekit.services.cache import CacheService
from provenancekit.services.model_loader import load_state_dict

# ── Hierarchy tests ───────────────────────────────────────────────


def test_exception_hierarchy():
    assert issubclass(ModelLoadError, ProvenanceError)
    assert issubclass(ExtractionError, ProvenanceError)
    assert issubclass(CacheError, ProvenanceError)


def test_model_load_error_is_extraction_error():
    assert issubclass(ModelLoadError, ExtractionError)


def test_raise_and_catch():
    try:
        raise ModelLoadError("model not found")
    except ProvenanceError as e:
        assert "model not found" in str(e)


# ── Structured context (details, to_dict, model_id) ──────────────


class TestExceptionContext:
    def test_default_details_is_empty_dict(self) -> None:
        exc = ProvenanceError("boom")
        assert exc.details == {}
        assert exc.message == "boom"

    def test_details_passed_through(self) -> None:
        exc = ExtractionError(
            "bad tensor",
            details={"model_id": "org/m", "stage": "streaming"},
        )
        assert exc.details["model_id"] == "org/m"
        assert exc.details["stage"] == "streaming"

    def test_to_dict_structure(self) -> None:
        exc = CacheError("corrupt", details={"path": "/tmp/x.json"})
        d = exc.to_dict()
        assert d["error_type"] == "CacheError"
        assert d["message"] == "corrupt"
        assert d["details"]["path"] == "/tmp/x.json"

    def test_model_load_error_carries_model_id(self) -> None:
        exc = ModelLoadError(
            "config failed",
            details={"stage": "config_load"},
            model_id="org/broken",
        )
        assert exc.model_id == "org/broken"
        assert exc.details["stage"] == "config_load"
        assert "config failed" in str(exc)

    def test_model_load_error_default_model_id(self) -> None:
        exc = ModelLoadError("generic fail")
        assert exc.model_id == ""
        assert exc.details == {}

    def test_to_dict_on_subclass(self) -> None:
        exc = ModelLoadError(
            "load failed",
            details={"strategy": "automodel"},
            model_id="org/m",
        )
        d = exc.to_dict()
        assert d["error_type"] == "ModelLoadError"
        assert d["details"]["strategy"] == "automodel"


# ── Integration: model_loader raises ModelLoadError ───────────────


class TestModelLoadErrorIntegration:
    def test_config_failure_raises_model_load_error(self) -> None:
        with (
            patch(
                "provenancekit.services.model_loader.estimate_model_params",
                return_value=0,
            ),
            patch(
                "provenancekit.services.model_loader.AutoConfig",
            ) as mock_cfg,
        ):
            mock_cfg.from_pretrained.side_effect = OSError("offline")
            with pytest.raises(ModelLoadError, match="config"):
                load_state_dict("bad/model")

    def test_automodel_exhausted_raises_model_load_error(self) -> None:
        with (
            patch(
                "provenancekit.services.model_loader.estimate_model_params",
                return_value=int(1e8),
            ),
            patch(
                "provenancekit.services.model_loader.AutoConfig",
            ) as mock_cfg,
            patch(
                "provenancekit.services.model_loader._try_safetensors",
                return_value=None,
            ),
            patch(
                "provenancekit.services.model_loader.AutoModelForCausalLM",
            ) as mock_causal,
            patch(
                "provenancekit.services.model_loader.AutoModel",
            ) as mock_auto,
        ):
            mock_cfg.from_pretrained.return_value = MagicMock()
            mock_causal.from_pretrained.side_effect = ValueError("bad")
            mock_auto.from_pretrained.side_effect = RuntimeError("worse")
            with pytest.raises(ModelLoadError, match="All AutoModel"):
                load_state_dict("bad/model")


# ── Integration: weight_signals raises ExtractionError ────────────


class TestExtractionErrorIntegration:
    def test_no_weight_map_raises_extraction_error(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ExtractionError, match="No supported weight format"):
            extract_signals_streaming(str(tmp_path))

    def test_config_fail_raises_extraction_error(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        idx = {"weight_map": {"w": "model.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=OSError("boom"),
            ),
            pytest.raises(ExtractionError, match="config"),
        ):
            extract_signals_streaming(str(tmp_path))


# ── Integration: cache raises/catches CacheError ─────────────────


class TestCacheErrorIntegration:
    def test_corrupt_file_raises_cache_error_internally(
        self,
        tmp_path: Path,
    ) -> None:
        svc = CacheService(cache_dir=tmp_path)
        path = svc._cache_path("bad")
        path.write_text("NOT JSON {{{", encoding="utf-8")
        assert svc.get("bad") is None

    def test_load_disk_raises_cache_error(self, tmp_path: Path) -> None:
        from provenancekit.services.cache import CacheService

        svc = CacheService(cache_dir=tmp_path)
        path = svc._cache_path("bad")
        path.write_text("NOT JSON {{{", encoding="utf-8")
        with pytest.raises(CacheError, match="Corrupt cache"):
            svc._load_disk("bad")

    def test_save_disk_raises_cache_error_on_os_error(
        self,
        tmp_path: Path,
    ) -> None:
        from provenancekit.models.results import CachedEntry
        from provenancekit.services.cache import CacheService

        svc = CacheService(cache_dir=tmp_path)
        entry = CachedEntry(model_id="x")
        with (
            patch(
                "provenancekit.services.cache.tempfile.mkstemp",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(CacheError, match="Failed to write"),
        ):
            svc._save_disk("x", entry)

    def test_put_swallows_cache_error(self, tmp_path: Path) -> None:
        from provenancekit.models.results import CachedEntry
        from provenancekit.services.cache import CacheService

        svc = CacheService(cache_dir=tmp_path)
        entry = CachedEntry(model_id="x")
        with patch(
            "provenancekit.services.cache.tempfile.mkstemp",
            side_effect=OSError("disk full"),
        ):
            svc.put("x", entry)
