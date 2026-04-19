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

"""Tests for services/model_loader.py — HuggingFace Hub loading."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from provenancekit.exceptions import ModelLoadError
from provenancekit.models.parsing import ModelFormat
from provenancekit.models.results import LoadStrategy
from provenancekit.services.model_loader import (
    detect_model_format,
    estimate_model_params,
    is_local_hf_snapshot,
    load_state_dict,
    local_pytorch_weight_map,
    local_safetensors_weight_map,
    resolve_hf_model_ref,
)

# ── estimate_model_params (offline) ────────────────────────────────


class TestEstimateModelParams:
    def test_returns_zero_on_failure(self) -> None:
        with patch(
            "provenancekit.services.model_loader.AutoConfig",
        ) as mock_cfg:
            mock_cfg.from_pretrained.side_effect = Exception("no network")
            assert estimate_model_params("nonexistent/model") == 0

    def test_uses_config_fields(self) -> None:
        cfg = MagicMock()
        cfg.hidden_size = 768
        cfg.num_hidden_layers = 12
        cfg.num_attention_heads = 12
        cfg.num_key_value_heads = None
        cfg.head_dim = None
        cfg.vocab_size = 50257
        cfg.intermediate_size = 3072
        cfg.tie_word_embeddings = None
        cfg.hidden_act = "gelu_new"
        cfg.model_type = "gpt2"
        with patch(
            "provenancekit.services.model_loader.AutoConfig",
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = cfg
            params = estimate_model_params("test-model")
            assert params > 0

    def test_formula_correctness(self) -> None:
        cfg = MagicMock()
        h, n_layers, v, inter, n_heads = 768, 12, 50257, 3072, 12
        cfg.hidden_size = h
        cfg.num_hidden_layers = n_layers
        cfg.num_attention_heads = n_heads
        cfg.num_key_value_heads = None
        cfg.head_dim = None
        cfg.vocab_size = v
        cfg.intermediate_size = inter
        cfg.tie_word_embeddings = None
        cfg.hidden_act = "gelu_new"
        cfg.model_type = "gpt2"
        hd = h // n_heads
        n_kv = n_heads
        qkv = h * hd * (n_heads + 2 * n_kv)
        o_proj = n_heads * hd * h
        mlp = 2 * h * inter
        per_layer = qkv + o_proj + mlp
        expected = v * h + n_layers * per_layer
        with patch(
            "provenancekit.services.model_loader.AutoConfig",
        ) as mock_cls:
            mock_cls.from_pretrained.return_value = cfg
            assert estimate_model_params("gpt2") == expected


# ── load_state_dict (offline) ──────────────────────────────────────


class TestLoadStateDict:
    def test_returns_streaming_for_huge_model(self) -> None:
        with patch(
            "provenancekit.services.model_loader.estimate_model_params",
            return_value=int(20e9),
        ):
            result = load_state_dict("huge/model")
            assert result.strategy == LoadStrategy.streaming
            assert result.state_dict is None
            assert "too_large" in result.source

    def test_raises_model_load_error_on_config_failure(self) -> None:
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
                load_state_dict("broken/model")

    def test_returns_full_with_safetensors(self) -> None:
        fake_state: dict[str, Any] = {"weight": MagicMock()}
        fake_config = MagicMock()

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
            ) as mock_st,
        ):
            mock_cfg.from_pretrained.return_value = fake_config
            from provenancekit.models.results import LoadResult

            mock_st.return_value = LoadResult(
                state_dict=fake_state,
                config=fake_config,
                strategy=LoadStrategy.full,
                source="safetensors",
            )
            result = load_state_dict("small/model")
            assert result.strategy == LoadStrategy.full
            assert result.state_dict == fake_state
            assert result.source == "safetensors"

    def test_falls_back_to_automodel(self) -> None:
        fake_state: dict[str, Any] = {"weight": MagicMock()}
        fake_config = MagicMock()

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
                "provenancekit.services.model_loader._try_automodel",
            ) as mock_auto,
        ):
            mock_cfg.from_pretrained.return_value = fake_config
            from provenancekit.models.results import LoadResult

            mock_auto.return_value = LoadResult(
                state_dict=fake_state,
                config=fake_config,
                strategy=LoadStrategy.full,
                source="automodel_causal",
            )
            result = load_state_dict("small/model")
            assert result.strategy == LoadStrategy.full
            assert result.source == "automodel_causal"

    def test_raises_model_load_error_on_automodel_failure(self) -> None:
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
            mock_auto.from_pretrained.side_effect = RuntimeError("also bad")
            with pytest.raises(ModelLoadError, match="All AutoModel strategies failed"):
                load_state_dict("broken/model")

    def test_local_safetensors_single_file_no_hub_download(
        self,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        save_file({"w": torch.randn(4)}, str(tmp_path / "model.safetensors"))
        fake_config = MagicMock()

        def _no_hub(*_a: Any, **_k: Any) -> None:
            raise AssertionError("hf_hub_download must not be used for local dir")

        with (
            patch(
                "provenancekit.services.model_loader.estimate_model_params",
                return_value=int(1e8),
            ),
            patch(
                "provenancekit.services.model_loader.AutoConfig",
            ) as mock_cfg,
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=_no_hub,
            ),
        ):
            mock_cfg.from_pretrained.return_value = fake_config
            result = load_state_dict(str(tmp_path))
        assert result.strategy == LoadStrategy.full
        assert result.source == "safetensors"
        assert result.state_dict is not None
        assert "w" in result.state_dict


# ── HF model ref helpers ───────────────────────────────────────────


class TestHfModelRef:
    def test_resolve_local_returns_absolute_path(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        out = resolve_hf_model_ref(str(tmp_path))
        assert Path(out) == tmp_path.resolve()

    def test_resolve_hub_id_unchanged(self) -> None:
        assert resolve_hf_model_ref("  org/model  ") == "org/model"

    def test_is_local_hf_snapshot(self, tmp_path: Path) -> None:
        assert not is_local_hf_snapshot("org/model")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert is_local_hf_snapshot(str(tmp_path))

    def test_local_weight_map_from_index(self, tmp_path: Path) -> None:
        idx = {"weight_map": {"layer.bias": "shard-000.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )
        wm = local_safetensors_weight_map(str(tmp_path))
        assert wm == {"layer.bias": "shard-000.safetensors"}

    def test_local_weight_map_single_shard(self, tmp_path: Path) -> None:
        save_file(
            {"layer.weight": torch.zeros(2, 3)}, str(tmp_path / "model.safetensors")
        )
        wm = local_safetensors_weight_map(str(tmp_path))
        assert wm == {"layer.weight": "model.safetensors"}


# ── local_pytorch_weight_map ───────────────────────────────────────


class TestLocalPytorchWeightMap:
    def test_returns_none_when_no_pytorch_files(self, tmp_path: Path) -> None:
        assert local_pytorch_weight_map(str(tmp_path)) is None

    def test_reads_index_json(self, tmp_path: Path) -> None:
        idx = {"weight_map": {"layer.weight": "pytorch_model-00001.bin"}}
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )
        wm = local_pytorch_weight_map(str(tmp_path))
        assert wm == {"layer.weight": "pytorch_model-00001.bin"}

    def test_reads_single_bin(self, tmp_path: Path) -> None:
        sd = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}
        torch.save(sd, str(tmp_path / "pytorch_model.bin"))
        wm = local_pytorch_weight_map(str(tmp_path))
        assert wm is not None
        assert set(wm.keys()) == {"layer.weight", "layer.bias"}
        assert all(v == "pytorch_model.bin" for v in wm.values())

    def test_index_takes_priority_over_single(self, tmp_path: Path) -> None:
        idx = {"weight_map": {"a": "shard.bin"}}
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )
        torch.save({"b": torch.randn(2)}, str(tmp_path / "pytorch_model.bin"))
        wm = local_pytorch_weight_map(str(tmp_path))
        assert wm == {"a": "shard.bin"}


# ── detect_model_format ───────────────────────────────────────────


class TestDetectModelFormat:
    def test_safetensors_single(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        save_file({"w": torch.randn(4)}, str(tmp_path / "model.safetensors"))
        assert detect_model_format(str(tmp_path)) == ModelFormat.SAFETENSORS

    def test_safetensors_sharded(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        idx = {"weight_map": {"w": "model-00001-of-00002.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )
        assert detect_model_format(str(tmp_path)) == ModelFormat.SAFETENSORS

    def test_pytorch_single(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        torch.save({"w": torch.randn(4)}, str(tmp_path / "pytorch_model.bin"))
        assert detect_model_format(str(tmp_path)) == ModelFormat.PYTORCH

    def test_pytorch_sharded(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        idx = {"weight_map": {"w": "pytorch_model-00001.bin"}}
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps(idx),
            encoding="utf-8",
        )
        assert detect_model_format(str(tmp_path)) == ModelFormat.PYTORCH

    def test_safetensors_preferred_over_pytorch(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        save_file({"w": torch.randn(4)}, str(tmp_path / "model.safetensors"))
        torch.save({"w": torch.randn(4)}, str(tmp_path / "pytorch_model.bin"))
        assert detect_model_format(str(tmp_path)) == ModelFormat.SAFETENSORS

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert detect_model_format(str(tmp_path)) is None

    def test_returns_none_for_nonexistent(self) -> None:
        assert detect_model_format("nonexistent/model") is None


# ── LoadResult model ──────────────────────────────────────────────


class TestLoadResult:
    def test_strategy_enum_values(self) -> None:
        assert LoadStrategy.full == "full"
        assert LoadStrategy.streaming == "streaming"

    def test_default_fields(self) -> None:
        from provenancekit.models.results import LoadResult

        result = LoadResult(strategy=LoadStrategy.streaming)
        assert result.state_dict is None
        assert result.config is None
        assert result.source == ""


# ── Online golden tests ───────────────────────────────────────────


@pytest.mark.slow
class TestModelLoaderOnline:
    def test_load_gpt2_full(self) -> None:
        result = load_state_dict("gpt2")
        assert result.strategy == LoadStrategy.full
        assert result.state_dict is not None
        assert len(result.state_dict) > 0
        assert result.config is not None

    def test_estimate_gpt2_params(self) -> None:
        params = estimate_model_params("gpt2")
        assert 100e6 < params < 200e6
