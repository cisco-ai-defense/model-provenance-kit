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

"""Quantized checkpoints: packed weights must not be read as real weights."""

import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from provenancekit.core.signals.config_consistency import (
    check_config_consistency,
    collect_shapes,
)
from provenancekit.core.signals.weight_signals import (
    end_similarity,
    extract_signals,
    extract_signals_streaming,
    lep_similarity,
    nlf_similarity,
    wvc_similarity,
)
from provenancekit.utils.tensor import is_quantization_aux_name
from tests.core.signals.test_tensor_order import _llama_state, _write_config

StateDict = dict[str, torch.Tensor]
Quantizer = Callable[[StateDict], StateDict]

_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _is_projection(name: str) -> bool:
    return name.endswith(".weight") and any(p in name for p in _PROJ)


def _bnb_4bit(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """bitsandbytes-4bit layout: uint8 [N, 1] packs plus float side tensors."""
    g = torch.Generator().manual_seed(7)
    out: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        if not _is_projection(name):
            out[name] = t
            continue
        n = t.numel()
        out[name] = torch.randint(0, 255, (n // 2, 1), dtype=torch.uint8, generator=g)
        out[f"{name}.absmax"] = torch.randint(0, 255, (n // 64,), dtype=torch.uint8)
        out[f"{name}.nested_absmax"] = torch.rand(2048, generator=g)
        out[f"{name}.quant_map"] = torch.rand(16, generator=g)
        out[f"{name}.quant_state.bitsandbytes__nf4"] = torch.zeros(
            170, dtype=torch.uint8
        )
    return out


def _awq(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """AWQ layout: int32 qweight / qzeros plus fp16 scales."""
    g = torch.Generator().manual_seed(8)
    out: dict[str, torch.Tensor] = {}
    for name, t in state.items():
        if not _is_projection(name):
            out[name] = t
            continue
        base = name.removesuffix(".weight")
        rows, cols = t.shape
        out[f"{base}.qweight"] = torch.randint(
            -(2**31), 2**31 - 1, (cols, rows // 8), dtype=torch.int32, generator=g
        )
        out[f"{base}.qzeros"] = torch.zeros(cols // 32, rows // 8, dtype=torch.int32)
        out[f"{base}.scales"] = torch.rand(cols // 32, rows, generator=g).half()
    return out


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=100, hidden_size=64, num_hidden_layers=12, intermediate_size=128
    )


class TestQuantizationAuxNames:
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.mlp.down_proj.qweight",
            "model.layers.0.mlp.down_proj.qzeros",
            "model.layers.0.mlp.down_proj.scales",
            "model.layers.0.mlp.down_proj.g_idx",
            "model.layers.0.mlp.down_proj.weight.absmax",
            "model.layers.0.mlp.down_proj.weight.nested_absmax",
            "model.layers.0.mlp.down_proj.weight.quant_map",
            "model.layers.0.mlp.down_proj.weight.quant_state.bitsandbytes__nf4",
            "model.layers.0.mlp.down_proj.weight_scale_inv",
        ],
    )
    def test_side_tensors_are_recognised(self, name: str) -> None:
        assert is_quantization_aux_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.embed_tokens.weight",
            "model.layers.0.attn.scale_factor",
        ],
    )
    def test_regular_tensors_are_not(self, name: str) -> None:
        assert not is_quantization_aux_name(name)


@pytest.mark.parametrize("quantize", [_bnb_4bit, _awq], ids=["bnb-4bit", "awq"])
class TestQuantizedSignals:
    def test_packed_layers_give_no_per_layer_signals(self, quantize: Quantizer) -> None:
        feats = extract_signals(quantize(_llama_state()), config=None)
        assert feats is not None
        assert feats.lep_profile is None
        assert feats.wvc_layer_sigs is None

    def test_unquantized_tensors_still_match_original(
        self, quantize: Quantizer
    ) -> None:
        state = _llama_state()
        original = extract_signals(state, config=None)
        quantized = extract_signals(quantize(state), config=None)
        assert original is not None and quantized is not None
        assert nlf_similarity(original, quantized) == pytest.approx(1.0)
        assert end_similarity(original, quantized) == pytest.approx(1.0)
        assert math.isnan(lep_similarity(original, quantized))
        assert math.isnan(wvc_similarity(original, quantized))

    def test_streaming_matches_full_load(
        self, quantize: Quantizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from provenancekit.core.signals import weight_signals

        state = quantize(_llama_state())
        _write_config(tmp_path)
        save_file(
            {n: t.contiguous() for n, t in state.items()},
            str(tmp_path / "model.safetensors"),
        )
        full = extract_signals(state, config=None)
        assert full is not None
        for threshold in (20e9, 0):
            monkeypatch.setattr(
                weight_signals, "_SLICE_DISK_THRESHOLD_BYTES", threshold
            )
            streamed = extract_signals_streaming(str(tmp_path))
            assert streamed is not None
            assert streamed.lep_profile is None
            assert streamed.wvc_layer_sigs is None
            assert full.nlf_vector is not None and streamed.nlf_vector is not None
            np.testing.assert_array_equal(full.nlf_vector, streamed.nlf_vector)

    def test_config_check_has_no_false_findings(self, quantize: Quantizer) -> None:
        shapes = collect_shapes(quantize(_llama_state()))
        assert check_config_consistency(_config(), shapes) == []

    def test_config_check_still_flags_edited_vocab(self, quantize: Quantizer) -> None:
        shapes = collect_shapes(quantize(_llama_state()))
        cfg = _config()
        cfg.vocab_size = 90
        assert [f.field for f in check_config_consistency(cfg, shapes)] == [
            "vocab_size"
        ]
