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

"""Weight signals must not depend on tensor order or load path."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from provenancekit.core.signals.weight_signals import (
    extract_signals,
    extract_signals_streaming,
    nlf_similarity,
    wvc_similarity,
)
from provenancekit.models.signals import WeightSignalFeatures
from provenancekit.utils.tensor import canonical_tensor_key

_LAYERS = 12
_DIM = 64


def _llama_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """Module-ordered Llama-style state dict with >= 10 layers."""
    g = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(100, _DIM, generator=g)
    }
    for i in range(_LAYERS):
        p = f"model.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            state[f"{p}.self_attn.{proj}.weight"] = torch.randn(_DIM, _DIM, generator=g)
        state[f"{p}.mlp.gate_proj.weight"] = torch.randn(128, _DIM, generator=g)
        state[f"{p}.mlp.up_proj.weight"] = torch.randn(128, _DIM, generator=g)
        state[f"{p}.mlp.down_proj.weight"] = torch.randn(_DIM, 128, generator=g)
        state[f"{p}.input_layernorm.weight"] = torch.randn(_DIM, generator=g)
        state[f"{p}.post_attention_layernorm.weight"] = torch.randn(_DIM, generator=g)
    state["model.norm.weight"] = torch.randn(_DIM, generator=g)
    return state


def _write_config(path: Path) -> None:
    from transformers import LlamaConfig

    LlamaConfig(
        vocab_size=100,
        hidden_size=_DIM,
        intermediate_size=128,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        architectures=["LlamaForCausalLM"],
    ).save_pretrained(path)


def _assert_same_order_signals(
    a: WeightSignalFeatures, b: WeightSignalFeatures
) -> None:
    assert a.nlf_vector is not None and b.nlf_vector is not None
    np.testing.assert_array_equal(a.nlf_vector, b.nlf_vector)
    assert a.wvc_layer_sigs is not None and b.wvc_layer_sigs is not None
    assert a.wvc_layer_sigs.keys() == b.wvc_layer_sigs.keys()
    for layer in a.wvc_layer_sigs:
        np.testing.assert_array_equal(a.wvc_layer_sigs[layer], b.wvc_layer_sigs[layer])
    assert nlf_similarity(a, b) == pytest.approx(1.0)
    assert wvc_similarity(a, b) == pytest.approx(1.0)


class TestCanonicalTensorKey:
    def test_layer_numbers_sort_numerically(self) -> None:
        names = ["model.layers.10.w", "model.layers.2.w", "model.layers.1.w"]
        assert sorted(names, key=canonical_tensor_key) == [
            "model.layers.1.w",
            "model.layers.2.w",
            "model.layers.10.w",
        ]

    def test_mixed_names_are_comparable(self) -> None:
        names = ["lm_head.weight", "model.layers.3.x", "model.norm.weight", "h.0.ln_1"]
        assert sorted(names, key=canonical_tensor_key)[0] == "h.0.ln_1"


class TestWvcSliceSampling:
    @pytest.mark.parametrize("shape", [(1000, 777), (300, 7, 11), (64, 64)])
    def test_slice_sample_matches_in_memory_sample(
        self, tmp_path: Path, shape: tuple[int, ...]
    ) -> None:
        from safetensors import safe_open

        from provenancekit.core.signals.weight_signals import (
            _wvc_sample,
            _wvc_sample_from_slice,
        )

        tensor = torch.randn(*shape).to(torch.bfloat16)
        save_file({"w": tensor}, str(tmp_path / "t.safetensors"))
        with safe_open(str(tmp_path / "t.safetensors"), "pt") as f:
            sl = f.get_slice("w")
            sliced = _wvc_sample_from_slice(sl, sl.get_shape(), 512)
        np.testing.assert_array_equal(sliced, _wvc_sample(tensor, 512))


class TestFullLoadOrderInvariance:
    def test_shuffled_state_dict_gives_identical_signals(self) -> None:
        state = _llama_state()
        names = list(state)
        random.Random(1).shuffle(names)
        shuffled = {n: state[n] for n in names}
        _assert_same_order_signals(
            extract_signals(state, config=None), extract_signals(shuffled, config=None)
        )

    def test_alphabetical_vs_module_order(self) -> None:
        state = _llama_state()
        alphabetical = {n: state[n] for n in sorted(state)}
        _assert_same_order_signals(
            extract_signals(state, config=None),
            extract_signals(alphabetical, config=None),
        )


class TestStreamingMatchesFullLoad:
    def test_safetensors_sharded_streaming(self, tmp_path: Path) -> None:
        state = _llama_state()
        _write_config(tmp_path)
        names = list(state)
        random.Random(2).shuffle(names)
        shards = {
            "model-00001-of-00002.safetensors": names[::2],
            "model-00002-of-00002.safetensors": names[1::2],
        }
        weight_map = {}
        for shard, members in shards.items():
            save_file(
                {n: state[n].contiguous() for n in members}, str(tmp_path / shard)
            )
            weight_map.update(dict.fromkeys(members, shard))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )

        streamed = extract_signals_streaming(str(tmp_path))
        full = extract_signals(state, config=None)
        assert streamed is not None and full is not None
        _assert_same_order_signals(full, streamed)

    def test_row_sliced_streaming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Large-model row-sliced path: same WVC samples, same LEP scale."""
        from provenancekit.core.signals import weight_signals

        monkeypatch.setattr(weight_signals, "_SLICE_DISK_THRESHOLD_BYTES", 0)
        state = _llama_state()
        for name in state:
            layer = canonical_tensor_key(name)[1] if ".layers." in name else 0
            state[name] = state[name] * (1 + int(layer))
        _write_config(tmp_path)
        save_file(
            {n: t.contiguous() for n, t in state.items()},
            str(tmp_path / "model.safetensors"),
        )

        streamed = extract_signals_streaming(str(tmp_path))
        full = extract_signals(state, config=None)
        assert streamed is not None and full is not None
        _assert_same_order_signals(full, streamed)
        assert streamed.lep_profile is not None and full.lep_profile is not None
        np.testing.assert_allclose(streamed.lep_profile, full.lep_profile, rtol=0.05)

    def test_pytorch_bin_streaming(self, tmp_path: Path) -> None:
        state = _llama_state()
        _write_config(tmp_path)
        torch.save(state, tmp_path / "pytorch_model.bin")

        streamed = extract_signals_streaming(str(tmp_path))
        full = extract_signals({n: state[n] for n in sorted(state)}, config=None)
        assert streamed is not None and full is not None
        _assert_same_order_signals(full, streamed)
