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

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from provenancekit.core.signals.config_consistency import (
    check_config_consistency,
    collect_shapes,
)
from provenancekit.core.signals.metadata import similarity
from provenancekit.core.signals.weight_signals import extract_signals
from provenancekit.models.signals import ConfigInconsistency, WeightSignalFeatures
from tests.core.signals.test_metadata import _make_fp

# -- Fixture builders --------------------------------------------------


def _llama_config(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 176,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _llama_shapes(
    vocab: int = 1000,
    hidden: int = 64,
    layers: int = 4,
    inter: int = 176,
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for i in range(layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (hidden, hidden)
        shapes[f"{p}.self_attn.k_proj.weight"] = (hidden // 4, hidden)
        shapes[f"{p}.mlp.gate_proj.weight"] = (inter, hidden)
        shapes[f"{p}.mlp.up_proj.weight"] = (inter, hidden)
        shapes[f"{p}.mlp.down_proj.weight"] = (hidden, inter)
        shapes[f"{p}.input_layernorm.weight"] = (hidden,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (hidden,)
    return shapes


def _fields(findings: list[ConfigInconsistency]) -> list[str]:
    return [f.field for f in findings]


# -- check_config_consistency ------------------------------------------


class TestConsistentConfigs:
    def test_matching_config_has_no_findings(self) -> None:
        assert check_config_consistency(_llama_config(), _llama_shapes()) == []

    def test_empty_shapes_has_no_findings(self) -> None:
        assert check_config_consistency(_llama_config(), {}) == []

    def test_missing_config_fields_are_skipped(self) -> None:
        cfg = SimpleNamespace()
        assert check_config_consistency(cfg, _llama_shapes()) == []

    def test_text_backbone_config_is_resolved(self) -> None:
        cfg = SimpleNamespace(hidden_size=0, text_config=_llama_config())
        assert check_config_consistency(cfg, _llama_shapes()) == []


class TestVocabSize:
    def test_edited_vocab_size_is_flagged(self) -> None:
        cfg = _llama_config(vocab_size=900)
        findings = check_config_consistency(cfg, _llama_shapes())
        assert findings == [
            ConfigInconsistency(field="vocab_size", declared=900, observed=[1000])
        ]

    def test_small_edit_below_stored_rows_is_flagged(self) -> None:
        cfg = _llama_config(vocab_size=990)
        assert _fields(check_config_consistency(cfg, _llama_shapes())) == ["vocab_size"]

    def test_embedding_padded_to_multiple_of_64_is_accepted(self) -> None:
        cfg = _llama_config(vocab_size=1000)
        shapes = _llama_shapes(vocab=1024)
        assert "vocab_size" not in _fields(check_config_consistency(cfg, shapes))

    def test_padding_beyond_tolerance_is_flagged(self) -> None:
        cfg = _llama_config(vocab_size=1000)
        shapes = _llama_shapes(vocab=1280)
        assert "vocab_size" in _fields(check_config_consistency(cfg, shapes))

    def test_embedding_smaller_than_vocab_size_is_flagged(self) -> None:
        cfg = _llama_config(vocab_size=1010)
        assert _fields(check_config_consistency(cfg, _llama_shapes())) == ["vocab_size"]


class TestHiddenSize:
    def test_edited_hidden_size_is_flagged(self) -> None:
        cfg = _llama_config(hidden_size=128)
        findings = check_config_consistency(cfg, _llama_shapes())
        assert findings == [
            ConfigInconsistency(field="hidden_size", declared=128, observed=[64])
        ]

    def test_factorised_embedding_uses_layer_norm_width(self) -> None:
        shapes = _llama_shapes()
        shapes["model.embed_tokens.weight"] = (1000, 16)
        assert check_config_consistency(_llama_config(), shapes) == []


class TestNumHiddenLayers:
    def test_more_layers_declared_than_stored_is_flagged(self) -> None:
        cfg = _llama_config(num_hidden_layers=8)
        findings = check_config_consistency(cfg, _llama_shapes())
        assert findings == [
            ConfigInconsistency(field="num_hidden_layers", declared=8, observed=[4])
        ]

    def test_fewer_layers_declared_than_stored_is_flagged(self) -> None:
        cfg = _llama_config(num_hidden_layers=2)
        assert _fields(check_config_consistency(cfg, _llama_shapes())) == [
            "num_hidden_layers"
        ]

    def test_one_extra_trailing_layer_is_accepted(self) -> None:
        cfg = _llama_config(num_hidden_layers=3)
        assert check_config_consistency(cfg, _llama_shapes(layers=4)) == []

    def test_encoder_decoder_counts_each_stack(self) -> None:
        shapes = {
            "model.shared.weight": (1000, 64),
            **{f"model.encoder.layers.{i}.fc1.weight": (176, 64) for i in range(12)},
            **{f"model.decoder.layers.{i}.fc1.weight": (176, 64) for i in range(6)},
        }
        cfg = _llama_config(num_hidden_layers=12)
        assert check_config_consistency(cfg, shapes) == []

        cfg = _llama_config(num_hidden_layers=9)
        findings = check_config_consistency(cfg, shapes)
        assert findings == [
            ConfigInconsistency(field="num_hidden_layers", declared=9, observed=[6, 12])
        ]


def _bart_config(encoder_layers: int = 12, decoder_layers: int = 6) -> SimpleNamespace:
    return _llama_config(
        is_encoder_decoder=True,
        num_hidden_layers=encoder_layers,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )


def _bart_shapes(encoder: int = 12, decoder: int = 6) -> dict[str, tuple[int, ...]]:
    return {
        "model.shared.weight": (1000, 64),
        **{f"model.encoder.layers.{i}.fc1.weight": (176, 64) for i in range(encoder)},
        **{f"model.decoder.layers.{i}.fc1.weight": (176, 64) for i in range(decoder)},
    }


class TestEncoderDecoderLayers:
    def test_matching_stacks_have_no_findings(self) -> None:
        assert check_config_consistency(_bart_config(), _bart_shapes()) == []

    def test_decoder_only_load_is_not_flagged(self) -> None:
        shapes = {k: v for k, v in _bart_shapes().items() if "encoder" not in k}
        assert check_config_consistency(_bart_config(), shapes) == []

    def test_causal_lm_wrapper_config_is_still_encoder_decoder(self) -> None:
        cfg = _bart_config()
        cfg.is_encoder_decoder = False
        shapes = {k: v for k, v in _bart_shapes().items() if "encoder" not in k}
        assert check_config_consistency(cfg, shapes) == []

    def test_edited_decoder_layers_is_flagged(self) -> None:
        findings = check_config_consistency(
            _bart_config(decoder_layers=12), _bart_shapes()
        )
        assert findings == [
            ConfigInconsistency(field="decoder_layers", declared=12, observed=[6])
        ]

    def test_edited_encoder_layers_is_flagged(self) -> None:
        findings = check_config_consistency(
            _bart_config(encoder_layers=6), _bart_shapes()
        )
        assert findings == [
            ConfigInconsistency(field="encoder_layers", declared=6, observed=[12])
        ]

    def test_t5_style_fields_and_block_stacks(self) -> None:
        cfg = SimpleNamespace(
            is_encoder_decoder=True, num_layers=4, num_decoder_layers=2
        )
        shapes = {
            **{f"encoder.block.{i}.layer.0.weight": (8, 8) for i in range(4)},
            **{f"decoder.block.{i}.layer.0.weight": (8, 8) for i in range(3)},
        }
        assert check_config_consistency(cfg, shapes) == [
            ConfigInconsistency(field="num_decoder_layers", declared=2, observed=[3])
        ]


def _albert_config(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "model_type": "albert",
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": 12,
        "num_hidden_groups": 1,
        "inner_group_num": 1,
        "intermediate_size": 176,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _albert_shapes(groups: int = 1, inner: int = 1) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "albert.embeddings.word_embeddings.weight": (1000, 16),
        "albert.encoder.embedding_hidden_mapping_in.weight": (64, 16),
    }
    for g in range(groups):
        for i in range(inner):
            p = f"albert.encoder.albert_layer_groups.{g}.albert_layers.{i}"
            shapes[f"{p}.attention.query.weight"] = (64, 64)
            shapes[f"{p}.attention.LayerNorm.weight"] = (64,)
            shapes[f"{p}.ffn.weight"] = (176, 64)
            shapes[f"{p}.ffn_output.weight"] = (64, 176)
    return shapes


class TestAlbertLayerGroups:
    def test_one_shared_group_for_twelve_layers_is_consistent(self) -> None:
        assert check_config_consistency(_albert_config(), _albert_shapes()) == []

    def test_multiple_groups_are_consistent(self) -> None:
        cfg = _albert_config(num_hidden_groups=2, inner_group_num=3)
        assert check_config_consistency(cfg, _albert_shapes(groups=2, inner=3)) == []

    def test_edited_num_hidden_groups_is_flagged(self) -> None:
        cfg = _albert_config(num_hidden_groups=3)
        assert check_config_consistency(cfg, _albert_shapes()) == [
            ConfigInconsistency(field="num_hidden_groups", declared=3, observed=[1])
        ]

    def test_edited_inner_group_num_is_flagged(self) -> None:
        cfg = _albert_config(inner_group_num=2)
        assert check_config_consistency(cfg, _albert_shapes()) == [
            ConfigInconsistency(field="inner_group_num", declared=2, observed=[1])
        ]

    def test_group_fields_on_other_models_do_not_skip_layer_check(self) -> None:
        cfg = _llama_config(num_hidden_layers=8, num_hidden_groups=1, inner_group_num=4)
        assert _fields(check_config_consistency(cfg, _llama_shapes())) == [
            "num_hidden_layers"
        ]


class TestIntermediateSize:
    def test_edited_intermediate_size_is_flagged(self) -> None:
        cfg = _llama_config(intermediate_size=256)
        assert _fields(check_config_consistency(cfg, _llama_shapes())) == [
            "intermediate_size"
        ]

    def test_fused_gate_up_projection_is_accepted(self) -> None:
        shapes = {
            k: v
            for k, v in _llama_shapes().items()
            if "gate_proj" not in k and "up_proj" not in k and "down_proj" not in k
        }
        for i in range(4):
            shapes[f"model.layers.{i}.mlp.gate_up_proj.weight"] = (2 * 176, 64)
        assert check_config_consistency(_llama_config(), shapes) == []

    @pytest.mark.parametrize(
        "moe_field", ["num_experts", "num_local_experts", "n_routed_experts"]
    )
    def test_moe_models_skip_intermediate_check(self, moe_field: str) -> None:
        cfg = _llama_config(intermediate_size=256, **{moe_field: 8})
        assert check_config_consistency(cfg, _llama_shapes()) == []


# -- collect_shapes ----------------------------------------------------


class TestCollectShapes:
    def test_collects_tensor_shapes(self) -> None:
        sd = {"a": torch.zeros(3, 4), "b": torch.zeros(5), "meta": "x"}
        assert collect_shapes(sd) == {"a": (3, 4), "b": (5,)}


# -- similarity(metadata_trusted=False) --------------------------------


class TestUntrustedMetadataSkipsTiers:
    def test_tier1_pair_becomes_tier3(self) -> None:
        fp = _make_fp(arch_hash="same", family_hash="fam")
        assert similarity(fp, fp).tier == 1
        result = similarity(fp, fp, metadata_trusted=False)
        assert result.tier == 3
        assert result.match_type == "soft_match"

    def test_tier2_pair_becomes_tier3(self) -> None:
        fp_a = _make_fp(arch_hash="a", family_hash="fam")
        fp_b = _make_fp(arch_hash="b", family_hash="fam")
        assert similarity(fp_a, fp_b).tier == 2
        assert similarity(fp_a, fp_b, metadata_trusted=False).tier == 3


# -- Extraction and cache round-trip -----------------------------------


def _tiny_state_dict(layers: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        name: torch.randn(*shape)
        for name, shape in _llama_shapes(vocab=200, layers=layers).items()
    }


class TestExtractSignalsRunsCheck:
    def test_consistent_model_yields_empty_list(self) -> None:
        cfg = _llama_config(vocab_size=200, num_hidden_layers=2)
        ws = extract_signals(_tiny_state_dict(), cfg)
        assert ws is not None
        assert ws.config_inconsistencies == []

    def test_edited_config_is_reported(self) -> None:
        cfg = _llama_config(vocab_size=200, num_hidden_layers=6)
        ws = extract_signals(_tiny_state_dict(), cfg)
        assert ws is not None
        assert _fields(ws.config_inconsistencies or []) == ["num_hidden_layers"]

    def test_no_config_means_not_checked(self) -> None:
        ws = extract_signals(_tiny_state_dict(), None)
        assert ws is not None
        assert ws.config_inconsistencies is None


class TestCacheRoundTrip:
    @staticmethod
    def _ws(ci: list[ConfigInconsistency] | None) -> WeightSignalFeatures:
        return WeightSignalFeatures(
            hidden_size=64,
            num_layers=2,
            eas_self_sim=None,
            eas_anchor_count=0,
            nlf_vector=np.ones(4),
            nlf_mode="direct",
            nlf_num_layers=2,
            lep_profile=None,
            end_histogram=None,
            wsp_signature=None,
            wvc_layer_sigs=None,
            config_inconsistencies=ci,
        )

    def test_findings_survive_round_trip(self) -> None:
        ci = [ConfigInconsistency(field="vocab_size", declared=900, observed=[1000])]
        restored = WeightSignalFeatures.from_cache_dict(self._ws(ci).to_cache_dict())
        assert restored.config_inconsistencies == ci

    def test_unchecked_survives_round_trip(self) -> None:
        restored = WeightSignalFeatures.from_cache_dict(self._ws(None).to_cache_dict())
        assert restored.config_inconsistencies is None

    def test_legacy_cache_entry_is_unchecked(self) -> None:
        legacy = self._ws(None).to_cache_dict()
        del legacy["config_inconsistencies"]
        restored = WeightSignalFeatures.from_cache_dict(legacy)
        assert restored.config_inconsistencies is None
