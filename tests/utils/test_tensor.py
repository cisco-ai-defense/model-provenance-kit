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

import math

import numpy as np

from provenancekit.utils.tensor import (
    EMBEDDING_CANDIDATES,
    TENSOR_CATEGORIES,
    classify_tensor_name,
    cosine_clamp,
    extract_layer_index,
    find_embedding_in_state_dict,
    find_embedding_name_in_weight_map,
    is_norm_tensor_name,
    norm_vector_to_stats,
)

# ── classify_tensor_name ──────────────────────────────────────────


class TestClassifyTensorName:
    def test_attention_projections(self):
        assert (
            classify_tensor_name("model.layers.0.self_attn.q_proj.weight") == "q_proj"
        )
        assert (
            classify_tensor_name("model.layers.0.self_attn.k_proj.weight") == "k_proj"
        )
        assert (
            classify_tensor_name("model.layers.0.self_attn.v_proj.weight") == "v_proj"
        )
        assert (
            classify_tensor_name("model.layers.0.self_attn.o_proj.weight") == "o_proj"
        )

    def test_mlp_projections(self):
        assert (
            classify_tensor_name("model.layers.0.mlp.gate_proj.weight") == "gate_proj"
        )
        assert classify_tensor_name("model.layers.0.mlp.up_proj.weight") == "up_proj"
        assert (
            classify_tensor_name("model.layers.0.mlp.down_proj.weight") == "down_proj"
        )

    def test_embedding(self):
        assert classify_tensor_name("model.embed_tokens.weight") == "embedding"
        assert classify_tensor_name("transformer.wte.weight") == "embedding"

    def test_norm(self):
        assert classify_tensor_name("model.layers.0.input_layernorm.weight") == "norm"

    def test_lm_head(self):
        assert classify_tensor_name("lm_head.weight") == "lm_head"

    def test_unknown_returns_other(self):
        assert classify_tensor_name("some.random.thing") == "other"

    def test_case_insensitive(self):
        assert (
            classify_tensor_name("MODEL.LAYERS.0.SELF_ATTN.Q_PROJ.weight") == "q_proj"
        )

    def test_categories_constant_is_populated(self):
        assert len(TENSOR_CATEGORIES) >= 10


# ── extract_layer_index ───────────────────────────────────────────


class TestExtractLayerIndex:
    def test_standard_layers_pattern(self):
        assert extract_layer_index("model.layers.5.self_attn.q_proj.weight") == 5

    def test_double_digit_layer(self):
        assert extract_layer_index("model.layers.12.mlp.gate_proj.weight") == 12

    def test_h_pattern(self):
        assert extract_layer_index("transformer.h.3.attn.c_attn.weight") == 3

    def test_block_pattern(self):
        assert extract_layer_index("encoder.block.7.layer.0.weight") == 7

    def test_no_layer_returns_none(self):
        assert extract_layer_index("model.embed_tokens.weight") is None
        assert extract_layer_index("lm_head.weight") is None


# ── is_norm_tensor_name ───────────────────────────────────────────


class TestIsNormTensorName:
    def test_layernorm(self):
        assert is_norm_tensor_name("model.layers.0.input_layernorm.weight") is True

    def test_rmsnorm(self):
        assert is_norm_tensor_name("model.layers.0.rmsnorm.weight") is True

    def test_ln_prefix(self):
        assert is_norm_tensor_name("transformer.h.0.ln_1.weight") is True

    def test_post_attention(self):
        assert (
            is_norm_tensor_name("model.layers.0.post_attention_layernorm.weight")
            is True
        )

    def test_non_norm(self):
        assert is_norm_tensor_name("model.layers.0.self_attn.q_proj.weight") is False

    def test_embedding_is_not_norm(self):
        assert is_norm_tensor_name("model.embed_tokens.weight") is False


# ── Embedding helpers ─────────────────────────────────────────────


class TestEmbeddingCandidates:
    def test_has_common_names(self):
        assert "model.embed_tokens.weight" in EMBEDDING_CANDIDATES
        assert "transformer.wte.weight" in EMBEDDING_CANDIDATES
        assert len(EMBEDDING_CANDIDATES) > 10


class TestFindEmbeddingInStateDict:
    def test_exact_match(self):
        state = {
            "model.embed_tokens.weight": np.ones((100, 64)),
            "model.layers.0.self_attn.q_proj.weight": np.ones((64, 64)),
        }
        result = find_embedding_in_state_dict(state)
        assert result is not None
        assert result.shape == (100, 64)

    def test_heuristic_fallback(self):
        state = {
            "custom.embedding_layer.weight": np.ones((100, 64)),
        }
        result = find_embedding_in_state_dict(state)
        assert result is not None

    def test_returns_none_when_missing(self):
        state = {
            "model.layers.0.self_attn.q_proj.weight": np.ones((64, 64)),
        }
        assert find_embedding_in_state_dict(state) is None

    def test_skips_position_embeddings(self):
        state = {
            "model.position_embeddings.weight": np.ones((512, 64)),
        }
        assert find_embedding_in_state_dict(state) is None

    def test_skips_1d_tensors(self):
        state = {"embed_tokens.bias": np.ones(64)}
        assert find_embedding_in_state_dict(state) is None


class TestFindEmbeddingNameInWeightMap:
    def test_exact_match(self):
        wm = {
            "model.embed_tokens.weight": "shard-0.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "shard-0.safetensors",
        }
        assert find_embedding_name_in_weight_map(wm) == "model.embed_tokens.weight"

    def test_heuristic_fallback(self):
        wm = {"custom.embed_stuff.weight": "shard-0.safetensors"}
        assert find_embedding_name_in_weight_map(wm) == "custom.embed_stuff.weight"

    def test_returns_none_when_missing(self):
        wm = {"model.layers.0.self_attn.q_proj.weight": "shard-0.safetensors"}
        assert find_embedding_name_in_weight_map(wm) is None


# ── cosine_clamp ──────────────────────────────────────────────────


class TestCosineClamp:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_clamp(a, a) == 1.0

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_clamp(a, b) == 0.0

    def test_zero_vector_returns_nan(self):
        assert math.isnan(cosine_clamp(np.zeros(3), np.array([1, 2, 3])))

    def test_both_zero_returns_nan(self):
        assert math.isnan(cosine_clamp(np.zeros(3), np.zeros(3)))

    def test_negative_cosine_clamped_to_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_clamp(a, b) == 0.0

    def test_result_clamped_between_0_and_1(self):
        rng = np.random.RandomState(42)
        for _ in range(50):
            a = rng.randn(128)
            b = rng.randn(128)
            result = cosine_clamp(a, b)
            assert 0.0 <= result <= 1.0


# ── norm_vector_to_stats ─────────────────────────────────────────


class TestNormVectorToStats:
    def test_basic_split(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = norm_vector_to_stats(vec, num_layers=2)
        assert len(result) == 8  # 2 layers × 4 stats

    def test_single_layer(self):
        vec = np.array([1.0, 2.0, 3.0])
        result = norm_vector_to_stats(vec, num_layers=1)
        assert len(result) == 4
        assert result[0] == float(vec.mean())

    def test_zero_layers_returns_original(self):
        vec = np.array([1.0, 2.0])
        result = norm_vector_to_stats(vec, num_layers=0)
        np.testing.assert_array_equal(result, vec)

    def test_negative_layers_returns_original(self):
        vec = np.array([1.0, 2.0])
        result = norm_vector_to_stats(vec, num_layers=-1)
        np.testing.assert_array_equal(result, vec)

    def test_stats_are_mean_std_max_min(self):
        vec = np.array([10.0, 20.0, 30.0])
        result = norm_vector_to_stats(vec, num_layers=1)
        assert result[0] == float(vec.mean())
        assert result[1] == float(vec.std())
        assert result[2] == float(vec.max())
        assert result[3] == float(vec.min())
