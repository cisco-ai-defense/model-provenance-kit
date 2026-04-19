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

"""Tests for core/signals/weight_signals.py — weight-level signals."""

import math

import numpy as np
import pytest
import torch

from provenancekit.core.signals.weight_signals import (
    eas_similarity,
    end_similarity,
    extract_signals,
    lep_similarity,
    nlf_similarity,
    wsp_similarity,
    wvc_similarity,
)
from provenancekit.models.signals import WeightSignalFeatures

# ── Synthetic state_dict helpers ───────────────────────────────────


def _make_synthetic_state(
    vocab: int = 100,
    dim: int = 64,
    layers: int = 2,
) -> dict[str, torch.Tensor]:
    """Small synthetic state_dict for offline testing."""
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab, dim),
    }
    for i in range(layers):
        prefix = f"model.layers.{i}"
        state[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(dim, dim)
        state[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(dim, dim)
        state[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(dim, dim)
        state[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(dim, dim)
        state[f"{prefix}.input_layernorm.weight"] = torch.randn(dim)
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(dim)
    return state


def _make_features(**overrides: object) -> WeightSignalFeatures:
    """Build a minimal WeightSignalFeatures with optional overrides."""
    defaults: dict[str, object] = {
        "hidden_size": 64,
        "num_layers": 2,
        "eas_self_sim": None,
        "eas_anchor_count": 0,
        "nlf_vector": None,
        "nlf_mode": None,
        "nlf_num_layers": 0,
        "lep_profile": None,
        "end_histogram": None,
        "wsp_signature": None,
        "wvc_layer_sigs": None,
    }
    defaults.update(overrides)
    return WeightSignalFeatures(**defaults)  # type: ignore[arg-type]


# ── extract_signals (offline) ─────────────────────────────────────


class TestExtractSignals:
    def test_returns_weight_signal_features(self) -> None:
        state = _make_synthetic_state()
        feats = extract_signals(state, config=None)
        assert isinstance(feats, WeightSignalFeatures)

    def test_eas_populated_with_embedding(self) -> None:
        state = _make_synthetic_state(vocab=200, dim=64)
        feats = extract_signals(state, config=None)
        assert feats is not None
        assert feats.eas_anchor_count > 0

    def test_nlf_populated(self) -> None:
        state = _make_synthetic_state()
        feats = extract_signals(state, config=None)
        assert feats is not None
        assert feats.nlf_vector is not None
        assert feats.nlf_num_layers > 0

    def test_lep_populated(self) -> None:
        state = _make_synthetic_state()
        feats = extract_signals(state, config=None)
        assert feats is not None
        assert feats.lep_profile is not None
        assert len(feats.lep_profile) > 0

    def test_wvc_populated(self) -> None:
        state = _make_synthetic_state()
        feats = extract_signals(state, config=None)
        assert feats is not None
        assert feats.wvc_layer_sigs is not None

    def test_wsp_populated(self) -> None:
        state = _make_synthetic_state()
        feats = extract_signals(state, config=None)
        assert feats is not None
        assert feats.wsp_signature is not None

    def test_end_histogram_populated(self) -> None:
        state = _make_synthetic_state(vocab=200, dim=64)
        feats = extract_signals(state, config=None)
        assert feats is not None
        if feats.eas_anchor_count > 0:
            assert feats.end_histogram is not None

    def test_light_mode(self) -> None:
        state = _make_synthetic_state()
        feats = extract_signals(state, config=None, mode="light")
        assert isinstance(feats, WeightSignalFeatures)

    def test_empty_state_dict(self) -> None:
        feats = extract_signals({}, config=None)
        assert feats is not None
        assert feats.eas_self_sim is None
        assert feats.nlf_vector is None
        assert feats.lep_profile is None


# ── EAS similarity (offline) ──────────────────────────────────────


class TestEasSimilarity:
    def test_nan_on_none(self) -> None:
        assert math.isnan(eas_similarity(None, None))

    def test_nan_on_missing_sim(self) -> None:
        f = _make_features(eas_self_sim=None)
        assert math.isnan(eas_similarity(f, f))

    def test_identical_is_one(self) -> None:
        k = 32
        rng = np.random.RandomState(0)
        raw = rng.randn(k, k)
        mat = ((raw + raw.T) / 2).astype(np.float64)
        np.fill_diagonal(mat, 1.0)
        f = _make_features(eas_self_sim=mat, eas_anchor_count=k)
        score = eas_similarity(f, f)
        assert score == 1.0

    def test_different_below_one(self) -> None:
        rng = np.random.RandomState(0)
        m1 = rng.randn(32, 32).astype(np.float32)
        m2 = rng.randn(32, 32).astype(np.float32)
        f1 = _make_features(eas_self_sim=m1, eas_anchor_count=32)
        f2 = _make_features(eas_self_sim=m2, eas_anchor_count=32)
        score = eas_similarity(f1, f2)
        assert not math.isnan(score)
        assert score < 1.0


# ── NLF similarity (offline) ──────────────────────────────────────


class TestNlfSimilarity:
    def test_nan_on_none(self) -> None:
        assert math.isnan(nlf_similarity(None, None))

    def test_identical_direct(self) -> None:
        vec = np.random.randn(128).astype(np.float32)
        f = _make_features(nlf_vector=vec, nlf_mode="direct", nlf_num_layers=2)
        score = nlf_similarity(f, f)
        assert score == 1.0

    def test_different_lengths_stats_fallback(self) -> None:
        v1 = np.random.randn(128).astype(np.float32)
        v2 = np.random.randn(256).astype(np.float32)
        f1 = _make_features(nlf_vector=v1, nlf_mode="direct", nlf_num_layers=2)
        f2 = _make_features(nlf_vector=v2, nlf_mode="direct", nlf_num_layers=4)
        score = nlf_similarity(f1, f2)
        assert not math.isnan(score)


# ── LEP similarity (offline) ──────────────────────────────────────


class TestLepSimilarity:
    def test_nan_on_none(self) -> None:
        assert math.isnan(lep_similarity(None, None))

    def test_identical_is_one(self) -> None:
        profile = np.array([0.1, 0.5, 1.0, 0.8, 0.3])
        f = _make_features(lep_profile=profile)
        assert lep_similarity(f, f) == 1.0

    def test_different_lengths_interpolated(self) -> None:
        p1 = np.array([0.1, 0.5, 1.0, 0.8, 0.3])
        p2 = np.array([0.2, 0.6, 0.9])
        f1 = _make_features(lep_profile=p1)
        f2 = _make_features(lep_profile=p2)
        score = lep_similarity(f1, f2)
        assert not math.isnan(score)


# ── END similarity (offline) ──────────────────────────────────────


class TestEndSimilarity:
    def test_nan_on_none(self) -> None:
        assert math.isnan(end_similarity(None, None))

    def test_identical_is_one(self) -> None:
        hist = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        f = _make_features(end_histogram=hist)
        assert end_similarity(f, f) == 1.0


# ── WVC similarity (offline) ──────────────────────────────────────


class TestWvcSimilarity:
    def test_nan_on_none(self) -> None:
        assert math.isnan(wvc_similarity(None, None))

    def test_identical_is_one(self) -> None:
        sigs = {
            0: np.random.randn(128).astype(np.float32),
            1: np.random.randn(128).astype(np.float32),
        }
        f = _make_features(wvc_layer_sigs=sigs)
        assert wvc_similarity(f, f) == 1.0

    def test_nan_on_single_layer(self) -> None:
        sigs = {0: np.random.randn(128).astype(np.float32)}
        f = _make_features(wvc_layer_sigs=sigs)
        assert math.isnan(wvc_similarity(f, f))


# ── WSP similarity (offline) ──────────────────────────────────────


class TestWspSimilarity:
    def test_nan_on_none(self) -> None:
        assert math.isnan(wsp_similarity(None, None))

    def test_identical_is_one(self) -> None:
        sig = np.array([0.6, 0.01, 0.55, 0.02, 0.58, 0.015, 0.52, 0.03])
        f = _make_features(wsp_signature=sig)
        assert wsp_similarity(f, f) == 1.0


# ── NLF stats-mode fallback (mixed norm sizes) ───────────────────


class TestNlfStatsMode:
    def test_mixed_norm_sizes_use_stats(self) -> None:
        state: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.input_layernorm.weight": torch.randn(64),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.1.input_layernorm.weight": torch.randn(128),
        }
        feats = extract_signals(state, config=None)
        assert feats is not None
        assert feats.nlf_mode == "stats"
        assert feats.nlf_num_layers == 2


# ── Light mode with large tensors ────────────────────────────────


class TestLightModeLargeTensors:
    def test_light_mode_samples_large_tensor(self) -> None:
        state: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
            "model.layers.0.input_layernorm.weight": torch.randn(512),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(512, 512),
            "model.layers.1.input_layernorm.weight": torch.randn(512),
        }
        feats = extract_signals(state, config=None, mode="light")
        assert feats is not None
        assert feats.lep_profile is not None
        assert feats.wsp_signature is not None


# ── Two different models produce score < 1.0 ─────────────────────


class TestDifferentModels:
    def test_different_state_dicts_below_one(self) -> None:
        torch.manual_seed(0)
        state_a = _make_synthetic_state(vocab=200, dim=64, layers=3)
        torch.manual_seed(42)
        state_b = _make_synthetic_state(vocab=200, dim=64, layers=3)

        feats_a = extract_signals(state_a, config=None)
        feats_b = extract_signals(state_b, config=None)
        assert feats_a is not None
        assert feats_b is not None

        if feats_a.nlf_vector is not None and feats_b.nlf_vector is not None:
            score = nlf_similarity(feats_a, feats_b)
            assert not math.isnan(score)
            assert score < 1.0

        if feats_a.lep_profile is not None and feats_b.lep_profile is not None:
            score = lep_similarity(feats_a, feats_b)
            assert not math.isnan(score)
            assert score < 1.0

        if feats_a.wvc_layer_sigs is not None and feats_b.wvc_layer_sigs is not None:
            score = wvc_similarity(feats_a, feats_b)
            assert not math.isnan(score)
            assert score < 1.0


# ── Integration with extract_signals ──────────────────────────────


class TestExtractAndCompare:
    def test_self_similarity_all_signals(self) -> None:
        state = _make_synthetic_state(vocab=200, dim=64, layers=3)
        feats = extract_signals(state, config=None)
        assert feats is not None

        if feats.eas_self_sim is not None:
            assert eas_similarity(feats, feats) == 1.0
        if feats.nlf_vector is not None:
            assert nlf_similarity(feats, feats) == 1.0
        if feats.lep_profile is not None:
            assert lep_similarity(feats, feats) == 1.0
        if feats.end_histogram is not None:
            assert end_similarity(feats, feats) == 1.0
        if feats.wvc_layer_sigs is not None:
            assert wvc_similarity(feats, feats) == 1.0
        if feats.wsp_signature is not None:
            assert wsp_similarity(feats, feats) == 1.0


# ── Online golden tests ───────────────────────────────────────────


@pytest.mark.slow
class TestWeightSignalsOnline:
    def test_extract_gpt2(self) -> None:
        from provenancekit.services.model_loader import load_state_dict

        result = load_state_dict("gpt2")
        assert result.state_dict is not None
        feats = extract_signals(result.state_dict, result.config)
        assert feats is not None
        assert feats.eas_anchor_count == 64
        assert feats.nlf_num_layers > 0
        assert feats.lep_profile is not None

    def test_eas_gpt2_self(self) -> None:
        from provenancekit.services.model_loader import load_state_dict

        result = load_state_dict("gpt2")
        assert result.state_dict is not None
        feats = extract_signals(result.state_dict, result.config)
        assert eas_similarity(feats, feats) == 1.0

    def test_streaming_tiny_gpt2(self) -> None:
        from provenancekit.core.signals.weight_signals import (
            extract_signals_streaming,
        )

        feats = extract_signals_streaming("hf-internal-testing/tiny-random-gpt2")
        assert feats is not None
        assert feats.eas_anchor_count > 0
        # tiny-random-gpt2 has hidden_size=32 so norm tensors have < 64
        # elements and are skipped by the NLF extractor; only check >= 0.
        assert feats.nlf_num_layers >= 0
        assert feats.lep_profile is not None
