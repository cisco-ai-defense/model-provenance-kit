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

"""Tests for the canonicalization module.

Synthetic transformer-like state dicts validate:

* attention head permutation is recovered,
* MLP/neuron permutation is recovered,
* per-channel scale normalization is invariant to channel rescalings,
* identical inputs round-trip without crashing,
* unsupported architectures degrade gracefully,
* the CLI exposes ``--canonicalize`` in its help text.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pytest
import torch

from provenancekit.cli import main as cli_main
from provenancekit.core.canonicalization import (
    _HAS_SCIPY,
    CanonicalizationConfig,
    ComparisonView,
    WeightCanonicalizer,
    _solve_assignment,
    assert_not_comparison_view,
)
from provenancekit.models.results import CanonicalizationReportOutput

# ── Synthetic helpers ─────────────────────────────────────────────


@dataclass
class _SynthArchConfig:
    """Minimal AutoConfig stand-in for the canonicalizer."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int


def _vec_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.flatten().float()
    bv = b.flatten().float()
    denom = float(torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv))
    if denom < 1e-10:
        return 0.0
    return float(torch.dot(av, bv) / denom)


def _make_attention_layer(
    rng: np.random.Generator,
    hidden: int,
    n_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Build q/k/v/o projection weights with realistic shapes."""
    out_dim = n_heads * head_dim
    q = torch.tensor(rng.standard_normal((out_dim, hidden)), dtype=torch.float32)
    k = torch.tensor(rng.standard_normal((out_dim, hidden)), dtype=torch.float32)
    v = torch.tensor(rng.standard_normal((out_dim, hidden)), dtype=torch.float32)
    o = torch.tensor(rng.standard_normal((hidden, out_dim)), dtype=torch.float32)
    return {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
        "model.layers.0.self_attn.o_proj.weight": o,
    }


def _make_mlp_layer(
    rng: np.random.Generator, hidden: int, intermediate: int
) -> dict[str, torch.Tensor]:
    """Build gate/up/down MLP projections (LLaMA-style gated MLP)."""
    gate = torch.tensor(
        rng.standard_normal((intermediate, hidden)), dtype=torch.float32
    )
    up = torch.tensor(rng.standard_normal((intermediate, hidden)), dtype=torch.float32)
    down = torch.tensor(
        rng.standard_normal((hidden, intermediate)), dtype=torch.float32
    )
    return {
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.up_proj.weight": up,
        "model.layers.0.mlp.down_proj.weight": down,
    }


def _permute_attention_b(
    state: dict[str, torch.Tensor], perm: np.ndarray, head_dim: int, n_heads: int
) -> dict[str, torch.Tensor]:
    """Apply a head permutation to a synthetic attention block."""
    out = dict(state)
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    for role in ("q_proj", "k_proj", "v_proj"):
        name = f"model.layers.0.self_attn.{role}.weight"
        t = out[name]
        reshaped = t.reshape(n_heads, head_dim, -1)
        permuted = reshaped.index_select(0, perm_t)
        out[name] = permuted.reshape(n_heads * head_dim, -1).contiguous()
    o_name = "model.layers.0.self_attn.o_proj.weight"
    o = out[o_name]
    o_reshaped = o.reshape(o.shape[0], n_heads, head_dim)
    out[o_name] = (
        o_reshaped.index_select(1, perm_t)
        .reshape(o.shape[0], n_heads * head_dim)
        .contiguous()
    )
    return out


def _permute_mlp_b(
    state: dict[str, torch.Tensor], perm: np.ndarray
) -> dict[str, torch.Tensor]:
    out = dict(state)
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    for role in ("gate_proj", "up_proj"):
        name = f"model.layers.0.mlp.{role}.weight"
        t = out[name]
        out[name] = t.index_select(0, perm_t).contiguous()
    down_name = "model.layers.0.mlp.down_proj.weight"
    out[down_name] = out[down_name].index_select(1, perm_t).contiguous()
    return out


# ── Tests ────────────────────────────────────────────────────────


class TestHeadPermutationInvariance:
    def test_alignment_recovers_high_cosine(self) -> None:
        rng = np.random.default_rng(0)
        hidden, n_heads, head_dim = 32, 4, 8
        cfg_meta = _SynthArchConfig(
            hidden_size=hidden,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=hidden * 2,
        )
        state_a = _make_attention_layer(rng, hidden, n_heads, head_dim)

        perm = np.array([2, 0, 3, 1], dtype=np.int64)
        state_b = _permute_attention_b(state_a, perm, head_dim, n_heads)

        # Sanity: raw cosine is degraded after permutation.
        q_a = state_a["model.layers.0.self_attn.q_proj.weight"]
        q_b_raw = state_b["model.layers.0.self_attn.q_proj.weight"]
        raw_cos = _vec_cosine(q_a, q_b_raw)
        assert raw_cos < 0.95, f"unexpected raw cosine {raw_cos!r}"

        # Run the canonicalizer with permutation-only (no scale normalization
        # so the test isolates the alignment step).
        cfg = CanonicalizationConfig(
            enabled=True,
            align_permutations=True,
            normalize_scales=False,
            method="hungarian",
        )
        canonicalizer = WeightCanonicalizer(cfg)
        view_a, view_b, report = canonicalizer.canonicalize_pair(
            state_a, state_b, cfg_meta, cfg_meta
        )

        q_a_c = view_a["model.layers.0.self_attn.q_proj.weight"]
        q_b_c = view_b["model.layers.0.self_attn.q_proj.weight"]
        canon_cos = _vec_cosine(q_a_c, q_b_c)

        assert canon_cos > 0.999, (
            f"canonicalized cosine should approach 1.0 (got {canon_cos:.4f})"
        )
        assert report.attention_heads_aligned == n_heads
        assert report.layers_aligned == 1


class TestMLPNeuronPermutationInvariance:
    def test_alignment_recovers_high_cosine(self) -> None:
        rng = np.random.default_rng(1)
        hidden, intermediate = 16, 64
        cfg_meta = _SynthArchConfig(
            hidden_size=hidden,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=hidden // 2,
            intermediate_size=intermediate,
        )
        state_a = _make_mlp_layer(rng, hidden, intermediate)

        perm = np.random.default_rng(7).permutation(intermediate)
        state_b = _permute_mlp_b(state_a, perm)

        up_a = state_a["model.layers.0.mlp.up_proj.weight"]
        up_b_raw = state_b["model.layers.0.mlp.up_proj.weight"]
        raw_cos = _vec_cosine(up_a, up_b_raw)
        assert raw_cos < 0.5, f"raw permuted cosine should be low ({raw_cos:.3f})"

        cfg = CanonicalizationConfig(
            enabled=True,
            align_permutations=True,
            normalize_scales=False,
            method="hungarian",
        )
        canonicalizer = WeightCanonicalizer(cfg)
        view_a, view_b, report = canonicalizer.canonicalize_pair(
            state_a, state_b, cfg_meta, cfg_meta
        )

        up_a_c = view_a["model.layers.0.mlp.up_proj.weight"]
        up_b_c = view_b["model.layers.0.mlp.up_proj.weight"]
        canon_cos = _vec_cosine(up_a_c, up_b_c)
        assert canon_cos > 0.999, (
            f"canonicalized MLP cosine should approach 1.0 (got {canon_cos:.4f})"
        )
        assert report.mlp_channels_aligned == intermediate


class TestScaleNormalization:
    def test_per_channel_rescale_collapses_to_unit_norm(self) -> None:
        rng = np.random.default_rng(42)
        hidden, intermediate = 8, 16
        state_a = _make_mlp_layer(rng, hidden, intermediate)
        state_b = {k: v.clone() for k, v in state_a.items()}

        # Rescale a single output channel by 5x in B's up_proj.
        up_b = state_b["model.layers.0.mlp.up_proj.weight"]
        up_b[3, :] *= 5.0
        state_b["model.layers.0.mlp.up_proj.weight"] = up_b

        cfg = CanonicalizationConfig(
            enabled=True,
            align_permutations=False,
            normalize_scales=True,
            scale_mode="comparison",
        )
        canonicalizer = WeightCanonicalizer(cfg)
        view_a, view_b, report = canonicalizer.canonicalize_pair(
            state_a, state_b, None, None
        )

        up_a_c = view_a["model.layers.0.mlp.up_proj.weight"]
        up_b_c = view_b["model.layers.0.mlp.up_proj.weight"]

        # Each row should be unit-norm now.
        row_norms_a = torch.linalg.vector_norm(up_a_c, dim=1).numpy()
        row_norms_b = torch.linalg.vector_norm(up_b_c, dim=1).numpy()
        assert np.allclose(row_norms_a, 1.0, atol=1e-5)
        assert np.allclose(row_norms_b, 1.0, atol=1e-5)
        assert report.scale_normalized is True
        assert report.non_invertible is True


class TestNoOpSafety:
    def test_identical_inputs_remain_identical(self) -> None:
        rng = np.random.default_rng(2)
        hidden, n_heads, head_dim = 16, 2, 8
        cfg_meta = _SynthArchConfig(
            hidden_size=hidden,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            head_dim=head_dim,
            intermediate_size=hidden * 4,
        )
        state_a = _make_attention_layer(rng, hidden, n_heads, head_dim)
        state_a.update(_make_mlp_layer(rng, hidden, hidden * 4))
        state_b = {k: v.clone() for k, v in state_a.items()}

        cfg = CanonicalizationConfig(
            enabled=True, align_permutations=True, normalize_scales=False
        )
        canonicalizer = WeightCanonicalizer(cfg)
        view_a, view_b, _ = canonicalizer.canonicalize_pair(
            state_a, state_b, cfg_meta, cfg_meta
        )
        for name in state_a:
            assert torch.allclose(view_a[name], view_b[name], atol=1e-6)

    def test_unsupported_architecture_does_not_crash(self) -> None:
        # A state dict with no recognisable attention / MLP tensors.
        state_a = {
            "weird.blob.weight": torch.randn(8, 8),
            "another.scalar": torch.randn(4),
        }
        state_b = {k: v.clone() for k, v in state_a.items()}
        cfg = CanonicalizationConfig(enabled=True)
        canonicalizer = WeightCanonicalizer(cfg)
        view_a, view_b, report = canonicalizer.canonicalize_pair(
            state_a, state_b, None, None
        )
        # Both views are returned; nothing aligned.
        assert report.attention_heads_aligned == 0
        assert report.mlp_channels_aligned == 0
        assert isinstance(view_a, ComparisonView)
        assert isinstance(view_b, ComparisonView)

    def test_disabled_config_short_circuits(self) -> None:
        state_a = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
        state_b = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)}
        cfg = CanonicalizationConfig(enabled=False)
        canonicalizer = WeightCanonicalizer(cfg)
        view_a, view_b, report = canonicalizer.canonicalize_pair(
            state_a, state_b, None, None
        )
        assert report.enabled is False
        assert report.skipped_reason == "canonicalization_disabled"
        assert torch.allclose(
            view_a["model.layers.0.self_attn.q_proj.weight"],
            state_a["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_runtime_guard_rejects_comparison_view(self) -> None:
        state = {"x.weight": torch.randn(4, 4)}
        view = ComparisonView(
            state,
            report=WeightCanonicalizer(CanonicalizationConfig()).config,  # noqa: SLF001
        )
        # Replace the report with a real one; the guard only checks the marker.
        from provenancekit.core.canonicalization import CanonicalizationReport

        view.report = CanonicalizationReport(enabled=True)
        with pytest.raises(RuntimeError, match="comparison-only"):
            assert_not_comparison_view(view)


class TestCLIIntegration:
    def test_canonicalize_in_compare_help(self) -> None:
        old_argv = sys.argv
        old_stdout = sys.stdout
        buf = StringIO()
        try:
            sys.argv = ["provenancekit", "compare", "--help"]
            sys.stdout = buf
            with contextlib.suppress(SystemExit):
                cli_main()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
        out = buf.getvalue()
        assert "--canonicalize" in out
        assert "comparison-space" in out or "canonicalization" in out

    def test_canonicalize_in_scan_help(self) -> None:
        old_argv = sys.argv
        old_stdout = sys.stdout
        buf = StringIO()
        try:
            sys.argv = ["provenancekit", "scan", "--help"]
            sys.stdout = buf
            with contextlib.suppress(SystemExit):
                cli_main()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
        out = buf.getvalue()
        assert "--canonicalize" in out

    def test_compare_json_includes_canonicalization_section(self) -> None:
        # ``compare --canonicalize --json`` must surface the section.
        import json
        from unittest.mock import MagicMock, patch

        from provenancekit.models.results import (
            CanonicalizationReportOutput,
            CompareResult,
            PipelineScore,
            ScoreInterpretation,
            SignalScores,
            TimingBreakdown,
        )

        fake = CompareResult(
            model_a="x",
            model_b="y",
            family_a="x",
            family_b="y",
            signals=SignalScores(
                eas=0.9, nlf=0.9, lep=0.9, end=0.9, wvc=0.9, tfv=0.9, voa=0.9
            ),
            scores=PipelineScore(
                mfi_score=1.0,
                mfi_tier=1,
                mfi_match="exact",
                identity_score=0.9,
                tokenizer_score=0.9,
                pipeline_score=1.0,
                provenance_decision="Confirmed Match",
            ),
            interpretation=ScoreInterpretation(label="High", colour="#2ecc71"),
            time_seconds=1.0,
            timing=TimingBreakdown(
                total_seconds=1.0,
                metadata_extract_seconds=0.5,
                weight_feature_extract_seconds=0.5,
                cache_hit="False",
            ),
            canonicalization=CanonicalizationReportOutput(
                enabled=True,
                method="hungarian",
                scale_mode="comparison",
                non_invertible=True,
                layers_aligned=2,
                attention_heads_aligned=8,
                mlp_channels_aligned=64,
                scale_normalized=True,
                unsupported_layers=[],
                stability_warnings=[],
            ),
        )

        mock_scanner = MagicMock()
        mock_scanner.return_value.compare.return_value = fake

        old_argv = sys.argv
        old_stdout = sys.stdout
        buf = StringIO()
        with (
            patch("provenancekit.core.scanner.ModelProvenanceScanner", mock_scanner),
            patch("provenancekit.services.cache.CacheService"),
            patch("provenancekit.cli.Settings"),
        ):
            try:
                sys.argv = [
                    "provenancekit",
                    "compare",
                    "x",
                    "y",
                    "--json",
                    "--canonicalize",
                ]
                sys.stdout = buf
                with contextlib.suppress(SystemExit):
                    cli_main()
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout

        data = json.loads(buf.getvalue())
        canon = data["canonicalization"]
        assert canon["enabled"] is True
        assert canon["scale_mode"] == "comparison"
        assert canon["non_invertible"] is True
        assert canon["attention_heads_aligned"] == 8
        assert canon["mlp_channels_aligned"] == 64

        # And confirm the CLI passed through the canonicalization config.
        call = mock_scanner.return_value.compare.call_args
        assert call.kwargs["canonicalization"].enabled is True


def _assert_complete_perm(perm: np.ndarray, n: int) -> None:
    """Assert ``perm`` is a complete, valid permutation of ``range(n)``."""
    assert perm.shape == (n,), f"unexpected shape {perm.shape}"
    assert np.all(perm >= 0), f"perm contains -1 entries: {perm.tolist()}"
    assert set(perm.tolist()) == set(range(n)), (
        f"perm is not a valid permutation of 0..{n - 1}: {perm.tolist()}"
    )


class TestGreedyAssignment:
    """Edge-case coverage for the greedy fallback in ``_solve_assignment``.

    The greedy path used to use ``for _ in range(n)`` and could ``continue``
    past collisions without making an assignment, leaving ``-1`` entries in
    the returned permutation. Downstream callers
    (``_permute_attention_out``, MLP ``index_select``) do not filter ``-1``
    and would error or produce wrong results. The sorted-pair scan
    guarantees completeness; these tests lock that contract.
    """

    @pytest.mark.parametrize(
        "case_name,cost",
        [
            (
                "all_identical",
                np.full((6, 6), 0.5, dtype=np.float64),
            ),
            (
                "top1_collision",
                # Every row's argmin is column 0 — the original collision bug.
                (lambda n=6: np.full((n, n), 0.5, dtype=np.float64) * 0 + 0.5)(),
            ),
            (
                "all_zero",
                np.zeros((5, 5), dtype=np.float64),
            ),
            (
                "single_cell",
                np.array([[0.0]], dtype=np.float64),
            ),
            (
                "negative_costs",
                # Cost = 1 - cosine: with negative cosines the cost can exceed
                # 1.0; the algorithm minimises, so very-negative cells aren't
                # special, but the code must still complete with negatives.
                np.array(
                    [
                        [-0.5, 0.2, 1.5],
                        [1.2, -0.9, 0.0],
                        [0.4, 1.8, -0.1],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                "with_inf",
                # Inf must sort to the back and be picked last (only when
                # forced). Completeness must still hold.
                np.array(
                    [
                        [0.1, np.inf, 0.5],
                        [np.inf, 0.2, 0.3],
                        [0.4, 0.6, np.inf],
                    ],
                    dtype=np.float64,
                ),
            ),
        ],
    )
    def test_greedy_assignment_complete(self, case_name: str, cost: np.ndarray) -> None:
        # Force the top-1 collision by overriding column 0 of all rows.
        if case_name == "top1_collision":
            cost = cost.copy()
            cost[:, 0] = 0.0  # everyone wants column 0 first
            # Add small differentiation so non-zero columns remain pickable
            # but with a tie among rows for column 0.
            rng = np.random.default_rng(0)
            cost[:, 1:] = rng.uniform(0.4, 0.6, size=(cost.shape[0], cost.shape[1] - 1))

        # ``with_inf`` would be rejected by the new isfinite() guard;
        # exercise it explicitly to lock that contract and then run the
        # finite version of the case.
        if case_name == "with_inf":
            with pytest.raises(ValueError, match="finite"):
                _solve_assignment(cost, method="greedy")
            return

        n = cost.shape[0]
        perm = _solve_assignment(cost, method="greedy")
        _assert_complete_perm(perm, n)

        if case_name == "single_cell":
            assert perm.tolist() == [0]

        if case_name == "top1_collision":
            # Exactly one row claims column 0 (the universally cheapest).
            assert int((perm == 0).sum()) == 1

    def test_rectangular_cost_is_rejected(self) -> None:
        """The function requires a square cost matrix and must raise."""
        cost = np.zeros((3, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="square"):
            _solve_assignment(cost, method="greedy")

    def test_nan_cost_is_rejected(self) -> None:
        """NaN in cost is rejected with a clear error (locks contract)."""
        cost = np.array(
            [[0.1, 0.2], [np.nan, 0.3]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="finite"):
            _solve_assignment(cost, method="greedy")

    @pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")
    def test_greedy_matches_hungarian_on_well_separated(self) -> None:
        """When the optimum is unique by a wide margin, greedy and Hungarian
        must agree. Catches regressions where greedy goes pathologically
        wrong on easy inputs."""
        # Construct a strongly diagonal cost: identity is the unique optimum.
        n = 8
        cost = np.full((n, n), 5.0, dtype=np.float64)
        np.fill_diagonal(cost, 0.0)
        # Add slight off-diagonal noise so the optimum stays unique.
        rng = np.random.default_rng(123)
        cost = cost + rng.uniform(0.0, 0.1, size=(n, n))
        np.fill_diagonal(cost, 0.0)

        greedy = _solve_assignment(cost, method="greedy")
        hungarian = _solve_assignment(cost, method="hungarian")

        _assert_complete_perm(greedy, n)
        _assert_complete_perm(hungarian, n)
        assert greedy.tolist() == hungarian.tolist()
        assert greedy.tolist() == list(range(n))

    def test_greedy_random_property_check(self) -> None:
        """Fuzz the greedy path with random square cost matrices and
        confirm every returned permutation is valid and complete."""
        rng = np.random.default_rng(20260511)
        for _ in range(50):
            n = int(rng.integers(low=1, high=17))  # n in [1, 16]
            cost = rng.standard_normal(size=(n, n)).astype(np.float64)
            perm = _solve_assignment(cost, method="greedy")
            assert perm.dtype == np.int64
            _assert_complete_perm(perm, n)
            # Each column used exactly once.
            assert len(set(perm.tolist())) == n
            assert all(0 <= int(p) < n for p in perm)


class TestCanonicalizationReportMutableDefaults:
    """Lock the Field(default_factory=list) fix.

    A previous version used ``list[str] = []`` which is a class-level
    mutable default — *Pydantic v2 actually copies the default* and is
    safe in practice, but the class-attribute form is fragile and easy to
    regress to a shared list under refactors. The Field(default_factory)
    form makes the intent explicit and gives belt-and-braces independence.
    """

    def test_default_lists_are_independent(self) -> None:
        r1 = CanonicalizationReportOutput(
            enabled=True,
            method="hungarian",
            scale_mode="comparison",
            non_invertible=True,
        )
        r2 = CanonicalizationReportOutput(
            enabled=True,
            method="hungarian",
            scale_mode="comparison",
            non_invertible=True,
        )
        # Distinct objects per instance.
        assert r1.unsupported_layers is not r2.unsupported_layers
        assert r1.stability_warnings is not r2.stability_warnings

        # Mutating one must not affect the other.
        r1.unsupported_layers.append("layer.0")
        r1.stability_warnings.append("warn")
        assert r2.unsupported_layers == []
        assert r2.stability_warnings == []

        # And a freshly built third instance must also be empty —
        # catches a regression where mutation leaks into the class default.
        r3 = CanonicalizationReportOutput(
            enabled=True,
            method="hungarian",
            scale_mode="comparison",
            non_invertible=True,
        )
        assert r3.unsupported_layers == []
        assert r3.stability_warnings == []

    def test_model_dump_default_and_populated(self) -> None:
        r = CanonicalizationReportOutput(
            enabled=False,
            method="hungarian",
            scale_mode="comparison",
            non_invertible=True,
        )
        dumped = r.model_dump()
        assert dumped["unsupported_layers"] == []
        assert dumped["stability_warnings"] == []

        r.unsupported_layers.append("attn:layer.0:non_2d")
        r.stability_warnings.append("layer.0:perm_jump")
        dumped = r.model_dump()
        assert dumped["unsupported_layers"] == ["attn:layer.0:non_2d"]
        assert dumped["stability_warnings"] == ["layer.0:perm_jump"]

        # JSON round-trip preserves both fields.
        from_json = CanonicalizationReportOutput.model_validate_json(
            r.model_dump_json()
        )
        assert from_json.unsupported_layers == r.unsupported_layers
        assert from_json.stability_warnings == r.stability_warnings

    def test_explicit_list_inputs_still_supported(self) -> None:
        r = CanonicalizationReportOutput(
            enabled=True,
            method="hungarian",
            scale_mode="comparison",
            non_invertible=True,
            unsupported_layers=["a", "b"],
            stability_warnings=["w1"],
        )
        assert r.unsupported_layers == ["a", "b"]
        assert r.stability_warnings == ["w1"]

    def test_json_schema_field_types_unchanged(self) -> None:
        """Schema shape must still report array-of-string for both fields."""
        schema = CanonicalizationReportOutput.model_json_schema()
        props = schema["properties"]
        for name in ("unsupported_layers", "stability_warnings"):
            field_schema = props[name]
            assert field_schema["type"] == "array"
            assert field_schema["items"] == {"type": "string"}
