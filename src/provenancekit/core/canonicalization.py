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

"""Comparison-space weight canonicalization for similarity scoring.

This module produces *comparison-only* views of model weights so that
weight-level provenance signals (EAS, WVC, embedding-anchor cosines) are
robust to cheap function-preserving evasions:

* attention-head permutation,
* MLP/neuron permutation,
* adjacent-layer/channel rescaling,
* layer-norm gamma absorption.

Scale normalization operates in a comparison space and is not
function-preserving. The resulting representation is non-invertible and
must not be used for inference or model reconstruction. This design
intentionally trades invertibility for invariance to common evasion
strategies (channel rescaling and layer-norm absorption).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator, KeysView, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import structlog
import torch

try:
    from scipy.optimize import (  # type: ignore[import-untyped]
        linear_sum_assignment as _scipy_lsa,
    )

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised when scipy is absent
    _HAS_SCIPY = False

log = structlog.get_logger()

ScaleMode = Literal["comparison", "function_preserving"]
AlignMethod = Literal["greedy", "hungarian"]

_LAYER_INDEX_RE = re.compile(r"(?:layers?|h|block)\.\s*(\d+)\.")

_ATTN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "q": ("q_proj", "self.query", ".query.", "wq"),
    "k": ("k_proj", "self.key", ".key.", "wk"),
    "v": ("v_proj", "self.value", ".value.", "wv"),
    "o": ("o_proj", "out_proj", "attention.output.dense", "wo"),
}

_MLP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gate": ("gate_proj",),
    "up": ("up_proj", "fc1", "dense_h_to_4h", "intermediate.dense", "wi_0", "wi"),
    "down": ("down_proj", "fc2", "dense_4h_to_h", "output.dense", "wi_1", "wo_mlp"),
}


# ── Public configuration / report types ───────────────────────────


@dataclass(frozen=True)
class CanonicalizationConfig:
    """Configuration knobs for comparison-space canonicalization.

    Attributes:
        enabled: Master switch. When ``False``, every helper is a no-op.
        align_permutations: Enable attention-head and MLP-channel
            permutation alignment of model B into model A's basis.
        normalize_scales: Enable per-channel scale normalization of
            paired linear layers.
        max_layers: Optional cap on the number of layers processed.
        head_alignment: Align attention heads when architecture metadata
            is available.
        mlp_alignment: Align MLP/intermediate channels.
        max_mlp_width: Skip MLP-channel permutation alignment when the
            intermediate width exceeds this many channels. The assignment
            solver is O(n*m) in the intermediate width, so wide layers
            are gated off by default (8192) and recorded in
            ``unsupported_layers``. ``None`` disables the gate and aligns
            MLP layers of any width.
        eps: Numerical floor used to guard against division by zero.
        method: Assignment solver. ``"hungarian"`` uses
            :func:`scipy.optimize.linear_sum_assignment`; falls back to
            greedy max-matching when scipy is unavailable.
        scale_mode: ``"comparison"`` (default) performs per-channel
            unit-norm normalization independently on each model and is
            *non-invertible*. ``"function_preserving"`` divides W_in by
            the channel norm and multiplies W_out by the same factor,
            preserving the forward pass. The function-preserving form is
            stricter and slower, and is offered for callers who need to
            reuse the canonicalized weights downstream.
    """

    enabled: bool = False
    align_permutations: bool = True
    normalize_scales: bool = True
    max_layers: int | None = None
    head_alignment: bool = True
    mlp_alignment: bool = True
    max_mlp_width: int | None = 8192
    eps: float = 1e-8
    method: AlignMethod = "hungarian"
    scale_mode: ScaleMode = "comparison"


@dataclass
class CanonicalizationReport:
    """Outcome metadata from a single canonicalization pass."""

    enabled: bool = False
    method: str = "hungarian"
    scale_mode: str = "comparison"
    non_invertible: bool = True
    layers_aligned: int = 0
    attention_heads_aligned: int = 0
    mlp_channels_aligned: int = 0
    scale_normalized: bool = False
    unsupported_layers: list[str] = field(default_factory=list)
    stability_warnings: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report as a JSON-friendly dict."""
        return {
            "enabled": self.enabled,
            "method": self.method,
            "scale_mode": self.scale_mode,
            "non_invertible": self.non_invertible,
            "layers_aligned": self.layers_aligned,
            "attention_heads_aligned": self.attention_heads_aligned,
            "mlp_channels_aligned": self.mlp_channels_aligned,
            "scale_normalized": self.scale_normalized,
            "unsupported_layers": list(self.unsupported_layers),
            "stability_warnings": list(self.stability_warnings),
            "skipped_reason": self.skipped_reason,
        }


# ── Comparison-only carriers ──────────────────────────────────────


@dataclass(frozen=True)
class CanonicalizedTensor:
    """Comparison-only tensor view produced by :class:`WeightCanonicalizer`.

    **IMPORTANT** — the wrapped tensor is comparison-only:
    outputs are NOT function-preserving transformations, MUST NOT be
    used for inference, serialization, or model export, and scale
    normalization intentionally discards invertibility to collapse
    equivalence classes for fingerprinting.
    """

    tensor: torch.Tensor
    name: str = ""

    @property
    def is_comparison_only(self) -> bool:
        """Marker flag used by runtime guards in inference paths."""
        return True


class ComparisonView(Mapping[str, torch.Tensor]):
    """A state-dict-like dict tagged as comparison-only, non-invertible.

    The class implements :class:`collections.abc.Mapping` so it can be
    iterated like a regular ``state_dict`` for similarity-extraction
    code, while carrying a ``is_comparison_only=True`` flag and the
    canonicalization report. Inference / model-export paths should call
    :func:`assert_not_comparison_view` before consuming a state dict.
    """

    is_comparison_only: bool = True

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        report: CanonicalizationReport,
    ) -> None:
        """Wrap *tensors* as a comparison-only state-dict view."""
        self._tensors: dict[str, torch.Tensor] = tensors
        self.report: CanonicalizationReport = report

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return the tensor stored under *key*."""
        return self._tensors[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over tensor names in insertion order."""
        return iter(self._tensors)

    def __len__(self) -> int:
        """Return the number of tensors carried by this view."""
        return len(self._tensors)

    def keys(self) -> KeysView[str]:
        """Return a ``KeysView`` over tensor names."""
        return self._tensors.keys()


def assert_not_comparison_view(state_dict: Any) -> None:
    """Runtime guard for code paths that must not see comparison views.

    Raises:
        RuntimeError: When *state_dict* (or any object) carries the
            ``is_comparison_only`` marker that the canonicalizer attaches
            to its outputs. Use this in inference / serialization /
            export entry points so a comparison view cannot be silently
            substituted for real weights.
    """
    if isinstance(state_dict, CanonicalizedTensor) or getattr(
        state_dict, "is_comparison_only", False
    ):
        raise RuntimeError(
            "Canonicalized tensors are comparison-only and cannot be used "
            "outside similarity scoring."
        )


# ── Tensor classification helpers ─────────────────────────────────


def _layer_index(name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(name)
    return int(match.group(1)) if match else None


def _matches_any(name: str, keywords: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in keywords)


def _attn_role(name: str) -> str | None:
    for role, keywords in _ATTN_KEYWORDS.items():
        if _matches_any(name, keywords):
            return role
    return None


def _mlp_role(name: str) -> str | None:
    lower = name.lower()
    # ``output.dense`` appears both in attention output and in MLP output
    # in BERT-family models. Disambiguate by checking whether ``intermediate``
    # appeared in the same prefix path.
    for role, keywords in _MLP_KEYWORDS.items():
        if _matches_any(name, keywords):
            if role == "down" and "attention" in lower and "output" in lower:
                # attention.output.dense is the attention O projection
                return None
            return role
    return None


# ── Solver helpers ────────────────────────────────────────────────


def _solve_assignment(
    cost: np.ndarray,
    method: AlignMethod,
) -> np.ndarray:
    """Return the column permutation that minimises the assignment cost.

    The returned array has length equal to ``cost.shape[0]``; entry ``i``
    is the column index of B that should be mapped onto A's row ``i``.
    """
    n = cost.shape[0]
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("assignment requires a square cost matrix")
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix must contain only finite values")

    if method == "hungarian" and _HAS_SCIPY:
        _, col = _scipy_lsa(cost)
        return np.asarray(col, dtype=np.int64)

    # Greedy fallback: scan every (row, col) cell in ascending cost order
    # and assign whenever both row and column are still free. Guarantees a
    # complete permutation since the cost matrix is square.
    #
    # The cells are ordered by sorting flat indices rather than building a
    # list of ``(cost, row, col)`` tuples: the tuple form materialised
    # ~n*m Python objects (~24 bytes each), which OOMs at LLM MLP widths
    # (n = intermediate_size). A stable argsort over the flattened float32
    # cost reproduces the same ordering — ties break by flat index, i.e.
    # lexicographically by (row, col), exactly as the tuple sort did — at
    # 4 bytes/cell plus the int index array.
    m = cost.shape[1]
    flat = cost.astype(np.float32, copy=False).reshape(-1)
    order = np.argsort(flat, kind="stable")
    perm = np.full(n, -1, dtype=np.int64)
    row_used = np.zeros(n, dtype=bool)
    col_used = np.zeros(m, dtype=bool)
    assigned = 0
    for idx in order:
        i, j = divmod(int(idx), m)
        if row_used[i] or col_used[j]:
            continue
        perm[i] = j
        row_used[i] = True
        col_used[j] = True
        assigned += 1
        if assigned == n:
            break
    assert all(int(p) >= 0 for p in perm), "greedy assignment incomplete"
    return perm


def _build_cost_matrix(sigs_a: np.ndarray, sigs_b: np.ndarray) -> np.ndarray:
    """Compute ``1 - cosine`` between every row of A and every row of B.

    The cost matrix is built in ``float32`` rather than the NumPy default
    ``float64``. This halves peak memory of the assignment step, which is
    O(n*m) and dominates on LLM-width MLP layers (n = intermediate_size).
    The precision loss is immaterial: the matrix holds cosine distances
    consumed by a permutation solver, not values carried into scoring.
    """
    a = sigs_a.astype(np.float32, copy=False)
    b = sigs_b.astype(np.float32, copy=False)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    a_unit = a / a_norm
    b_unit = b / b_norm
    sim = a_unit @ b_unit.T
    cost: np.ndarray = (1.0 - sim).astype(np.float32, copy=False)
    return cost


# ── Per-channel scale helpers ─────────────────────────────────────


def _per_channel_unit_norm(tensor: torch.Tensor, axis: int, eps: float) -> torch.Tensor:
    """Divide *tensor* by the L2 norm along *axis* (per-channel)."""
    flat_axis = axis if axis >= 0 else tensor.dim() + axis
    other_axes = [i for i in range(tensor.dim()) if i != flat_axis]
    norms = torch.linalg.vector_norm(tensor, dim=other_axes, keepdim=True).clamp(
        min=eps
    )
    result: torch.Tensor = tensor / norms
    return result


def _function_preserving_pair(
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function-preserving rescale of ``W_in`` (output dim) / ``W_out`` (input dim).

    Uses the per-channel L2 norm of ``W_in`` along its output axis as
    α; divides ``W_in`` rows by α and multiplies the corresponding
    columns of ``W_out`` by α. Composition ``W_out @ W_in`` is preserved
    up to numerical precision. Both tensors are then *also* unit-normed
    per-channel for comparison.
    """
    if w_in.dim() < 2 or w_out.dim() < 2:
        return w_in, w_out
    if w_in.shape[0] != w_out.shape[1]:
        return w_in, w_out
    alpha = (
        torch.linalg.vector_norm(w_in, dim=tuple(range(1, w_in.dim())))
        .clamp(min=eps)
        .reshape(-1)
    )
    new_in = w_in / alpha.reshape((-1,) + (1,) * (w_in.dim() - 1))
    new_out = w_out * alpha.reshape((1,) * (w_out.dim() - 1) + (-1,))
    return new_in, new_out


# ── Architecture metadata extraction ──────────────────────────────


def _resolve_arch_meta(
    config_a: Any | None,
    config_b: Any | None,
) -> dict[str, int | None]:
    """Collect the architecture fields we need for head/MLP alignment.

    Returns the *intersection* of A and B's hyper-parameters: alignment
    is only safe when the head and MLP shapes match exactly.
    """

    def _read(cfg: Any, name: str) -> int | None:
        if cfg is None:
            return None
        val = getattr(cfg, name, None)
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    fields = (
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
    )
    a_vals = {name: _read(config_a, name) for name in fields}
    b_vals = {name: _read(config_b, name) for name in fields}
    out: dict[str, int | None] = {}
    for name in fields:
        a, b = a_vals[name], b_vals[name]
        out[name] = a if a is not None and a == b else None
    return out


# ── Core canonicalizer ────────────────────────────────────────────


class WeightCanonicalizer:
    """Produces comparison-space views of model weights for similarity scoring.

    **IMPORTANT** — comparison-only contract, must not be relaxed:
    outputs are NOT function-preserving transformations, MUST NOT be
    used for inference, serialization, or model export, and scale
    normalization intentionally discards invertibility to collapse
    equivalence classes for fingerprinting.
    """

    def __init__(self, config: CanonicalizationConfig) -> None:
        """Bind a configuration object."""
        self._config = config

    @property
    def config(self) -> CanonicalizationConfig:
        """Active configuration."""
        return self._config

    # ── Public entry point ────────────────────────────────────────

    def canonicalize_pair(
        self,
        state_dict_a: Mapping[str, torch.Tensor],
        state_dict_b: Mapping[str, torch.Tensor],
        config_a: Any | None = None,
        config_b: Any | None = None,
    ) -> tuple[ComparisonView, ComparisonView, CanonicalizationReport]:
        """Return comparison-space views for two state dicts.

        Model B is aligned into model A's basis (head and channel
        permutations). Both models are then independently scale-normalized
        (per-channel) to collapse cheap rescaling evasions.

        Args:
            state_dict_a: Reference model's state dict.
            state_dict_b: Suspect model's state dict.
            config_a: Optional ``AutoConfig``-like object for model A.
            config_b: Optional ``AutoConfig``-like object for model B.

        Returns:
            A tuple ``(comparison_view_a, comparison_view_b, report)``.
            Both views are :class:`ComparisonView` instances with
            ``is_comparison_only=True``; passing them to anything other
            than the similarity-scoring path is a contract violation.
        """
        cfg = self._config
        report = CanonicalizationReport(
            enabled=cfg.enabled,
            method=cfg.method,
            scale_mode=cfg.scale_mode,
            non_invertible=cfg.scale_mode == "comparison",
        )

        if not cfg.enabled:
            report.skipped_reason = "canonicalization_disabled"
            return (
                ComparisonView(_clone_state(state_dict_a), report),
                ComparisonView(_clone_state(state_dict_b), report),
                report,
            )

        out_a = _clone_state(state_dict_a)
        out_b = _clone_state(state_dict_b)

        arch_meta = _resolve_arch_meta(config_a, config_b)

        if cfg.align_permutations:
            self._align_layers(out_a, out_b, arch_meta, report)

        if cfg.normalize_scales:
            self._normalize_scales(out_a, report.unsupported_layers)
            self._normalize_scales(out_b, report.unsupported_layers)
            report.scale_normalized = True

        return (
            ComparisonView(out_a, report),
            ComparisonView(out_b, report),
            report,
        )

    # ── Permutation alignment ─────────────────────────────────────

    def _align_layers(
        self,
        out_a: dict[str, torch.Tensor],
        out_b: dict[str, torch.Tensor],
        arch_meta: dict[str, int | None],
        report: CanonicalizationReport,
    ) -> None:
        cfg = self._config
        layers_a = _group_by_layer(out_a)
        layers_b = _group_by_layer(out_b)
        common = sorted(set(layers_a.keys()) & set(layers_b.keys()))
        if cfg.max_layers is not None:
            common = common[: cfg.max_layers]

        for li in common:
            a_names = layers_a[li]
            b_names = layers_b[li]

            if cfg.head_alignment:
                aligned_heads = self._align_attention_heads(
                    out_a, out_b, a_names, b_names, arch_meta, report
                )
                if aligned_heads:
                    report.attention_heads_aligned += aligned_heads
                    report.layers_aligned += 1

            if cfg.mlp_alignment:
                aligned_channels = self._align_mlp_channels(
                    out_a, out_b, a_names, b_names, arch_meta, report
                )
                if aligned_channels:
                    report.mlp_channels_aligned += aligned_channels

    def _align_attention_heads(
        self,
        out_a: dict[str, torch.Tensor],
        out_b: dict[str, torch.Tensor],
        a_names: dict[str, list[str]],
        b_names: dict[str, list[str]],
        arch_meta: dict[str, int | None],
        report: CanonicalizationReport,
    ) -> int:
        a_attn = a_names.get("attn", [])
        b_attn = b_names.get("attn", [])
        if not a_attn or not b_attn:
            return 0

        raw_a = {role: _first_with_role(a_attn, _attn_role, role) for role in "qkv"}
        raw_b = {role: _first_with_role(b_attn, _attn_role, role) for role in "qkv"}
        if not all(raw_a.values()) or not all(raw_b.values()):
            return 0
        roles_a: dict[str, str] = {k: v for k, v in raw_a.items() if v is not None}
        roles_b: dict[str, str] = {k: v for k, v in raw_b.items() if v is not None}
        if len(roles_a) != 3 or len(roles_b) != 3:
            return 0

        q_a_name = roles_a["q"]
        q_a, q_b = out_a[q_a_name], out_b[roles_b["q"]]
        k_a, k_b = out_a[roles_a["k"]], out_b[roles_b["k"]]
        v_a, v_b = out_a[roles_a["v"]], out_b[roles_b["v"]]

        # Need 2-D tensors with matching shapes between A and B.
        if any(t.dim() != 2 for t in (q_a, q_b, k_a, k_b, v_a, v_b)):
            report.unsupported_layers.append(f"attn:{q_a_name}:non_2d")
            return 0
        if q_a.shape != q_b.shape or k_a.shape != k_b.shape or v_a.shape != v_b.shape:
            report.unsupported_layers.append(f"attn:{q_a_name}:shape_mismatch")
            return 0

        n_heads = arch_meta.get("num_attention_heads")
        head_dim = arch_meta.get("head_dim")
        hidden = arch_meta.get("hidden_size") or q_a.shape[1]
        if n_heads is None:
            report.unsupported_layers.append(f"attn:{q_a_name}:missing_num_heads")
            return 0
        n_kv = arch_meta.get("num_key_value_heads") or n_heads
        if head_dim is None and hidden:
            head_dim = hidden // max(n_heads, 1)
        if head_dim is None or head_dim <= 0 or n_heads <= 0:
            report.unsupported_layers.append(f"attn:{q_a_name}:invalid_head_dim")
            return 0
        if q_a.shape[0] != n_heads * head_dim:
            report.unsupported_layers.append(f"attn:{q_a_name}:head_shape_mismatch")
            return 0

        sigs_a, sigs_b = _attention_signatures(
            q_a, k_a, v_a, q_b, k_b, v_b, n_heads, head_dim, n_kv or n_heads
        )
        if sigs_a is None or sigs_b is None:
            return 0

        cost = _build_cost_matrix(sigs_a, sigs_b)
        perm = _solve_assignment(cost, self._config.method)

        # Stability check: average self-cosine before vs after on B.
        before = float(np.mean(np.diag(1.0 - cost)))
        # Compute average aligned cosine using perm
        after_sims = [
            float(1.0 - cost[i, perm[i]]) for i in range(len(perm)) if perm[i] >= 0
        ]
        after = float(np.mean(after_sims)) if after_sims else 0.0
        if after - before > 0.5:
            report.stability_warnings.append(
                f"layer:{q_a_name}:perm_jump:{before:.3f}->{after:.3f}"
            )

        # Derive the KV-head permutation *before* mutating ``out_b``. A
        # GQA group-split bail must leave the comparison views byte-for-byte
        # untouched; permuting Q up-front would silently reorder B's q_proj
        # while the report still claims nothing was aligned.
        if n_heads == n_kv:
            kv_perm = perm
        else:
            # GQA/MQA: derive the KV-head permutation group-wise. Every Q
            # head in A-side group ``g`` must map to the same B-side KV
            # group; otherwise the Q permutation is incompatible with the
            # KV grouping and the canonicalizer must not claim alignment.
            ratio = n_heads // n_kv
            kv_perm = np.empty(n_kv, dtype=np.int64)
            for g in range(n_kv):
                targets = {int(perm[g * ratio + r]) // ratio for r in range(ratio)}
                if len(targets) != 1:
                    report.unsupported_layers.append(f"attn:{q_a_name}:gqa_group_split")
                    return 0
                kv_perm[g] = targets.pop()
            if sorted(kv_perm.tolist()) != list(range(n_kv)):
                report.unsupported_layers.append(f"attn:{q_a_name}:gqa_perm_invalid")
                return 0

        # All permutations validated — apply to Q/K/V (output dim) and to
        # O (input dim).
        out_b[roles_b["q"]] = _permute_attention_out(q_b, perm, head_dim)
        out_b[roles_b["k"]] = _permute_attention_out(k_b, kv_perm, head_dim)
        out_b[roles_b["v"]] = _permute_attention_out(v_b, kv_perm, head_dim)

        o_a_name = _first_with_role(a_attn, _attn_role, "o")
        o_b_name = _first_with_role(b_attn, _attn_role, "o")
        if o_a_name and o_b_name:
            o_b = out_b[o_b_name]
            if o_b.dim() == 2 and o_b.shape[1] == n_heads * head_dim:
                out_b[o_b_name] = _permute_attention_in(o_b, perm, head_dim)
            else:
                report.unsupported_layers.append(f"attn:{o_b_name}:o_shape_mismatch")
        return n_heads

    def _align_mlp_channels(
        self,
        out_a: dict[str, torch.Tensor],
        out_b: dict[str, torch.Tensor],
        a_names: dict[str, list[str]],
        b_names: dict[str, list[str]],
        arch_meta: dict[str, int | None],
        report: CanonicalizationReport,
    ) -> int:
        a_mlp = a_names.get("mlp", [])
        b_mlp = b_names.get("mlp", [])
        if not a_mlp or not b_mlp:
            return 0

        up_a_name_opt = _first_with_role(a_mlp, _mlp_role, "up")
        up_b_name_opt = _first_with_role(b_mlp, _mlp_role, "up")
        down_a_name_opt = _first_with_role(a_mlp, _mlp_role, "down")
        down_b_name_opt = _first_with_role(b_mlp, _mlp_role, "down")
        if (
            up_a_name_opt is None
            or up_b_name_opt is None
            or down_a_name_opt is None
            or down_b_name_opt is None
        ):
            return 0
        up_a_name: str = up_a_name_opt
        up_b_name: str = up_b_name_opt
        down_a_name: str = down_a_name_opt
        down_b_name: str = down_b_name_opt

        up_a, up_b = out_a[up_a_name], out_b[up_b_name]
        down_a, down_b = out_a[down_a_name], out_b[down_b_name]
        if any(t.dim() != 2 for t in (up_a, up_b, down_a, down_b)):
            report.unsupported_layers.append(f"mlp:{up_a_name}:non_2d")
            return 0
        if up_a.shape != up_b.shape or down_a.shape != down_b.shape:
            report.unsupported_layers.append(f"mlp:{up_a_name}:shape_mismatch")
            return 0

        intermediate = up_a.shape[0]
        max_width = self._config.max_mlp_width
        if max_width is not None and intermediate > max_width:
            # The assignment cost matrix and its solve are O(n*m) in the
            # intermediate width. Gate wide MLP layers off rather than
            # build a multi-GB cost matrix; record the skip so callers
            # can see alignment was not attempted for this block.
            report.unsupported_layers.append(f"mlp:{up_a_name}:mlp_width_exceeded")
            return 0
        meta_inter = arch_meta.get("intermediate_size")
        if meta_inter is not None and meta_inter != intermediate:
            report.unsupported_layers.append(f"mlp:{up_a_name}:intermediate_mismatch")
            return 0
        if down_a.shape[1] != intermediate:
            report.unsupported_layers.append(f"mlp:{up_a_name}:down_shape_mismatch")
            return 0

        gate_a_name = _first_with_role(a_mlp, _mlp_role, "gate")
        gate_b_name = _first_with_role(b_mlp, _mlp_role, "gate")
        gate_a = out_a[gate_a_name] if gate_a_name else None
        gate_b = out_b[gate_b_name] if gate_b_name else None

        sigs_a = _mlp_signatures(up_a, down_a, gate_a)
        sigs_b = _mlp_signatures(up_b, down_b, gate_b)
        if sigs_a.shape != sigs_b.shape:
            return 0

        cost = _build_cost_matrix(sigs_a, sigs_b)
        perm = _solve_assignment(cost, self._config.method)

        before = float(np.mean(np.diag(1.0 - cost)))
        after_sims = [
            float(1.0 - cost[i, perm[i]]) for i in range(len(perm)) if perm[i] >= 0
        ]
        after = float(np.mean(after_sims)) if after_sims else 0.0
        if after - before > 0.5:
            report.stability_warnings.append(
                f"layer:{up_a_name}:perm_jump:{before:.3f}->{after:.3f}"
            )

        perm_t = torch.as_tensor(perm, dtype=torch.long)
        out_b[up_b_name] = up_b.index_select(0, perm_t).contiguous()
        out_b[down_b_name] = down_b.index_select(1, perm_t).contiguous()
        if gate_b_name and gate_b is not None:
            out_b[gate_b_name] = gate_b.index_select(0, perm_t).contiguous()

        return intermediate

    # ── Scale normalization ───────────────────────────────────────

    def _normalize_scales(
        self,
        state: dict[str, torch.Tensor],
        unsupported: list[str],
    ) -> None:
        cfg = self._config
        layers = _group_by_layer(state)

        for _, names in layers.items():
            attn = names.get("attn", [])
            mlp = names.get("mlp", [])

            if cfg.scale_mode == "function_preserving":
                self._scale_pairs_function_preserving(state, attn, mlp, unsupported)
            else:
                self._scale_pairs_comparison(state, attn, mlp)

        # LayerNorm gamma vectors: tag and unit-norm them so they cannot
        # dominate cosine similarity once concatenated with other signals.
        for name, tensor in state.items():
            lower = name.lower()
            if tensor.dim() == 1 and ("layernorm" in lower or "rmsnorm" in lower):
                norm = float(torch.linalg.vector_norm(tensor).item())
                if norm > cfg.eps:
                    state[name] = tensor / norm

    def _scale_pairs_comparison(
        self,
        state: dict[str, torch.Tensor],
        attn: list[str],
        mlp: list[str],
    ) -> None:
        cfg = self._config
        # Q/K/V: per-row (output channel) unit-norm.
        for role in ("q", "k", "v", "o"):
            name = _first_with_role(attn, _attn_role, role)
            if not name:
                continue
            t = state[name]
            if t.dim() == 2:
                axis = 1 if role == "o" else 0
                state[name] = _per_channel_unit_norm(t, axis=axis, eps=cfg.eps)

        # Gate / up: per-row; down: per-column (input channel).
        for role in ("gate", "up"):
            name = _first_with_role(mlp, _mlp_role, role)
            if not name:
                continue
            t = state[name]
            if t.dim() == 2:
                state[name] = _per_channel_unit_norm(t, axis=0, eps=cfg.eps)

        down = _first_with_role(mlp, _mlp_role, "down")
        if down:
            t = state[down]
            if t.dim() == 2:
                state[down] = _per_channel_unit_norm(t, axis=1, eps=cfg.eps)

    def _scale_pairs_function_preserving(
        self,
        state: dict[str, torch.Tensor],
        attn: list[str],
        mlp: list[str],
        unsupported: list[str],
    ) -> None:
        cfg = self._config
        # Attention V → O is the canonical adjacent-linear pair.
        v_name = _first_with_role(attn, _attn_role, "v")
        o_name = _first_with_role(attn, _attn_role, "o")
        if v_name and o_name:
            v_t = state[v_name]
            o_t = state[o_name]
            if v_t.dim() == 2 and o_t.dim() == 2 and v_t.shape[0] == o_t.shape[1]:
                v_t, o_t = _function_preserving_pair(v_t, o_t, cfg.eps)
                state[v_name] = v_t
                state[o_name] = o_t
            else:
                unsupported.append(f"attn_pair:{v_name}|{o_name}")

        up_name = _first_with_role(mlp, _mlp_role, "up")
        down_name = _first_with_role(mlp, _mlp_role, "down")
        if up_name and down_name:
            u_t = state[up_name]
            d_t = state[down_name]
            if u_t.dim() == 2 and d_t.dim() == 2 and u_t.shape[0] == d_t.shape[1]:
                u_t, d_t = _function_preserving_pair(u_t, d_t, cfg.eps)
                state[up_name] = u_t
                state[down_name] = d_t
            else:
                unsupported.append(f"mlp_pair:{up_name}|{down_name}")


# ── Internal helpers ──────────────────────────────────────────────


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a shallow-clone dict (tensors are detached & cloned)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone()
        else:  # pragma: no cover - defensive
            out[k] = v
    return out


def _group_by_layer(
    state: Mapping[str, torch.Tensor],
) -> dict[int, dict[str, list[str]]]:
    """Group tensor names by layer index and broad role (attn/mlp)."""
    groups: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: {"attn": [], "mlp": []}
    )
    for name in state:
        li = _layer_index(name)
        if li is None:
            continue
        if _attn_role(name) is not None:
            groups[li]["attn"].append(name)
        elif _mlp_role(name) is not None:
            groups[li]["mlp"].append(name)
    return groups


def _first_with_role(
    names: list[str],
    classifier: Any,
    role: str,
) -> str | None:
    """Return the first *names* entry whose ``classifier`` returns *role*."""
    for n in names:
        if classifier(n) == role:
            return n
    return None


def _attention_signatures(
    q_a: torch.Tensor,
    k_a: torch.Tensor,
    v_a: torch.Tensor,
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    n_heads: int,
    head_dim: int,
    n_kv: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build per-head signature matrices for assignment cost scoring."""
    if n_heads <= 0 or head_dim <= 0:
        return None, None

    def _split(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> np.ndarray:
        q_view = q.reshape(n_heads, head_dim, -1).reshape(n_heads, -1)
        # KV may have fewer heads under GQA: tile or slice to n_heads.
        if k.shape[0] == n_heads * head_dim:
            k_view = k.reshape(n_heads, head_dim, -1).reshape(n_heads, -1)
            v_view = v.reshape(n_heads, head_dim, -1).reshape(n_heads, -1)
        elif k.shape[0] == n_kv * head_dim and n_kv > 0 and n_heads % n_kv == 0:
            ratio = n_heads // n_kv
            k_view = (
                k.reshape(n_kv, head_dim, -1)
                .repeat_interleave(ratio, dim=0)
                .reshape(n_heads, -1)
            )
            v_view = (
                v.reshape(n_kv, head_dim, -1)
                .repeat_interleave(ratio, dim=0)
                .reshape(n_heads, -1)
            )
        else:
            return q_view.float().cpu().numpy()
        cat = torch.cat([q_view, k_view, v_view], dim=1)
        return cat.float().cpu().numpy()

    return _split(q_a, k_a, v_a), _split(q_b, k_b, v_b)


def _mlp_signatures(
    up: torch.Tensor,
    down: torch.Tensor,
    gate: torch.Tensor | None,
) -> np.ndarray:
    """Per-channel MLP signature combining up/gate rows with down columns."""
    sigs = [up.float()]
    if gate is not None and gate.shape == up.shape:
        sigs.append(gate.float())
    # Down's columns are intermediate channels — transpose so each row is one channel.
    sigs.append(down.float().t())
    cat = torch.cat(sigs, dim=1)
    return cat.cpu().numpy()


def _permute_attention_out(
    tensor: torch.Tensor, perm: np.ndarray, head_dim: int
) -> torch.Tensor:
    """Permute output-dim heads of an attention projection (rows)."""
    n_heads = tensor.shape[0] // head_dim
    if perm.shape[0] != n_heads:
        # Mismatched perm — leave tensor unchanged.
        return tensor
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    reshaped = tensor.reshape(n_heads, head_dim, *tensor.shape[1:])
    permuted = reshaped.index_select(0, perm_t)
    return permuted.reshape(n_heads * head_dim, *tensor.shape[1:]).contiguous()


def _permute_attention_in(
    tensor: torch.Tensor, perm: np.ndarray, head_dim: int
) -> torch.Tensor:
    """Permute input-dim heads of the attention output projection (cols)."""
    n_heads = tensor.shape[1] // head_dim
    if perm.shape[0] != n_heads:
        return tensor
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    reshaped = tensor.reshape(tensor.shape[0], n_heads, head_dim)
    permuted = reshaped.index_select(1, perm_t)
    return permuted.reshape(tensor.shape[0], n_heads * head_dim).contiguous()
