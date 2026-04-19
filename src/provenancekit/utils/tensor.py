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

"""Pure tensor-name, numeric, and parameter-estimation helpers.

This module is the shared foundation imported by ``core/signals/`` and other
weight-processing code. It has **no domain imports** — only stdlib, numpy,
and typing — so it can never create circular dependencies.
"""

import re
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

# ── Tensor classification ─────────────────────────────────────────

TENSOR_CATEGORIES: dict[str, list[str]] = {
    "q_proj": ["q_proj", "query", "self.query"],
    "k_proj": ["k_proj", "key", "self.key"],
    "v_proj": ["v_proj", "value", "self.value"],
    "o_proj": ["o_proj", "dense", "out_proj"],
    "gate_proj": ["gate_proj", "gate"],
    "up_proj": ["up_proj", "fc1", "dense_h_to_4h", "wi_0"],
    "down_proj": [
        "down_proj",
        "fc2",
        "dense_4h_to_h",
        "wi_1",
        "wo",
    ],
    "embedding": ["embed_tokens", "wte", "word_embeddings", "shared.weight"],
    "norm": ["layernorm", "rmsnorm", "ln_", "norm"],
    "lm_head": ["lm_head", "cls"],
}


def classify_tensor_name(name: str) -> str:
    """Classify a tensor name into a functional category.

    Matches against :data:`TENSOR_CATEGORIES` keywords (case-insensitive).
    Returns ``"other"`` when no keyword matches.
    """
    lower = name.lower()
    for category, keywords in TENSOR_CATEGORIES.items():
        if any(kw in lower for kw in keywords):
            return category
    return "other"


# ── Layer index extraction ────────────────────────────────────────

_LAYER_INDEX_RE = re.compile(r"(?:layers?|h|block)\.\s*(\d+)\.")


def extract_layer_index(name: str) -> int | None:
    """Extract the numeric layer index from a parameter name.

    Recognises patterns like ``layers.5.``, ``layer.12.``, ``h.0.``,
    ``block.3.``.  Returns ``None`` when no layer index is found
    (e.g. embedding or final-norm tensors).
    """
    match = _LAYER_INDEX_RE.search(name)
    return int(match.group(1)) if match else None


# ── Norm tensor detection ─────────────────────────────────────────

_NORM_KEYWORDS: list[str] = [
    "layernorm",
    "rmsnorm",
    "input_layernorm",
    "post_attention",
    "ln_",
    "norm",
]


def is_norm_tensor_name(name: str) -> bool:
    """Return ``True`` if *name* corresponds to a normalisation-layer weight."""
    lower = name.lower()
    return any(kw in lower for kw in _NORM_KEYWORDS)


# ── Embedding tensor helpers ──────────────────────────────────────

EMBEDDING_CANDIDATES: list[str] = [
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "embeddings.word_embeddings.weight",
    "gpt_neox.embed_in.weight",
    "transformer.word_embeddings.weight",
    "shared.weight",
    "model.shared.weight",
    "encoder.embed_tokens.weight",
    "model.encoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "decoder.embed_tokens.weight",
    "bert.embeddings.word_embeddings.weight",
    "roberta.embeddings.word_embeddings.weight",
    "deberta.embeddings.word_embeddings.weight",
    "distilbert.embeddings.word_embeddings.weight",
    "electra.embeddings.word_embeddings.weight",
    "albert.embeddings.word_embeddings.weight",
    "xlm-roberta.embeddings.word_embeddings.weight",
    "camembert.embeddings.word_embeddings.weight",
    "word_embeddings.weight",
    "wte.weight",
]


def find_embedding_in_state_dict(
    state: dict[str, Any],
) -> Any | None:
    """Locate the token-embedding weight tensor in a state dict.

    Tries the well-known names in :data:`EMBEDDING_CANDIDATES` first,
    then falls back to a heuristic search for any 2-D tensor whose name
    contains ``embed`` or ``wte`` (excluding positional embeddings).

    Returns ``None`` when no embedding tensor is found.
    """
    for name in EMBEDDING_CANDIDATES:
        if name in state:
            return state[name]

    for name, param in state.items():
        lower = name.lower()
        is_embed = "embed" in lower or "wte" in lower
        is_2d = hasattr(param, "shape") and len(param.shape) == 2
        if is_embed and is_2d and "position" not in lower:
            return param

    return None


def find_embedding_name_in_weight_map(
    weight_map: dict[str, str],
) -> str | None:
    """Locate the embedding tensor *name* in a safetensors weight-map index.

    The *weight_map* maps ``tensor_name → shard_filename``.
    Returns ``None`` when no embedding tensor is found.
    """
    for name in EMBEDDING_CANDIDATES:
        if name in weight_map:
            return name

    for name in weight_map:
        lower = name.lower()
        if ("embed" in lower or "wte" in lower) and "position" not in lower:
            return name

    return None


# ── Numeric helpers ───────────────────────────────────────────────


def cosine_clamp(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity clamped to [0, 1].

    Returns ``NaN`` when either vector has near-zero norm.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    norm_product = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if norm_product < 1e-10:
        return float("nan")
    cos = float(np.dot(arr_a, arr_b) / norm_product)
    return round(max(0.0, min(1.0, cos)), 4)


def norm_vector_to_stats(
    vec: np.ndarray,
    num_layers: int,
) -> np.ndarray:
    """Convert a raw norm-layer weight vector to per-layer statistics.

    Splits *vec* into *num_layers* equal chunks and computes
    ``(mean, std, max, min)`` for each chunk.  Used by NLF similarity
    when two models have different hidden sizes.
    """
    if num_layers <= 0:
        return vec
    chunk_size = len(vec) // num_layers
    if chunk_size <= 0:
        return vec

    stats: list[float] = []
    for i in range(num_layers):
        chunk = vec[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            stats.extend(
                [
                    float(chunk.mean()),
                    float(chunk.std()),
                    float(chunk.max()),
                    float(chunk.min()),
                ]
            )
    return np.array(stats)


# ── Parameter estimation ─────────────────────────────────────────


class _HasArchFields(Protocol):
    """Structural typing for objects carrying architecture dimensions.

    Both ``ScanModelInfo`` and ``MFIFingerprint`` satisfy this protocol,
    so callers never need to convert between types.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool | None
    hidden_act: str | None
    head_dim: int | None


_GATED_ACTIVATIONS: frozenset[str] = frozenset({"silu", "swiglu", "gegelu"})

_GATED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "llama",
        "mistral",
        "mixtral",
        "phi3",
        "phi4",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "deepseek_v2",
        "deepseek_v3",
        "cohere",
        "cohere2",
        "command-r",
        "starcoder2",
    }
)


def is_gated_mlp(info: _HasArchFields) -> bool:
    """Detect whether the architecture uses a gated MLP (3 projections).

    Checks both the activation function name and the ``model_type``
    attribute (when present) for known gated families such as LLaMA,
    Gemma, Qwen, etc.
    """
    act = (getattr(info, "hidden_act", None) or "").lower()
    if act in _GATED_ACTIVATIONS:
        return True
    mtype = getattr(info, "model_type", None)
    return bool(mtype and str(mtype).lower() in _GATED_MODEL_TYPES)


def estimate_param_count(info: _HasArchFields) -> int:
    """Estimate total parameter count from architectural dimensions.

    Accounts for GQA/MQA key-value heads, explicit ``head_dim``
    overrides, gated MLPs (SwiGLU/GeGLU), and tied word embeddings.

    Accepts any object whose attributes satisfy :class:`_HasArchFields`
    (e.g. ``ScanModelInfo``, ``MFIFingerprint``, ``AutoConfig``).
    Uses ``getattr`` with safe defaults so that objects missing optional
    attributes (e.g. ``AutoConfig`` for older architectures) still work.
    """
    h: int = getattr(info, "hidden_size", 0)
    n_layers: int = getattr(info, "num_hidden_layers", 0)
    v: int = getattr(info, "vocab_size", 0)
    n_heads: int = getattr(info, "num_attention_heads", 0)
    n_kv: int | None = getattr(info, "num_key_value_heads", None)
    if not n_kv:
        n_kv = n_heads
    inter: int = (
        getattr(info, "intermediate_size", 0) or getattr(info, "n_inner", 0) or 4 * h
    )
    tied = getattr(info, "tie_word_embeddings", None)
    if tied is None:
        tied = True
    head_dim: int | None = getattr(info, "head_dim", None)

    hd = head_dim if head_dim else (h // n_heads if n_heads else h)

    qkv = h * hd * (n_heads + 2 * n_kv)
    o_proj = n_heads * hd * h

    mlp = (3 if is_gated_mlp(info) else 2) * h * inter

    per_layer = qkv + o_proj + mlp

    embed = v * h
    lm_head = 0 if tied else v * h

    return embed + n_layers * per_layer + lm_head


_BUCKET_THRESHOLDS: list[tuple[int, str]] = [
    (1_000_000_000, "<=1B"),
    (10_000_000_000, "1-10B"),
    (40_000_000_000, "10-40B"),
]


def param_count_to_bucket(param_count: int) -> str:
    """Map a parameter count to a human-readable size bucket string."""
    for threshold, label in _BUCKET_THRESHOLDS:
        if param_count < threshold:
            return label
    return "40B+"


def compute_param_bucket(info: _HasArchFields) -> str:
    """Estimate parameters from architecture fields and return a size bucket.

    Convenience wrapper combining :func:`estimate_param_count` and
    :func:`param_count_to_bucket`.
    """
    return param_count_to_bucket(estimate_param_count(info))


def format_param_count(
    count: int,
    *,
    approximate: bool = False,
) -> str:
    """Format a parameter count as a human-readable string.

    Parameters
    ----------
    count:
        Raw parameter count.
    approximate:
        When ``True``, prepend ``~`` to indicate an estimate.
    """
    prefix = "~" if approximate else ""
    if count >= 1_000_000_000:
        return f"{prefix}{count / 1_000_000_000:.1f}B"
    return f"{prefix}{count / 1_000_000:.0f}M"
