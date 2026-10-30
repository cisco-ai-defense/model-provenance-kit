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

"""Cross-check declared ``config.json`` dimensions against stored tensor shapes.

MFI Tier 1/2 are computed from ``config.json`` alone, which can be edited
without touching the weights.  This module derives the same dimensions
from the tensors themselves so that a contradicting config can be
detected and the metadata gate skipped.

Checks (each is skipped when the needed tensors or fields are absent):

* ``vocab_size``        vs token-embedding row count
* ``hidden_size``       vs token-embedding column count or per-layer norm size
* ``num_hidden_layers`` vs layer count of a stored layer stack
  (encoder-decoder: ``encoder_layers`` / ``decoder_layers`` vs their own stack;
  ALBERT: ``num_hidden_groups`` / ``inner_group_num`` vs the shared layer groups)
* ``intermediate_size`` vs an MLP projection dimension (dense models only)
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import structlog

from provenancekit.core.signals.metadata import (
    _resolve_intermediate_size,
    _resolve_text_backbone,
)
from provenancekit.models.signals import ConfigInconsistency
from provenancekit.utils.tensor import (
    EMBEDDING_CANDIDATES,
    is_norm_tensor_name,
    is_quantization_aux_name,
)

log = structlog.get_logger()

Shapes = Mapping[str, Sequence[int]]

# Some checkpoints pad the embedding table past ``vocab_size`` up to a
# multiple of 64 or 128 for kernel efficiency.
VOCAB_PADDING_MULTIPLE = 64
VOCAB_PADDING_TOLERANCE = 128

# Multi-token-prediction heads (e.g. DeepSeek-V3) are stored as one extra
# trailing layer that ``num_hidden_layers`` does not count.
EXTRA_TRAILING_LAYERS_TOLERANCE = 1

_EMBEDDING_SUFFIXES: tuple[str, ...] = (
    "embed_tokens.weight",
    "word_embeddings.weight",
    "wte.weight",
    "embed_in.weight",
)

_LAYER_STACK_RE = re.compile(r"^(.*?(?:layers?|h|block))\.\s*(\d+)\.")

_ENCODER_DECODER_LAYER_FIELDS: dict[str, tuple[str, ...]] = {
    "encoder": ("encoder_layers", "num_layers"),
    "decoder": ("decoder_layers", "num_decoder_layers"),
}

_ALBERT_GROUP = "albert_layer_groups."

_MOE_CONFIG_FIELDS: tuple[str, ...] = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
)


def collect_shapes(state_dict: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return ``{tensor_name: shape}`` for every tensor in *state_dict*."""
    return {
        name: tuple(int(d) for d in t.shape)
        for name, t in state_dict.items()
        if hasattr(t, "shape")
    }


def check_config_consistency(config: Any, shapes: Shapes) -> list[ConfigInconsistency]:
    """Compare declared config dimensions with tensor-derived values.

    Args:
        config: ``transformers.PretrainedConfig`` for the model.  The text
            backbone is resolved the same way as MFI fingerprinting.
        shapes: ``{tensor_name: shape}`` covering at least the token
            embedding and all per-layer tensors.

    Returns:
        One :class:`ConfigInconsistency` per contradicted field; empty
        when everything that could be checked agrees.
    """
    cfg = _resolve_text_backbone(config)
    embedding = _find_embedding_shape(shapes)
    findings: list[ConfigInconsistency] = []

    if _is_encoder_decoder(cfg):
        layer_checks = _check_encoder_decoder_layers(cfg, shapes)
    elif _is_albert(cfg):
        layer_checks = _check_albert_layer_groups(cfg, shapes)
    else:
        layer_checks = [_check_num_hidden_layers(cfg, shapes)]
    for finding in (
        _check_vocab_size(cfg, embedding),
        _check_hidden_size(cfg, embedding, shapes),
        *layer_checks,
        _check_intermediate_size(cfg, shapes),
    ):
        if finding is not None:
            findings.append(finding)

    if findings:
        log.warning(
            "config_inconsistent",
            fields=[f.field for f in findings],
            details=[f.model_dump() for f in findings],
        )
    return findings


# -- Individual checks ------------------------------------------------


def _check_vocab_size(
    cfg: Any, embedding: tuple[int, int] | None
) -> ConfigInconsistency | None:
    declared = _positive_int(getattr(cfg, "vocab_size", 0))
    if declared is None or embedding is None:
        return None
    rows = embedding[0]
    padded = (
        0 < rows - declared < VOCAB_PADDING_TOLERANCE
        and rows % VOCAB_PADDING_MULTIPLE == 0
    )
    if rows == declared or padded:
        return None
    return ConfigInconsistency(field="vocab_size", declared=declared, observed=[rows])


def _check_hidden_size(
    cfg: Any, embedding: tuple[int, int] | None, shapes: Shapes
) -> ConfigInconsistency | None:
    declared = _positive_int(getattr(cfg, "hidden_size", 0))
    if declared is None:
        return None
    # Factorised embeddings (ALBERT, OPT-350m) have a narrower embedding
    # than hidden_size, so per-layer norm widths are accepted as evidence.
    observed: set[int] = set(_layer_norm_widths(shapes))
    if embedding is not None:
        observed.add(embedding[1])
    if not observed or declared in observed:
        return None
    return ConfigInconsistency(
        field="hidden_size", declared=declared, observed=sorted(observed)
    )


def _check_num_hidden_layers(cfg: Any, shapes: Shapes) -> ConfigInconsistency | None:
    declared = _positive_int(getattr(cfg, "num_hidden_layers", 0))
    if declared is None:
        return None
    stacks = _layer_stack_counts(shapes)
    if not stacks:
        return None
    counts = sorted(set(stacks.values()))
    if any(declared <= c <= declared + EXTRA_TRAILING_LAYERS_TOLERANCE for c in counts):
        return None
    return ConfigInconsistency(
        field="num_hidden_layers", declared=declared, observed=counts
    )


def _check_encoder_decoder_layers(
    cfg: Any, shapes: Shapes
) -> list[ConfigInconsistency | None]:
    # num_hidden_layers aliases the encoder depth here, and some loaders
    # (e.g. *ForCausalLM) keep only the decoder, so each side is checked
    # against its own stack and skipped when that stack is absent.
    stacks = _layer_stack_counts(shapes)
    findings: list[ConfigInconsistency | None] = []
    for side, fields in _ENCODER_DECODER_LAYER_FIELDS.items():
        field, declared = _first_declared(cfg, fields)
        counts = sorted({c for prefix, c in stacks.items() if side in prefix})
        if declared is None or not counts or declared in counts:
            continue
        findings.append(
            ConfigInconsistency(field=field, declared=declared, observed=counts)
        )
    return findings


def _check_albert_layer_groups(
    cfg: Any, shapes: Shapes
) -> list[ConfigInconsistency | None]:
    # ALBERT reuses num_hidden_groups stored groups of inner_group_num layers
    # across all num_hidden_layers positions, so the depth itself is not
    # recoverable from the weights; only the stored group layout is checked.
    group_counts = [
        c
        for prefix, c in _layer_stack_counts(shapes).items()
        if _ALBERT_GROUP in prefix
    ]
    if not group_counts:
        return []
    findings: list[ConfigInconsistency | None] = []
    groups = _positive_int(getattr(cfg, "num_hidden_groups", None)) or 1
    if len(group_counts) != groups:
        findings.append(
            ConfigInconsistency(
                field="num_hidden_groups", declared=groups, observed=[len(group_counts)]
            )
        )
    inner = _positive_int(getattr(cfg, "inner_group_num", None)) or 1
    if set(group_counts) != {inner}:
        findings.append(
            ConfigInconsistency(
                field="inner_group_num",
                declared=inner,
                observed=sorted(set(group_counts)),
            )
        )
    return findings


def _check_intermediate_size(cfg: Any, shapes: Shapes) -> ConfigInconsistency | None:
    if _is_moe(cfg) or getattr(cfg, "block_auto_adjust_ff_dim", False):
        return None
    declared = _positive_int(_resolve_intermediate_size(cfg))
    if declared is None:
        return None
    dims = _layer_matrix_dims(shapes)
    if not dims:
        return None
    # Fused gate/up projections store 2 x intermediate_size rows.
    if declared in dims or 2 * declared in dims:
        return None
    return ConfigInconsistency(
        field="intermediate_size",
        declared=declared,
        observed=sorted(dims),
    )


# -- Shape helpers ----------------------------------------------------


def _find_embedding_shape(shapes: Shapes) -> tuple[int, int] | None:
    for name in EMBEDDING_CANDIDATES:
        shape = shapes.get(name)
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    for name, shape in shapes.items():
        if name.endswith(_EMBEDDING_SUFFIXES) and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return None


def _layer_stack_counts(shapes: Shapes) -> dict[str, int]:
    """Map each layer-stack prefix to ``max layer index + 1``."""
    max_index: dict[str, int] = defaultdict(lambda: -1)
    for name in shapes:
        match = _LAYER_STACK_RE.match(name)
        if match:
            prefix, idx = match.group(1), int(match.group(2))
            max_index[prefix] = max(max_index[prefix], idx)
    return {prefix: idx + 1 for prefix, idx in max_index.items()}


def _layer_norm_widths(shapes: Shapes) -> set[int]:
    return {
        int(shape[0])
        for name, shape in shapes.items()
        if len(shape) == 1
        and _LAYER_STACK_RE.match(name)
        and is_norm_tensor_name(name)
        and not is_quantization_aux_name(name)
    }


def _layer_matrix_dims(shapes: Shapes) -> set[int]:
    # bitsandbytes 4-bit packs each matrix into a flat [N, 1] byte tensor and
    # quantizers add scale / zero-point side tensors; neither shape reflects
    # the projection dimensions.
    dims: set[int] = set()
    for name, shape in shapes.items():
        if (
            len(shape) == 2
            and int(shape[1]) != 1
            and _LAYER_STACK_RE.match(name)
            and not is_quantization_aux_name(name)
        ):
            dims.update(int(d) for d in shape)
    return dims


def _is_moe(cfg: Any) -> bool:
    return any(_positive_int(getattr(cfg, f, 0)) for f in _MOE_CONFIG_FIELDS)


def _is_albert(cfg: Any) -> bool:
    return getattr(cfg, "model_type", None) == "albert"


def _is_encoder_decoder(cfg: Any) -> bool:
    # *ForCausalLM wrappers reset is_encoder_decoder on the loaded config,
    # so a declared decoder depth is also accepted as evidence.
    if getattr(cfg, "is_encoder_decoder", False) is True:
        return True
    _, decoder_layers = _first_declared(cfg, _ENCODER_DECODER_LAYER_FIELDS["decoder"])
    return decoder_layers is not None


def _first_declared(cfg: Any, fields: Sequence[str]) -> tuple[str, int | None]:
    for field in fields:
        value = _positive_int(getattr(cfg, field, None))
        if value is not None:
            return field, value
    return fields[0], None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)
