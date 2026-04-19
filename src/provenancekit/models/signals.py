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

"""Pydantic models for signal extraction outputs.

Covers Metadata Family Identification (MFI), Tokenizer Feature Vector (TFV),
Vocabulary Overlap Analysis (VOA), and weight-level signal features.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class MFIFingerprint(BaseModel):
    """Metadata Family Identification fingerprint (26 fields).

    Extracted from a model's HuggingFace ``AutoConfig`` and
    ``AutoTokenizer``.  Supports three-tier provenance classification
    (exact / family / soft).
    """

    model_type: str
    architectures: list[str]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    hidden_act: str | None
    rope_theta: float | None
    rope_scaling: dict[str, Any] | None
    tie_word_embeddings: bool | None
    bos_token_id: int | None
    eos_token_id: int | None

    gqa_ratio: float
    attention_style: str
    norm_type: str
    attention_bias: bool | None
    qk_norm: bool
    pos_encoding: str

    head_dim: int | None = None

    tokenizer_hash: str
    token_id_signature: str
    arch_hash: str
    family_hash: str


class MFISimilarity(BaseModel):
    """Three-tier MFI pairwise similarity result.

    Attributes:
        score: Similarity value in ``[0, 1]``.
        tier: Classification tier (1=exact, 2=family, 3=soft).
        match_type: Human-readable tier label.
    """

    score: float
    tier: int
    match_type: str


class TokenizerFeatures(BaseModel):
    """Tokenizer Feature Vector (TFV) extraction output (18 fields).

    Multi-dimensional fingerprint covering vocabulary composition, script
    distribution, merge rules, and special tokens.

    Fields used by ``tfv_similarity`` are always required.  Metadata-only
    fields default to ``None`` / empty so that DB feature bundles (which
    may store a subset) can be loaded via ``model_validate`` without
    requiring every key.
    """

    vocab_size: int
    tokenizer_class: str
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    num_added_tokens: int | None = None
    num_special_tokens: int | None = None
    num_merges: int = 0
    first_5_merges: list[str] = Field(default_factory=list)
    merge_rule_hash: str = ""
    all_merges_str: str = ""
    special_token_ids: dict[str, int | None] = Field(default_factory=dict)
    pct_single_char: float | None = None
    avg_token_length: float = 0.0
    max_token_length: int | None = None
    pct_whitespace_prefix: float = 0.0
    pct_byte_tokens: float = 0.0
    script_distribution: dict[str, float] = Field(default_factory=dict)


class VocabOverlap(BaseModel):
    """Vocabulary Overlap Analysis (VOA) output.

    Jaccard similarity and set-level statistics between two vocabularies.
    """

    jaccard: float
    vocab_a_size: int
    vocab_b_size: int
    intersection: int
    union: int
    only_a: int
    only_b: int
    overlap_pct_a: float
    overlap_pct_b: float


class WeightSignalFeatures(BaseModel):
    """Weight-level signal features for EAS, NLF, LEP, END, WSP, and WVC.

    Holds numpy arrays from weight analysis.  Provides ``to_cache_dict``
    / ``from_cache_dict`` for JSON-safe persistence of the feature
    vectors without losing fidelity.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_size: int
    num_layers: int
    eas_self_sim: np.ndarray | None
    eas_anchor_count: int
    nlf_vector: np.ndarray | None
    nlf_mode: str | None
    nlf_num_layers: int
    lep_profile: np.ndarray | None
    end_histogram: np.ndarray | None
    wsp_signature: np.ndarray | None
    wvc_layer_sigs: dict[int, np.ndarray] | None

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (numpy arrays → lists)."""

        def _arr(v: np.ndarray | None) -> list[float] | None:
            return v.tolist() if v is not None else None

        wvc: dict[str, list[float]] | None = None
        if self.wvc_layer_sigs is not None:
            wvc = {str(k): v.tolist() for k, v in self.wvc_layer_sigs.items()}

        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "eas_self_sim": _arr(self.eas_self_sim),
            "eas_anchor_count": self.eas_anchor_count,
            "nlf_vector": _arr(self.nlf_vector),
            "nlf_mode": self.nlf_mode,
            "nlf_num_layers": self.nlf_num_layers,
            "lep_profile": _arr(self.lep_profile),
            "end_histogram": _arr(self.end_histogram),
            "wsp_signature": _arr(self.wsp_signature),
            "wvc_layer_sigs": wvc,
        }

    @classmethod
    def from_cache_dict(cls, d: dict[str, Any]) -> "WeightSignalFeatures":
        """Reconstruct from a JSON-safe dict (lists → numpy arrays)."""

        def _arr(v: list[float] | None) -> np.ndarray | None:
            return np.array(v) if v is not None else None

        wvc: dict[int, np.ndarray] | None = None
        raw_wvc = d.get("wvc_layer_sigs")
        if raw_wvc is not None:
            wvc = {int(k): np.array(v) for k, v in raw_wvc.items()}

        return cls(
            hidden_size=d["hidden_size"],
            num_layers=d["num_layers"],
            eas_self_sim=_arr(d.get("eas_self_sim")),
            eas_anchor_count=d.get("eas_anchor_count", 0),
            nlf_vector=_arr(d.get("nlf_vector")),
            nlf_mode=d.get("nlf_mode"),
            nlf_num_layers=d.get("nlf_num_layers", 0),
            lep_profile=_arr(d.get("lep_profile")),
            end_histogram=_arr(d.get("end_histogram")),
            wsp_signature=_arr(d.get("wsp_signature")),
            wvc_layer_sigs=wvc,
        )
