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

"""Metadata Family Identification (MFI) — fingerprint extraction and similarity.

Three-tier system for classifying model provenance from metadata:

* **Tier 1 (Exact):**   ``arch_hash`` match  → confidence 1.0
* **Tier 2 (Family):**  ``family_hash`` match (dimension-aware) → confidence 0.9
* **Tier 3 (Soft):**    weighted scoring across 11 curated features → 0–1

Tier 2 includes dimension-awareness: if hidden_size or num_hidden_layers
differ despite a family_hash match, the pair is demoted to Tier 3.
Dimension features (hidden_size, num_hidden_layers) participate directly
in Tier 3 scoring.
"""

import hashlib
import itertools
import json
from typing import Any

import structlog
from transformers import AutoConfig, AutoTokenizer

from provenancekit.config.constants import FAMILY_MAP
from provenancekit.models.signals import MFIFingerprint, MFISimilarity

log = structlog.get_logger()

DIM_SIM_THRESHOLD = 0.90

# ── Config field resolution ────────────────────────────────────────


def _resolve_intermediate_size(cfg: Any) -> int:
    """Resolve FFN intermediate dimension from config.

    Different architectures store this under different names:
      - intermediate_size  (Llama, Mistral, BERT, Gemma, Qwen, ...)
      - d_ff               (T5, mT5, FLAN-T5)
      - encoder_ffn_dim    (BART, mBART, Marian/OPUS-MT)
      - decoder_ffn_dim    (BART, mBART — fallback if encoder not present)
      - ffn_dim            (OPT)
      - hidden_dim         (DistilBERT — FFN intermediate, NOT hidden_size)

    Returns 0 only when no field is present in the config at all.
    """
    for attr in (
        "intermediate_size",
        "d_ff",
        "encoder_ffn_dim",
        "decoder_ffn_dim",
        "ffn_dim",
    ):
        val = getattr(cfg, attr, 0)
        if val and val > 0:
            return int(val)
    hidden_dim = getattr(cfg, "hidden_dim", 0)
    hidden_size = getattr(cfg, "hidden_size", 0)
    if hidden_dim and hidden_dim > 0 and hidden_dim != hidden_size:
        return int(hidden_dim)
    return 0


# ── Fingerprint extraction ─────────────────────────────────────────


def extract_fingerprint(
    model_name: str,
    config: Any = None,
    tokenizer: Any = None,
    *,
    trust_remote_code: bool = False,
) -> tuple[MFIFingerprint, Any]:
    """Extract MFI metadata fingerprint with all three-tier fields.

    Handles composite/multimodal models by resolving to the text
    backbone config (``text_config`` or ``decoder`` sub-config).

    Args:
        model_name: HuggingFace model identifier.
        config: Pre-loaded ``transformers.PretrainedConfig``, or None
            to auto-load.
        tokenizer: Pre-loaded ``transformers.PreTrainedTokenizer``, or
            None to auto-load.
        trust_remote_code: Allow execution of model-hosted Python code
            when loading config/tokenizer.  Defaults to ``False`` for
            security; pass ``True`` only for models that require it.

    Returns:
        ``(fingerprint, tokenizer)`` — the tokenizer is returned so
        callers can reuse it for TFV / VOA extraction without a
        redundant download.
    """
    if config is None:
        log.info("loading_config", model_id=model_name)
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    if tokenizer is None:
        log.info("loading_tokenizer", model_id=model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

    cfg = _resolve_text_backbone(config)

    fp: dict[str, Any] = {
        "model_type": getattr(config, "model_type", "unknown"),
        "architectures": getattr(config, "architectures", ["unknown"]),
        "hidden_size": getattr(cfg, "hidden_size", 0),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", 0),
        "num_attention_heads": getattr(cfg, "num_attention_heads", 0),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "intermediate_size": _resolve_intermediate_size(cfg),
        "vocab_size": getattr(cfg, "vocab_size", 0),
        "max_position_embeddings": getattr(
            cfg,
            "max_position_embeddings",
            0,
        ),
        "hidden_act": getattr(
            cfg,
            "hidden_act",
            getattr(cfg, "hidden_activation", None),
        ),
        "head_dim": getattr(cfg, "head_dim", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    _derive_attention_style(fp)
    _derive_norm_type(fp, cfg)
    _derive_attention_bias(fp, cfg)
    _derive_qk_norm(fp, cfg)
    _derive_pos_encoding(fp, cfg)
    _derive_tokenizer_hash(fp, tokenizer)
    _derive_hashes(fp)

    return MFIFingerprint(**fp), tokenizer


# ── Derived field helpers ──────────────────────────────────────────


def _resolve_text_backbone(config: Any) -> Any:
    """Resolve to the text backbone for composite/multimodal models."""
    cfg = config
    if not hasattr(cfg, "hidden_size") or getattr(cfg, "hidden_size", 0) == 0:
        for sub_attr in ("text_config", "decoder"):
            sub = getattr(cfg, sub_attr, None)
            if sub is not None and getattr(sub, "hidden_size", 0) > 0:
                return sub
    return cfg


def _derive_attention_style(fp: dict[str, Any]) -> None:
    """Populate gqa_ratio and attention_style."""
    nah: int = fp["num_attention_heads"] or 1
    nkv: int | None = fp["num_key_value_heads"]
    fp["gqa_ratio"] = round(nah / nkv, 1) if nkv else 1.0
    if nkv is None or nkv == nah:
        fp["attention_style"] = "MHA"
    elif nkv == 1:
        fp["attention_style"] = "MQA"
    else:
        fp["attention_style"] = "GQA"


def _derive_norm_type(fp: dict[str, Any], cfg: Any) -> None:
    """Populate norm_type."""
    if hasattr(cfg, "rms_norm_eps") or hasattr(cfg, "rms_norm_epsilon"):
        fp["norm_type"] = "RMSNorm"
    elif hasattr(cfg, "layer_norm_eps") or hasattr(cfg, "layer_norm_epsilon"):
        fp["norm_type"] = "LayerNorm"
    else:
        fp["norm_type"] = "unknown"


def _derive_attention_bias(fp: dict[str, Any], cfg: Any) -> None:
    """Populate attention_bias."""
    if hasattr(cfg, "attention_bias"):
        fp["attention_bias"] = cfg.attention_bias
    elif hasattr(cfg, "bias"):
        fp["attention_bias"] = cfg.bias
    else:
        fp["attention_bias"] = None


def _derive_qk_norm(fp: dict[str, Any], cfg: Any) -> None:
    """Populate qk_norm."""
    fp["qk_norm"] = bool(
        getattr(cfg, "qk_layernorm", False)
        or getattr(cfg, "qk_norm", False)
        or (getattr(cfg, "query_pre_attn_scalar", None) is not None)
    )


def _derive_pos_encoding(fp: dict[str, Any], cfg: Any) -> None:
    """Populate pos_encoding."""
    if fp["rope_theta"] is not None:
        fp["pos_encoding"] = "RoPE"
    elif hasattr(cfg, "position_embedding_type"):
        fp["pos_encoding"] = getattr(
            cfg,
            "position_embedding_type",
            "absolute",
        )
    else:
        fp["pos_encoding"] = "absolute"


def _derive_tokenizer_hash(fp: dict[str, Any], tokenizer: Any) -> None:
    """Populate tokenizer_hash and token_id_signature."""
    try:
        tok_str: str = tokenizer.backend_tokenizer.to_str()
    except Exception as exc:  # noqa: BLE001
        log.debug("backend_tokenizer_to_str_failed", error=str(exc))
        vocab_sample = sorted(itertools.islice(tokenizer.get_vocab().items(), 100))
        safe_sample = [
            (k.decode("utf-8", errors="replace") if isinstance(k, bytes) else k, v)
            for k, v in vocab_sample
        ]
        tok_str = json.dumps(safe_sample)
    fp["tokenizer_hash"] = hashlib.sha256(
        tok_str.encode(),
    ).hexdigest()
    fp["token_id_signature"] = f"bos={fp['bos_token_id']}|eos={fp['eos_token_id']}"


def _derive_hashes(fp: dict[str, Any]) -> None:
    """Populate arch_hash (Tier 1) and family_hash (Tier 2)."""
    arch_fields = json.dumps(
        [
            fp["model_type"],
            fp["architectures"],
            fp["hidden_size"],
            fp["num_hidden_layers"],
            fp["num_attention_heads"],
            fp["num_key_value_heads"],
            fp["intermediate_size"],
            fp["vocab_size"],
            fp["max_position_embeddings"],
        ],
        sort_keys=True,
    )
    fp["arch_hash"] = hashlib.sha256(
        arch_fields.encode(),
    ).hexdigest()

    rope_theta_bucket = "none"
    if fp["rope_theta"] is not None:
        rope_theta_bucket = str(round(fp["rope_theta"], -2))
    rope_scaling_type = "none"
    if fp.get("rope_scaling") and isinstance(fp["rope_scaling"], dict):
        rope_scaling_type = str(fp["rope_scaling"].get("type", "none"))
    family_fields = json.dumps(
        [
            fp["model_type"],
            fp["attention_style"],
            fp["hidden_act"],
            fp["norm_type"],
            fp["tokenizer_hash"],
            rope_theta_bucket,
            rope_scaling_type,
        ],
        sort_keys=True,
    )
    fp["family_hash"] = hashlib.sha256(
        family_fields.encode(),
    ).hexdigest()


# ── Classification ─────────────────────────────────────────────────


def classify(fp: MFIFingerprint) -> tuple[str, float]:
    """Classify a fingerprint into a known model family.

    Returns:
        ``(family_name, confidence)`` — confidence is 1.0 for known
        families, 0.0 for unknown.
    """
    mt = fp.model_type.lower()
    for family, types in FAMILY_MAP.items():
        if mt in types:
            return family, 1.0
    return "unknown", 0.0


# ── Similarity (dimension-aware) ───────────────────────────────────


def similarity(
    fp_a: MFIFingerprint,
    fp_b: MFIFingerprint,
) -> MFISimilarity:
    """Dimension-aware three-tier MFI similarity.

    Tier 2 checks dimension match: if ``hidden_size`` or
    ``num_hidden_layers`` differ despite a ``family_hash`` match,
    the pair is demoted to Tier 3 where dimension features
    participate directly in scoring.
    """
    if fp_a.arch_hash and fp_b.arch_hash and fp_a.arch_hash == fp_b.arch_hash:
        return MFISimilarity(score=1.0, tier=1, match_type="exact")

    if fp_a.family_hash and fp_b.family_hash and fp_a.family_hash == fp_b.family_hash:
        dim_match = (
            fp_a.hidden_size == fp_b.hidden_size
            and fp_a.num_hidden_layers == fp_b.num_hidden_layers
        )
        if dim_match:
            return MFISimilarity(score=0.9, tier=2, match_type="family")

    final = _tier3_soft_score(fp_a, fp_b)
    return MFISimilarity(score=final, tier=3, match_type="soft_match")


def _tier3_soft_score(
    fp_a: MFIFingerprint,
    fp_b: MFIFingerprint,
) -> float:
    """Compute Tier 3 weighted soft-match score across 11 features."""
    score, total = 0.0, 0.0

    # token_id_sig (w=2.0)
    w = 2.0
    total += w
    bos_match = 1.0 if fp_a.bos_token_id == fp_b.bos_token_id else 0.0
    eos_match = 1.0 if fp_a.eos_token_id == fp_b.eos_token_id else 0.0
    score += w * (bos_match + eos_match) / 2

    # hidden_size (w=1.8)
    w = 1.8
    ha, hb = fp_a.hidden_size, fp_b.hidden_size
    if ha is not None and hb is not None and ha > 0 and hb > 0:
        total += w
        sim = 1.0 - abs(ha - hb) / max(ha, hb)
        if sim >= DIM_SIM_THRESHOLD:
            score += w

    # qk_norm (w=1.5)
    w = 1.5
    total += w
    if fp_a.qk_norm == fp_b.qk_norm:
        score += w

    # num_hidden_layers (w=1.5)
    w = 1.5
    la, lb = fp_a.num_hidden_layers, fp_b.num_hidden_layers
    if la is not None and lb is not None and la > 0 and lb > 0:
        total += w
        sim = 1.0 - abs(la - lb) / max(la, lb)
        if sim >= DIM_SIM_THRESHOLD:
            score += w

    # vocab_size (w=1.5)
    w = 1.5
    va, vb = fp_a.vocab_size, fp_b.vocab_size
    if va and vb:
        total += w
        score += w * (1.0 - abs(va - vb) / max(va, vb))

    # attention_style (w=1.5)
    w = 1.5
    total += w
    if fp_a.attention_style == fp_b.attention_style:
        score += w

    # rope_theta (w=1.2)
    w = 1.2
    rt_a, rt_b = fp_a.rope_theta, fp_b.rope_theta
    if rt_a is not None and rt_b is not None and rt_a > 0 and rt_b > 0:
        total += w
        score += w * (1.0 - abs(rt_a - rt_b) / max(rt_a, rt_b))
    else:
        total += w
        if rt_a is None and rt_b is None:
            score += w

    # norm_type (w=0.5)
    w = 0.5
    total += w
    if fp_a.norm_type == fp_b.norm_type:
        score += w

    # posenc (w=0.4)
    w = 0.4
    total += w
    if fp_a.pos_encoding == fp_b.pos_encoding:
        score += w

    # linear_bias_present (w=0.4)
    w = 0.4
    total += w
    if fp_a.attention_bias == fp_b.attention_bias:
        score += w

    # naming_bonus (w=0.1)
    w = 0.1
    total += w
    if fp_a.model_type == fp_b.model_type:
        score += w

    final = round(score / total, 4) if total > 0 else 0.0

    return final
