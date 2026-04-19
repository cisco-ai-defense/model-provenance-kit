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

"""Tokenizer-level signals: TFV and VOA.

TFV = Tokenizer Feature Vector, VOA = Vocabulary Overlap Analysis.

TFV extracts a multi-dimensional fingerprint from tokenizer structure
(vocabulary composition, script distribution, merge rules, special tokens)
and computes pairwise similarity via 11 weighted components.

VOA computes Jaccard similarity and set-level statistics between vocabularies.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
import structlog
from transformers import AutoTokenizer

from provenancekit.models.signals import TokenizerFeatures, VocabOverlap

log = structlog.get_logger()

# ── Unicode / script helpers ───────────────────────────────────────


def classify_script(char: str) -> str:
    """Classify a Unicode character into a broad script family.

    Returns one of ``"Latin"``, ``"CJK"``, ``"Cyrillic"``, ``"Arabic"``,
    ``"Devanagari"``, or ``"Other"``.
    """
    cp = ord(char)
    if cp <= 0x007F:
        return "Latin"
    try:
        name = unicodedata.name(char, "")
    except ValueError:
        return "Other"
    for script, keywords in [
        ("CJK", ["CJK", "HANGUL", "HIRAGANA", "KATAKANA", "BOPOMOFO"]),
        ("Cyrillic", ["CYRILLIC"]),
        ("Arabic", ["ARABIC"]),
        ("Devanagari", ["DEVANAGARI"]),
        ("Latin", ["LATIN"]),
    ]:
        if any(kw in name for kw in keywords):
            return script
    return "Other"


def compute_script_distribution(tokens: list[str]) -> dict[str, float]:
    """Compute script distribution over a list of token strings.

    Returns a mapping from script name to proportion (0--1).
    """
    counts: Counter[str] = Counter()
    total = 0
    for token in tokens:
        for ch in token:
            if ch.isalpha():
                counts[classify_script(ch)] += 1
                total += 1
    if not total:
        return {}
    return {k: round(v / total, 4) for k, v in counts.most_common()}


# ── TFV extraction ─────────────────────────────────────────────────


def extract_tokenizer_features(
    model_name: str,
    tokenizer: Any = None,
    *,
    trust_remote_code: bool = False,
) -> TokenizerFeatures:
    """Extract enhanced TFV with merge-rule identity features.

    Args:
        model_name: HuggingFace model identifier.
        tokenizer: Pre-loaded ``transformers.PreTrainedTokenizer``,
            or ``None`` to auto-load.
        trust_remote_code: Allow execution of model-hosted Python code
            when loading the tokenizer.  Defaults to ``False``.

    Returns:
        A ``TokenizerFeatures`` model containing all 18 TFV fields.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    vocab: dict[str, int] = tokenizer.get_vocab()
    tokens = [
        t.decode("utf-8", errors="replace") if isinstance(t, bytes) else t
        for t in vocab
    ]
    n = len(tokens)

    if n == 0:
        return TokenizerFeatures(vocab_size=0, tokenizer_class=type(tokenizer).__name__)

    f: dict[str, Any] = {}
    f["vocab_size"] = n
    f["tokenizer_class"] = type(tokenizer).__name__
    f["bos_token_id"] = tokenizer.bos_token_id
    f["eos_token_id"] = tokenizer.eos_token_id
    f["pad_token_id"] = tokenizer.pad_token_id
    f["num_added_tokens"] = len(tokenizer.added_tokens_encoder)
    f["num_special_tokens"] = len(tokenizer.all_special_tokens)

    merges: list[str] = []
    all_merges_str = ""
    try:
        tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
        raw_merges = tok_json.get("model", {}).get("merges", [])
        merges = [" ".join(m) if isinstance(m, list) else str(m) for m in raw_merges]
        f["num_merges"] = len(merges)
        f["first_5_merges"] = merges[:5]
        all_merges_str = "\n".join(merges)
    except Exception as exc:  # noqa: BLE001
        log.debug("tokenizer_merges_extraction_failed", error=str(exc))
        f["num_merges"] = 0
        f["first_5_merges"] = []

    first_50 = "\n".join(merges[:50]) if merges else ""
    f["merge_rule_hash"] = hashlib.sha256(
        first_50.encode(),
    ).hexdigest()

    f["all_merges_str"] = all_merges_str

    f["special_token_ids"] = {
        "pad": tokenizer.pad_token_id,
        "unk": getattr(tokenizer, "unk_token_id", None),
        "mask": getattr(tokenizer, "mask_token_id", None),
    }

    f["pct_single_char"] = round(
        sum(1 for t in tokens if len(t) == 1) / n,
        4,
    )
    lengths = [len(t) for t in tokens]
    f["avg_token_length"] = round(float(np.mean(lengths)), 2)
    f["max_token_length"] = max(lengths) if lengths else 0
    f["pct_whitespace_prefix"] = round(
        sum(1 for t in tokens if len(t) > 0 and t[0] in ("\u2581", "\u0120", " ")) / n,
        4,
    )
    f["pct_byte_tokens"] = round(
        sum(1 for t in tokens if re.match(r"^<0x[0-9A-Fa-f]{2}>$", t)) / n,
        4,
    )
    f["script_distribution"] = compute_script_distribution(tokens)

    return TokenizerFeatures(**f)


# ── TFV similarity ─────────────────────────────────────────────────


def tfv_similarity(
    fa: TokenizerFeatures,
    fb: TokenizerFeatures,
) -> float:
    """Compute enhanced TFV similarity with 11 weighted components.

    Component weights (sum = 1.0)::

        tokenizer_class        0.10
        vocab_size             0.10
        bos/eos IDs            0.08
        avg_token_length       0.07
        whitespace_prefix      0.05
        script_distribution    0.10
        byte_token_presence    0.05
        num_merges             0.05
        merge_rule_hash        0.15
        special_token_match    0.10
        merge_rule_identity    0.15
    """
    score = 0.0

    score += 0.10 * (1.0 if fa.tokenizer_class == fb.tokenizer_class else 0.0)
    max_vocab = max(fa.vocab_size, fb.vocab_size)
    score += 0.10 * (
        1.0 - abs(fa.vocab_size - fb.vocab_size) / max_vocab if max_vocab > 0 else 0.0
    )

    bos = 1.0 if fa.bos_token_id == fb.bos_token_id else 0.0
    eos = 1.0 if fa.eos_token_id == fb.eos_token_id else 0.0
    score += 0.08 * (bos + eos) / 2

    atl_a, atl_b = fa.avg_token_length, fb.avg_token_length
    score += 0.07 * max(
        0,
        1.0 - abs(atl_a - atl_b) / max(atl_a, atl_b, 1e-8),
    )

    score += 0.05 * max(
        0,
        1.0 - abs(fa.pct_whitespace_prefix - fb.pct_whitespace_prefix),
    )

    score += _script_distribution_sim(fa, fb)

    score += 0.05 * (
        1.0 if (fa.pct_byte_tokens > 0) == (fb.pct_byte_tokens > 0) else 0.0
    )

    score += _merge_count_sim(fa, fb)
    score += _merge_rule_hash_sim(fa, fb)
    score += _special_token_sim(fa, fb)
    score += _merge_identity_sim(fa, fb)

    return round(score, 4)


def _script_distribution_sim(
    fa: TokenizerFeatures,
    fb: TokenizerFeatures,
) -> float:
    """Script distribution cosine similarity (weight 0.10)."""
    all_s = sorted(set(fa.script_distribution) | set(fb.script_distribution))
    if not all_s:
        return 0.0
    va = np.array([fa.script_distribution.get(s, 0) for s in all_s])
    vb = np.array([fb.script_distribution.get(s, 0) for s in all_s])
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return 0.10 * (float(np.dot(va, vb) / norm) if norm > 1e-8 else 0.0)


def _merge_count_sim(
    fa: TokenizerFeatures,
    fb: TokenizerFeatures,
) -> float:
    """Merge count similarity (weight 0.05)."""
    ma, mb = fa.num_merges, fb.num_merges
    if ma > 0 and mb > 0:
        return 0.05 * (1.0 - abs(ma - mb) / max(ma, mb))
    if ma == 0 and mb == 0:
        return 0.05
    return 0.0


def _merge_rule_hash_sim(
    fa: TokenizerFeatures,
    fb: TokenizerFeatures,
) -> float:
    """Merge rule hash similarity (weight 0.15)."""
    if fa.merge_rule_hash and fb.merge_rule_hash:
        return 0.15 * (1.0 if fa.merge_rule_hash == fb.merge_rule_hash else 0.0)
    return 0.0


def _special_token_sim(
    fa: TokenizerFeatures,
    fb: TokenizerFeatures,
) -> float:
    """Special token exact match (weight 0.10)."""
    sp_matches = 0
    sp_total = 0
    for key in ("pad", "unk", "mask"):
        id_a = fa.special_token_ids.get(key)
        id_b = fb.special_token_ids.get(key)
        if id_a is not None and id_b is not None:
            sp_total += 1
            if id_a == id_b:
                sp_matches += 1
        elif id_a is None and id_b is None:
            sp_total += 1
            sp_matches += 1
    return 0.10 * (sp_matches / sp_total if sp_total > 0 else 0.5)


def _merge_identity_sim(
    fa: TokenizerFeatures,
    fb: TokenizerFeatures,
) -> float:
    """Merge rule identity — binary full-merge comparison (weight 0.15)."""
    merges_a = fa.all_merges_str
    merges_b = fb.all_merges_str
    if merges_a and merges_b:
        return 0.15 * (1.0 if merges_a == merges_b else 0.0)
    if not merges_a and not merges_b:
        return 0.15 * 0.5
    return 0.0


# ── VOA (Vocabulary Overlap Analysis) ──────────────────────────────


def vocab_overlap(
    model_a: str,
    model_b: str,
    tok_a: Any = None,
    tok_b: Any = None,
    vocab_a: set[str] | None = None,
    vocab_b: set[str] | None = None,
) -> VocabOverlap:
    """Compute Jaccard similarity and set statistics between vocabularies.

    Three input paths (in priority order):

    1. Pre-extracted vocab sets (``vocab_a`` / ``vocab_b``).
    2. Pre-loaded tokenizers (``tok_a`` / ``tok_b``).
    3. Model names only — tokenizers are auto-loaded.

    Args:
        model_a: HuggingFace model identifier for model A.
        model_b: HuggingFace model identifier for model B.
        tok_a: Pre-loaded tokenizer for model A.
        tok_b: Pre-loaded tokenizer for model B.
        vocab_a: Pre-extracted vocabulary token strings for model A.
        vocab_b: Pre-extracted vocabulary token strings for model B.
    """
    va = _resolve_vocab(model_a, tok_a, vocab_a)
    vb = _resolve_vocab(model_b, tok_b, vocab_b)

    inter = va & vb
    union = va | vb
    return VocabOverlap(
        jaccard=round(len(inter) / len(union), 4) if union else 0.0,
        vocab_a_size=len(va),
        vocab_b_size=len(vb),
        intersection=len(inter),
        union=len(union),
        only_a=len(va - vb),
        only_b=len(vb - va),
        overlap_pct_a=round(len(inter) / len(va), 4) if va else 0.0,
        overlap_pct_b=round(len(inter) / len(vb), 4) if vb else 0.0,
    )


def _resolve_vocab(
    model_name: str,
    tokenizer: Any,
    vocab: set[str] | None,
) -> set[str]:
    """Resolve vocabulary from the first available source."""
    if vocab is not None:
        return set(vocab)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False,
        )
    return {
        k.decode("utf-8", errors="replace") if isinstance(k, bytes) else k
        for k in tokenizer.get_vocab()
    }
