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

"""Anchor token selection for EAS (Embedding Anchor Similarity).

Selects K anchor token IDs using a script-aware allocation strategy:

1. Fixed punctuation / digit anchors.
2. Script-proportional allocation (Latin, CJK, Cyrillic, Arabic,
   Devanagari) based on the model's vocabulary distribution.
3. Random backfill from the first 50k token IDs to reach K.
"""

from typing import Any

import numpy as np
import structlog

from provenancekit.config.settings import Settings
from provenancekit.core.signals.tokenizer import compute_script_distribution

log = structlog.get_logger()

# ── Anchor token constants ─────────────────────────────────────────

ANCHOR_PUNCT: list[str] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ".",
    ",",
    "!",
    "?",
    ":",
    ";",
]

ANCHOR_LATIN: list[str] = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "that",
    "for",
    "it",
    "was",
    "on",
    "are",
    "with",
    "as",
    "be",
    "at",
    "ing",
    "tion",
    "er",
    "al",
    "ed",
    "es",
    "re",
    "en",
]

ANCHOR_CJK: list[str] = [
    "\u7684",
    "\u4e00",
    "\u662f",
    "\u4e0d",
    "\u4eba",
    "\u6211",
    "\u5728",
    "\u6709",
    "\u4e86",
    "\u8fd9",
    "\u4e2d",
    "\u5927",
]

ANCHOR_CYRILLIC: list[str] = [
    "\u0438",
    "\u0432",
    "\u043d\u0435",
    "\u043d\u0430",
    "\u044f",
    "\u0447\u0442\u043e",
    "\u043e\u043d",
    "\u043a",
]

ANCHOR_ARABIC: list[str] = [
    "\u0645\u0646",
    "\u0641\u064a",
    "\u0639\u0644\u0649",
    "\u0623\u0646",
    "\u0647\u0630\u0627",
    "\u0627\u0644\u0644\u0647",
    "\u0625\u0644\u0649",
    "\u0644\u0627",
]

ANCHOR_DEVANAGARI: list[str] = [
    "\u0915\u0947",
    "\u0939\u0948",
    "\u0915\u093e",
    "\u0915\u0940",
    "\u092f\u0939",
    "\u0938\u0947",
    "\u0915\u094b",
    "\u092a\u0930",
]

SCRIPT_ANCHORS: dict[str, list[str]] = {
    "Latin": ANCHOR_LATIN,
    "CJK": ANCHOR_CJK,
    "Cyrillic": ANCHOR_CYRILLIC,
    "Arabic": ANCHOR_ARABIC,
    "Devanagari": ANCHOR_DEVANAGARI,
}

_FIXED_SEED: int = 42
_SCRIPT_THRESHOLD: float = 0.05


# ── Public API ─────────────────────────────────────────────────────


def get_anchor_ids(
    tokenizer: Any,
    vocab: set[str] | list[str] | None,
    vocab_size: int,
    anchor_k: int | None = None,
) -> list[int]:
    """Select *anchor_k* anchor token IDs via script-aware allocation.

    Args:
        tokenizer: A HuggingFace tokenizer (needs ``.encode()``).
            ``None`` skips script-based anchors and uses random only.
        vocab: Vocabulary token strings for script distribution.
        vocab_size: Total vocabulary size (upper bound for valid IDs).
        anchor_k: Number of anchor tokens to select. Defaults to
            ``Settings().anchor_k`` when not provided.
    """
    if anchor_k is None:
        anchor_k = Settings().anchor_k

    ids: list[int] = []
    seen: set[int] = set()

    _try_encode_anchors(ANCHOR_PUNCT, tokenizer, vocab_size, ids, seen)

    script_dist = _vocab_script_distribution(vocab)
    active_scripts = {
        s: pct
        for s, pct in script_dist.items()
        if pct >= _SCRIPT_THRESHOLD and s in SCRIPT_ANCHORS
    }
    if not active_scripts:
        active_scripts = {"Latin": 1.0}

    budget = anchor_k - len(ids)
    total_pct = sum(active_scripts.values())

    allocations: dict[str, int] = {}
    allocated = 0
    for script, pct in active_scripts.items():
        n = max(2, round(budget * pct / total_pct))
        allocations[script] = n
        allocated += n

    diff = allocated - budget
    if diff != 0:
        largest = max(allocations, key=allocations.get)  # type: ignore[arg-type]
        allocations[largest] = max(2, allocations[largest] - diff)

    for script, _ in sorted(
        active_scripts.items(),
        key=lambda x: -x[1],
    ):
        n_slots = allocations[script]
        candidates = SCRIPT_ANCHORS[script][:n_slots]
        _try_encode_anchors(
            candidates,
            tokenizer,
            vocab_size,
            ids,
            seen,
        )

    # RandomState (legacy MT19937) is used intentionally here.
    # default_rng (PCG64) produces different backfill anchor IDs for the
    # same seed, which changes EAS scores vs the original implementation.
    rng = np.random.RandomState(_FIXED_SEED)  # noqa: NPY002
    n_remaining = anchor_k - len(ids)
    if n_remaining > 0:
        pool = [i for i in range(min(vocab_size, 50000)) if i not in seen]
        if pool:
            sample = rng.choice(
                pool,
                size=min(n_remaining, len(pool)),
                replace=False,
            )
            ids.extend(sample.tolist())

    return ids[:anchor_k]


# ── Private helpers ────────────────────────────────────────────────


def _try_encode_anchors(
    tokens: list[str],
    tokenizer: Any,
    vocab_size: int,
    ids: list[int],
    seen: set[int],
) -> None:
    """Encode each anchor string and append valid IDs."""
    if tokenizer is None:
        return
    for token_str in tokens:
        try:
            encoded: list[int] = tokenizer.encode(
                token_str,
                add_special_tokens=False,
            )
            if encoded:
                tid = encoded[0]
                if 0 <= tid < vocab_size and tid not in seen:
                    ids.append(tid)
                    seen.add(tid)
        except Exception as exc:  # noqa: BLE001
            log.debug("anchor_encode_failed", token=token_str, error=str(exc))


def _vocab_script_distribution(
    vocab: set[str] | list[str] | None,
) -> dict[str, float]:
    """Compute script distribution from vocabulary token strings."""
    if vocab is None:
        return {}
    tokens = list(vocab) if not isinstance(vocab, list) else vocab
    if not tokens:
        return {}
    return compute_script_distribution(tokens)
