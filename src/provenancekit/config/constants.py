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

"""Constants, weights, and taxonomy for model provenance scoring."""

# ── Family taxonomy ─────────────────────────────────────────────────
# Maps canonical family names to all known model_type strings.

FAMILY_MAP: dict[str, list[str]] = {
    "llama": ["llama"],
    "qwen": [
        "qwen2",
        "qwen2_moe",
        "qwen",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3",
        "qwen3_moe",
    ],
    "phi": ["phi", "phi3", "phi-msft", "phi3small", "mixformer-sequential"],
    "gemma": ["gemma", "gemma2", "gemma3", "paligemma"],
    "mistral": ["mistral"],
    "deepseek": ["deepseek_v3", "deepseek_v2"],
    "gpt2": ["gpt2"],
    "falcon": ["falcon"],
    "starcoder": ["gpt_bigcode"],
    "bert": ["bert"],
    "roberta": ["roberta"],
    "t5": ["t5"],
    "bart": ["bart"],
    "marian": ["marian"],
    "bloom": ["bloom"],
    "gpt_neox": ["gpt_neox"],
    "albert": ["albert"],
    "deberta": ["deberta-v2", "deberta"],
    "xlm_roberta": ["xlm-roberta"],
    "nllb": ["m2m_100"],
}


# ── Identity signal weights ───────────────────────────────────────
# Empirically calibrated from Tier 3 Cohen's d on 111-pair benchmark.
# Includes WVC (d=1.21, 2nd strongest after EAS).

IDENTITY_WEIGHTS: dict[str, float] = {
    "eas": 0.36,  # d=2.03 — Embedding Anchor Similarity (strongest)
    "nlf": 0.08,  # d=0.46 — Norm Layer Fingerprint
    "lep": 0.16,  # d=0.92 — Layer Energy Profile
    "end": 0.19,  # d=1.05 — Embedding Norm Distribution
    "wvc": 0.21,  # d=1.21 — Weight Vector Correlation
}

# ── Tokenizer signal weights ───────────────────────────────────────
# Supplementary context — tokenizer is a tool, not a weight.

TOKENIZER_WEIGHTS: dict[str, float] = {"tfv": 0.25, "voa": 0.75}

# ── Decision threshold ────────────────────────────────────────────
# Calibrated on 111-pair benchmark (best F1 via threshold sweep).

SIMILARITY_THRESHOLD: float = 0.70
