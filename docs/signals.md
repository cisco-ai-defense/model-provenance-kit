# Signal Reference

This document describes each of the 8 provenance signals in detail: what they
extract, how similarity is computed, fallback behavior, and typical score
ranges.

For a high-level overview of how these signals fit into the pipeline, see
[Architecture](architecture.md). For how they are combined into a final score,
see [Scoring and Model Loading](scoring-and-model-loading.md).

## Signal Overview

| # | Signal | Full Name | Category | What It Captures |
|---|--------|-----------|----------|------------------|
| 1 | **MFI** | Metadata Family Identification | Metadata | Architecture config + two structural hashes for fast family matching |
| 2 | **TFV** | Tokenizer Feature Vector | Tokenizer | 14 tokenizer features (vocab size, merge rules, special tokens, script distribution) |
| 3 | **VOA** | Vocabulary Overlap Analysis | Tokenizer | Jaccard overlap of token string sets |
| 4 | **EAS** | Embedding Anchor Similarity | Weight | Self-similarity matrix of 64 anchor token embeddings |
| 5 | **NLF** | Norm Layer Fingerprint | Weight | Concatenated LayerNorm/RMSNorm weight vectors |
| 6 | **LEP** | Layer Energy Profile | Weight | Per-layer Frobenius norm of weight tensors |
| 7 | **END** | Embedding Norm Distribution | Weight | 20-bin histogram of embedding row norms |
| 8 | **WVC** | Weight-Value Cosine | Weight | Layer-by-layer subsampled cosine similarity of weight tensors |

---

## 1. MFI — Metadata Family Identification

MFI reads the model's `config.json` and computes two structural hashes. It
acts as a **fast gate** that can short-circuit expensive weight analysis.

### What It Extracts

22 fields from the model config, organized as:

| Category | Fields |
|----------|--------|
| Architecture | `model_type`, `architectures` |
| Dimensions | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings` |
| Design choices | `hidden_act`, `attention_style` (MHA/GQA/MQA), `norm_type` (RMSNorm/LayerNorm), `attention_bias`, `qk_norm`, `pos_encoding` (RoPE/absolute/learned) |
| RoPE config | `rope_theta`, `rope_scaling` |
| Token IDs | `bos_token_id`, `eos_token_id` |
| Identity | `tokenizer_hash` (SHA-256 of tokenizer backend) |

From these, two hashes are computed:

| Hash | Includes | Purpose |
|------|----------|---------|
| `arch_hash` | All dimension fields + model_type + architectures | Exact architecture match (Tier 1) |
| `family_hash` | model_type, attention_style, hidden_act, norm_type, tokenizer_hash, rope_theta_bucket, rope_scaling_type | Same family, ignoring dimensions (Tier 2) |

### Three-Tier Similarity

```
Tier 1: arch_hash_a == arch_hash_b?
  YES → score = 1.0, tier = 1, DONE

Tier 2: family_hash_a == family_hash_b?
  YES + same dimensions → score = 0.9, tier = 2, DONE
  YES + diff dimensions → demote to Tier 3
  NO  → fall through to Tier 3

Tier 3: Soft match
  9 weighted features → score/total + family_bonus (+0.15 if demoted)
```

Tier 3 compares 11 features including token IDs, `qk_norm`, `vocab_size`,
`attention_style`, `rope_theta`, `hidden_size`, `num_hidden_layers`,
`norm_type`, `pos_encoding`, `attention_bias`, and `model_type`. Each feature
uses either binary matching or ratio similarity and has its own weight in the
final soft score.

**Dimension gate**: when Tier 2 matches but `hidden_size` or
`num_hidden_layers` differ (e.g., Llama-2-7B vs Llama-2-13B), the pair is
demoted to Tier 3. This prevents same-family/different-size pairs from getting
an automatic 0.9.

---

## 2. TFV — Tokenizer Feature Vector

TFV creates a 14-feature structural fingerprint of a tokenizer — its
vocabulary size, how it splits words, what special tokens it uses, and its
merge rules.

### What It Extracts

| Feature | What It Measures | Example (Llama-2) |
|---------|------------------|-------------------|
| `vocab_size` | Total number of tokens | 32000 |
| `tokenizer_class` | Python class used | `LlamaTokenizerFast` |
| `bos_token_id` | Beginning-of-sequence token ID | 1 |
| `eos_token_id` | End-of-sequence token ID | 2 |
| `pad_token_id` | Padding token ID | 0 |
| `num_merges` | Number of BPE merge rules | 31743 |
| `first_5_merges` | First 5 merge rules | `["t h", "e n", ...]` |
| `pct_single_char` | Fraction of single-character tokens | 0.0089 |
| `avg_token_length` | Average string length of all tokens | 6.21 |
| `pct_whitespace_prefix` | Fraction starting with space marker | 0.5124 |
| `pct_byte_tokens` | Fraction matching byte-fallback pattern | 0.0080 |
| `script_distribution` | Writing systems in vocabulary | `{"Latin": 0.72, "CJK": 0.08}` |
| `merge_rule_hash` | SHA-256 of first 50 merge rules | `a3f2b1c9...` |
| `special_token_ids` | IDs for pad, unk, mask tokens | `{"pad": 0, "unk": 3}` |

### How Similarity Works

TFV similarity is computed as a weighted sum of 11 components, each comparing
one aspect of the two tokenizers: tokenizer class, vocab size, bos/eos IDs,
average token length, whitespace prefix handling, script distribution, byte
token presence, merge rule count, merge rule hash (first 50 rules), special
token IDs, and full merge rule identity.

Each component produces a score between 0 and 1 using either exact match,
ratio similarity, or cosine similarity. The final TFV score is a weighted
average across all components. Range: 0 to 1.

### Typical Values

| Pair Type | TFV |
|-----------|-----|
| Same tokenizer (Llama-2 vs Llama-2-Chat) | ~1.0 |
| Different tokenizer, same language (Llama vs Mistral) | ~0.55 |
| Different language families | ~0.20 |

---

## 3. VOA — Vocabulary Overlap Analysis

VOA is the **most discriminative tokenizer signal**. It measures how much the
token string sets of two models overlap.

### What It Extracts

The full vocabulary as a set of strings:

```python
vocab = set(tokenizer.get_vocab().keys())
# e.g., {"the", "▁of", "<0x0A>", ...}
```

### How Similarity Works — Jaccard Index

```
jaccard = |tokens in BOTH A and B| / |tokens in A or B or both|
```

**Example:**

```
Model A vocabulary: {"the", "cat", "sat", "on", "mat"}
Model B vocabulary: {"the", "dog", "sat", "in", "mat"}

Intersection: {"the", "sat", "mat"} → 3
Union:        {"the", "cat", "sat", "on", "mat", "dog", "in"} → 7

Jaccard = 3 / 7 = 0.43
```

### Typical Values

| Pair Type | VOA |
|-----------|-----|
| Same tokenizer (Llama-2 vs Llama-2-Chat) | 0.95–1.00 |
| Same language, different tokenizer (Llama vs Mistral) | 0.30–0.60 |
| Different languages (English-only vs Chinese-only) | 0.01–0.10 |

---

## 4. EAS — Embedding Anchor Similarity

EAS is the **strongest identity signal**. It compares the learned geometry of
embedding spaces using a fixed set of anchor tokens.

### Core Idea

Every token has a learned embedding vector. EAS picks 64 anchor tokens, builds
a self-similarity matrix (how similar each anchor's embedding is to every
other), and compares these matrices between two models. The self-similarity
matrix acts as a training fingerprint — models from the same checkpoint produce
nearly identical matrices even if they have different hidden sizes.

### Why Self-Similarity?

Raw embedding vectors depend on `hidden_size` (4096 for 7B, 5120 for 13B) —
you cannot directly compare vectors of different lengths. Self-similarity
matrices are always 64x64 regardless of hidden size, making EAS
**dimension-independent**.

### Anchor Selection (Script-Aware)

```
Phase 1 — Universal anchors (always included):
  Digits and punctuation: "0","1",...,"9",".",",","!","?",...
  → ~16 anchors that exist in virtually every tokenizer

Phase 2 — Detect vocabulary script distribution:
  Count what fraction of tokens belong to each script.
  Keep only scripts above 5% threshold.
  Example: Llama-2 → {Latin: 92%, CJK: 3%, Cyrillic: 2%, ...}

Phase 3 — Allocate remaining budget proportionally:
  Latin:      "the", "of", "and", "to", "in", "is", ...  (24 candidates)
  CJK:        "的", "一", "是", "不", "人", "我", ...      (12 candidates)
  Cyrillic:   "и", "в", "не", "на", "я", ...              (8 candidates)
  Arabic:     "من", "في", "على", "أن", ...                (8 candidates)
  Devanagari: "के", "है", "का", "की", ...                (8 candidates)
  Each active script gets at least 2 anchors.

Phase 4 — Fill remaining slots with random token IDs
  (deterministic, seed=42, from first 50K positions)
```

### Extraction Steps

1. Find the embedding matrix in the model weights (V x d table).
2. Select K=64 anchor tokens via script-aware allocation above.
3. Look up the embedding vector for each anchor (64 vectors of length d).
4. Compute a 64x64 self-similarity matrix using cosine similarity.

### How Similarity Is Computed

```
1. Align: K = min(anchors_A, anchors_B). Trim both matrices to KxK.
2. Flatten: extract upper triangle (K*(K-1)/2 unique pairs).
3. Correlate: Pearson correlation between the two flat vectors.
4. Rescale: EAS = (correlation + 1) / 2  →  range [0, 1]
```

**Fallback**: returns 0.5 (neutral) if fewer than 8 anchors or constant
vectors.

### Typical Values

| Pair Type | EAS |
|-----------|-----|
| Fine-tune (Llama-2 vs Llama-2-Chat) | ~0.97 |
| Independent training (StableLM vs Pythia) | ~0.36 |
| Completely unrelated | ~0.50 |

---

## 5. NLF — Norm Layer Fingerprint

NLF captures the learned normalization layer weights — tiny vectors (~500 KB
total) that are remarkably specific to each training run and barely change
during fine-tuning.

### Background

Normalization layers (LayerNorm or RMSNorm) rescale intermediate values
between transformer operations. Each has a learned weight vector of size
`hidden_size`. A typical 32-layer model has ~64–128 such vectors.

### What It Extracts

```
Step 1: Scan weights for normalization layers
        (names containing "layernorm", "rmsnorm", "ln_", "norm", etc.)
        Keep only 1D vectors with >= 64 elements.

Step 2: Choose representation mode:
        IF all vectors have the same length (typical):
          → "direct" mode: concatenate into one long vector
          Example: 64 norm layers x 4096 = 262,144 elements

        IF vectors have different lengths (rare):
          → "stats" mode: summarize each as [mean, std, max, min]
```

### How Similarity Is Computed

```
Both "direct" mode, same length:
  NLF = cosine_similarity(vector_A, vector_B)

Different modes or lengths:
  Convert to per-layer stats, truncate, then cosine similarity.
```

**Fallback**: returns 0.5 if either vector is missing or too short (< 4
elements).

### Typical Values

| Pair Type | NLF |
|-----------|-----|
| Fine-tune (Llama-2 vs Llama-2-Chat) | ~0.998 |
| Independent training (Llama vs Mistral) | ~0.50 |

---

## 6. LEP — Layer Energy Profile

LEP measures the "energy" (total weight magnitude) of each transformer layer,
producing a curve that acts as a fingerprint of the optimizer's trajectory.

### Background — Frobenius Norm

The Frobenius norm of a matrix is the square root of the sum of all squared
elements — "how big are the numbers in this matrix overall."

```
Matrix W = [[1, 2], [3, 4]]
Frobenius norm = sqrt(1² + 2² + 3² + 4²) = sqrt(30) ≈ 5.48
```

### What It Extracts

```
Step 1: For each weight tensor with a layer index and >= 2 dimensions:
          energy[layer_index] += Frobenius_norm(tensor)

Step 2: Build an energy array for layers 0 through max_layer.

Step 3: Normalize so the maximum is 1.0.
          → [0.988, 0.978, 1.000, 0.980, ..., 0.958]
```

### How Similarity Is Computed

```
1. If different layer counts, interpolate both to the shorter length.
2. Compute Pearson correlation between the two energy curves.
3. Rescale: LEP = (correlation + 1) / 2
```

**Fallback**: returns 0.5 if either profile is missing or constant.

### Typical Values

| Pair Type | LEP |
|-----------|-----|
| Fine-tune (Llama-2 vs Llama-2-Chat) | ~0.98 |
| Independent training (Llama vs Mistral) | ~0.50 |

---

## 7. END — Embedding Norm Distribution

END measures the distribution of embedding vector lengths across the entire
vocabulary, complementing EAS which captures geometry of 64 specific anchors.

### What It Extracts

```
Step 1: Load the embedding matrix (same tensor as EAS).

Step 2: Compute L2 norm of every row:
          norm[i] = sqrt(embed[i,0]² + ... + embed[i,d-1]²)

Step 3: Build a 20-bin histogram of these norms.

Step 4: Normalize so the histogram sums to 1.0.
```

### Why END Complements EAS

EAS compares the *geometry* (angles between embeddings) of 64 anchor tokens.
END compares the *magnitude distribution* across the entire vocabulary. A
model could have similar geometry but different magnitudes, or vice versa.

### How Similarity Is Computed

```
END = cosine_similarity(histogram_A, histogram_B)
```

Clamped to [0, 1]. **Fallback**: returns 0.5 if either histogram is missing.

### Typical Values

| Pair Type | END |
|-----------|-----|
| Fine-tune (Llama-2 vs Llama-2-Chat) | ~0.96 |
| Independent training (Falcon vs Llama) | ~0.50 |

---

## 8. WVC — Weight-Value Cosine

WVC directly compares the raw learned parameter values in attention and MLP
layers. Unlike EAS/NLF/LEP/END which capture structural fingerprints, WVC
tests whether two models have numerically identical (or near-identical) weights.

### What It Extracts

```
Step 1: Classify each weight tensor by role:
          "attention": q_proj, k_proj, v_proj, o_proj, etc.
          "mlp": gate_proj, up_proj, down_proj, fc1, fc2, etc.
          "identity": other layer-indexed 2D+ tensors
          SKIP: norm layers, embeddings, lm_head

Step 2: For each layer, collect all eligible tensors.

Step 3: Subsample each tensor to 4,096 elements (deterministic stride).
          → One flat vector per layer per model.
```

### How Similarity Is Computed

```
1. Find common layers between Model A and Model B
   (must have >= 2 common layers with >= 64 elements each).

2. For each common layer:
     cosine[layer] = cosine_similarity(subsample_A, subsample_B)

3. WVC = mean(cosine across all common layers)
```

**Fallback**: returns NaN if fewer than 2 common layers or insufficient data.
When WVC is NaN, it is excluded from the identity score and remaining signal
weights are rescaled.

### Typical Values

| Pair Type | WVC |
|-----------|-----|
| Fine-tune (Llama-2 vs Llama-2-Chat) | ~0.99 |
| Shared tokenizer, independent training (StableLM vs Pythia) | ~0.02 |
| Completely unrelated | ~0.00 |

---

## Signal Summary

| Signal | Extraction Cost | Similarity Method | Fallback |
|--------|----------------|-------------------|----------|
| **MFI** | Config only (no weights) | Hash tiers + weighted features | N/A |
| **TFV** | Tokenizer only | Weighted sum of 11 components | N/A |
| **VOA** | Tokenizer only | Jaccard index | N/A |
| **EAS** | 1 tensor (~250 MB for 7B) | Pearson of self-sim upper triangles | 0.5 |
| **NLF** | Tiny 1D vectors (~500 KB) | Cosine of concatenated vectors | 0.5 |
| **LEP** | All 2D tensors (streaming OK) | Pearson of energy profiles | 0.5 |
| **END** | Same tensor as EAS | Cosine of 20-bin histograms | 0.5 |
| **WVC** | Subsampled 2D tensors | Mean cosine across layers | NaN |

## Further Reading

- [Architecture](architecture.md) — how signals fit into the pipeline
- [Scoring and Model Loading](scoring-and-model-loading.md) — how signal
  scores are weighted and combined, and how 100 GB+ models are loaded via
  row-slicing
