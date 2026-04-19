# Scoring and Model Loading

This document covers two topics: how individual signal scores are combined
into a pipeline decision, and how models of any size are loaded for signal
extraction.

For what each signal measures and how similarity is computed, see
[Signals](signals.md). For how these fit into the overall pipeline, see
[Architecture](architecture.md).

---

## Scoring

### Identity Score

The identity score is the primary metric for provenance detection. It combines
five weight-level signals into a single value using a NaN-aware weighted
average:

```
identity_score = weighted_average(EAS, NLF, LEP, END, WVC)
```

The signals are not equally weighted — each signal's weight reflects its
discriminative power (measured by Cohen's d on a benchmark of Tier 3 pairs).
EAS contributes the most because embedding geometry is the strongest indicator
of shared training. WVC and END follow, then LEP and NLF. The exact weights
are defined in `src/provenancekit/core/scoring.py`.

### NaN-Aware Rescaling

All identity scores handle missing signals gracefully. If any signal returns
NaN (e.g., WVC returns NaN when models have incompatible layer counts), that
signal is excluded and the remaining weights are proportionally rescaled to
sum to 1.0.

**Example**: if WVC = NaN, the four remaining signals (EAS, NLF, LEP, END)
are rescaled so their weights still sum to 1.0, and the identity score is
computed from those four alone.

When all five identity signals fall back to neutral (0.5) or NaN, the pipeline
score lands in the 0.50 neutral zone — the pipeline cannot make a confident
determination without weight data.

### Tokenizer Score

The tokenizer score measures how likely two models share the same tokenizer.
It is **reported as a separate column** but does **not** contribute to the
pipeline score.

```
tokenizer_score = weighted_average(TFV, VOA)
```

VOA (vocabulary overlap) carries the larger weight because it is the most
discriminative tokenizer signal.

**Why exclude tokenizer from the pipeline decision?** Provenance is about
weight initialization, not tokenizer identity. Two independently trained
models can share a tokenizer (e.g., StableLM and Pythia both use GPT-NeoX
tokenizer, tokenizer score ~1.0, but they have no weight lineage). Conversely,
a model can be derived from a base model and have its tokenizer rebuilt during
continued pretraining.

### MFI Gate

The MFI gate short-circuits scoring for structurally matched pairs:

```
Tier 1 (arch_hash match)           → pipeline_score = 1.0
Tier 2 (family_hash + same dims)   → pipeline_score = 0.9
Tier 3 (soft match / demoted)      → pipeline_score = identity_score
```

Tier 1 and Tier 2 pairs are overwhelmingly likely to share provenance based
on metadata alone — expensive weight analysis is skipped. Tier 3 pairs
(including same-family/different-size demotions) require the full identity
score.

### Score Interpretation

| Pipeline Score | Verdict |
|----------------|---------|
| >= 0.85 | Same family / direct derivative |
| >= 0.65 | Likely same family or closely related |
| >= 0.45 | Possibly related |
| >= 0.25 | Weakly related |
| < 0.25 | Different families |

### Worked Examples

**Fine-tune** (Llama-2-7B → Llama-2-7B-Chat):

All weight signals score in the high 0.90s. The identity score is ~0.97.
Tokenizer score is ~1.00 (same tokenizer). Pipeline verdict: "Same family /
direct derivative."

**Shared tokenizer, independent training** (StableLM-3B vs Pythia-2.8B):

Weight signals are low (EAS ~0.36, WVC ~0.02) — weights are uncorrelated
because the models were trained from scratch independently. Identity score is
~0.35. Tokenizer score is ~1.00 (identical GPT-NeoX tokenizer). The pipeline
correctly returns "Weakly related" based on weights alone, despite the
tokenizer being identical.

**Vocabulary-modified derivation** (tokenizer rebuilt during continued
pretraining):

Some weight signals remain high (NLF ~0.80, WVC ~0.65) because core weights
still carry the parent's values. EAS is partially preserved (~0.75). Identity
score is ~0.63. Pipeline verdict: "Possibly related."

**Completely unrelated** (Falcon-7B vs Mistral-7B):

All weight signals are near baseline. Identity score ~0.20. Pipeline verdict:
"Different families."

---

## Model Loading

### Two-Tier Loading Strategy

Models of any size are supported through a loading strategy based on **disk
size** (not parameter count, since param estimates can be inaccurate — e.g.,
GLM-4.5-Air is estimated at 8.5B params but occupies 220 GB on disk):

| Model Disk Size | Loading Strategy | Peak Memory |
|-----------------|------------------|-------------|
| <= 20 GB | Full tensor load via safetensors | ~1–2x model size |
| > 20 GB | Safetensors row-slicing (32 rows per tensor) | ~1–2 GB |

### Row-Slicing Mechanism

For models above the disk threshold, instead of loading entire tensors, the
pipeline reads **32 evenly-spaced rows** from each 2D tensor using
safetensors' random-access `get_slice()` API:

```
For a tensor with shape [8192, 5120]:
  stride = 8192 / 32 = 256
  Load rows: [0, 256, 512, ..., 7936]  → 32 x 5120 = 163,840 elements
  vs full load: 8192 x 5120 = 41,943,040 elements (99.6% reduction)
```

### Signal Behavior Under Row-Slicing

All five weight signals are computed — nothing is skipped or returned as NaN
due to model size:

| Signal | Full Load | Row-Sliced | Notes |
|--------|-----------|------------|-------|
| **EAS** | Full embedding tensor | Full embedding (always loaded) | Embedding is always loaded in full |
| **END** | Full embedding tensor | Full embedding (always loaded) | Same tensor as EAS |
| **NLF** | Full 1D norm vectors | Full 1D norm vectors | Norm vectors are small (~500 KB total) |
| **LEP** | Full 2D tensors → Frobenius norm | 32 rows → scaled energy | Energy scaled by `total_rows / 32` |
| **WVC** | 4096-element subsample | Subsample from loaded rows | Deterministic stride within loaded rows |

### Shard-Aware Streaming

Large models are stored across multiple safetensors shard files. The pipeline
groups tensors by shard using the weight map, opens only shards containing
layer-indexed tensors, and runs `gc.collect()` after each shard to free
memory:

```
Step 1: get_safetensors_metadata(model_name)
        → weight_map: {tensor_name → shard_filename}

Step 2: Compute total disk size from shard files.

Step 3: IF total_disk > 20 GB → use row-slicing
        FOR each shard (grouped by weight_map):
          Open shard with safe_open(shard_path):
            FOR each tensor in this shard:
              Load 32 rows via get_slice()
              Accumulate LEP energy, WVC subsample
              Free tensor immediately
          gc.collect()

Peak memory ≈ 1–2 GB even for 220 GB models.
```

### Stall Detection and Recovery

All pipeline phases use stall detection. If no task completes within the
timeout, stalled tasks are cancelled and retried sequentially:

| Phase | Timeout | Retry Strategy |
|-------|---------|----------------|
| Phase 1a (base features) | 180s (3 min) | Sequential retry |
| Phase 1b (weight signals) | 600s (10 min) | Sequential retry |
| Phase 2 (pair evaluation) | 120s (2 min) | Sequential retry |

### Performance Settings

| Setting | Value | Purpose |
|---------|-------|---------|
| Phase 1a parallelism | ThreadPoolExecutor | I/O-bound config/tokenizer downloads |
| Phase 1b parallelism | max(2, n_cpus // 4) workers | Reduced parallelism for memory-heavy weight loading |
| Phase 2 parallelism | ThreadPoolExecutor | Pair evaluation (all data cached) |
| Disk threshold | 20 GB | Triggers row-slicing for signal extraction |

### Verified Large Models

The streaming pipeline has been validated on:

- **chuck-norris-llm** (25B params, 27 shards, 130 GB): 14.6s, all 5 weight signals extracted
- **GLM-4.5-Air** (8.5B estimated params, 47 shards, 220 GB): 218s, all 5 weight signals extracted

---

## Neutral Fallbacks

All weight-level similarity functions return **0.5** (neutral) when data is
unavailable. Neutral means the signal neither helps nor hurts — the remaining
signals determine the outcome.

| Condition | Affected Signals | Fallback |
|-----------|-----------------|----------|
| No safetensors files available | EAS, NLF, LEP, END, WVC | 0.5 (or NaN for WVC) |
| Embedding tensor not found | EAS, END | 0.5 |
| < 8 anchor tokens found | EAS | 0.5 |
| No norm layers found | NLF | 0.5 |
| No layer indices extractable | LEP | 0.5 |
| Standard deviation < 1e-8 | EAS, LEP | 0.5 |
| < 2 common layers or < 64 elements | WVC | NaN |

## Further Reading

- [Architecture](architecture.md) — pipeline phases and data flow
- [Signals](signals.md) — detailed reference for each signal
- [Database and Caching](database-and-caching.md) — cache layers and reference DB
- [Python API](python-api.md) — programmatic usage
