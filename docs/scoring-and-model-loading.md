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

In `scan`, VOA is not computed: the database stores only each model's
vocabulary size, not its tokens. The scan's `tokenizer_score` is therefore
partial (0.25 × TFV, at most 0.25) and is not comparable with `compare`'s.
Use the `tfv` field for tokenizer similarity in scan results, or run
`compare` for the full score.

**Why exclude tokenizer from the pipeline decision?** Provenance is about
weight initialization, not tokenizer identity. Two independently trained
models can share a tokenizer (e.g., StableLM and Pythia both use GPT-NeoX
tokenizer, tokenizer score ~1.0, but they have no weight lineage). Conversely,
a model can be derived from a base model and have its tokenizer rebuilt during
continued pretraining.

### Weights Decide, Metadata Supports

MFI tiers come from `config.json` alone, which is trivially edited, so a
metadata match can never produce a match verdict on its own. Whenever
sufficient weight evidence exists, `pipeline_score = identity_score`; the MFI
score is reported separately in `mfi_score` / `mfi_tier`. The quantized-model
exception is described below.

```
Tier 1 (arch_hash match)           → metadata match
Tier 2 (family_hash + same dims)   → metadata match
Tier 3 (soft match / demoted)      → no metadata match
```

When there is no weight evidence (identity is NaN, e.g. no deep-signal
fingerprint or the weights could not be loaded), `pipeline_score` falls back
to the MFI score so results can still be ranked, and the decision is marked
**unverified**.

`config.json` is also checked against the stored tensor shapes
(`vocab_size`, `hidden_size`, `num_hidden_layers`, `intermediate_size`).
If a model's config contradicts its own weights, MFI Tier 1/2 and the scan's
hash labels are skipped for that model, and the fields are listed in
`config_inconsistent` in the output.

### Score Interpretation

| MFI Tier | Identity Score (I) | Verdict | `verification` |
|----------|--------------------|---------|----------------|
| 1–2 | I > 0.75 | Confirmed Match | `weights_agree` |
| 1–2 | 0.65 < I ≤ 0.75 | Weak Match (metadata match, weak weight evidence) | `weights_weak` |
| 1–2 | I ≤ 0.65 | Not Matched (metadata/weight conflict) | `weights_conflict` |
| 3 | I > 0.75 | High-Confidence Match | `weights_only` |
| 3 | 0.65 < I ≤ 0.75 | Weak Match | `weights_only` |
| 3 | I ≤ 0.65 | Not Matched | `weights_only` |
| 1–2, or 3 with MFI > 0.65 | NaN | Metadata Match (unverified) | `unverified` |
| 3 with MFI ≤ 0.65 | NaN | Not Matched (unverified) | `unverified` |

Two rules adjust this table: the unrelated-weights floor and the
shallow-evidence guard, both described below.

In scan mode, weight-backed matches are ranked ahead of unverified ones.

### Unrelated-Weights Floor

Models of the same family that were trained separately (for example
Llama-2-7B and Llama-2-13B, or two Pythia sizes) share architecture and
tokenizer. That lifts EAS, NLF, LEP, and END, so the identity score can
exceed 0.65 even though no weights are shared. WVC still separates them:
unrelated weights give WVC around 0.003, while fine-tunes, distillations,
and continued pretraining stay above 0.1.

When WVC is computed and is below 0.01, `pipeline_score` is capped at 0.65,
so the verdict is **Not Matched** (Tier 3) or **Not Matched
(metadata/weight conflict)** (Tier 1/2). The raw `identity_score` is still
reported. When WVC is missing (NaN), the floor does not apply.

### Quantized Checkpoints and Shallow Evidence

For bitsandbytes, AWQ, and GPTQ checkpoints, packed integer weights and
quantizer side tensors are not comparable to the original floating-point
weights. MPK excludes them from LEP and WVC while retaining comparable
floating-point embeddings, norms, and output heads for EAS, NLF, and END.

When both LEP and WVC are unavailable, a full match requires all of:

- MFI Tier 1 or 2;
- identity score above 0.75; and
- tokenizer feature similarity (TFV) of at least 0.9.

Such a result is marked `weights_shallow`, rather than `weights_agree`, because
it lacks deep projection-weight evidence. If the guard does not pass, the raw
`identity_score` is preserved but `pipeline_score` is capped at 0.75 and the
verdict is **Weak Match (unverified, no deep weight signals)**.

FP8 and MLX checkpoints currently return no weight evidence and therefore an
unverified metadata verdict. Their packed/scaled tensors require
format-specific dequantization before MPK can compare them reliably.

Known limitation: two quantized sibling checkpoints with the same architecture
and tokenizer may pass the shallow-evidence guard even when neither is the
direct parent of the other. The result establishes strong shared-family/base
evidence, not lineage direction. LEP/WVC recovery through scale-aware
dequantization is required to reduce this ambiguity.

### Known Limitations

- **Knowledge transfer without shared weights is not detected.** phi-2 was
  trained from phi-1.5's knowledge but not from its weights (WVC 0.003), so
  the pair is Not Matched.
- **Matches without LEP and WVC can still be Weak.** A quantized model or
  ALBERT (one layer shared across depth) has no LEP/WVC, so neither the
  floor nor deep evidence applies. Against an unrelated model of similar
  shape these can reach Weak Match (unverified) at 0.68–0.75.
- **Distillation that changes dimensions** (e.g. TinyBERT) is not matched
  to its teacher.
- **SentencePiece-only tokenizers.** A repo that ships only
  `tokenizer.model` (no `tokenizer.json`) can be converted without the
  leading-space marker, so EAS anchors do not line up with the parent and
  the verdict can drop to Weak.

### Worked Examples

**Fine-tune** (Llama-2-7B → Llama-2-7B-Chat):

All weight signals score in the high 0.90s. The identity score is ~0.97.
Tokenizer score is ~1.00 (same tokenizer). Pipeline verdict: "High-Confidence
Match."

**Same family, separate training** (Llama-2-7B vs Llama-2-13B):

EAS, NLF, LEP, and END are high because architecture and tokenizer are
shared, giving an identity score of ~0.75. WVC is ~0.003, so the
unrelated-weights floor caps the score at 0.65. Pipeline verdict: "Not
Matched."

**Shared tokenizer, independent training** (StableLM-3B vs Pythia-2.8B):

Weight signals are low (EAS ~0.36, WVC ~0.02) — weights are uncorrelated
because the models were trained from scratch independently. Identity score is
~0.35. Tokenizer score is ~1.00 (identical GPT-NeoX tokenizer). The pipeline
correctly returns "Not Matched" based on weights alone, despite the
tokenizer being identical.

**Vocabulary-modified derivation** (tokenizer rebuilt during continued
pretraining):

Some weight signals remain high (NLF ~0.80, WVC ~0.65) because core weights
still carry the parent's values. EAS is partially preserved (~0.75). Identity
score is ~0.63. Pipeline verdict: "Not Matched."

**Completely unrelated** (Falcon-7B vs Mistral-7B):

All weight signals are near baseline. Identity score ~0.20. Pipeline verdict:
"Not Matched."

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
