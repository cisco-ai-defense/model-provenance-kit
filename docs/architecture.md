# Pipeline Architecture

Model Provenance Kit determines whether two models share a **weight-lineage
relationship** — i.e., whether one model's weights were initialized from the
other's checkpoint (through fine-tuning, distillation, quantization, continued
pretraining, or any combination).

The pipeline takes two model identifiers and produces a Pipeline Score (0–1)
plus a human-readable verdict.

## Signal Families

Eight signal families are extracted from each model, organized into three
roles:

| Role | Signals | What They Measure | Usage |
|------|---------|-------------------|-------|
| **Gate** (metadata) | MFI | Architecture config match (structural hashes) | Short-circuits Tier 1–2 pairs without weight loading |
| **Identity** (primary) | EAS, NLF, LEP, END, WVC | Weight-level training fingerprints and direct numerical comparison | Determines the Pipeline Score |
| **Tokenizer** (reporting) | TFV, VOA | Tokenizer structure and vocabulary overlap | Reported separately; does **not** influence the Pipeline Score |

The identity-primary design reflects a core principle: **provenance is about
weights**. A shared tokenizer does not imply provenance, and a different
tokenizer does not rule it out.

## Pipeline Phases

The pipeline runs in four phases:

```
┌──────────────────────────────────────────────────────────────────┐
│                   INPUT: Model A, Model B                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1a — Base Feature Extraction (per model, cached)          │
│                                                                  │
│    ├── Load config       → MFI fingerprint (22 fields, 2 hashes) │
│    ├── Load tokenizer    → TFV features (14 fields)              │
│    └── Extract vocabulary → token string set (for VOA)           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1b — Weight Signal Extraction (per model, cached)         │
│                                                                  │
│    Disk size <= 20 GB?                                           │
│      YES → full tensor load (CPU, float16)                       │
│      NO  → safetensors row-slicing (32 rows per tensor)          │
│                                                                  │
│    Extract: EAS, NLF, LEP, END, WVC                              │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Pairwise Similarity                                   │
│                                                                  │
│    Compute 8 similarity scores (each 0–1):                       │
│    MFI, TFV, VOA, EAS, NLF, LEP, END, WVC                       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Scoring and Decision                                  │
│                                                                  │
│    MFI Tier 1 (arch_hash match)        → pipeline = 1.0   DONE   │
│    MFI Tier 2 (family_hash + same dim) → pipeline = 0.9   DONE   │
│    MFI Tier 3 (soft match / demoted)   → pipeline = identity_score│
│                                                                  │
│    identity = 0.36×EAS + 0.08×NLF + 0.16×LEP + 0.19×END + 0.21×WVC│
│    tokenizer = 0.25×TFV + 0.75×VOA  (reporting only)            │
│                                                                  │
│    Decision = interpret(pipeline_score)                           │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 1a — Base Feature Extraction

For each model (run once, then cached):

- **MFI fingerprint**: reads 22 config fields and computes `arch_hash`
  (exact architecture) and `family_hash` (same family, ignoring dimensions).
- **TFV features**: extracts 14 tokenizer properties (vocab size, merge rules,
  special tokens, script distribution).
- **Vocabulary**: dumps the full token string set for VOA overlap analysis.

### Phase 1b — Weight Signal Extraction

For each model (run once, then cached):

- Models up to 20 GB on disk are loaded fully into memory.
- Models over 20 GB use safetensors row-slicing (32 rows per tensor) to keep
  peak memory around 1–2 GB. See
  [Scoring and Model Loading](scoring-and-model-loading.md#row-slicing-mechanism)
  for details.
- Five weight signals are extracted: **EAS** (embedding geometry), **NLF**
  (normalization weights), **LEP** (layer energy curve), **END** (embedding
  norm histogram), and **WVC** (per-layer weight cosine).

### Phase 2 — Pairwise Similarity

For each model pair, the cached features are compared to produce 8 similarity
scores (one per signal), each in the range 0–1.

### Phase 3 — Scoring and Decision

Similarity scores are combined into composite scores and a final verdict:

1. **MFI gate** checks structural match first:
   - Tier 1 (exact `arch_hash` match) → pipeline score = 1.0, done.
   - Tier 2 (`family_hash` match + same dimensions) → pipeline score = 0.9,
     done.
   - Tier 3 (soft match or demoted) → continue to identity scoring.

2. **Identity score** (NaN-aware weighted average of 5 weight signals):

   ```
   identity_score = 0.36×EAS + 0.08×NLF + 0.16×LEP + 0.19×END + 0.21×WVC
   ```

   If any signal returns NaN, it is excluded and the remaining weights are
   rescaled to sum to 1.0.

3. **Tokenizer score** (reporting only, not used in the decision):

   ```
   tokenizer_score = 0.25×TFV + 0.75×VOA
   ```

4. **Verdict** is derived from the pipeline score:

   | Pipeline Score | Verdict |
   |----------------|---------|
   | >= 0.85 | Same family / direct derivative |
   | >= 0.65 | Likely same family or closely related |
   | >= 0.45 | Possibly related |
   | >= 0.25 | Weakly related |
   | < 0.25 | Different families |

See [Scoring and Model Loading](scoring-and-model-loading.md) for detailed
formulas, worked examples, and large model support.

## Compare vs Scan

The pipeline supports two modes that share the same extraction and scoring
logic but differ in how candidates are selected.

### Compare (pairwise)

Takes two model identifiers, extracts features for both, and runs the full
scoring pipeline once. The flow above describes this mode.

```
provenancekit compare gpt2 distilgpt2
```

### Scan (one-vs-many)

Takes a single model identifier and matches it against a **reference database**
of known base-model fingerprints through a 3-stage narrowing process:

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: query model                                              │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Extract features (Phase 1a + 1b)                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  3-STAGE DATABASE LOOKUP                                         │
│                                                                  │
│  Stage 1 — Param Filter                                          │
│    Size-bucket filtering (± 1 adjacent bucket)                   │
│                         │                                        │
│                         ▼                                        │
│  Stage 2 — Hash Check                                            │
│    Annotate candidates: exact / family / none                    │
│                         │                                        │
│                         ▼                                        │
│  Stage 3 — Full Similarity                                       │
│    Run scoring pipeline per candidate (MFI gate + identity score)│
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Rank by pipeline score → return top-k above threshold           │
└──────────────────────────────────────────────────────────────────┘
```

1. **Stage 1 — Param Filter**: narrows the database to models in the same
   size bucket (±1 adjacent bucket), eliminating obviously incompatible
   candidates (e.g., a 7B model will not match a 70B model).

2. **Stage 2 — Hash Check**: annotates each remaining candidate as exact
   match, family match, or no match using the MFI `arch_hash` and
   `family_hash`.

3. **Stage 3 — Full Similarity**: runs the complete scoring pipeline (Phase 2
   + Phase 3) for each candidate that passes the earlier stages.

Results are ranked by pipeline score and the top-k matches above the threshold
are returned.

```
provenancekit scan bigscience/bloom-560m
provenancekit scan gpt2 --top-k 10 --threshold 0.30
```

The reference database ships with the package under
`src/provenancekit/data/database/` and can be overridden with `--db-root`.
See [Database and Caching](database-and-caching.md) for the database structure.

## Data Flow Summary

```
Model ID
  → config.json          → MFI fingerprint (22 fields, 2 hashes)
  → tokenizer files      → TFV features (14 fields) + vocabulary set
  → weight tensors       → EAS, NLF, LEP, END, WVC features
      ↓
  All features cached (in-memory + disk JSON)
      ↓
  Compare: pairwise similarity → scoring → verdict
  Scan:    3-stage DB lookup → per-candidate scoring → ranked matches
```

## Caching

Both base features (Phase 1a) and weight signals (Phase 1b) are cached per
model. On a warm cache, ProvenanceKit skips model loading and feature
extraction entirely, reducing comparison time from minutes to seconds. See
[Database and Caching](database-and-caching.md) for cache internals.

## Further Reading

- [Signals](signals.md) — detailed reference for each of the 8 signals
- [Scoring and Model Loading](scoring-and-model-loading.md) — formulas, NaN handling, worked examples, streaming and row-slicing for 100 GB+ models
- [Python API](python-api.md) — programmatic usage with `ModelProvenanceScanner`
