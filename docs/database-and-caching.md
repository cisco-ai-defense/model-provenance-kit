# Database and Caching

This document describes the reference database that ships with the package,
how deep-signal fingerprints are downloaded and managed, and the two-layer
caching system used for extracted features.

## Reference Database

### Structure

The provenance seed database lives under `src/provenancekit/data/database/`
and follows a sharded layout:

```
data/database/
├── catalog/
│   ├── manifest.json                     # Shard registry
│   └── by-family/
│       ├── llama.json                    # Catalog shard for Llama family
│       ├── bloom.json                    # Catalog shard for BLOOM family
│       ├── gemma.json                    # ...
│       └── ...
├── features/
│   ├── base/
│   │   └── by-family/
│   │       ├── llama/
│   │       │   ├── <asset_id>_features.json
│   │       │   └── ...
│   │       ├── bloom/
│   │       │   └── ...
│   │       └── ...
│   └── deep-signals/                     # Downloaded separately
│       └── by-family/
│           ├── llama/
│           │   ├── <asset_id>_deep-signals.parquet
│           │   └── ...
│           └── ...
└── README.md
```

### Catalog

The catalog provides a registry of known base models organized by family.

- **`manifest.json`**: lists all catalog shards with their IDs and timestamps.
- **Family shards** (`by-family/<family_id>.json`): contain family metadata,
  model records, and asset records. Each asset has a `param_bucket` field used
  for size-based filtering during scan.

The catalog is loaded into memory at startup and provides O(1) lookups by
family ID, model ID, and asset ID.

### Base Features

Base features (`features/base/`) are pre-extracted fingerprints that ship with
the package. Each asset has a JSON file containing MFI fingerprint, tokenizer
features (TFV), and vocabulary data. These enable the metadata and tokenizer
stages of the scan pipeline without downloading any model weights.

### Deep-Signal Fingerprints

Deep-signal fingerprints (`features/deep-signals/`) are pre-computed
weight-level features stored as parquet files. They enable the full
weight-signal matching pipeline during scan. Without them, scan results rely
only on metadata and tokenizer signals.

Deep signals are **not** bundled with the package (they are too large). They
are downloaded separately:

```bash
provenancekit download-deepsignals-fingerprint
```

### Deep-Signal Download Process

1. Downloads `deep-signals.zip` from HuggingFace Hub over HTTPS only (non-HTTPS
   redirects are blocked).
2. Verifies SHA-256 integrity of the downloaded archive (unless `--no-verify`
   is used, which requires dev mode).
3. Extracts parquet files with safety checks:
   - Size limits on individual files and total extraction size.
   - Path traversal protection (rejects paths with `..` or absolute paths).
   - Symlink rejection.
4. Performs an **atomic swap** of the `by-family/` directory — the old
   directory is replaced in a single rename operation, preventing partial
   state.
5. Writes an installation marker (`.deep-signals-installed`) for status
   checks.

Check installation status or update at any time:

```bash
provenancekit download-deepsignals-fingerprint --status
provenancekit download-deepsignals-fingerprint --update
```

### Custom Database

To use your own database instead of the bundled one:

```bash
provenancekit scan gpt2 --db-root /path/to/my/database
```

Or set the environment variable:

```bash
export PROVENANCEKIT_DB_ROOT=/path/to/my/database
```

The custom database must follow the same directory structure as the bundled
one (catalog shards, base features, and optionally deep-signal parquets).

---

## Scan Lookup Process

When `provenancekit scan` runs, it performs a 3-stage lookup against the
database:

**Stage 1 — Param Filter**: narrows candidates to models in the same size
bucket (plus one adjacent bucket on each side). A 7B query model will not be
compared against 70B database entries.

**Stage 2 — Hash Check**: annotates each remaining candidate using the MFI
`arch_hash` and `family_hash` — marking it as exact match, family match, or
no match.

**Stage 3 — Full Similarity**: runs the complete scoring pipeline for each
candidate that passes the earlier stages, using both base features and
deep-signal fingerprints (if installed).

Results are ranked by pipeline score and the top-k matches above the threshold
are returned.

---

## Feature Caching

ProvenanceKit uses a two-layer cache to avoid re-extracting features for
models that have already been analyzed.

### Layer 1 — In-Memory Cache

A thread-safe Python dictionary that provides instant lookups within the same
process session. Features are stored as typed `CachedEntry` objects. The
in-memory cache is checked first on every `get()` call.

### Layer 2 — Disk Cache

JSON files stored under `~/.provenancekit/cache/` (configurable via
`--cache-dir` or `PROVENANCEKIT_CACHE_DIR`). Each model gets one JSON file
containing its MFI fingerprint, tokenizer features, vocabulary, and weight
signals.

Disk cache files are serialized via Pydantic's `model_dump` / `model_validate`
for type safety.

### HMAC Integrity

Disk cache files are protected by HMAC-SHA-256 to detect tampering or
corruption:

- A per-installation HMAC key is generated on first use and stored in
  `.cache_key` with restricted permissions (`0o600` — owner read/write only).
- Every cache file is signed on write and verified on read.
- If verification fails, the corrupt entry is discarded and the feature is
  re-extracted.

### Cache Lookup Flow

```
get(model_id)
  → check in-memory dict
    → HIT: return immediately
    → MISS: check disk JSON
      → HIT + HMAC valid: load into memory, return
      → HIT + HMAC invalid: discard, return None (re-extract)
      → MISS: return None (re-extract)

put(model_id, entry)
  → write to in-memory dict
  → serialize to JSON + compute HMAC
  → atomic write to disk (write to temp file, then rename)
```

### NullCache

When `--no-cache` is passed on the CLI, a `NullCache` object is used instead
of `CacheService`. It implements the same `get()` / `put()` / `clear()`
interface but does nothing — every run extracts features fresh.

### Cache Controls

```bash
# Use a custom cache directory
provenancekit compare gpt2 gpt2 --cache-dir /tmp/pk-cache

# Disable caching entirely
provenancekit compare gpt2 gpt2 --no-cache
```

### What Gets Cached

| Data | Stored In | Cached Per |
|------|-----------|------------|
| MFI fingerprint | Base features JSON | Model |
| TFV features | Base features JSON | Model |
| Vocabulary set | Base features JSON | Model |
| EAS, NLF, LEP, END, WVC | Weight signals JSON | Model |

On a warm cache, ProvenanceKit skips model downloading, weight loading, and
feature extraction entirely — reducing comparison time from minutes to seconds.

---

## Further Reading

- [Architecture](architecture.md) — how database and cache fit into the
  pipeline phases
- [Signals](signals.md) — what features are extracted and cached
- [Scoring and Model Loading](scoring-and-model-loading.md) — how cached
  features are scored
