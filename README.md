# Model Provenance Kit

> Part of [Cisco AI Defense](https://github.com/cisco-ai-defense) - Open-source AI security scanners, developer tools, and research from Cisco.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.com/invite/nKWtDcXxtx)
[![Cisco AI Defense](https://img.shields.io/badge/Cisco-AI%20Defense-049fd9?logo=cisco&logoColor=white)](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html)
[![AI Security and Safety Framework](https://img.shields.io/badge/AI%20Security-Framework-orange)](https://learn-cloudsecurity.cisco.com/ai-security-framework)

Model Provenance Kit is a Python toolkit and CLI for detecting model provenance. It determines whether a machine learning model derives from a known base model family by comparing multi-signal fingerprints extracted from weights, tokenizers, and architecture metadata.

![Model Provenance Kit Demo](images/demo.gif)

## Key Features

- **Pairwise comparison**: compare any two models head-to-head across 8 provenance signals.
- **Database scan**: scan a model against a reference database of known base-model fingerprints.
- **Deep-signal fingerprints**: download pre-computed weight fingerprints for weight-level matching.
- **Multi-signal pipeline**: combines metadata (MFI), tokenizer (TFV, VOA), and weight signals (EAS, NLF, LEP, END, WVC) into a single pipeline score.
- **MFI gate**: architecture metadata acts as a fast structural gate before expensive weight analysis.
- **Two-layer caching**: in-memory + disk JSON cache for fast repeat runs.
- **Multiple output formats**: Rich terminal table (default), JSON, or plain text.
- **Streaming support**: models over 20 GB are loaded via streaming to limit memory usage.

## Reference Database

The bundled reference database contains fingerprints for **~150 base models** spanning **45+ model families** from **20+ publishers**, ranging from 135M to 70B+ parameters. Covered publishers include:

Meta, Google, Alibaba, Microsoft, Mistral AI, DeepSeek, TII, Zhipu AI, NVIDIA, IBM, BigScience, OpenAI, Allen AI, Facebook AI, Stability AI, Hugging Face, Cohere, Databricks, Tencent, Moonshot AI, MiniMax, and more.

The database covers text generation, fill-mask, text-to-text, embedding, and translation architectures across four size buckets (<=1B, 1B–10B, 10B–40B, 40B+).

## Documentation

For deeper technical details, see the guides in [`docs/`](docs/):

| Guide | Description |
|-------|-------------|
| [Pipeline Architecture](docs/architecture.md) | End-to-end data flow, compare vs scan modes, phase breakdown |
| [Signal Reference](docs/signals.md) | Extraction, similarity, and behaviour of all 8 provenance signals |
| [Scoring and Model Loading](docs/scoring-and-model-loading.md) | Identity/tokenizer scores, MFI gate, NaN handling, large-model streaming |
| [Database and Caching](docs/database-and-caching.md) | Seed database layout, deep-signal download, two-layer cache, HMAC integrity |

## Installation

### Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Install from source

```bash
git clone https://github.com/cisco-ai-defense/model-provenance-kit.git
cd model-provenance-kit
uv sync
```

### Install as a CLI tool

```bash
uv tool install .
```

After installation the `provenancekit` command is available:

```bash
provenancekit --help
```

## Quick Start

### 1. Download deep-signal fingerprints (one-time setup)

Deep-signal fingerprints are pre-computed weight-level features stored as parquet files. They enable the full weight-signal matching pipeline during `scan`. Without them, scan results rely only on metadata and tokenizer signals.

The fingerprints are hosted on Hugging Face: [cisco-ai/model-provenance-kit](https://huggingface.co/datasets/cisco-ai/model-provenance-kit).

```bash
provenancekit download-deepsignals-fingerprint
```

Check installation status at any time:

```bash
provenancekit download-deepsignals-fingerprint --status
```

To update to the latest fingerprints:

```bash
provenancekit download-deepsignals-fingerprint --update
```

### 2. Scan a model against known base models

```bash
provenancekit scan bigscience/bloom-560m
```

This extracts features from the model, runs a 3-stage lookup against the reference database, and returns ranked matches with scores and decision labels.

### 3. Compare two models head-to-head

```bash
provenancekit compare gpt2 distilgpt2
```

## Usage

### Commands

| Command | Purpose |
|---------|---------|
| `provenancekit compare MODEL_A MODEL_B` | Pairwise comparison of two models |
| `provenancekit scan MODEL_ID` | Scan one model against the reference database |
| `provenancekit download-deepsignals-fingerprint` | Download/manage deep-signal weight fingerprints |

### Output Formats

All commands that produce results support three output modes:

```bash
# Rich terminal table (default)
provenancekit compare gpt2 gpt2

# JSON (machine-readable, suitable for piping)
provenancekit compare gpt2 gpt2 --json

# Plain text (no colour, CI-friendly)
provenancekit compare gpt2 gpt2 --plain
```

### Verbose Logging

Enable structured logging to stderr with the top-level `--verbose` flag. It must come **before** the subcommand:

```bash
provenancekit --verbose scan bigscience/bloom-560m
provenancekit --verbose compare gpt2 distilgpt2
```

## CLI Reference

### `provenancekit compare`

```text
provenancekit [--verbose] compare MODEL_A MODEL_B [options]
```

| Option | Description |
|--------|-------------|
| `MODEL_A` | First model: HuggingFace repo ID (e.g. `gpt2`) or local path |
| `MODEL_B` | Second model: HuggingFace repo ID or local snapshot path |
| `--json` | Output as JSON |
| `--plain` | Output as plain key-value text (no colour) |
| `--cache-dir PATH` | Override the default cache directory |
| `--no-cache` | Disable feature caching entirely |
| `--timing` | Show high-level phase timings |

**Examples:**

```bash
# Basic comparison
provenancekit compare gpt2 distilgpt2

# Compare with JSON output
provenancekit compare bigscience/bloom-560m bigscience/bloomz-560m --json

# Compare with custom cache
provenancekit compare gpt2 gpt2 --cache-dir /tmp/pk-cache

# Compare without caching
provenancekit compare gpt2 gpt2 --no-cache
```

### `provenancekit scan`

```text
provenancekit [--verbose] scan MODEL_ID [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_ID` | | Model to scan: HuggingFace repo ID or local snapshot path |
| `--json` | | Output as JSON |
| `--plain` | | Output as plain key-value text (no colour) |
| `--top-k N` | `3` | Maximum number of matches to return |
| `--threshold F` | `0.50` | Minimum pipeline score for inclusion (0.0–1.0) |
| `--db-root PATH` | bundled DB | Override the provenance database root directory |
| `--cache-dir PATH` | `~/.provenancekit/cache` | Override the default cache directory |
| `--no-cache` | | Disable feature caching |
| `--timing` | | Show phase-level timing breakdown |

**Examples:**

```bash
# Basic scan
provenancekit scan bigscience/bloom-560m

# Scan with more results and lower threshold
provenancekit scan gpt2 --top-k 10 --threshold 0.30

# Scan with JSON output
provenancekit scan bigscience/bloom-560m --json

# Scan with a custom database
provenancekit scan gpt2 --db-root /path/to/my/database
```

**Scan workflow:**

1. Extract model fingerprint and features (MFI, tokenizer, weight signals).
2. Run a 3-stage lookup against the provenance database:
   - **Stage 1 (Param Filter)**: size-bucket filtering (±1 adjacent bucket).
   - **Stage 2 (Hash Check)**: annotate candidates with exact/family/none match.
   - **Stage 3 (Similarity)**: full scoring per candidate with MFI gate.
3. Return ranked matches with scores, decision labels, and signal breakdowns.

### `provenancekit download-deepsignals-fingerprint`

```text
provenancekit download-deepsignals-fingerprint [options]
```

| Option | Description |
|--------|-------------|
| `--db-root PATH` | Override the provenance database root directory |
| `--update` | Re-download and replace existing fingerprints with the latest |
| `--no-verify` | Skip SHA-256 integrity check after download |
| `--status` | Show current deep-signals installation status and exit |

**What it does:**

1. Downloads `deep-signals.zip` from HuggingFace Hub (HTTPS only).
2. Verifies SHA-256 integrity (unless `--no-verify`).
3. Extracts parquet files with safety checks (size limits, path traversal protection, symlink rejection).
4. Performs an atomic swap of the `by-family/` directory.
5. Writes an installation marker for subsequent status checks.

**Examples:**

```bash
# First-time install
provenancekit download-deepsignals-fingerprint

# Check what's installed
provenancekit download-deepsignals-fingerprint --status

# Force update to latest
provenancekit download-deepsignals-fingerprint --update

# Install to a custom database location
provenancekit download-deepsignals-fingerprint --db-root /data/provenance-db
```

## Signals and Scoring

Model ProvenanceKit combines three categories of evidence into a single pipeline score.

### Metadata Signal

| Signal | Full Name | Description |
|--------|-----------|-------------|
| MFI | Metadata Family Identification | 3-tier gate from `config.json`: Tier 1 (exact arch hash), Tier 2 (family hash + dimension check), Tier 3 (weighted soft match across 11 feature groups) |

### Tokenizer Signals

| Signal | Full Name | Description |
|--------|-----------|-------------|
| TFV | Tokenizer Feature Vector | 11-component structural similarity (class, vocab size, BOS/EOS, script distribution, merge rules, etc.) |
| VOA | Vocabulary Overlap Analysis | Jaccard similarity between vocabulary sets |

### Weight Signals

| Signal | Full Name | Description |
|--------|-----------|-------------|
| EAS | Embedding Anchor Similarity | Pairwise cosine of script-aware anchor embedding rows → self-similarity matrix → Pearson on upper triangle |
| NLF | Norm Layer Fingerprint | Concatenated LayerNorm/RMSNorm weight vectors → cosine similarity |
| LEP | Layer Energy Profile | Frobenius norm per layer → 1D profile → Pearson correlation |
| END | Embedding Norm Distribution | Row-wise L2 norms of embeddings → histogram → cosine similarity |
| WVC | Weight Vector Correlation | Per-layer statistical signature → mean cosine over common layers |

### Scoring

**Identity score**: NaN-aware weighted average of the 5 weight signals (EAS, NLF, LEP, END, WVC). Signal weights are calibrated via Cohen's d on a 111-pair benchmark. When a signal returns NaN, it is excluded and remaining weights are proportionally rescaled.

**Tokenizer score**: supplementary context, 25% TFV + 75% VOA. Reported alongside identity but not used in the pipeline decision.

**Pipeline score**: final decision score using the MFI gate:
- MFI Tier 1-2 (structural match): pipeline score = MFI score
- MFI Tier 3 (no structural match): pipeline score = identity score

### Score Interpretation

| Pipeline Score | Verdict |
|----------------|---------|
| S = 1.0 or MFI Tier ≤ 2 | Confirmed Match |
| S > 0.75 | High-Confidence Match |
| 0.65 < S ≤ 0.75 | Weak Match |
| S ≤ 0.65 | Not Matched |

## Caching

Model ProvenanceKit uses a two-layer feature cache to speed up repeat comparisons:

1. **In-memory cache**: session-scoped Python dict for instant lookups within the same process.
2. **Disk cache**: JSON files under `~/.provenancekit/cache/` (configurable) storing per-model MFI fingerprints, tokenizer features, vocabularies, and weight signals.

On a warm cache, Model ProvenanceKit skips expensive model loading and feature extraction, reducing comparison time from minutes to seconds.

**Cache controls:**

```bash
# Use a custom cache directory
provenancekit compare gpt2 gpt2 --cache-dir /tmp/pk-cache

# Disable caching entirely (always extract fresh)
provenancekit compare gpt2 gpt2 --no-cache
```

The HuggingFace Hub also caches downloaded model files and tokenizer assets locally. Both caches work together to minimize network usage and computation.

## Environment Variables

All settings use the `PROVENANCEKIT_` prefix and can be set as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVENANCEKIT_CACHE_DIR` | `~/.provenancekit/cache` | Feature cache directory |
| `PROVENANCEKIT_DB_ROOT` | bundled database | Path to the provenance seed database |
| `PROVENANCEKIT_SCAN_TOP_K` | `3` | Max matches for scan |
| `PROVENANCEKIT_SCAN_THRESHOLD` | `0.50` | Min pipeline score for scan results |

**Example:**

```bash
export PROVENANCEKIT_CACHE_DIR=/tmp/pk-cache
export PROVENANCEKIT_SCAN_TOP_K=10
provenancekit scan gpt2
```

## Benchmark

The `benchmarks/run_benchmark.ipynb` notebook runs a structured evaluation across similar and dissimilar model pairs.

### Jupyter Kernel Setup

```bash
cd model-provenance-kit
uv pip install ipykernel
uv run python -m ipykernel install --user --name provenancekit --display-name "ProvenanceKit (.venv)"
```

Then select **"ProvenanceKit (.venv)"** as the kernel in VS Code / Cursor / JupyterLab.

### Configuration

The notebook exposes three knobs in the **Configuration** cell:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_WORKERS` | `2` | Parallel comparison workers |
| `PAIR_LIMIT` | `None` | Max pairs to evaluate (`None` = all) |
| `PAIR_FILTER` | `"all"` | `"all"`, `"similar"`, or `"dissimilar"` |

## Development

### Run tests

```bash
# Fast tests only
uv run pytest -m "not slow" --tb=short -q

# All tests (includes model downloads)
uv run pytest -m slow

# With coverage
uv run pytest -m "not slow" --cov=provenancekit --cov-report=term-missing
```

### Lint and type-check

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

### Pre-commit hooks

```bash
# Install hooks
uv run pre-commit install

# Run all hooks against all files
uv run pre-commit run --all-files
```

## Troubleshooting

### Model downloads hang or stall

Recent versions of `huggingface_hub` (≥ 0.27) include **Xet**, a storage backend for large files. On some networks and VPNs the Xet transfer protocol can stall or produce errors when downloading model weights, tokenizer files, or other repository assets (e.g. `Byte range not sequential`, `Can't load tokenizer`).

If you experience download hangs, corrupted file errors, or tokenizer-loading failures, disable Xet and fall back to standard HTTPS:

```bash
# One-off
HF_HUB_DISABLE_XET=1 provenancekit scan bigscience/bloom-560m

# Persistent (add to your shell profile)
export HF_HUB_DISABLE_XET=1
```

This also applies when running tests:

```bash
HF_HUB_DISABLE_XET=1 uv run pytest tests/
```

> **Tip:** If the issue persists, try clearing the cached files for the affected model and retrying:
> ```bash
> rm -rf ~/.cache/huggingface/hub/models--<org>--<model>
> ```

### Streaming extraction is slow

The first run downloads safetensors shards from the Hub. Subsequent runs reuse the HuggingFace cache and complete in seconds.

### Deep-signal fingerprints not installed

If `scan` shows a hint about missing deep-signal fingerprints:

```bash
provenancekit download-deepsignals-fingerprint
```

This enables the full weight-signal matching pipeline. Without deep signals, scan results rely on metadata and tokenizer signals only.

### Scan returns few or no matches

- Try lowering the threshold: `--threshold 0.30`
- Try increasing top-k: `--top-k 10`
- Verify deep signals are installed: `provenancekit download-deepsignals-fingerprint --status`

## Notes and Limitations

- Model comparisons depend on available model artifacts and configs on HuggingFace.
- The `scan` command uses a bundled seed database. For custom deployments, use `--db-root` or `PROVENANCEKIT_DB_ROOT` to point to your own database directory.
- Results provide strong evidence of provenance but are not absolute proof.
- Weight signals require loading model weights into memory (or streaming for large models). First-run performance depends on network speed and model size.

## Contributing

Contributions are welcome. Please read the following before submitting a pull request:

- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Security

To report a security vulnerability, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
