# Changelog

## 1.2.0 (unreleased)

1.1.0 was set in `pyproject.toml` but never published; its changes (NVIDIA
Nemotron fingerprints and dependency updates) are included here.

This release makes verdicts depend on weight evidence rather than declared
metadata, and fixes several cases where scores depended on how a checkpoint
was stored rather than on its weights. Verdicts are stricter than in 1.0.0:
some pairs that were previously Confirmed or Weak are now Weak, unverified,
or Not Matched.

### Upgrade notes

- **Re-download the deep-signal fingerprints.** The fingerprint database was
  rebuilt with the fixed extractor (`extractor_version` 2), and `scan` refuses
  fingerprints from an older release:

  ```bash
  provenancekit download-deepsignals-fingerprint --update
  ```

  The download is now `deep-signals-v2.zip`; `deep-signals.zip` stays in place
  for 1.0.0.
- Cached features from 1.0.0 are re-extracted automatically on first use.
- JSON output has a new `verification` value, `weights_shallow`, and a new
  `config_inconsistent` list.

### Security and integrity fixes

Reported through coordinated disclosure by Huan Bui
([@huuhuannt1998](https://github.com/huuhuannt1998), hbui11@charlotte.edu).

- **Metadata no longer confirms a match on its own.** A Tier 1/2 metadata
  match previously gave Confirmed Match with no weight evidence, and overrode
  weight evidence that disagreed. Weights now decide the verdict; metadata
  only supports it, and a match without weight evidence is marked
  `unverified`.
- **`config.json` is checked against the tensor shapes.** Editing a config
  field to the claimed parent's value could turn Not Matched into Confirmed
  Match. `vocab_size`, `hidden_size`, `num_hidden_layers`, and
  `intermediate_size` are now compared with the stored weights; a
  contradicting model gets no Tier 1/2 and the fields are reported in
  `config_inconsistent`.
- **Scores no longer depend on file layout.**
  - NLF and WVC compare tensors in a canonical order, so re-saving identical
    weights in a different order no longer changes the score.
  - Shards in the standard Hugging Face cache layout (symlinks into `blobs/`)
    are accepted instead of silently falling back to another loader.
  - Full and streaming (row-sliced) loads sample LEP and WVC the same way.
  - Raw `pytorch_model.bin` tensors are loaded before the `AutoModel`
    fallback, and the fallback is logged and recorded.

### Fixes

- A missing `architectures` key in `config.json` no longer aborts
  verification.
- GPT-2-style `c_attn`/`c_fc`/`c_proj` tensors now count towards WVC (missing
  since 1.0.0; affected gpt2 and starcoder2 models).
- Tokenizers that return `bytes` vocabulary keys (e.g. tiktoken-based
  glm-4) no longer crash feature extraction.
- ALBERT's shared layer groups are no longer reported as a config
  inconsistency.

### Scoring changes

- **Unrelated-weights floor:** when WVC is computed and below 0.01, the score
  is capped at 0.65 (Not Matched). Separately trained models of the same
  family, such as Llama-2-7B and Llama-2-13B, were previously Weak or
  High-Confidence matches.
- **Quantized checkpoints:** packed bitsandbytes, AWQ, and GPTQ weights are
  excluded from LEP and WVC. Without them, a match needs MFI Tier 1/2 and
  matching tokenizers and is marked `weights_shallow`; otherwise it is capped
  at Weak Match (unverified). Scan ranks these matches by signals, not by the
  verification label.
- **FP8 and MLX checkpoints** return no weight evidence and an unverified
  verdict until format-specific dequantization is added. MLX models that
  declare only a `quantization` key are detected.

### Database

- Rebuilt with the fixed extractor: 179 models, each pinned to a Hugging Face
  commit (`source_revision`), including the NVIDIA Nemotron fingerprints
  added since 1.0.0.
- Not yet rebuilt and therefore not scanned against in this release:
  Llama-3.1-70B-Instruct, Qwen2.5-72B-Instruct, Nemotron-Flash-1B, Nemotron-Flash-3B-Instruct, and
  Llama-3.1-Nemotron-Nano-VL-8B-V1.
- New `scripts/build_database.py` rebuilds the database and the
  deep-signals archive from an existing catalog.

### Known limitations

- Knowledge transfer without shared weights (e.g. phi-1.5 to phi-2) and
  distillation that changes dimensions are not detected.
- Quantized checkpoints and ALBERT have no LEP/WVC; against an unrelated model
  of similar shape they can reach Weak Match (unverified).
- Repos that ship only a SentencePiece `tokenizer.model` can lose EAS anchor
  alignment and score lower against their parent.
- In `scan`, vocabulary overlap (VOA) is not computed, so `tokenizer_score` is
  partial (0.25 × TFV, at most 0.25). The scan table now says so; use the `tfv`
  field or `compare` for tokenizer similarity.

## 1.0.0

Initial release.
