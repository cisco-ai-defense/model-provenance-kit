# Canonicalization (Comparison-Space Hardening)

ProvenanceKit's weight-level signals (EAS, NLF, LEP, END, WVC) compute
cosine / correlation scores in raw weight space. Raw weight space is
*basis-sensitive*: two functionally identical models can produce very
different scores after a cheap, function-preserving transformation such
as

* attention-head permutation,
* MLP/neuron permutation,
* adjacent-layer (channel-wise) rescaling,
* layer-norm gamma absorption.

Canonicalization is an optional pre-processing pass that aligns model B
into model A's basis (heads + channels) and normalizes per-channel
scales before similarity scoring. It is opt-in via the `--canonicalize`
flag and disabled by default.

## What it does

| Step | Behaviour |
|------|-----------|
| Permutation alignment (attention) | Builds per-head signatures from Q/K/V (and the corresponding columns of the attention-output projection), solves a Hungarian assignment between A and B's heads, and applies the resulting permutation to B's Q/K/V output rows and to O's input columns. |
| Permutation alignment (MLP) | Builds per-channel signatures from `up_proj` (and `gate_proj` when present) rows together with `down_proj` columns, solves a Hungarian assignment, and applies the permutation to B's `up`, `gate`, and `down` projections. |
| Scale normalization (`comparison`, default) | Divides every per-channel slice by its L2 norm. Applied independently to A and B. **Non-invertible.** |
| Scale normalization (`function_preserving`) | Divides W_in by the per-channel norm α and multiplies W_out by α, preserving forward-pass equivalence. Stricter and slower; offered for callers who want to reuse the canonicalized weights. |
| LayerNorm gamma | Each LayerNorm/RMSNorm gamma vector is unit-normed independently so it cannot dominate cosine similarity once concatenated with other signals. |
| Stability check | Per-layer cosine before vs after alignment is compared; large jumps surface in `stability_warnings` to flag partially aligned layers, architecture mismatches, or bad tensor mapping. |

When SciPy is unavailable the assignment falls back to a greedy
max-matching solver. Pass `--canonicalize-method greedy` to force it.

## Important: comparison-only output

> Scale normalization operates in a comparison space and is not
> function-preserving. The resulting representation is non-invertible
> and must not be used for inference or model reconstruction.

> This design intentionally trades invertibility for invariance to
> common evasion strategies (channel rescaling and layer-norm
> absorption).

The canonicalizer returns `ComparisonView` objects tagged with
`is_comparison_only=True`. Inference, serialization, or model-export
code paths should call
`provenancekit.core.canonicalization.assert_not_comparison_view` on any
state-dict-shaped input as a runtime guard.

## CLI

```
provenancekit compare base-model suspect-model --canonicalize --json
provenancekit compare base-model suspect-model --canonicalize --canonicalize-method greedy
provenancekit compare base-model suspect-model --canonicalize --no-scale-normalize
provenancekit compare base-model suspect-model --canonicalize --canonicalize-scale-mode function_preserving
```

| Flag | Meaning |
|------|---------|
| `--canonicalize` | Enable the pass. Off by default. |
| `--canonicalize-method {hungarian,greedy}` | Assignment solver. `hungarian` requires SciPy. |
| `--canonicalize-scale-mode {comparison,function_preserving}` | Scale handling. Default is `comparison` (non-invertible). |
| `--no-scale-normalize` | Skip per-channel scale normalization. |
| `--no-permutation-align` | Skip head / channel permutation alignment. |

## JSON output additions

When `--canonicalize` is set, `compare`'s JSON output gains a
`canonicalization` section:

```json
"canonicalization": {
  "enabled": true,
  "method": "hungarian",
  "scale_mode": "comparison",
  "non_invertible": true,
  "layers_aligned": 32,
  "attention_heads_aligned": 1024,
  "mlp_channels_aligned": 11008,
  "scale_normalized": true,
  "unsupported_layers": [],
  "stability_warnings": [],
  "skipped_reason": null
}
```

`non_invertible: true` is intentional and not cosmetic. When
`scale_mode` is `comparison`, downstream consumers must treat the
canonicalized representation as a comparison artifact only.

## Limitations

Canonicalization reduces false negatives caused by permutation and
scale symmetries in functionally equivalent weight representations. It
**does not** prove lineage under distillation, retraining, model
merging, or behavioral imitation. Treat it as one additional defence
against trivial obfuscation, not as a behavioral fingerprint.

* Architecture metadata for both models must be compatible (same head
  count, same head dimension, same intermediate size). When metadata is
  missing or mismatched, alignment is skipped for that layer and
  recorded in `unsupported_layers`.
* Streaming-only loads (very large models) cannot be canonicalized
  in-place at comparison time; the report is returned with
  `skipped_reason="state_dict_unavailable"`.
* Pairwise canonicalization is performed at comparison time. Cached
  feature bundles remain unchanged so existing cache keys stay valid.
