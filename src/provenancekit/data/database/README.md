# Provenance Seed Database

This folder contains the bundled seed database used by the package.

Seed structure:

- `catalog/manifest.json`: shard registry
- `catalog/by-family/<family_id>.json`: family/model/asset shards (UUID-identified, minimal fields)
- `features/base/by-family/<family_id>/<asset_id>_features.json`: primary extracted payload
- `features/deep-signals/by-family/<family_id>/<asset_id>_deep-signals.parquet`: optional heavy deep-scan payload (single merged parquet per asset)
- `features.json -> artifact_refs`: large-field artifact references

Deep-signals parquet format (long-form):

- One row stores one scalar value from one signal payload.
- Columns:
  - `signal` (string): signal name (`eas_self_sim`, `nlf_vector`, `lep_profile`, `end_histogram`, `wsp_signature`, `wvc_layer_sigs`)
  - `layer` (nullable int32): layer index when applicable (commonly set for `wvc_layer_sigs`; may be null for global vectors)
  - `row` (int32): row index within the signal payload
  - `col` (nullable int32): column index for matrix-like payloads (null for 1D payloads)
  - `value` (float32): scalar numeric value
- Rows are typically grouped by signal. A top-of-file preview often shows many `eas_self_sim` rows first; this does not mean other signals are missing.
- For models where some signals are unavailable due architecture/extraction constraints, the authoritative available set is `artifact_refs[].signals` in the corresponding base feature bundle.

Each catalog shard has a `shard_id` (UUID) and `updated_at` timestamp.
`publisher` is stored on the family record. `family_id` is not repeated
on model/asset rows (derived from the shard file at load time).

Each asset row includes a `param_bucket` field for scan-time structural
filtering.

Large arrays should be externalized via artifact refs rather than always kept
inline in `features.json`.
