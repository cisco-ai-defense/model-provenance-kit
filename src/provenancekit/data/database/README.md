# Provenance Seed Database

This folder contains the bundled seed database used by the package.

Seed structure:

- `catalog/manifest.json`: shard registry
- `catalog/by-family/<family_id>.json`: family/model/asset shards (UUID-identified, minimal fields)
- `features/base/by-family/<family_id>/<asset_id>_features.json`: primary extracted payload
- `features/deep-signals/by-family/<family_id>/<asset_id>_deep-signals.parquet`: optional heavy deep-scan payload (single merged parquet per asset)
- `features.json -> artifact_refs`: large-field artifact references

Each catalog shard has a `shard_id` (UUID) and `updated_at` timestamp.
`publisher` is stored on the family record. `family_id` is not repeated
on model/asset rows (derived from the shard file at load time).

Each asset row includes a `param_bucket` field for scan-time structural
filtering.

Large arrays should be externalized via artifact refs rather than always kept
inline in `features.json`.
