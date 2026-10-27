#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

r"""Rebuild the fingerprint database with the current extractor.

Reads the model list (families, asset ids, Hugging Face repos) from an
existing database, re-extracts every model with the same code path that
``provenancekit scan`` uses, and writes a new database stamped with
``EXTRACTOR_VERSION`` and the extraction settings.

Each model is downloaded at a pinned commit (recorded as
``source_revision``) and extracted from that local snapshot.  Finished
models are skipped on re-runs, so an interrupted build can be resumed.

Usage::

    # Pilot: five models into a scratch database
    python scripts/build_database.py --out /tmp/db-pilot \
        --only openai-community/gpt2 Qwen/Qwen2.5-0.5B

    # Full rebuild plus the deep-signals archive for Hugging Face
    python scripts/build_database.py --out /data/db-v2 \
        --zip /data/deep-signals-v2.zip --delete-downloads

Gated repos need a Hugging Face login (``hf auth login`` or ``HF_TOKEN``).
Repos with custom modelling code are only loaded when listed with
``--trust-remote-code-for`` after their code has been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

from provenancekit.config.constants import EXTRACTOR_VERSION
from provenancekit.config.settings import Settings
from provenancekit.core.scanner import ModelProvenanceScanner
from provenancekit.core.signals.weight_signals import (
    eas_similarity,
    end_similarity,
    lep_similarity,
    nlf_similarity,
    wvc_similarity,
)
from provenancekit.models.signals import WeightSignalFeatures
from provenancekit.models.storage import (
    ArtifactRef,
    AssetRecord,
    CatalogManifest,
    CatalogShard,
    FeatureBundle,
    ShardRef,
    SignalSummary,
)
from provenancekit.services.cache import NullCache
from provenancekit.services.database import DatabaseService
from provenancekit.utils.tensor import compute_param_bucket

log = structlog.get_logger()

_REPO_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE = _REPO_DIR / "src" / "provenancekit" / "data" / "database"
_HF_REPO_RE = re.compile(r"huggingface\.co/([^/]+/[^/?#]+)")
_SIGNALS = (
    "eas_self_sim",
    "nlf_vector",
    "lep_profile",
    "end_histogram",
    "wsp_signature",
    "wvc_layer_sigs",
)
_META_SUFFIXES = (".json", ".model", ".txt", ".tiktoken", ".py")


@dataclass
class _Job:
    """One asset to rebuild, with its catalog context."""

    shard: CatalogShard
    asset: AssetRecord
    repo_id: str

    @property
    def family_id(self) -> str:
        return self.shard.family.family_id

    def bundle_rel(self) -> str:
        return (
            f"features/base/by-family/{self.family_id}/"
            f"{self.asset.asset_id}_features.json"
        )

    def parquet_rel(self) -> str:
        return (
            f"features/deep-signals/by-family/{self.family_id}/"
            f"{self.asset.asset_id}_deep-signals.parquet"
        )


# ?? Model list ????????????????????????????????????????????????????


def _load_jobs(source: Path) -> list[_Job]:
    """Read every asset from the source database catalog."""
    manifest = DatabaseService(source).load_manifest()
    jobs: list[_Job] = []
    for ref in manifest.shards:
        shard = CatalogShard.model_validate(
            json.loads((source / ref.shard_path).read_text(encoding="utf-8"))
        )
        for asset in shard.assets:
            match = _HF_REPO_RE.search(asset.source_uri)
            if match is None:
                log.warning("asset_without_hf_repo", asset_id=asset.asset_id)
                continue
            jobs.append(_Job(shard=shard, asset=asset, repo_id=match.group(1)))
    return jobs


def _select(jobs: list[_Job], only: list[str] | None, limit: int | None) -> list[_Job]:
    if only:
        wanted = {o.lower() for o in only}
        jobs = [
            j
            for j in jobs
            if j.repo_id.lower() in wanted or j.asset.model_id.lower() in wanted
        ]
        found = {j.repo_id.lower() for j in jobs} | {
            j.asset.model_id.lower() for j in jobs
        }
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"Not in the source catalog: {', '.join(missing)}")
    return jobs[:limit] if limit else jobs


# ?? Download ??????????????????????????????????????????????????????


def _weight_files(root_files: list[str]) -> tuple[list[str], str]:
    """Pick the root-level weight files the loader will read."""
    safetensors = [
        f
        for f in root_files
        if f.endswith(".safetensors") or f == "model.safetensors.index.json"
    ]
    if any(f.endswith(".safetensors") for f in safetensors):
        return safetensors, "safetensors"
    bins = [
        f
        for f in root_files
        if (f.startswith("pytorch_model") and f.endswith(".bin"))
        or f == "pytorch_model.bin.index.json"
    ]
    return bins, "pytorch"


def _download(api: HfApi, repo_id: str) -> tuple[Path, str, str]:
    """Download a model at its current commit; return path, revision, format."""
    info = api.model_info(repo_id)
    revision = info.sha
    if not revision:
        raise RuntimeError(f"No commit hash reported for {repo_id}")
    root_files = [s.rfilename for s in info.siblings or [] if "/" not in s.rfilename]
    weights, fmt = _weight_files(root_files)
    if not weights:
        raise RuntimeError(f"No safetensors or pytorch_model.bin files in {repo_id}")
    meta = [f for f in root_files if f.endswith(_META_SUFFIXES)]
    path = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=sorted(set(weights) | set(meta)),
    )
    return Path(path), revision, fmt


def _delete_download(revision: str) -> None:
    """Remove a downloaded revision from the Hugging Face cache."""
    try:
        strategy = scan_cache_dir().delete_revisions(revision)
        strategy.execute()
        log.info(
            "download_deleted",
            revision=revision,
            freed=strategy.expected_freed_size_str,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("download_delete_failed", revision=revision, error=str(exc))


# ?? Bundle and parquet ????????????????????????????????????????????


def _available(ws: WeightSignalFeatures | None) -> list[str]:
    if ws is None:
        return []
    return [s for s in _SIGNALS if getattr(ws, s) is not None]


def _summary(ws: WeightSignalFeatures | None) -> SignalSummary:
    if ws is None:
        return SignalSummary()

    def _len(value: np.ndarray | None) -> int:
        return int(len(value)) if value is not None else 0

    wvc = ws.wvc_layer_sigs or {}
    first_wvc = next(iter(wvc.values()), None)
    return SignalSummary(
        hidden_size=ws.hidden_size,
        num_layers=ws.num_layers,
        eas_anchor_count=ws.eas_anchor_count,
        nlf_mode=ws.nlf_mode,
        nlf_num_layers=ws.nlf_num_layers,
        eas_self_sim_shape=(
            list(ws.eas_self_sim.shape) if ws.eas_self_sim is not None else []
        ),
        end_histogram_bins=_len(ws.end_histogram),
        nlf_vector_length=_len(ws.nlf_vector),
        lep_profile_length=_len(ws.lep_profile),
        wsp_signature_length=_len(ws.wsp_signature),
        wvc_layer_count=len(wvc),
        wvc_layer_vector_length=_len(first_wvc),
    )


def _long_form(ws: WeightSignalFeatures) -> pd.DataFrame:
    """Flatten weight signals into the long-form parquet schema."""
    frames: list[pd.DataFrame] = []

    def _add(signal: str, values: np.ndarray, *, layer=None, rows=None, cols=None):  # type: ignore[no-untyped-def]
        n = len(values)
        frames.append(
            pd.DataFrame(
                {
                    "signal": signal,
                    "layer": pd.array([layer] * n, dtype="Int32"),
                    "row": np.arange(n, dtype=np.int32) if rows is None else rows,
                    "col": pd.array(
                        [None] * n if cols is None else cols, dtype="Int32"
                    ),
                    "value": values.astype(np.float32),
                }
            )
        )

    if ws.eas_self_sim is not None:
        k = ws.eas_self_sim.shape[0]
        rows, cols = np.divmod(np.arange(k * k, dtype=np.int32), k)
        _add(
            "eas_self_sim",
            ws.eas_self_sim.reshape(-1),
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
        )
    for name in ("nlf_vector", "lep_profile", "end_histogram", "wsp_signature"):
        value = getattr(ws, name)
        if value is not None:
            _add(name, np.asarray(value).reshape(-1))
    for layer, value in sorted((ws.wvc_layer_sigs or {}).items()):
        _add("wvc_layer_sigs", np.asarray(value).reshape(-1), layer=int(layer))
    return pd.concat(frames, ignore_index=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_roundtrip(
    out: Path, job: _Job, bundle: FeatureBundle, ws: WeightSignalFeatures
) -> None:
    """Reload the written parquet and require self-similarity of 1.0."""
    db = DatabaseService(out)
    stored = db.reconstruct_weight_features(
        db.load_deep_signals(job.parquet_rel()), bundle
    )
    scores = {
        "eas": eas_similarity(ws, stored),
        "nlf": nlf_similarity(ws, stored),
        "lep": lep_similarity(ws, stored),
        "end": end_similarity(ws, stored),
        "wvc": wvc_similarity(ws, stored),
    }
    bad = {k: v for k, v in scores.items() if not np.isnan(v) and v < 0.9999}
    if bad:
        raise RuntimeError(f"Stored fingerprint does not reproduce itself: {bad}")


def _build_one(
    job: _Job,
    out: Path,
    settings: Settings,
    api: HfApi,
    *,
    trust_remote_code: bool,
    delete_download: bool,
) -> dict[str, Any]:
    t0 = time.monotonic()
    local, revision, fmt = _download(api, job.repo_id)
    try:
        scanner = ModelProvenanceScanner(
            settings=settings.model_copy(
                update={"trust_remote_code": trust_remote_code}
            ),
            cache=NullCache(),
        )
        data = scanner._extract_model(str(local))  # noqa: SLF001
    finally:
        if delete_download:
            _delete_download(revision)

    ws = data.ws
    available = _available(ws)
    missing = [s for s in _SIGNALS if s not in available]
    artifact_refs: list[ArtifactRef] = []
    if ws is not None and available:
        parquet = out / job.parquet_rel()
        parquet.parent.mkdir(parents=True, exist_ok=True)
        _long_form(ws).to_parquet(parquet, index=False)
        artifact_refs.append(
            ArtifactRef(
                artifact_id=f"{job.asset.model_id}-deep-signals",
                type="deep_signals",
                path=job.parquet_rel(),
                signals=available,
                checksum=f"sha256:{_sha256(parquet)}",
            )
        )

    bundle = FeatureBundle(
        bundle_id=str(uuid.uuid4()),
        model_id=job.asset.model_id,
        asset_id=job.asset.asset_id,
        family_id=job.family_id,
        source_id="build_database",
        status="ok" if not missing else "partial",
        updated_at=datetime.now(UTC).isoformat(),
        extractor_version=EXTRACTOR_VERSION,
        source_revision=revision,
        ws_loader=data.ws_loader,
        missing_signals=missing,
        mfi=data.fp.model_dump(),
        tfv=data.tfv.model_dump(),
        vocab={"size": data.tfv.vocab_size},
        signals=_summary(ws),
        artifact_refs=artifact_refs,
    )
    if ws is not None and artifact_refs:
        _verify_roundtrip(out, job, bundle, ws)

    bundle_path = out / job.bundle_rel()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = bundle_path.with_suffix(".tmp")
    tmp.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(bundle_path)

    inconsistent = (
        [c.model_dump() for c in (ws.config_inconsistencies or [])] if ws else []
    )
    return {
        "repo": job.repo_id,
        "asset_id": job.asset.asset_id,
        "revision": revision,
        "format": fmt,
        "param_bucket": compute_param_bucket(data.fp),
        "status": bundle.status,
        "missing_signals": missing,
        "loader": data.ws_loader,
        "config_inconsistencies": inconsistent,
        "seconds": round(time.monotonic() - t0, 1),
    }


def _is_built(out: Path, job: _Job) -> bool:
    path = out / job.bundle_rel()
    if not path.exists():
        return False
    try:
        bundle = FeatureBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return bundle.extractor_version == EXTRACTOR_VERSION


# ?? Catalog and archive ???????????????????????????????????????????


def _write_catalog(out: Path, jobs: list[_Job], settings: Settings) -> int:
    """Write shards and manifest for every asset that has a bundle in *out*."""
    now = datetime.now(UTC).isoformat()
    by_family: dict[str, list[_Job]] = {}
    for job in jobs:
        if _is_built(out, job):
            by_family.setdefault(job.family_id, []).append(job)

    refs: list[ShardRef] = []
    for family_id, family_jobs in sorted(by_family.items()):
        assets = []
        for job in family_jobs:
            bundle = FeatureBundle.model_validate_json(
                (out / job.bundle_rel()).read_text(encoding="utf-8")
            )
            fp_bucket = compute_param_bucket(_MFIView(bundle.mfi))
            assets.append(
                job.asset.model_copy(
                    update={"feature_path": job.bundle_rel(), "param_bucket": fp_bucket}
                )
            )
        shard = family_jobs[0].shard.model_copy(
            update={"updated_at": now, "assets": assets}
        )
        shard_rel = f"catalog/by-family/{family_id}.json"
        shard_path = out / shard_rel
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(
            shard.model_dump_json(
                indent=2, exclude={"assets": {"__all__": {"family_id"}}}
            ),
            encoding="utf-8",
        )
        refs.append(
            ShardRef(family_id=family_id, shard_path=shard_rel, asset_count=len(assets))
        )

    manifest = CatalogManifest(
        updated_at=now,
        extractor_version=EXTRACTOR_VERSION,
        extraction_params=settings.extraction_params(),
        shards=refs,
    )
    (out / "catalog" / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return sum(r.asset_count for r in refs)


class _MFIView:
    """Attribute view over a stored MFI dict for the bucket estimator."""

    def __init__(self, mfi: dict[str, Any]) -> None:
        self.__dict__.update(mfi)


def _write_zip(out: Path, zip_path: Path) -> str:
    """Archive ``features/deep-signals/by-family`` for Hugging Face."""
    root = out / "features" / "deep-signals"
    files = sorted((root / "by-family").rglob("*.parquet"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(root).as_posix())
    return _sha256(zip_path)


# ?? CLI ???????????????????????????????????????????????????????????


def main(argv: list[str] | None = None) -> int:
    """Run the database build."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-db", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="Repo ids or model ids to build")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Rebuild finished models")
    parser.add_argument("--zip", type=Path, help="Write the deep-signals archive here")
    parser.add_argument(
        "--delete-downloads",
        action="store_true",
        help="Remove each model from the Hugging Face cache after extraction",
    )
    parser.add_argument(
        "--trust-remote-code-for",
        nargs="+",
        default=[],
        metavar="REPO",
        help="Reviewed repos whose custom modelling code may run",
    )
    args = parser.parse_args(argv)

    if args.out.resolve() == args.source_db.resolve():
        raise SystemExit("--out must differ from --source-db")

    settings = Settings()
    jobs = _load_jobs(args.source_db)
    selected = _select(jobs, args.only, args.limit)
    trusted = {r.lower() for r in args.trust_remote_code_for}
    api = HfApi()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "build_report.jsonl"

    print(
        f"Extractor version {EXTRACTOR_VERSION}, settings "
        f"{settings.extraction_params()}; "
        f"building {len(selected)} of {len(jobs)} models"
    )
    failures = 0
    for i, job in enumerate(selected, 1):
        if not args.force and _is_built(args.out, job):
            print(f"[{i}/{len(selected)}] {job.repo_id}: already built")
            continue
        print(f"[{i}/{len(selected)}] {job.repo_id} ...", flush=True)
        try:
            row = _build_one(
                job,
                args.out,
                settings,
                api,
                trust_remote_code=job.repo_id.lower() in trusted,
                delete_download=args.delete_downloads,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            row = {
                "repo": job.repo_id,
                "asset_id": job.asset.asset_id,
                "error": str(exc),
            }
            print(f"    FAILED: {exc}", file=sys.stderr)
        else:
            print(
                f"    {row['status']}, revision {row['revision'][:12]}, "
                f"{row['loader']}, {row['seconds']}s"
            )
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    count = _write_catalog(args.out, jobs, settings)
    print(f"Catalog written: {count} models in {args.out}")
    if args.zip:
        digest = _write_zip(args.out, args.zip)
        print(f"Deep-signals archive: {args.zip}\n  sha256={digest}")
    if failures:
        print(f"{failures} model(s) failed; see {report_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
