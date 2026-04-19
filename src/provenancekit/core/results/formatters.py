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

"""Output formatters for CompareResult and ScanResult.

Three rendering modes per result type:

* ``format_table`` / ``format_scan_table`` — Rich terminal table (default CLI).
* ``format_json``  / ``format_scan_json``  — Machine-readable JSON.
* ``format_plain`` / ``format_scan_plain`` — Colourless key-value text for piping.
"""

import json
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from provenancekit.core.scoring import interpret_score
from provenancekit.models.results import CompareResult, ScanModelInfo, ScanResult
from provenancekit.utils.tensor import estimate_param_count, format_param_count


def _fmt(value: float | None, precision: int = 4) -> str:
    """Format a float or None as a fixed-width string."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _format_params(info: ScanModelInfo) -> str:
    """Human-readable parameter count (exact from Hub or estimated)."""
    if info.num_parameters is not None:
        return format_param_count(info.num_parameters)
    return format_param_count(estimate_param_count(info), approximate=True)


def format_json(result: CompareResult) -> str:
    """Render *result* as indented JSON."""
    payload = result.model_dump()
    payload.pop("family_a", None)
    payload.pop("family_b", None)
    return json.dumps(payload, indent=2) + "\n"


def format_table(result: CompareResult, include_timing: bool = False) -> str:
    """Render *result* as a Rich terminal table."""
    s = result.scores
    sig = result.signals
    interp = result.interpretation

    table = Table(
        title=(f"Provenance Comparison: {result.model_a} vs {result.model_b}"),
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    decision_text = Text(s.provenance_decision)
    decision_text.stylize(f"bold {interp.colour}")
    table.add_row("Provenance Decision", decision_text)

    verdict = Text(
        f"{_fmt(s.pipeline_score)}  ({interp.label})",
    )
    verdict.stylize(f"bold {interp.colour}")
    table.add_row("Final Pipeline Score", verdict)
    table.add_row(
        "Metadata Feature Identifier",
        f"{_fmt(s.mfi_score)}  (Tier {s.mfi_tier} · {s.mfi_match})",
    )
    table.add_row("Weight Identity Score", _fmt(s.identity_score))
    table.add_row("Tokenizer Score", _fmt(s.tokenizer_score))

    table.add_section()
    table.add_row(Text("Weight Feature Scores", style="bold underline"), "")
    for label, val in [
        ("EAS", sig.eas),
        ("NLF", sig.nlf),
        ("LEP", sig.lep),
        ("END", sig.end),
        ("WVC", sig.wvc),
    ]:
        table.add_row(label, _fmt(val))

    table.add_section()
    table.add_row(Text("Tokenizer Feature Scores", style="bold underline"), "")
    for label, val in [
        ("TFV", sig.tfv),
        ("VOA", sig.voa),
    ]:
        table.add_row(label, _fmt(val))

    table.add_section()
    if include_timing and result.timing is not None:
        table.add_row(Text("Time Taken Breakdown", style="bold underline"), "")
        table.add_row("Total Time", f"{result.timing.total_seconds:.1f}s")
        table.add_row(
            "Model Metadata Extract Time",
            f"{result.timing.metadata_extract_seconds:.1f}s",
        )
        table.add_row(
            "Model Weight Feature Extract Time",
            f"{result.timing.weight_feature_extract_seconds:.1f}s",
        )
        table.add_row("Cache Hit", str(result.timing.cache_hit))
    else:
        table.add_row("Time", f"{result.time_seconds:.1f}s")

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100)
    console.print(table)
    return buf.getvalue()


def format_plain(result: CompareResult, include_timing: bool = False) -> str:
    """Render *result* as colourless key-value text."""
    s = result.scores
    sig = result.signals
    interp = result.interpretation

    lines = [
        f"model_a:         {result.model_a}",
        f"model_b:         {result.model_b}",
        f"pipeline_score:  {_fmt(s.pipeline_score)}",
        f"verdict:         {interp.label}",
        f"mfi_score:       {_fmt(s.mfi_score)}",
        f"mfi_tier:        {s.mfi_tier}",
        f"mfi_match:       {s.mfi_match}",
        f"identity_score:  {_fmt(s.identity_score)}",
        f"tokenizer_score: {_fmt(s.tokenizer_score)}",
        f"eas:             {_fmt(sig.eas)}",
        f"nlf:             {_fmt(sig.nlf)}",
        f"lep:             {_fmt(sig.lep)}",
        f"end:             {_fmt(sig.end)}",
        f"wvc:             {_fmt(sig.wvc)}",
        f"tfv:             {_fmt(sig.tfv)}",
        f"voa:             {_fmt(sig.voa)}",
        f"provenance_decision: {s.provenance_decision}",
        f"time_seconds:    {result.time_seconds:.1f}",
    ]
    if include_timing and result.timing is not None:
        lines.extend(
            [
                f"total_time_s:                 {result.timing.total_seconds:.1f}",
                f"metadata_extract_time_s:       "
                f"{result.timing.metadata_extract_seconds:.1f}",
                f"weight_feature_extract_time_s: "
                f"{result.timing.weight_feature_extract_seconds:.1f}",
                f"cache_hit:                    {result.timing.cache_hit}",
            ],
        )
    return "\n".join(lines) + "\n"


# ── Scan formatters ──────────────────────────────────────────────


def _decision_colour(pipeline_score: float | None) -> str:
    """Map a pipeline score to a Rich colour string."""
    if pipeline_score is None:
        return "#999999"
    return interpret_score(pipeline_score).colour


def format_scan_json(result: ScanResult) -> str:
    """Render a scan result as indented JSON."""
    return json.dumps(result.model_dump(), indent=2) + "\n"


def format_scan_table(
    result: ScanResult,
    *,
    include_timing: bool = False,
) -> str:
    """Render a scan result as a Rich panel with model info and match cards."""
    info = result.model_info

    table = Table(
        show_header=True,
        header_style="bold",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    # ── Query model info ──
    table.add_row("Scanned Model", info.model_path)
    param_str = _format_params(info)
    table.add_row("Parameters", f"{param_str}  ({info.param_bucket} bucket)")
    table.add_row("Layers", str(info.num_hidden_layers))
    table.add_row(
        "Weight signals",
        "yes" if info.has_weight_signals else "no",
    )

    # ── Matches ──
    table.add_section()
    if not result.matches:
        table.add_row(
            Text("Matches", style="bold underline"),
            "None above threshold",
        )
    else:
        table.add_row(
            Text(
                f"Top {result.match_count} Provenance Match(es)",
                style="bold underline",
            ),
            "",
        )
        for rank, m in enumerate(result.matches, 1):
            s = m.scores
            colour = _decision_colour(s.pipeline_score)

            decision_text = Text(m.provenance_decision)
            decision_text.stylize(f"bold {colour}")

            pipeline_text = Text(_fmt(s.pipeline_score))
            pipeline_text.stylize(f"bold {colour}")

            if rank > 1:
                table.add_section()

            table.add_row(f"  #{rank} Model", m.model_id)
            table.add_row(
                "  Family",
                f"{m.family_name} ({m.family_id})"
                + (f"  [{m.param_bucket}]" if m.param_bucket else ""),
            )
            table.add_row("  Provenance Decision", decision_text)
            table.add_row("  Pipeline Score", pipeline_text)
            table.add_row(
                "  MFI Score",
                f"{s.mfi_score:.4f}  (Tier {s.mfi_tier} - {s.mfi_match_type})",
            )
            table.add_row(
                "  Weight Score",
                f"{_fmt(s.identity_score)}  "
                f"(EAS={_fmt(s.eas, 2)} NLF={_fmt(s.nlf, 2)}"
                f" LEP={_fmt(s.lep, 2)} END={_fmt(s.end, 2)}"
                f" WVC={_fmt(s.wvc, 2)})",
            )
            table.add_row("  Tokenizer Score (TFV only)", _fmt(s.tokenizer_score))

    # ── Metadata-only hint ──
    has_mfi_only = any(m.match_type == "mfi_only" for m in result.matches)
    if has_mfi_only:
        table.add_section()
        table.add_row(
            Text("Note", style="bold yellow"),
            Text(
                "Some matches are based on metadata (MFI) only. "
                "Install deep-signal fingerprints for weight-level scoring:\n"
                "  provenancekit download-deepsignals-fingerprint",
                style="yellow",
            ),
        )

    # ── Footer ──
    table.add_section()
    table.add_row("Time", f"{result.elapsed_ms:.0f} ms")
    if include_timing:
        table.add_row("  Feature Extraction", f"{result.extract_seconds:.1f}s")
        table.add_row("  DB Lookup", f"{result.lookup_seconds:.1f}s")

    panel = Panel(
        table,
        title="[bold]Provenance Scan[/bold]",
        border_style="bright_blue",
        padding=(0, 1),
    )

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=92)
    console.print(panel)
    return buf.getvalue()


def format_scan_plain(
    result: ScanResult,
    *,
    include_timing: bool = False,
) -> str:
    """Render a scan result as colourless key-value text."""
    info = result.model_info
    lines = [
        f"model:            {info.model_path}",
        f"model_type:       {info.model_type}",
        f"architectures:    {', '.join(info.architectures)}",
        f"hidden_size:      {info.hidden_size}",
        f"num_hidden_layers: {info.num_hidden_layers}",
        f"vocab_size:       {info.vocab_size}",
        f"param_bucket:     {info.param_bucket}",
        f"arch_hash:        {info.arch_hash}",
        f"family_hash:      {info.family_hash}",
        f"weight_signals:   {'yes' if info.has_weight_signals else 'no'}",
        f"match_count:      {result.match_count}",
        f"elapsed_ms:       {result.elapsed_ms:.0f}",
    ]
    if include_timing:
        lines.append(f"extract_seconds:  {result.extract_seconds:.1f}")
        lines.append(f"lookup_seconds:   {result.lookup_seconds:.1f}")

    for rank, m in enumerate(result.matches, 1):
        s = m.scores
        lines.append("---")
        lines.append(f"match_{rank}_model_id:       {m.model_id}")
        lines.append(f"match_{rank}_family:         {m.family_name} ({m.family_id})")
        lines.append(f"match_{rank}_provenance_decision: {m.provenance_decision}")
        lines.append(f"match_{rank}_match_type:     {m.match_type}")
        lines.append(f"match_{rank}_pipeline_score: {_fmt(s.pipeline_score)}")
        lines.append(f"match_{rank}_mfi_score:      {s.mfi_score:.4f}")
        lines.append(f"match_{rank}_mfi_tier:       {s.mfi_tier}")
        lines.append(f"match_{rank}_identity_score: {_fmt(s.identity_score)}")
        lines.append(f"match_{rank}_tokenizer_score: {_fmt(s.tokenizer_score)}")
        lines.append(f"match_{rank}_eas:            {_fmt(s.eas)}")
        lines.append(f"match_{rank}_nlf:            {_fmt(s.nlf)}")
        lines.append(f"match_{rank}_lep:            {_fmt(s.lep)}")
        lines.append(f"match_{rank}_end:            {_fmt(s.end)}")
        lines.append(f"match_{rank}_wvc:            {_fmt(s.wvc)}")
        lines.append(f"match_{rank}_tfv:            {_fmt(s.tfv)}")

    has_mfi_only = any(m.match_type == "mfi_only" for m in result.matches)
    if has_mfi_only:
        lines.append("---")
        lines.append(
            "note: some matches are metadata-only (no deep-signal fingerprints). "
            "Run: provenancekit download-deepsignals-fingerprint"
        )

    return "\n".join(lines) + "\n"
