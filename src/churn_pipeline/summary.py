"""Render the ingestion summary as JSON (for machines) and Markdown (for people)."""
from __future__ import annotations

import json
from pathlib import Path

from .synthetic import NOTICE


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _kb(n: int) -> str:
    return f"{n / 1024:,.1f} KiB"


def render_markdown(s: dict) -> str:
    lines = [
        "# Ingestion summary",
        "",
        f"- Run id: `{s['run_id']}`",
        f"- Finished: {s['finished_at']}",
        "",
        "## Sources",
        "",
    ]
    rows = []
    for name, src in s["sources"].items():
        rows.append([name, src["file"], src["records_read"], src["valid"], src["rejected"],
                     f"`{src['sha256'][:12]}`"])
    lines.append(_table(["source", "file", "read", "valid", "rejected", "sha256"], rows))
    lines += ["", f"> Sources B and C are synthetic. {NOTICE}", ""]

    lines += ["## Rejected records by reason (full-refresh sources, this run)", ""]
    rej = [[n, reason, c] for n, src in s["sources"].items()
           for reason, c in src["reject_reasons"].items()]
    lines.append(_table(["source", "reason", "count"], rej) if rej else "_None._")

    ev = s["events"]
    lines += ["", "## Event ingestion (incremental)", ""]
    lines.append(_table(
        ["file", "status", "valid", "inserted", "duplicates", "conflicts", "rejected"],
        [[f["file_name"], f["status"], f["valid_records"], f["inserted"], f["duplicates"],
          f["conflicts"], f["rejected"]] for f in ev["files"]],
    ))
    lines += [
        "",
        f"- New events inserted this run: **{ev['inserted_this_run']}**",
        f"- Events in store (all runs): **{ev['store_events']}** "
        f"from {ev['store_files']} ingested files",
        "",
        "Event rejects in store, by reason (all runs):",
        "",
        _table(["reason", "count"], [[r, c] for r, c in ev["store_reject_reasons"].items()])
        if ev["store_reject_reasons"] else "_None._",
    ]

    j = s["joins"]
    lines += [
        "",
        "## Joins (spine = IBM customers)",
        "",
        _table(["check", "count"], [
            ["IBM customers (spine)", j["customers"]],
            ["with profile", j["customers_with_profile"]],
            ["without profile (kept, profile fields null)", j["customers_without_profile"]],
            ["orphan profiles (not in IBM data, excluded)", j["orphan_profiles"]],
            ["with >= 1 event", j["customers_with_events"]],
            ["without events (counts = 0)", j["customers_without_events"]],
            ["orphan events (stored, excluded from features)", j["orphan_events"]],
            ["customer master rows", j["master_rows"]],
        ]),
        "",
        f"Row-count invariant (master rows == IBM customers): "
        f"**{'PASS' if j['master_rows'] == j['customers'] else 'FAIL'}**",
    ]

    lines += ["", "## Outputs", ""]
    lines.append(_table(["output", "rows", "size", "path"],
                        [[k, o["rows"], _kb(o["bytes"]), o["path"]]
                         for k, o in s["outputs"].items()]))

    if s["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in s["warnings"]]
    return "\n".join(lines) + "\n"


def write_summary(s: dict, md_path: Path, json_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(s, indent=2, default=str))
    md_path.write_text(render_markdown(s))
