#!/usr/bin/env python3
"""Build a filtered code-repair candidate pool from SWE-PolyBench CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUTS = [
    "agentic-rl/data/external/swe_polybench/verified/test.csv",
    "agentic-rl/data/external/swe_polybench/pb500/test.csv",
]


def patch_stats(patch: str) -> dict[str, Any]:
    files: set[str] = set()
    added = 0
    deleted = 0
    current_file = None

    for line in str(patch).splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[2][2:] if parts[2].startswith("a/") else parts[2]
                files.add(current_file)
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added += 1
            continue
        if line.startswith("-"):
            deleted += 1
            continue

    file_exts = sorted({Path(f).suffix.lower().lstrip(".") for f in files if Path(f).suffix})
    return {
        "changed_files": len(files),
        "changed_file_paths": sorted(files),
        "changed_file_exts": file_exts,
        "added_lines": added,
        "deleted_lines": deleted,
        "changed_lines": added + deleted,
    }


def difficulty(row: pd.Series) -> str:
    changed_files = int(row["changed_files"])
    changed_lines = int(row["changed_lines"])
    nodes = int(row.get("num_nodes") or 0)

    if changed_files <= 1 and changed_lines <= 40 and nodes <= 8:
        return "easy"
    if changed_files <= 3 and changed_lines <= 80 and nodes <= 20:
        return "medium"
    return "hard_light"


def normalize_record(row: pd.Series, source: str) -> dict[str, Any]:
    keep_columns = [
        "repo",
        "pull_number",
        "instance_id",
        "issue_numbers",
        "base_commit",
        "problem_statement",
        "hints_text",
        "created_at",
        "language",
        "Dockerfile",
        "test_command",
        "task_category",
        "is_func_only",
        "is_class_only",
        "is_mixed",
        "num_func_changes",
        "num_class_changes",
        "num_nodes",
        "is_single_func",
        "is_single_class",
        "modified_nodes",
    ]
    record = {col: row.get(col) for col in keep_columns}

    for key, value in list(record.items()):
        if pd.isna(value):
            record[key] = None

    record.update(
        {
            "source_dataset": source,
            "patch": row["patch"],
            "test_patch": row.get("test_patch"),
            "changed_files": int(row["changed_files"]),
            "changed_file_paths": row["changed_file_paths"],
            "changed_file_exts": row["changed_file_exts"],
            "added_lines": int(row["added_lines"]),
            "deleted_lines": int(row["deleted_lines"]),
            "changed_lines": int(row["changed_lines"]),
            "difficulty": row["difficulty"],
            "candidate_reason": row["candidate_reason"],
        }
    )
    return record


def load_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        csv_path = Path(path)
        source = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df["source_dataset"] = source
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["instance_id"], keep="first").reset_index(drop=True)

    stats = merged["patch"].map(patch_stats)
    stats_df = pd.DataFrame(stats.tolist())
    for col in stats_df.columns:
        merged[col] = stats_df[col]
    merged["difficulty"] = merged.apply(difficulty, axis=1)
    return merged


def filter_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mask = (
        df["language"].isin(args.languages)
        & df["task_category"].isin(args.categories)
        & df["changed_files"].between(args.min_changed_files, args.max_changed_files)
        & df["changed_lines"].between(args.min_changed_lines, args.max_changed_lines)
    )
    filtered = df[mask].copy()
    filtered["candidate_reason"] = (
        "language="
        + filtered["language"].astype(str)
        + ";category="
        + filtered["task_category"].astype(str)
        + ";files="
        + filtered["changed_files"].astype(str)
        + ";lines="
        + filtered["changed_lines"].astype(str)
    )
    filtered = filtered.sort_values(
        by=["difficulty", "language", "changed_files", "changed_lines", "instance_id"],
        kind="stable",
    ).reset_index(drop=True)
    return filtered


def summarize(df: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in frame[column].value_counts().sort_index().items()}

    return {
        "total_raw_rows_after_dedup": int(len(df)),
        "candidate_rows": int(len(candidates)),
        "raw_language_counts": counts(df, "language"),
        "candidate_language_counts": counts(candidates, "language"),
        "candidate_category_counts": counts(candidates, "task_category"),
        "candidate_difficulty_counts": counts(candidates, "difficulty"),
        "candidate_changed_files_quantiles": {
            str(k): float(v) for k, v in candidates["changed_files"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1]).items()
        },
        "candidate_changed_lines_quantiles": {
            str(k): float(v) for k, v in candidates["changed_lines"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1]).items()
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SWE-PolyBench Candidate Filter Report",
        "",
        "## Filter",
        "",
        f"- languages: {', '.join(args.languages)}",
        f"- categories: {', '.join(args.categories)}",
        f"- changed files: {args.min_changed_files}-{args.max_changed_files}",
        f"- changed lines: {args.min_changed_lines}-{args.max_changed_lines}",
        "",
        "## Summary",
        "",
        f"- raw rows after dedup: {summary['total_raw_rows_after_dedup']}",
        f"- candidate rows: {summary['candidate_rows']}",
        f"- candidate language counts: {summary['candidate_language_counts']}",
        f"- candidate category counts: {summary['candidate_category_counts']}",
        f"- candidate difficulty counts: {summary['candidate_difficulty_counts']}",
        f"- changed files quantiles: {summary['candidate_changed_files_quantiles']}",
        f"- changed lines quantiles: {summary['candidate_changed_lines_quantiles']}",
        "",
        "## Decision",
        "",
    ]
    if summary["candidate_rows"] >= 100:
        lines.append("The strict filter produces enough candidates for round-0 verifier and rollout work.")
    else:
        lines.append("The strict filter is too small; relax changed lines or include feature tasks before rollout.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default="agentic-rl/data/tasks/code_repair_candidates_v0.jsonl")
    parser.add_argument("--summary", default="agentic-rl/data/tasks/code_repair_candidates_v0.summary.json")
    parser.add_argument("--report", default="agentic-rl/reports/04_candidate_filter_report.md")
    parser.add_argument("--languages", nargs="+", default=["JavaScript", "TypeScript", "Python"])
    parser.add_argument("--categories", nargs="+", default=["Bug Fix"])
    parser.add_argument("--min-changed-files", type=int, default=1)
    parser.add_argument("--max-changed-files", type=int, default=3)
    parser.add_argument("--min-changed-lines", type=int, default=5)
    parser.add_argument("--max-changed-lines", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_inputs(args.inputs)
    candidates = filter_candidates(df, args)
    summary = summarize(df, candidates)
    rows = [
        normalize_record(row, str(row["source_dataset"]))
        for _, row in candidates.iterrows()
    ]

    write_jsonl(Path(args.output), rows)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(Path(args.report), summary, args)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
