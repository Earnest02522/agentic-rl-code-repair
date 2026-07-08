#!/usr/bin/env python3
"""Create reproducible train/val/test splits for code-repair candidates."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["difficulty"])].append(row)

    split_rows = {"train": [], "val": [], "test": []}
    for _, group in sorted(groups.items()):
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        split_rows["test"].extend(group[:n_test])
        split_rows["val"].extend(group[n_test : n_test + n_val])
        split_rows["train"].extend(group[n_test + n_val :])

    for split in split_rows:
        split_rows[split].sort(key=lambda r: (r["language"], r["difficulty"], r["changed_lines"], r["instance_id"]))
    return split_rows


def make_round0(train_rows: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 17)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        groups[(row["language"], row["difficulty"])].append(row)
    for group in groups.values():
        rng.shuffle(group)

    desired = [
        ("JavaScript", "easy"),
        ("TypeScript", "easy"),
        ("Python", "easy"),
        ("JavaScript", "medium"),
        ("TypeScript", "medium"),
        ("Python", "medium"),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    while len(selected) < target_size:
        progressed = False
        for key in desired:
            group = groups.get(key, [])
            while group and group[-1]["instance_id"] in seen:
                group.pop()
            if group and len(selected) < target_size:
                row = group.pop()
                selected.append(row)
                seen.add(row["instance_id"])
                progressed = True
        if not progressed:
            break

    selected.sort(key=lambda r: (r["language"], r["difficulty"], r["changed_lines"], r["instance_id"]))
    return selected


def summarize(rows_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, rows in rows_by_name.items():
        lang_counts: dict[str, int] = defaultdict(int)
        diff_counts: dict[str, int] = defaultdict(int)
        repo_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            lang_counts[row["language"]] += 1
            diff_counts[row["difficulty"]] += 1
            repo_counts[row["repo"]] += 1
        summary[name] = {
            "rows": len(rows),
            "language_counts": dict(sorted(lang_counts.items())),
            "difficulty_counts": dict(sorted(diff_counts.items())),
            "top_repos": sorted(repo_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
        }
    return summary


def write_report(path: Path, summary: dict[str, Any], seed: int) -> None:
    lines = [
        "# Code Repair Split Report",
        "",
        f"- seed: `{seed}`",
        "",
    ]
    for name, stats in summary.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- rows: {stats['rows']}",
                f"- language counts: {stats['language_counts']}",
                f"- difficulty counts: {stats['difficulty_counts']}",
                f"- top repos: {stats['top_repos']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="agentic-rl/data/tasks/code_repair_candidates_v0.jsonl")
    parser.add_argument("--output-dir", default="agentic-rl/data/tasks")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--round0-size", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    splits = stratified_split(rows, seed=args.seed)
    round0 = make_round0(splits["train"], target_size=args.round0_size, seed=args.seed)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "code_repair_train_v0.jsonl", splits["train"])
    write_jsonl(out_dir / "code_repair_val_v0.jsonl", splits["val"])
    write_jsonl(out_dir / "code_repair_test_v0.jsonl", splits["test"])
    write_jsonl(out_dir / "code_repair_round0_v0.jsonl", round0)

    summary = summarize({**splits, "round0": round0})
    (out_dir / "code_repair_split_v0.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(Path("agentic-rl/reports/05_split_report.md"), summary, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
