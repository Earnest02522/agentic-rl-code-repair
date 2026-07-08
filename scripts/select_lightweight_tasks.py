#!/usr/bin/env python3
"""Select the lightest tasks for workspace materialization checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score(row: dict[str, Any]) -> float:
    command = row.get("test_command") or ""
    penalty = 0.0
    if "nvm use" in command:
        penalty += 10.0
    if any(token in command for token in ("npm ", "yarn ", "npx ")):
        penalty += 6.0
    if "xvfb" in command:
        penalty += 10.0
    if "pytest" in command:
        penalty += 2.0
    return row["changed_files"] * 20 + row["changed_lines"] + penalty + len(command) / 500.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="agentic-rl/data/tasks/code_repair_round0_specs_v0.jsonl")
    parser.add_argument("--output", default="agentic-rl/data/tasks/code_repair_round0_light5_v0.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    for row in rows:
        row["selection_score"] = score(row)
        row["selection_reason"] = (
            f"files={row['changed_files']};lines={row['changed_lines']};"
            f"language={row['language']};difficulty={row['difficulty']}"
        )
    selected = sorted(rows, key=lambda row: (row["selection_score"], row["language"], row["task_id"]))[: args.limit]
    write_jsonl(Path(args.output), selected)
    print(json.dumps({
        "output": args.output,
        "rows": len(selected),
        "tasks": [
            {
                "task_id": row["task_id"],
                "repo": row["repo"],
                "language": row["language"],
                "changed_files": row["changed_files"],
                "changed_lines": row["changed_lines"],
                "score": row["selection_score"],
            }
            for row in selected
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
