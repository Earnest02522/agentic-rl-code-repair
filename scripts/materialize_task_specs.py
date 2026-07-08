#!/usr/bin/env python3
"""Convert filtered candidates into round-specific code-repair task specs."""

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


def convert(row: dict[str, Any], split: str, workspace_root: str) -> dict[str, Any]:
    task_id = row["instance_id"].replace("__", "_").replace("/", "_")
    return {
        "task_id": task_id,
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "language": row["language"],
        "difficulty": row["difficulty"],
        "split": split,
        "source_dataset": row["source_dataset"],
        "base_commit": row["base_commit"],
        "problem_statement": row["problem_statement"],
        "allowed_files": row["changed_file_paths"],
        "patch": row["patch"],
        "test_patch": row.get("test_patch"),
        "test_command": row["test_command"],
        "Dockerfile": row.get("Dockerfile"),
        "changed_files": row["changed_files"],
        "changed_lines": row["changed_lines"],
        "workspace": str(Path(workspace_root) / task_id),
        "metadata": {
            "task_category": row["task_category"],
            "created_at": row.get("created_at"),
            "hints_text": row.get("hints_text"),
            "modified_nodes": row.get("modified_nodes"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="agentic-rl/data/tasks/code_repair_round0_v0.jsonl")
    parser.add_argument("--split", default="round0")
    parser.add_argument("--workspace-root", default="agentic-rl/workspaces")
    parser.add_argument("--output", default="agentic-rl/data/tasks/code_repair_round0_specs_v0.jsonl")
    args = parser.parse_args()

    rows = [convert(row, args.split, args.workspace_root) for row in load_jsonl(Path(args.input))]
    write_jsonl(Path(args.output), rows)
    summary = {
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "workspace_root": args.workspace_root,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
