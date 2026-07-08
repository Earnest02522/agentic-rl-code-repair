#!/usr/bin/env python3
"""Run deterministic verifier commands for materialized local tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_EXEC = ROOT / "environments" / "code_exec"
sys.path.insert(0, str(CODE_EXEC))

from sandbox_runner import SandboxRunner  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    rows = []
    for task in load_jsonl(Path(args.tasks)):
        workspace = task.get("workspace")
        if not workspace:
            verdict = {
                "success": False,
                "reward": 0.0,
                "checks": {"workspace": False},
                "failure_summary": "task has no materialized workspace",
            }
        else:
            runner = SandboxRunner(workspace, timeout_seconds=args.timeout)
            verdict = runner.verify(task["test_command"], timeout_seconds=args.timeout).to_dict()
        rows.append({"task_id": task["task_id"], "instance_id": task["instance_id"], "verdict": verdict})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = len(rows)
    success = sum(1 for row in rows if row["verdict"]["success"])
    print(json.dumps({"total": total, "success": success, "success_rate": success / total if total else 0.0}, indent=2))


if __name__ == "__main__":
    main()
