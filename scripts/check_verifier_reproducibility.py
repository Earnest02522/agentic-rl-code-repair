#!/usr/bin/env python3
"""Run verifier commands repeatedly for materialized tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "code_exec"))

from sandbox_runner import SandboxRunner  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agentic-rl/data/tasks/code_repair_round0_light5_materialized_v0.jsonl")
    parser.add_argument("--out", default="agentic-rl/results/verifier_repro_light5_v0.json")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    reports = []
    for task in load_jsonl(Path(args.tasks)):
        task_report = {"task_id": task["task_id"], "instance_id": task["instance_id"], "runs": []}
        workspace = task.get("workspace")
        if not workspace or not Path(workspace).exists():
            task_report["runs"].append({"success": False, "failure_summary": "workspace missing"})
            reports.append(task_report)
            continue
        runner = SandboxRunner(workspace, timeout_seconds=args.timeout)
        for _ in range(args.repeats):
            verdict = runner.verify(task["test_command"], timeout_seconds=args.timeout).to_dict()
            if runner.steps:
                # verify() does not append to steps, so this branch is reserved for future rollout use.
                pass
            task_report["runs"].append(verdict)
        reports.append(task_report)
        print(json.dumps({
            "task_id": task["task_id"],
            "successes": [run["success"] for run in task_report["runs"]],
        }, ensure_ascii=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stable = 0
    for report in reports:
        values = [run.get("success") for run in report["runs"]]
        if values and len(set(values)) == 1:
            stable += 1
    print(json.dumps({"total": len(reports), "stable": stable}, indent=2))


if __name__ == "__main__":
    main()
