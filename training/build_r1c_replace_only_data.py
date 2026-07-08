#!/usr/bin/env python3
"""Build R1c data from replace-only rollouts and gold repairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from build_sft_data import (  # noqa: E402
    assistant_action,
    compact_observation,
    failure_to_repair_messages,
    gold_messages,
    make_sample,
    system_prompt,
    user_prompt,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_clean_success(row: dict[str, Any], max_tool_calls: int) -> bool:
    stats = row.get("tool_stats", {})
    return (
        bool(row.get("final_verdict", {}).get("success"))
        and int(stats.get("invalid_calls", 0)) == 0
        and int(stats.get("repeated_calls", 0)) == 0
        and int(stats.get("tool_calls", 99)) <= max_tool_calls
    )


def canonical_success_messages(task: dict[str, Any], traj: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(task)},
    ]
    for step in traj.get("steps", []):
        if step["tool"] not in {"view_file", "replace_file", "run_test", "final"}:
            continue
        arguments = dict(step["arguments"])
        if step["tool"] == "run_test":
            arguments.pop("timeout_seconds", None)
        messages.append({"role": "assistant", "content": assistant_action(step["tool"], arguments)})
        messages.append({"role": "user", "content": "Observation: " + compact_observation(step["observation"], max_chars=1200)})
    messages.append({"role": "assistant", "content": assistant_action("final", {"answer": "done"})})
    return messages


def train_pass_by_family(rollouts: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    for row in rollouts:
        family = row.get("metadata", {}).get("bug_family", "unknown")
        totals[family] += 1
        successes[family] += int(bool(row.get("final_verdict", {}).get("success")))
    return {family: successes[family] / total for family, total in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agentic-rl/data/tasks/controlled_py_v1_300_train.jsonl")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", default="agentic-rl/data/rl/controlled_py_v1_r1c_replace_only_train.jsonl")
    parser.add_argument("--report", default="agentic-rl/results/metrics/controlled_py_v1_r1c_replace_only_data_report.json")
    parser.add_argument("--hard-family-threshold", type=float, default=0.60)
    parser.add_argument("--hard-gold-copies", type=int, default=2)
    parser.add_argument("--max-clean-tool-calls", type=int, default=4)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    rollouts = load_jsonl(Path(args.rollouts))
    task_by_id = {task["task_id"]: task for task in tasks}
    rollout_by_id = {row["task_id"]: row for row in rollouts}
    pass_by_family = train_pass_by_family(rollouts)
    hard_families = {
        family
        for family, pass_rate in pass_by_family.items()
        if pass_rate < args.hard_family_threshold
    }

    samples: list[dict[str, Any]] = []
    for task in tasks:
        samples.append(make_sample(task, gold_messages(task), "gold_repair", "gold_patch"))

    for row in rollouts:
        task = task_by_id.get(row["task_id"])
        if task and is_clean_success(row, args.max_clean_tool_calls):
            samples.append(
                make_sample(
                    task,
                    canonical_success_messages(task, row),
                    "replace_only_clean_success",
                    "sft_replace_only_rollout",
                )
            )

    for task in tasks:
        family = task["metadata"].get("bug_family")
        if family in hard_families:
            for _ in range(args.hard_gold_copies):
                samples.append(make_sample(task, gold_messages(task), "hard_family_gold_repair", "gold_patch_oversample"))

    for row in rollouts:
        task = task_by_id.get(row["task_id"])
        if not task:
            continue
        family = task["metadata"].get("bug_family")
        if family in hard_families and not row.get("final_verdict", {}).get("success"):
            samples.append(
                make_sample(
                    task,
                    failure_to_repair_messages(task, row),
                    "replace_only_failure_to_gold",
                    "replace_only_failure_plus_gold",
                )
            )

    write_jsonl(Path(args.out), samples)
    report = {
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "samples": len(samples),
        "sample_type_counts": dict(Counter(sample["sample_type"] for sample in samples)),
        "hard_families": sorted(hard_families),
        "train_pass_by_family": {k: round(v * 100, 2) for k, v in sorted(pass_by_family.items())},
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
