#!/usr/bin/env python3
"""Build conservative reward-filtered data for RLVR/RFT refinement."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from build_rlvr_data import reward, trajectory_messages  # noqa: E402
from build_sft_data import gold_messages, make_sample  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_success(row: dict[str, Any], min_reward: float) -> bool:
    stats = row.get("tool_stats", {})
    return (
        bool(row.get("final_verdict", {}).get("success"))
        and row.get("_reward", -1.0) >= min_reward
        and int(stats.get("invalid_calls", 0)) == 0
        and int(stats.get("repeated_calls", 0)) == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agentic-rl/data/tasks/controlled_py_v1_300_train.jsonl")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--out", default="agentic-rl/data/rl/controlled_py_v1_rlvr_r1b_train.jsonl")
    parser.add_argument("--report", default="agentic-rl/results/metrics/controlled_py_v1_rlvr_r1b_data_report.json")
    parser.add_argument("--min-reward", type=float, default=2.2)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    task_by_id = {task["task_id"]: task for task in tasks}

    samples: list[dict[str, Any]] = []
    for task in tasks:
        samples.append(make_sample(task, gold_messages(task), "gold_repair", "gold_patch"))

    selected_rollouts: list[dict[str, Any]] = []
    for path in args.rollouts:
        for row in load_jsonl(Path(path)):
            if row.get("task_id") not in task_by_id:
                continue
            row["_reward"] = reward(row)
            if clean_success(row, args.min_reward):
                selected_rollouts.append(row)

    selected_rollouts = sorted(selected_rollouts, key=lambda row: (-row["_reward"], row["task_id"]))
    seen: set[str] = set()
    for row in selected_rollouts:
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        task = task_by_id[row["task_id"]]
        sample = {
            "task_id": task["task_id"],
            "sample_type": "rlvr_clean_success",
            "source": "reward_filtered_clean_success",
            "reward": row["_reward"],
            "messages": trajectory_messages(task, row),
            "metadata": {
                "bug_family": task["metadata"].get("bug_family"),
                "difficulty": task.get("difficulty"),
                "split": task.get("split"),
                "final_success": True,
                "public_success": bool(row.get("public_verdict", {}).get("success")),
                "tool_stats": row.get("tool_stats", {}),
            },
        }
        samples.append(sample)

    write_jsonl(Path(args.out), samples)
    report = {
        "tasks": len(tasks),
        "samples": len(samples),
        "sample_type_counts": dict(Counter(sample["sample_type"] for sample in samples)),
        "clean_success_rollouts": len(seen),
        "min_reward": args.min_reward,
        "avg_clean_reward": (
            sum(sample.get("reward", 0.0) for sample in samples if sample["sample_type"] == "rlvr_clean_success") / len(seen)
            if seen
            else 0.0
        ),
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
