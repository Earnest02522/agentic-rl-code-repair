#!/usr/bin/env python3
"""Build reward-filtered RLVR/RFT data from sampled rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_sft_data import assistant_action, compact_observation, system_prompt, user_prompt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reward(row: dict[str, Any]) -> float:
    stats = row.get("tool_stats", {})
    hidden_success = 1.0 if row.get("final_verdict", {}).get("success") else 0.0
    public_success = 1.0 if row.get("public_verdict", {}).get("success") else 0.0
    patch_changed = 1.0 if (row.get("patch_diff") or "").strip() else 0.0
    invalid = float(stats.get("invalid_calls", 0))
    repeated = float(stats.get("repeated_calls", 0))
    tool_calls = float(stats.get("tool_calls", 0))
    tokens = float(stats.get("total_tokens", 0))
    hidden_access = 0.0
    for call in row.get("model_calls", []):
        if "hidden" in call.get("text", "").lower() and "test_hidden" in call.get("text", "").lower():
            hidden_access = 1.0
            break
    value = (
        2.0 * hidden_success
        + 0.3 * public_success
        + 0.2 * patch_changed
        - 0.25 * invalid
        - 0.15 * repeated
        - 0.05 * max(tool_calls - 3.0, 0.0)
        - 0.10 * hidden_access
        - 0.00002 * tokens
    )
    return max(-1.0, min(2.5, value))


def trajectory_messages(task: dict[str, Any], traj: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(task)},
    ]
    calls = traj.get("model_calls", [])
    steps = traj.get("steps", [])
    for idx, step in enumerate(steps):
        if idx < len(calls):
            messages.append({"role": "assistant", "content": calls[idx]["text"]})
        else:
            messages.append({"role": "assistant", "content": assistant_action(step["tool"], step["arguments"])})
        messages.append({"role": "user", "content": "Observation: " + compact_observation(step["observation"], max_chars=1200)})
    messages.append({"role": "assistant", "content": assistant_action("final", {"answer": "done"})})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agentic-rl/data/tasks/controlled_py_v1_300_train.jsonl")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--out", default="agentic-rl/data/rl/controlled_py_v1_rlvr_train.jsonl")
    parser.add_argument("--report", default="agentic-rl/results/metrics/controlled_py_v1_rlvr_data_report.json")
    parser.add_argument("--min-reward", type=float, default=1.6)
    parser.add_argument("--include-best-failures", type=int, default=80)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    task_by_id = {task["task_id"]: task for task in tasks}
    candidates: list[dict[str, Any]] = []
    for path in args.rollouts:
        for row in load_jsonl(Path(path)):
            if row["task_id"] in task_by_id:
                row["_reward"] = reward(row)
                candidates.append(row)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_task[row["task_id"]].append(row)

    selected: list[dict[str, Any]] = []
    for task_id, rows in by_task.items():
        rows = sorted(rows, key=lambda item: item["_reward"], reverse=True)
        best = rows[0]
        if best["_reward"] >= args.min_reward or best.get("final_verdict", {}).get("success"):
            selected.append(best)

    selected_ids = {row["task_id"] for row in selected}
    failure_pool = [
        rows[0]
        for task_id, rows in by_task.items()
        if task_id not in selected_ids
        for rows in [sorted(rows, key=lambda item: item["_reward"], reverse=True)]
    ]
    failure_pool = sorted(failure_pool, key=lambda item: item["_reward"], reverse=True)[: args.include_best_failures]
    selected.extend(failure_pool)

    samples = []
    for row in selected:
        task = task_by_id[row["task_id"]]
        samples.append(
            {
                "task_id": task["task_id"],
                "sample_type": "rlvr_reward_selected_success" if row.get("final_verdict", {}).get("success") else "rlvr_best_effort_failure",
                "source": "reward_filtered_sampled_rollout",
                "reward": row["_reward"],
                "messages": trajectory_messages(task, row),
                "metadata": {
                    "bug_family": task["metadata"].get("bug_family"),
                    "difficulty": task.get("difficulty"),
                    "split": task.get("split"),
                    "final_success": bool(row.get("final_verdict", {}).get("success")),
                    "public_success": bool(row.get("public_verdict", {}).get("success")),
                    "tool_stats": row.get("tool_stats", {}),
                },
            }
        )

    write_jsonl(Path(args.out), samples)
    report = {
        "tasks": len(tasks),
        "candidate_rollouts": len(candidates),
        "tasks_with_candidates": len(by_task),
        "samples": len(samples),
        "sample_type_counts": dict(Counter(sample["sample_type"] for sample in samples)),
        "avg_selected_reward": sum(sample["reward"] for sample in samples) / len(samples) if samples else 0.0,
        "max_reward": max((row["_reward"] for row in candidates), default=None),
        "min_reward": min((row["_reward"] for row in candidates), default=None),
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
