#!/usr/bin/env python3
"""Build SFT chat data from base trajectories and gold repairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def system_prompt() -> str:
    return """You are a code repair agent in a constrained local repository.

You may call exactly one tool per turn by emitting one JSON object:

{"tool":"view_file","arguments":{"path":"app/example.py","start_line":1,"max_lines":120}}
{"tool":"replace_file","arguments":{"path":"app/example.py","content":"complete file content here"}}
{"tool":"run_test","arguments":{"command":"PYTHONPATH=. python -m pytest -q tests"}}
{"tool":"final","arguments":{"answer":"done"}}

Rules:
- Use only the allowed files.
- Start by viewing the allowed file.
- For these small tasks, prefer replace_file with the complete corrected file.
- Run public tests after editing.
- Do not try to inspect hidden tests.
"""


def user_prompt(task: dict[str, Any]) -> str:
    return f"""Repair this Python repository.

Task id: {task['task_id']}
Problem: {task['problem_statement']}
Allowed files: {', '.join(task['allowed_files'])}
Public test command: {task['metadata'].get('public_test_command', task['test_command'])}

Find the bug, patch the allowed file, run the public tests, then finish.
"""


def compact_observation(obs: dict[str, Any], max_chars: int = 1800) -> str:
    return json.dumps(
        {
            "ok": obs.get("ok"),
            "summary": obs.get("summary"),
            "exit_code": obs.get("exit_code"),
            "stdout_tail": (obs.get("stdout") or "")[-max_chars:],
            "stderr_tail": (obs.get("stderr") or "")[-max_chars:],
        },
        ensure_ascii=False,
    )


def assistant_action(tool: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False)


def gold_fixed_content(task: dict[str, Any]) -> str:
    added_lines: list[str] = []
    for line in task["patch"].splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
    if not added_lines:
        raise ValueError(f"could not extract fixed content for {task['task_id']}")
    return "\n".join(added_lines).rstrip() + "\n"


def gold_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    path = task["allowed_files"][0]
    public_cmd = task["metadata"].get("public_test_command", task["test_command"])
    fixed = gold_fixed_content(task)
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(task)},
        {"role": "assistant", "content": assistant_action("view_file", {"path": path, "start_line": 1, "max_lines": 120})},
        {"role": "user", "content": "Observation: inspect the current buggy implementation in the allowed file."},
        {"role": "assistant", "content": assistant_action("replace_file", {"path": path, "content": fixed})},
        {"role": "user", "content": "Observation: file replaced."},
        {"role": "assistant", "content": assistant_action("run_test", {"command": public_cmd})},
        {"role": "user", "content": "Observation: public tests passed."},
        {"role": "assistant", "content": assistant_action("final", {"answer": "done"})},
    ]


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
        messages.append({"role": "user", "content": "Observation: " + compact_observation(step["observation"])})
    messages.append({"role": "assistant", "content": assistant_action("final", {"answer": "done"})})
    return messages


def failure_to_repair_messages(task: dict[str, Any], traj: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(task)},
    ]
    calls = traj.get("model_calls", [])
    steps = traj.get("steps", [])
    included = 0
    for idx, step in enumerate(steps):
        messages.append({"role": "assistant", "content": calls[idx]["text"] if idx < len(calls) else assistant_action(step["tool"], step["arguments"])})
        messages.append({"role": "user", "content": "Observation: " + compact_observation(step["observation"], max_chars=900)})
        included += 1
        if included >= 3:
            break
    path = task["allowed_files"][0]
    public_cmd = task["metadata"].get("public_test_command", task["test_command"])
    fixed = gold_fixed_content(task)
    messages.append({"role": "user", "content": "The previous attempt did not pass. Use the observations and produce a corrected repair now."})
    messages.append({"role": "assistant", "content": assistant_action("replace_file", {"path": path, "content": fixed})})
    messages.append({"role": "user", "content": "Observation: file replaced."})
    messages.append({"role": "assistant", "content": assistant_action("run_test", {"command": public_cmd})})
    messages.append({"role": "user", "content": "Observation: public tests passed."})
    messages.append({"role": "assistant", "content": assistant_action("final", {"answer": "done"})})
    return messages


def make_sample(task: dict[str, Any], messages: list[dict[str, str]], sample_type: str, source: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "sample_type": sample_type,
        "source": source,
        "messages": messages,
        "metadata": {
            "bug_family": task["metadata"].get("bug_family"),
            "difficulty": task.get("difficulty"),
            "split": task.get("split"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agentic-rl/data/tasks/controlled_py_v1_300_train.jsonl")
    parser.add_argument("--rollouts", default="agentic-rl/results/rollouts/base_qwen_train/base_qwen_train.jsonl")
    parser.add_argument("--out", default="agentic-rl/data/sft/controlled_py_v1_sft_train.jsonl")
    parser.add_argument("--report", default="agentic-rl/results/metrics/controlled_py_v1_sft_data_report.json")
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    rollouts = load_jsonl(Path(args.rollouts))
    task_by_id = {task["task_id"]: task for task in tasks}
    rollout_by_id = {row["task_id"]: row for row in rollouts}
    rows: list[dict[str, Any]] = []

    for task in tasks:
        traj = rollout_by_id.get(task["task_id"])
        rows.append(make_sample(task, gold_messages(task), "gold_repair", "gold_patch"))
        if traj and traj.get("final_verdict", {}).get("success"):
            rows.append(make_sample(task, trajectory_messages(task, traj), "successful_base_trajectory", "base_rollout"))
        elif traj:
            rows.append(make_sample(task, failure_to_repair_messages(task, traj), "failure_to_repair", "base_failure_plus_gold"))

    write_jsonl(Path(args.out), rows)
    counts = Counter(row["sample_type"] for row in rows)
    report = {
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "samples": len(rows),
        "sample_type_counts": dict(counts),
        "missing_rollouts": sorted(set(task_by_id) - set(rollout_by_id)),
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
