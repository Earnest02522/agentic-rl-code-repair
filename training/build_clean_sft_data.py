#!/usr/bin/env python3
"""Build clean SFT data that avoids imitating failed tool-use prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from build_r1c_replace_only_data import canonical_success_messages, is_clean_success  # noqa: E402
from build_sft_data import gold_messages, load_jsonl, make_sample, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--rollouts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gold-copies", type=int, default=1)
    parser.add_argument("--max-clean-tool-calls", type=int, default=4)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    rollouts = load_jsonl(Path(args.rollouts)) if args.rollouts else []
    task_by_id = {task["task_id"]: task for task in tasks}

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for _ in range(args.gold_copies):
            rows.append(make_sample(task, gold_messages(task), "gold_repair", "gold_patch_clean"))

    clean_success = 0
    skipped_success = 0
    for rollout in rollouts:
        task = task_by_id.get(rollout.get("task_id"))
        if not task:
            continue
        if is_clean_success(rollout, args.max_clean_tool_calls):
            clean_success += 1
            rows.append(
                make_sample(
                    task,
                    canonical_success_messages(task, rollout),
                    "clean_success_trajectory",
                    "base_or_policy_clean_success",
                )
            )
        elif rollout.get("final_verdict", {}).get("success"):
            skipped_success += 1

    write_jsonl(Path(args.out), rows)
    report = {
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "samples": len(rows),
        "sample_type_counts": dict(Counter(row["sample_type"] for row in rows)),
        "gold_copies": args.gold_copies,
        "clean_success_added": clean_success,
        "successful_but_not_clean_skipped": skipped_success,
        "max_clean_tool_calls": args.max_clean_tool_calls,
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
