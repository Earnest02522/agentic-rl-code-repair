#!/usr/bin/env python3
"""Build SFT distillation data from successful routed rollouts."""

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
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--include-gold", action="store_true")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--max-clean-tool-calls", type=int, default=5)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    rollouts = load_jsonl(Path(args.rollouts))
    task_by_id = {task["task_id"]: task for task in tasks}

    rows: list[dict[str, Any]] = []
    if args.include_gold:
        for task in tasks:
            rows.append(make_sample(task, gold_messages(task), "gold_repair", "gold_patch"))

    skipped = Counter()
    for rollout in rollouts:
        task = task_by_id.get(rollout.get("task_id"))
        if task is None:
            skipped["missing_task"] += 1
            continue
        if args.success_only and not rollout.get("final_verdict", {}).get("success"):
            skipped["not_success"] += 1
            continue
        if args.clean_only and not is_clean_success(rollout, args.max_clean_tool_calls):
            skipped["not_clean"] += 1
            continue
        rows.append(
            make_sample(
                task,
                canonical_success_messages(task, rollout),
                "routed_success_distill",
                rollout.get("routed_policy", "rollout"),
            )
        )

    write_jsonl(Path(args.out), rows)
    report = {
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "samples": len(rows),
        "sample_type_counts": dict(Counter(row["sample_type"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "skipped": dict(skipped),
        "include_gold": args.include_gold,
        "success_only": args.success_only,
        "clean_only": args.clean_only,
        "max_clean_tool_calls": args.max_clean_tool_calls,
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
