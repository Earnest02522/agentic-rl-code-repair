#!/usr/bin/env python3
"""Build a family-targeted routed rollout file from existing policy rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_override(raw: str) -> tuple[str, set[str], Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError("override must be formatted as name:family1,family2:path")
    name, families_raw, path_raw = parts
    families = {item.strip() for item in families_raw.split(",") if item.strip()}
    if not name or not families:
        raise ValueError("override requires a non-empty name and at least one family")
    return name, families, Path(path_raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-name", required=True)
    parser.add_argument("--default-rollout", required=True)
    parser.add_argument("--override", action="append", default=[], help="name:family1,family2:path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    default_rows = load_jsonl(Path(args.default_rollout))
    routed_by_task = {row["task_id"]: (args.default_name, row) for row in default_rows}

    family_to_policy: dict[str, str] = {}
    override_rows_by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in args.override:
        name, families, path = parse_override(raw)
        rows = {row["task_id"]: row for row in load_jsonl(path)}
        override_rows_by_policy[name] = rows
        for family in families:
            if family in family_to_policy:
                raise ValueError(f"family {family!r} is overridden more than once")
            family_to_policy[family] = name

    routed_rows: list[dict[str, Any]] = []
    missing_overrides: list[dict[str, str]] = []
    for default_row in default_rows:
        task_id = default_row["task_id"]
        family = default_row.get("metadata", {}).get("bug_family", "unknown")
        policy = family_to_policy.get(family)
        if policy is None:
            chosen_policy, chosen_row = routed_by_task[task_id]
        else:
            chosen_row = override_rows_by_policy[policy].get(task_id)
            if chosen_row is None:
                missing_overrides.append({"task_id": task_id, "family": family, "policy": policy})
                chosen_policy, chosen_row = routed_by_task[task_id]
            else:
                chosen_policy = policy
        chosen_row = dict(chosen_row)
        chosen_row["routed_policy"] = chosen_policy
        routed_rows.append(chosen_row)

    write_jsonl(Path(args.out), routed_rows)

    policy_counts = Counter(row["routed_policy"] for row in routed_rows)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_success: dict[str, Counter[str]] = defaultdict(Counter)
    for row in routed_rows:
        family = row.get("metadata", {}).get("bug_family", "unknown")
        policy = row["routed_policy"]
        family_counts[family][policy] += 1
        family_success[family][policy] += int(bool(row.get("final_verdict", {}).get("success")))

    report = {
        "default_policy": args.default_name,
        "default_rollout": args.default_rollout,
        "family_to_policy": dict(sorted(family_to_policy.items())),
        "policy_counts": dict(policy_counts),
        "missing_overrides": missing_overrides,
        "by_family": {
            family: {
                policy: {
                    "total": total,
                    "success": int(family_success[family][policy]),
                }
                for policy, total in sorted(counts.items())
            }
            for family, counts in sorted(family_counts.items())
        },
        "output": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
