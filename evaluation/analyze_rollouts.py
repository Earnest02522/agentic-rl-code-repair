#!/usr/bin/env python3
"""Analyze code-repair rollout trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float) -> float:
    return round(value * 100, 2)


def safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    final_success = [bool(row.get("final_verdict", {}).get("success")) for row in rows]
    public_success = [bool(row.get("public_verdict", {}).get("success")) for row in rows]
    stats = [row.get("tool_stats", {}) for row in rows]
    tool_calls = [float(item.get("tool_calls", 0)) for item in stats]
    invalid_calls = [float(item.get("invalid_calls", 0)) for item in stats]
    repeated_calls = [float(item.get("repeated_calls", 0)) for item in stats]
    total_tokens = [float(item.get("total_tokens", 0)) for item in stats]
    successes = sum(final_success)
    tool_name_counts = Counter()
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    by_difficulty: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})

    for row, ok in zip(rows, final_success):
        for name in row.get("tool_stats", {}).get("tool_names", []):
            tool_name_counts[name] += 1
        family = row.get("metadata", {}).get("bug_family", "unknown")
        difficulty = row.get("metadata", {}).get("difficulty", "unknown")
        by_family[family]["total"] += 1
        by_family[family]["success"] += int(ok)
        by_difficulty[difficulty]["total"] += 1
        by_difficulty[difficulty]["success"] += int(ok)

    return {
        "total": total,
        "final_success": successes,
        "pass_at_1": pct(successes / total) if total else 0.0,
        "public_pass_rate": pct(sum(public_success) / total) if total else 0.0,
        "avg_tool_calls": safe_mean(tool_calls),
        "avg_invalid_calls": safe_mean(invalid_calls),
        "invalid_call_rate_per_step": pct(sum(invalid_calls) / max(sum(tool_calls) + sum(invalid_calls), 1)),
        "avg_repeated_calls": safe_mean(repeated_calls),
        "repeated_call_rate_per_tool_step": pct(sum(repeated_calls) / max(sum(tool_calls), 1)),
        "avg_total_tokens": safe_mean(total_tokens),
        "cost_per_success_tokens": round(sum(total_tokens) / successes, 2) if successes else None,
        "tool_name_counts": dict(tool_name_counts),
        "by_family": {
            key: {
                "total": value["total"],
                "success": value["success"],
                "pass_at_1": pct(value["success"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_family.items())
        },
        "by_difficulty": {
            key: {
                "total": value["total"],
                "success": value["success"],
                "pass_at_1": pct(value["success"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_difficulty.items())
        },
    }


def write_markdown(path: Path, summary: dict[str, Any], source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Rollout Analysis

Source: `{source}`

| Metric | Value |
| --- | ---: |
| Total tasks | {summary['total']} |
| Final success | {summary['final_success']} |
| Pass@1 | {summary['pass_at_1']}% |
| Public pass rate | {summary['public_pass_rate']}% |
| Avg tool calls | {summary['avg_tool_calls']} |
| Avg invalid calls | {summary['avg_invalid_calls']} |
| Invalid call rate | {summary['invalid_call_rate_per_step']}% |
| Avg repeated calls | {summary['avg_repeated_calls']} |
| Repeated call rate | {summary['repeated_call_rate_per_tool_step']}% |
| Avg total tokens | {summary['avg_total_tokens']} |
| Cost per success tokens | {summary['cost_per_success_tokens']} |

## By Difficulty

```json
{json.dumps(summary['by_difficulty'], indent=2, ensure_ascii=False)}
```

## By Bug Family

```json
{json.dumps(summary['by_family'], indent=2, ensure_ascii=False)}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--md-out")
    args = parser.parse_args()

    source = Path(args.input)
    rows = load_jsonl(source)
    summary = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.md_out:
        write_markdown(Path(args.md_out), summary, source)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
