#!/usr/bin/env python3
"""Run SWE-PolyBench verifier commands inside task-specific Docker images."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "cmd": cmd,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "elapsed_seconds": time.time() - start,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "cmd": cmd,
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_seconds": time.time() - start,
        }


def image_name(task: dict[str, Any]) -> str:
    safe = task["task_id"].lower().replace("_", "-").replace(".", "-")
    return f"agentic-rl-{safe}:v0"


def prepare_dockerfile(task: dict[str, Any], workspace: Path) -> Path:
    dockerfile = workspace / "Dockerfile.agentic"
    content = task.get("metadata", {}).get("Dockerfile") or task.get("Dockerfile")
    if not content:
        raise ValueError(f"task has no Dockerfile: {task['task_id']}")
    dockerfile.write_text(content, encoding="utf-8")
    return dockerfile


def run_task(task: dict[str, Any], timeout: int) -> dict[str, Any]:
    workspace = Path(task["workspace"]).resolve()
    dockerfile = prepare_dockerfile(task, workspace)
    image = image_name(task)
    build = run(["docker", "build", "-f", dockerfile.name, "-t", image, "."], cwd=workspace, timeout=timeout)
    if not build["ok"]:
        return {"task_id": task["task_id"], "success": False, "stage": "build", "build": build}

    command = task.get("original_test_command") or task["test_command"]
    docker_run = run(["docker", "run", "--rm", image, "bash", "-lc", command], timeout=timeout)
    return {
        "task_id": task["task_id"],
        "success": docker_run["ok"],
        "stage": "run",
        "image": image,
        "build": build,
        "run": docker_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", default="agentic-rl/results/docker_verifier_v0.json")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))[: args.limit]
    reports = []
    for task in tasks:
        report = run_task(task, timeout=args.timeout)
        reports.append(report)
        print(json.dumps({"task_id": task["task_id"], "success": report["success"], "stage": report["stage"]}, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(reports), "success": sum(1 for r in reports if r["success"])}, indent=2))


if __name__ == "__main__":
    main()
