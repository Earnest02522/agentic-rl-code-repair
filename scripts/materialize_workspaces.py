#!/usr/bin/env python3
"""Materialize real git workspaces for selected code-repair tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> dict[str, Any]:
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
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
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


def repo_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def materialize_task(task: dict[str, Any], cache_root: Path, workspace_root: Path, force: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_slug = task["repo"].replace("/", "__")
    cache_dir = cache_root / repo_slug
    workspace = workspace_root / task["task_id"]
    logs: list[dict[str, Any]] = []

    if not cache_dir.exists():
        clone = run(["git", "clone", "--no-tags", repo_url(task["repo"]), str(cache_dir)], timeout=1800)
        logs.append({"step": "clone_cache", **clone})
        if not clone["ok"]:
            return task, {"success": False, "task_id": task["task_id"], "logs": logs}
    else:
        fetch = run(["git", "fetch", "--all", "--prune"], cwd=cache_dir, timeout=600)
        logs.append({"step": "fetch_cache", **fetch})

    if workspace.exists() and force:
        shutil.rmtree(workspace)
    if not workspace.exists():
        clone_local = run(["git", "clone", "--shared", str(cache_dir), str(workspace)], timeout=600)
        logs.append({"step": "clone_workspace", **clone_local})
        if not clone_local["ok"]:
            return task, {"success": False, "task_id": task["task_id"], "logs": logs}

    checkout = run(["git", "checkout", "--force", task["base_commit"]], cwd=workspace, timeout=600)
    logs.append({"step": "checkout_base", **checkout})
    clean = run(["git", "clean", "-fdx"], cwd=workspace, timeout=600)
    logs.append({"step": "clean_workspace", **clean})

    test_patch_result = {"ok": True, "summary": "no test_patch"}
    if task.get("test_patch"):
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=task["test_patch"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        test_patch_result = {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    logs.append({"step": "apply_test_patch", **test_patch_result})

    task = dict(task)
    task["workspace"] = str(workspace.resolve())
    task["original_test_command"] = task["test_command"]
    task["test_command"] = task["test_command"].replace("/testbed", str(workspace.resolve()))
    success = bool(checkout["ok"] and clean["ok"] and test_patch_result["ok"])
    return task, {"success": success, "task_id": task["task_id"], "workspace": str(workspace), "logs": logs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agentic-rl/data/tasks/code_repair_round0_light5_v0.jsonl")
    parser.add_argument("--output", default="agentic-rl/data/tasks/code_repair_round0_light5_materialized_v0.jsonl")
    parser.add_argument("--report", default="agentic-rl/results/materialization_light5_v0.json")
    parser.add_argument("--cache-root", default="agentic-rl/external/repos")
    parser.add_argument("--workspace-root", default="agentic-rl/workspaces")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    workspace_root = Path(args.workspace_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    materialized = []
    reports = []
    for task in load_jsonl(Path(args.tasks)):
        new_task, report = materialize_task(task, cache_root, workspace_root, args.force)
        materialized.append(new_task)
        reports.append(report)
        print(json.dumps({"task_id": task["task_id"], "success": report["success"]}, ensure_ascii=False))

    write_jsonl(Path(args.output), materialized)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(reports), "success": sum(1 for r in reports if r["success"])}, indent=2))


if __name__ == "__main__":
    main()
