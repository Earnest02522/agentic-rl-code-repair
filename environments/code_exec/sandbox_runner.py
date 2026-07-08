"""Runner utilities for constrained code-repair trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from file_tools import CodeExecTools
from task_schema import TrajectoryStep, VerifierResult


ALLOWED_TOOLS = {"search_grep", "view_file", "apply_patch", "replace_file", "run_test"}


class SandboxRunner:
    def __init__(self, workspace: str | Path, max_turns: int = 6, max_tool_calls: int = 8, timeout_seconds: int = 60):
        self.workspace = Path(workspace).resolve()
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.tools = CodeExecTools(self.workspace, timeout_seconds=timeout_seconds)
        self.steps: list[TrajectoryStep] = []

    def call_tool(self, tool: str, arguments: dict[str, Any]) -> TrajectoryStep:
        if tool not in ALLOWED_TOOLS:
            observation = self.tools.run_test("false", timeout_seconds=1)
            observation.ok = False
            observation.summary = f"invalid tool: {tool}"
        elif len(self.steps) >= self.max_tool_calls:
            observation = self.tools.run_test("false", timeout_seconds=1)
            observation.ok = False
            observation.summary = "tool call budget exceeded"
        else:
            fn = getattr(self.tools, tool)
            observation = fn(**arguments)

        step = TrajectoryStep(
            turn=len(self.steps) + 1,
            tool=tool,
            arguments=arguments,
            observation=observation,
        )
        self.steps.append(step)
        return step

    def verify(self, command: str, timeout_seconds: int = 60) -> VerifierResult:
        obs = self.tools.run_test(command, timeout_seconds=timeout_seconds)
        return VerifierResult(
            success=obs.ok,
            reward=1.0 if obs.ok else 0.0,
            checks={
                "tests": obs.ok,
                "exit_code": obs.exit_code,
                "elapsed_seconds": obs.elapsed_seconds,
                "stdout_tail": obs.stdout[-4000:],
                "stderr_tail": obs.stderr[-4000:],
            },
            failure_summary=None if obs.ok else obs.summary,
        )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
