"""Shared schemas for constrained code-repair tasks and trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodeRepairTask:
    task_id: str
    instance_id: str
    repo: str
    language: str
    difficulty: str
    split: str
    source_dataset: str
    base_commit: str
    problem_statement: str
    allowed_files: list[str]
    patch: str
    test_patch: str | None
    test_command: str
    changed_files: int
    changed_lines: int
    workspace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    ok: bool
    summary: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    elapsed_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    turn: int
    tool: str
    arguments: dict[str, Any]
    observation: ToolObservation

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observation"] = self.observation.to_dict()
        return data


@dataclass
class VerifierResult:
    success: bool
    reward: float
    checks: dict[str, Any]
    failure_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    trajectory_id: str
    task_id: str
    model: str
    split: str
    steps: list[TrajectoryStep]
    final_verdict: VerifierResult
    tool_stats: dict[str, Any]
    workspace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        data["final_verdict"] = self.final_verdict.to_dict()
        return data


def safe_relative_path(root: Path, user_path: str) -> Path:
    path = (root / user_path).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError(f"path escapes workspace: {user_path}")
    return path
