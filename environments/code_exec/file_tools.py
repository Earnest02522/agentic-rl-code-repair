"""Four constrained tools for code-repair agents."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from task_schema import ToolObservation, safe_relative_path


MAX_VIEW_BYTES = 20000


class CodeExecTools:
    def __init__(self, workspace: str | Path, timeout_seconds: int = 60):
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def search_grep(self, pattern: str, include: str | None = None, max_matches: int = 50) -> ToolObservation:
        start = time.time()
        cmd = ["rg", "--line-number", "--no-heading", pattern, "."]
        if include:
            cmd.extend(["--glob", include])
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.workspace,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
            )
            lines = proc.stdout.splitlines()[:max_matches]
            return ToolObservation(
                ok=proc.returncode in (0, 1),
                summary=f"{len(lines)} matches returned",
                stdout="\n".join(lines),
                stderr=proc.stderr,
                exit_code=proc.returncode,
                elapsed_seconds=time.time() - start,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolObservation(False, f"search_grep failed: {exc}", elapsed_seconds=time.time() - start)

    def view_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> ToolObservation:
        start = time.time()
        try:
            file_path = safe_relative_path(self.workspace, path)
            if not file_path.exists() or not file_path.is_file():
                return ToolObservation(False, f"file not found: {path}", elapsed_seconds=time.time() - start)
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            start_idx = max(start_line - 1, 0)
            selected = lines[start_idx : start_idx + max_lines]
            text = "\n".join(f"{i + start_idx + 1}: {line}" for i, line in enumerate(selected))
            if len(text.encode("utf-8")) > MAX_VIEW_BYTES:
                text = text.encode("utf-8")[:MAX_VIEW_BYTES].decode("utf-8", errors="ignore")
            return ToolObservation(
                True,
                f"read {len(selected)} lines from {path}",
                stdout=text,
                elapsed_seconds=time.time() - start,
                extra={"total_lines": len(lines)},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolObservation(False, f"view_file failed: {exc}", elapsed_seconds=time.time() - start)

    def apply_patch(self, patch_text: str) -> ToolObservation:
        start = time.time()
        try:
            proc = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=self.workspace,
                input=patch_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
            )
            return ToolObservation(
                ok=proc.returncode == 0,
                summary="patch applied" if proc.returncode == 0 else "patch failed",
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                elapsed_seconds=time.time() - start,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolObservation(False, f"apply_patch failed: {exc}", elapsed_seconds=time.time() - start)

    def replace_file(self, path: str, content: str) -> ToolObservation:
        start = time.time()
        try:
            file_path = safe_relative_path(self.workspace, path)
            if not file_path.exists() or not file_path.is_file():
                return ToolObservation(False, f"file not found: {path}", elapsed_seconds=time.time() - start)
            file_path.write_text(content.rstrip() + "\n", encoding="utf-8")
            return ToolObservation(
                True,
                f"replaced {path}",
                stdout="",
                elapsed_seconds=time.time() - start,
                extra={"bytes": len(content.encode("utf-8"))},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolObservation(False, f"replace_file failed: {exc}", elapsed_seconds=time.time() - start)

    def run_test(self, command: str, timeout_seconds: int | None = None) -> ToolObservation:
        start = time.time()
        timeout = timeout_seconds or self.timeout_seconds
        try:
            self._clean_python_bytecode()
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.workspace,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            return ToolObservation(
                ok=proc.returncode == 0,
                summary="tests passed" if proc.returncode == 0 else "tests failed",
                stdout=proc.stdout[-20000:],
                stderr=proc.stderr[-20000:],
                exit_code=proc.returncode,
                elapsed_seconds=time.time() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolObservation(
                False,
                f"tests timed out after {timeout}s",
                stdout=(exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-20000:] if isinstance(exc.stderr, str) else "",
                elapsed_seconds=time.time() - start,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolObservation(False, f"run_test failed: {exc}", elapsed_seconds=time.time() - start)

    def _clean_python_bytecode(self) -> None:
        for path in self.workspace.rglob("__pycache__"):
            if path.is_dir():
                for child in path.glob("*.pyc"):
                    child.unlink(missing_ok=True)
