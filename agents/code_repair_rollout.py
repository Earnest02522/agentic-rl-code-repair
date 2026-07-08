#!/usr/bin/env python3
"""Run constrained code-repair rollouts with a local chat model."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE_EXEC = ROOT / "environments" / "code_exec"
sys.path.insert(0, str(CODE_EXEC))

from sandbox_runner import SandboxRunner  # noqa: E402


ACTION_RE = re.compile(r"<ACTION>\s*(\{.*?\})\s*</ACTION>", re.DOTALL)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ModelCall:
    turn: int
    prompt_tokens: int
    completion_tokens: int
    text: str
    parse_error: str | None = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
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
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def reset_workspace(workspace: Path) -> None:
    result = run_cmd(["git", "reset", "--hard", "HEAD"], cwd=workspace)
    if not result["ok"]:
        raise RuntimeError(f"git reset failed for {workspace}: {result['stderr']}")
    clean = run_cmd(["git", "clean", "-fd"], cwd=workspace)
    if not clean["ok"]:
        raise RuntimeError(f"git clean failed for {workspace}: {clean['stderr']}")


def copy_workspace(source: Path, dest_root: Path, task_id: str) -> Path:
    dest = dest_root / task_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    reset_workspace(dest)
    return dest


def compact_observation(observation: dict[str, Any], max_chars: int = 3500) -> str:
    stdout = observation.get("stdout") or ""
    stderr = observation.get("stderr") or ""
    text = {
        "ok": observation.get("ok"),
        "summary": observation.get("summary"),
        "exit_code": observation.get("exit_code"),
        "stdout_tail": stdout[-max_chars:],
        "stderr_tail": stderr[-max_chars:],
    }
    return json.dumps(text, ensure_ascii=False)


def build_system_prompt(tool_mode: str = "full") -> str:
    if tool_mode == "replace_only":
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

    return """You are a code repair agent in a constrained local repository.

You may call exactly one tool per turn by emitting one XML-wrapped JSON object:

<ACTION>{"tool":"search_grep","arguments":{"pattern":"...","include":"*.py"}}</ACTION>
<ACTION>{"tool":"view_file","arguments":{"path":"app/example.py","start_line":1,"max_lines":120}}</ACTION>
<ACTION>{"tool":"apply_patch","arguments":{"patch_text":"diff --git a/app/example.py b/app/example.py\\n--- a/app/example.py\\n+++ b/app/example.py\\n@@ ..."} }</ACTION>
<ACTION>{"tool":"replace_file","arguments":{"path":"app/example.py","content":"complete file content here"}}</ACTION>
<ACTION>{"tool":"run_test","arguments":{"command":"PYTHONPATH=. python -m pytest -q tests"}}</ACTION>
<ACTION>{"tool":"final","arguments":{"answer":"done"}}</ACTION>

Rules:
- Use only these tools: search_grep, view_file, apply_patch, replace_file, run_test, final.
- You may edit only the allowed files.
- The allowed file path is given in the task. Your first action should usually be view_file on that exact path, not search_grep.
- For these small tasks, prefer replace_file after viewing the allowed source file. The content must be the complete corrected file.
- Run the public tests after applying a patch.
- Hidden tests are not visible. Do not try to access hidden test paths.
- Output only the ACTION block, with no markdown fences.
"""


def build_initial_user(task: dict[str, Any], tool_mode: str = "full") -> str:
    allowed = ", ".join(task["allowed_files"])
    if tool_mode == "replace_only":
        return f"""Repair this Python repository.

Task id: {task['task_id']}
Problem: {task['problem_statement']}
Allowed files: {allowed}
Public test command: {task['metadata'].get('public_test_command', task['test_command'])}

Find the bug, patch the allowed file, run the public tests, then finish.
"""

    return f"""Repair this Python repository.

Task id: {task['task_id']}
Problem: {task['problem_statement']}
Allowed files: {allowed}
Public test command: {task['metadata'].get('public_test_command', task['test_command'])}

Find the bug, patch the allowed file, run the public tests, then finish.
Start by viewing the allowed file path exactly as provided.
"""


def parse_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    match = ACTION_RE.search(text)
    raw = None
    if match:
        raw = match.group(1)
    else:
        fenced = FENCED_JSON_RE.search(text)
        if fenced:
            raw = fenced.group(1)
        else:
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                raw = stripped
    if raw is None:
        return None, "missing action JSON block"
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid action json: {exc}"
    if not isinstance(action, dict):
        return None, "action is not an object"
    tool = action.get("tool")
    arguments = action.get("arguments", {})
    if not isinstance(tool, str):
        return None, "tool must be a string"
    if not isinstance(arguments, dict):
        return None, "arguments must be an object"
    return {"tool": tool, "arguments": arguments}, None


def load_model(model_path: str, dtype: str, adapter_path: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_action(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> tuple[str, int, int]:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, int(inputs["input_ids"].shape[1]), int(completion_ids.shape[0])


def enforce_tool_arguments(tool: str, arguments: dict[str, Any], task: dict[str, Any], tool_mode: str = "full") -> tuple[dict[str, Any], str | None]:
    allowed = set(task["allowed_files"])
    if tool_mode == "replace_only" and tool not in {"view_file", "replace_file", "run_test", "final"}:
        return arguments, f"invalid tool in replace_only mode: {tool}"
    if tool == "view_file":
        path = arguments.get("path")
        if not isinstance(path, str):
            return arguments, "view_file requires string path"
        return {
            "path": path,
            "start_line": int(arguments.get("start_line", 1)),
            "max_lines": min(int(arguments.get("max_lines", 160)), 220),
        }, None
    if tool == "search_grep":
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return arguments, "search_grep requires non-empty pattern"
        include = arguments.get("include")
        return {
            "pattern": pattern,
            "include": include if isinstance(include, str) else None,
            "max_matches": min(int(arguments.get("max_matches", 50)), 80),
        }, None
    if tool == "run_test":
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            command = task["metadata"].get("public_test_command", task["test_command"])
        hidden = str(task["metadata"].get("hidden_test_file", ""))
        if hidden and hidden in command:
            return {"command": task["metadata"].get("public_test_command", task["test_command"])}, "hidden test path is not allowed during rollout"
        return {"command": command, "timeout_seconds": 30}, None
    if tool == "apply_patch":
        patch_text = arguments.get("patch_text")
        if not isinstance(patch_text, str) or "diff --git" not in patch_text:
            return arguments, "apply_patch requires unified diff in patch_text"
        touched = set(re.findall(r"^\+\+\+ b/(.+)$", patch_text, flags=re.MULTILINE))
        if touched and not touched.issubset(allowed):
            return arguments, f"patch touches non-allowed files: {sorted(touched - allowed)}"
        return {"patch_text": patch_text}, None
    if tool == "replace_file":
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return arguments, "replace_file requires path and content strings"
        if path not in allowed:
            return arguments, f"replace_file targets non-allowed file: {path}"
        return {"path": path, "content": content}, None
    if tool == "final":
        return {"answer": str(arguments.get("answer", "done"))}, None
    return arguments, f"invalid tool: {tool}"


def classify_invalid_reason(parse_error: str | None, arg_error: str | None, tool: str | None) -> str | None:
    if parse_error:
        return "parse_error"
    if arg_error:
        if arg_error.startswith("invalid tool"):
            return "invalid_tool"
        if "hidden test" in arg_error:
            return "hidden_test_access"
        if "non-allowed" in arg_error:
            return "patch_outside_allowed_files"
        return "invalid_arguments"
    if tool is None:
        return "unknown"
    return None


def rollout_one(
    task: dict[str, Any],
    tokenizer,
    model,
    *,
    run_workspace_root: Path,
    model_name: str,
    repeat_index: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    tool_mode: str,
) -> dict[str, Any]:
    start_time = time.time()
    run_task_id = task["task_id"] if repeat_index == 0 else f"{task['task_id']}_r{repeat_index}"
    workspace = copy_workspace(Path(task["workspace"]), run_workspace_root, run_task_id)
    runner = SandboxRunner(workspace, max_turns=max_turns, max_tool_calls=max_turns, timeout_seconds=60)
    messages = [
        {"role": "system", "content": build_system_prompt(tool_mode)},
        {"role": "user", "content": build_initial_user(task, tool_mode)},
    ]
    model_calls: list[ModelCall] = []
    invalid_calls = 0
    repeated_calls = 0
    tool_signatures: set[str] = set()
    final_seen = False

    for turn in range(1, max_turns + 1):
        text, prompt_tokens, completion_tokens = generate_action(
            tokenizer, model, messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        action, parse_error = parse_action(text)
        model_calls.append(ModelCall(turn, prompt_tokens, completion_tokens, text, parse_error))
        if parse_error:
            invalid_calls += 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {parse_error}. Emit a valid ACTION JSON."})
            continue

        tool = action["tool"]
        arguments, arg_error = enforce_tool_arguments(tool, action["arguments"], task, tool_mode)
        invalid_reason = classify_invalid_reason(parse_error, arg_error, tool)
        signature = json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
        if signature in tool_signatures and tool != "final":
            repeated_calls += 1
        tool_signatures.add(signature)

        messages.append({"role": "assistant", "content": text})
        if arg_error:
            invalid_calls += 1
            messages.append({"role": "user", "content": f"Observation: invalid call ({invalid_reason}): {arg_error}"})
            continue

        if tool == "final":
            final_seen = True
            break

        step = runner.call_tool(tool, arguments)
        observation = step.observation.to_dict()
        messages.append({"role": "user", "content": "Observation: " + compact_observation(observation)})
        if tool == "run_test" and observation.get("ok"):
            final_seen = True
            break

    public_verdict = runner.verify(task["metadata"].get("public_test_command", task["test_command"]), timeout_seconds=60)
    hidden_verdict = runner.verify(task["metadata"]["verifier_command"], timeout_seconds=60)
    diff = run_cmd(["git", "diff", "--", *task["allowed_files"]], cwd=workspace)
    reset_workspace(workspace)

    prompt_tokens = sum(call.prompt_tokens for call in model_calls)
    completion_tokens = sum(call.completion_tokens for call in model_calls)
    tool_steps = [step.to_dict() for step in runner.steps]
    tool_names = [step["tool"] for step in tool_steps]
    return {
        "trajectory_id": f"{model_name.replace('/', '_')}_{run_task_id}",
        "task_id": task["task_id"],
        "repeat_index": repeat_index,
        "split": task["split"],
        "model": model_name,
        "workspace": str(workspace),
        "problem_statement": task["problem_statement"],
        "allowed_files": task["allowed_files"],
        "public_test_command": task["metadata"].get("public_test_command", task["test_command"]),
        "verifier_command": task["metadata"]["verifier_command"],
        "model_calls": [asdict(call) for call in model_calls],
        "steps": tool_steps,
        "final_seen": final_seen,
        "public_verdict": public_verdict.to_dict(),
        "final_verdict": hidden_verdict.to_dict(),
        "patch_diff": diff["stdout"],
        "tool_stats": {
            "tool_calls": len(tool_steps),
            "invalid_calls": invalid_calls,
            "repeated_calls": repeated_calls,
            "tool_names": tool_names,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_seconds": time.time() - start_time,
        },
        "metadata": {
            "bug_family": task["metadata"].get("bug_family"),
            "difficulty": task.get("difficulty"),
            "source_workspace": task["workspace"],
        },
    }


def select_tasks(rows: list[dict[str, Any]], limit: int | None, shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    selected = [row for idx, row in enumerate(rows) if idx % num_shards == shard_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"))
    parser.add_argument("--adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-workspace-root", default=str(ROOT / "workspaces" / "rollout_runs" / "base_qwen"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--tool-mode", choices=["full", "replace_only"], default="full")
    args = parser.parse_args()

    tasks = select_tasks(load_jsonl(Path(args.tasks)), args.limit, args.shard_id, args.num_shards)
    print(json.dumps({"tasks": len(tasks), "shard_id": args.shard_id, "num_shards": args.num_shards}, ensure_ascii=False), flush=True)
    tokenizer, model = load_model(args.model, args.dtype, args.adapter)
    model_name = Path(args.adapter).name if args.adapter else Path(args.model).name
    rows: list[dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks, start=1):
        print(f"[rollout] {index}/{len(tasks)} {task['task_id']}", flush=True)
        try:
            row = rollout_one(
                task,
                tokenizer,
                model,
                run_workspace_root=Path(args.run_workspace_root) / f"shard_{args.shard_id}",
                model_name=model_name,
                repeat_index=args.repeat_index,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                tool_mode=args.tool_mode,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "task_id": task["task_id"],
                "split": task["split"],
                "model": model_name,
                "error": repr(exc),
                "final_verdict": {"success": False, "reward": 0.0, "checks": {}, "failure_summary": "rollout_exception"},
                "tool_stats": {"tool_calls": 0, "invalid_calls": 1, "repeated_calls": 0, "total_tokens": 0},
            }
        rows.append(row)
        write_jsonl(out, rows)

    success = sum(1 for row in rows if row.get("final_verdict", {}).get("success"))
    print(json.dumps({"total": len(rows), "success": success, "success_rate": success / len(rows) if rows else 0.0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
