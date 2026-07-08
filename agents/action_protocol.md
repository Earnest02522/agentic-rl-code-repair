# Agent Action Protocol

The agent should interact with the environment using structured actions.

## Recommended Action Format

```json
{
  "action": "tool_call",
  "tool": "run_tsc",
  "arguments": {
    "project_dir": "."
  }
}
```

Final answer:

```json
{
  "action": "final",
  "answer": "The task is complete."
}
```

## Rules

- The agent should inspect existing files before editing when the task depends on starter code.
- The agent should run verification tools after edits.
- The agent should use observations to decide the next step.
- Repeating the same failed command without changing files should count as a repeated low-value call.
- Ignoring a clear compiler/test error should count as an observation-use failure.

