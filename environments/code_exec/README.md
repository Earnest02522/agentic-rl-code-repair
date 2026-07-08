# Code Execution Environment

This is the first environment to implement.

The environment should run small React / TypeScript tasks in isolated workspaces and expose deterministic tools to the agent.

## Initial Tools

| Tool | Purpose |
| --- | --- |
| `read_file` | Inspect existing source files. |
| `write_file` | Create or replace source files. |
| `run_tsc` | Run TypeScript checks. |
| `run_tests` | Run unit/component tests. |
| `run_lint` | Run lint checks when configured. |
| `inspect_dom` | Inspect rendered DOM state. |
| `take_screenshot` | Capture visual state for verifier checks. |

## Design Requirements

- Every tool call must be logged with arguments, observation, status, duration, and error text.
- Tool failures are observations, not hidden exceptions.
- The environment must enforce max turns, max tool calls, and per-tool timeouts.
- The verifier must be able to replay or inspect enough state to judge success.

