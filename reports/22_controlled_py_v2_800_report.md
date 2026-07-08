# Controlled Python Mini-Repo Benchmark v2 800

Date: 2026-07-06

## Result

`controlled_py_v2_800` expands the initial 30-task benchmark to 800 deterministic Python code-repair tasks with separated public and hidden tests.

| Check | Result |
| --- | ---: |
| Total tasks | 800 |
| Train split | 600 |
| Val split | 100 |
| Held-out test split | 100 |
| Buggy public tests fail | 800 / 800 |
| Buggy hidden tests fail | 800 / 800 |
| Gold patch applies | 800 / 800 |
| Gold public tests pass | 800 / 800 |
| Gold hidden tests pass | 800 / 800 |
| Stability repeated samples | 40 / 800 |
| All quality gates pass | True |

## Files

```text
agentic-rl/data/tasks/controlled_py_v2_800.jsonl
agentic-rl/data/tasks/controlled_py_v2_800_train.jsonl
agentic-rl/data/tasks/controlled_py_v2_800_val.jsonl
agentic-rl/data/tasks/controlled_py_v2_800_heldout_test.jsonl
agentic-rl/data/hidden_tests/controlled_py_v2_800/
agentic-rl/workspaces/controlled_py_v2_800/
agentic-rl/results/controlled_py_v2_800_build_report.json
```

## Design

The benchmark contains 20 generation families with 40 variants per family. They map to 19 bug-family labels because two generation families are path-handling variants. Each generation family contributes 30 train tasks, 5 validation tasks, and 5 held-out test tasks, so the held-out set has 100 tasks and every generation family appears in held-out evaluation.

Public tests live inside each task workspace under `tests/` and are the tests an agent may run during repair. Hidden tests live outside the workspace under `agentic-rl/data/hidden_tests/controlled_py_v2_800/`, so the four code-exec tools cannot inspect them by relative workspace paths. The final verifier runs both public and hidden tests.

## Quality Gates

Each task is accepted only if:

1. buggy code fails public tests
2. buggy code fails hidden tests
3. gold patch applies with `git apply`
4. patched code passes public tests
5. patched code passes hidden tests
6. patched code passes final public+hidden verifier
7. 40 stability samples, two from each generation family, repeat the public/hidden/final checks for stability

This is the dataset to use for base rollout, SFT data generation, GRPO/RLVR reward computation, and held-out evaluation.
