# Task Data

This directory stores D2C / Code Agent task definitions.

Each task should describe the user requirement, starter project, allowed tools, verifier checks, and expected success conditions.

Recommended schema:

```json
{
  "task_id": "d2c_button_0001",
  "split": "train",
  "category": "react_component",
  "difficulty": "easy",
  "instruction": "Build a primary button component with loading and disabled states.",
  "starter": {
    "template": "vite_react_ts",
    "files": {}
  },
  "allowed_tools": ["read_file", "write_file", "run_tsc", "run_tests", "inspect_dom"],
  "success_criteria": {
    "typescript": true,
    "tests": ["button_states.test.tsx"],
    "dom_assertions": [],
    "visual_assertions": []
  },
  "metadata": {
    "source": "manual_seed",
    "notes": ""
  }
}
```

Keep raw tasks immutable. If a task is repaired or re-labeled, write a new version instead of editing old experiment inputs.

