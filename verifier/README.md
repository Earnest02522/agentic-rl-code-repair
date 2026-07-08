# Verifier

The verifier automatically decides whether an execution trajectory solved the task.

For the initial D2C / Code Agent scope, combine deterministic checks:

- TypeScript passes.
- Required tests pass.
- Task-specific assertions pass.
- Optional DOM assertions pass.
- Optional screenshot or visual checks pass.

The verifier output should be structured:

```json
{
  "success": false,
  "checks": {
    "typescript": true,
    "tests": false,
    "dom": null,
    "screenshot": null
  },
  "failure_summary": "The disabled state is missing.",
  "evidence": [
    {
      "check": "tests",
      "message": "Expected disabled button to have aria-disabled=true."
    }
  ]
}
```

The verifier is the foundation for both evaluation metrics and RL rewards, so keep versions stable and record them in trajectories.

