# Evaluation

Evaluation should report both task quality and tool efficiency.

## Required Metrics

- Task Success Rate
- Repair Success Rate
- Avg Tool Calls
- Invalid Tool Call Rate
- Repeated Call Rate
- Cost per Success
- Error Attribution Accuracy
- Pass@1
- Pass@k

## Comparison Table

Use a table like this after experiments exist:

| Model | Success | Repair Success | Avg Calls | Invalid Calls | Repeated Calls | Cost/Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base prompt | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD | TBD | TBD | TBD |
| SFT + GRPO | TBD | TBD | TBD | TBD | TBD | TBD |
| Closed-loop round 2 | TBD | TBD | TBD | TBD | TBD | TBD |

Keep `Task Success Rate` separate from tool-efficiency metrics. A model can solve more tasks while becoming too expensive, so both sides matter.

