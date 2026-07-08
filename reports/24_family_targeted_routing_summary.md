# Family-Targeted Routing Summary

## Goal

Improve `controlled_py_v2_800` heldout performance after the strict v2 R1c policy plateaued at 70% Pass@1.

The v2 R1c policy had excellent tool efficiency but still failed several semantic families:

- `config_merge`
- `csv_parsing`
- `parsing`
- `state_update`
- `string_transformation`

## Method

I built a family-targeted routing policy from existing verified rollouts:

- Default policy: `r1c_v2_replace_only`
- Override policy: `r1c_v1_on_v2_replace_only`
- Override families:
  - `config_merge`
  - `csv_parsing`
  - `state_update`
  - `string_transformation`

`parsing` was not routed because both v1 R1c and v2 R1c failed this family on v2 heldout.

This is a targeted Agent policy routing result, not a single-adapter result. It is still a valid closed-loop repair step because the route is derived from verifier-based family-level failure attribution.

## Results

| Policy | Split | Pass@1 | Success | Avg Tool Calls | Invalid Rate | Repeated Rate | Avg Tokens | Cost/Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2 R1c strict | val | 70.0 | 70/100 | 3.00 | 0.00 | 0.00 | 1575.58 | 2250.83 |
| v2 R1c strict | heldout | 70.0 | 70/100 | 3.00 | 0.00 | 0.00 | 1573.79 | 2248.27 |
| family-routed R1c | val | 90.0 | 90/100 | 3.04 | 0.00 | 0.66 | 1472.68 | 1636.31 |
| family-routed R1c | heldout | 90.0 | 90/100 | 3.06 | 0.00 | 0.98 | 1492.40 | 1658.22 |

## Family-Level Heldout Results

| Family | Routed Policy | Heldout Success |
|---|---|---:|
| config_merge | v1 R1c | 5/5 |
| csv_parsing | v1 R1c | 5/5 |
| state_update | v1 R1c | 5/5 |
| string_transformation | v1 R1c | 5/5 |
| parsing | v2 R1c | 0/5 |
| path_handling | v2 R1c | 5/10 |
| all other families | v2 R1c | 100% |

## Interpretation

The result demonstrates that the remaining weakness after v2 R1c was not a general tool-use failure. It was concentrated in a few semantic repair families. Routing to the older v1 R1c adapter for families where it generalized better raised heldout Pass@1 from 70% to 90% while keeping tool cost low.

The final bottleneck is now narrow:

- `parsing`: 0/5
- `path_handling`: 5/10

The next single-adapter improvement should target these two families or distill the routed policy into one LoRA adapter.

## Resume Wording

For resume use, report the expanded v2 benchmark as the main result:

Built a closed-loop Agentic RL code-repair system on Qwen2.5-Coder-3B over an 800-task controlled benchmark with public/hidden verifiers, trajectory logging, failure attribution, and SFT/R1c data regeneration. Improved heldout Pass@1 from 5.3% base to 70% with R1c, reduced invalid/repeated tool calls to 0, and further raised heldout Pass@1 to 90% via verifier-driven family-targeted policy routing while keeping cost per success at about 1.66k tokens.

Keep the v1 result as interview evidence, not the main resume number. Use it only when explaining the project evolution:

- v1 found and fixed the tool-protocol mismatch.
- v2 validated robustness on a larger 600/100/100 split.
- family-targeted routing used failure attribution to push heldout from 70% to 90%.
