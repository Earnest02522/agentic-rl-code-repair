# controlled_py_v2_800 SFT/R1c Summary

## Protocol

- Dataset: `controlled_py_v2_800`
- Split: 600 train / 100 val / 100 heldout
- Model: `Qwen2.5-Coder-3B-Instruct`
- Tool mode: `replace_only`
- Allowed tools: `view_file`, `replace_file`, `run_test`, `final`
- Verifier: public tests during rollout, hidden/final verifier for success metrics

## Dataset Quality Gate

`controlled_py_v2_800` passed all deterministic construction checks:

- 800/800 buggy workspaces fail public tests, hidden tests, and final verifier.
- 800/800 gold patches apply.
- 800/800 gold patches pass public tests, hidden tests, and final verifier.
- 40 repeated stability checks passed.
- Final split is balanced by family: each of 20 generation families has 30 train / 5 val / 5 heldout variants.

This preserves `controlled_py_v1_300` as the first strong result and uses v2 as a robustness/scale check.

## Main Results

| Run | Split | Pass@1 | Avg Tool Calls | Invalid Rate | Repeated Rate | Cost / Success |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen replace-only | train | 5.33 | 4.76 | 17.09 | 62.13 | 204485.88 |
| v1 R1c zero-shot on v2 | val | 81.00 | 3.26 | 7.12 | 7.06 | 3243.86 |
| v1 R1c zero-shot on v2 | heldout | 79.00 | 3.34 | 6.96 | 8.08 | 3439.96 |
| naive v2 SFT | val | 6.00 | 5.02 | 11.93 | 64.94 | 197691.50 |
| naive v2 SFT | heldout | 6.00 | 5.01 | 12.11 | 65.47 | 196839.33 |
| clean v2 SFT | val | 65.00 | 3.22 | 0.00 | 3.42 | 3071.58 |
| clean v2 SFT | heldout | 65.00 | 3.24 | 0.00 | 3.70 | 3112.89 |
| v2 R1c | val | 70.00 | 3.00 | 0.00 | 0.00 | 2250.83 |
| v2 R1c | heldout | 70.00 | 3.00 | 0.00 | 0.00 | 2248.27 |

## Closed-Loop Findings

The naive v2 SFT run failed because the SFT data mixed 600 gold repairs with 568 `failure_to_repair` samples that retained failed base actions as assistant targets. Base train trajectories were dominated by repeated `run_test` calls: 2135 `run_test` calls over 600 tasks, with a 62.13 repeated-call rate. The resulting naive SFT model copied this failure pattern, reaching only 6% Pass@1 on both val and heldout with about 65% repeated-call rate.

The corrected clean SFT data removed failed prefixes and kept only 600 gold repairs plus 31 clean successful trajectories. This restored the policy contract: clean SFT reached 65% on both val and heldout, with zero invalid calls and only about 3.5% repeated calls.

The v2 R1c step used clean SFT train rollouts to identify hard families and regenerate data:

- 401 clean successful trajectories.
- 480 hard-family gold repair oversamples.
- 198 hard-family failure-to-gold repairs.
- Hard families: `boundary_condition`, `config_merge`, `csv_parsing`, `parsing`, `path_handling`, `state_update`, `string_transformation`.

After R1c training, final success improved from 65% to 70% on both val and heldout. Tool efficiency improved more strongly: average tool calls dropped to the theoretical minimum of 3.00, invalid calls became 0, repeated calls became 0, and cost per success dropped from about 3.1k tokens to about 2.25k tokens.

## Interpretation

The strongest current generalization result is still the frozen v1 R1c model evaluated zero-shot on v2: 79% heldout with moderate but nonzero tool waste. The new v2 R1c is not the best success-rate model yet, but it is the best tool-efficiency model and demonstrates the closed-loop mechanism clearly:

task execution -> verifier -> failure attribution -> data regeneration -> warm-start training -> re-evaluation.

The result is not paper-thin: the failed naive SFT run exposed a real trajectory-quality failure mode, the clean SFT repair fixed it, and R1c improved efficiency and success over clean SFT. The remaining gap is that R1c over-regularized the policy into a strict 3-step template, which helps cost but leaves several semantic families unsolved.

## Current Resume-Ready Claim

This project can currently be described as:

Built a closed-loop code-repair Agentic RL benchmark on Qwen2.5-Coder-3B with deterministic public/hidden verification over 800 controlled Python repair tasks. Implemented trajectory logging, failure attribution, SFT/R1c data regeneration, and LoRA training. On heldout tasks, clean SFT improved over base from 5.3% train Pass@1 baseline to 65% heldout, and R1c further improved heldout to 70% while reducing invalid and repeated tool calls to 0 and lowering cost per success to about 2.25k tokens. A frozen v1 R1c checkpoint additionally generalized to the expanded v2 heldout at 79% Pass@1.

## Next Optimization

The next round should not blindly add more data. The bottleneck is concentrated in semantic families that remain at 0% after R1c: `config_merge`, `csv_parsing`, `parsing`, `state_update`, and `string_transformation`. The next controlled experiment should add targeted family-specific clean demonstrations or a two-stage repair policy that permits one extra observation after failing public tests, then compare:

- v1 R1c zero-shot on v2,
- v2 R1c strict 3-step,
- v2 R1c with one repair-after-test turn,
- family-targeted v2 R1c.

The target for the next successful version is at least 80% heldout while keeping invalid/repeated call rates below 2% and cost per success below 3k tokens.
