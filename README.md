# Agentic RL Code Repair

This repository contains a lightweight public release of a closed-loop Agentic RL project for tool-augmented code repair. The system builds a controlled Python repair benchmark, runs tool-use rollouts, verifies success with public and hidden tests, attributes failure modes, regenerates SFT/RLVR-style data, and distills successful behavior into a single deployable LoRA adapter.

Large model weights, LoRA checkpoints, full rollout traces, hidden tests, generated workspaces, and server-specific launchers are intentionally excluded from this public release.

## Results

Benchmark: `Controlled-PyRepair-800`, a deterministic 800-task Python repair benchmark with 600 train, 100 validation, and 100 heldout test tasks across 20 bug families.

| Policy | Split | Pass@1 | Success | Avg tool calls | Invalid rate | Repeated rate | Cost / success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base rollout | train | 5.33% | 32/600 | 4.76 | 17.09% | 62.13% | 204485.88 |
| Naive SFT | heldout | 6.00% | 6/100 | 5.01 | 12.11% | 65.47% | 196839.33 |
| Clean SFT | heldout | 65.00% | 65/100 | 3.24 | 0.00% | 3.70% | 3112.89 |
| Curated RLVR-style data | heldout | 70.00% | 70/100 | 3.00 | 0.00% | 0.00% | 2248.27 |
| Family-routed teacher | heldout | 90.00% | 90/100 | 3.06 | 0.00% | 0.98% | 1658.22 |
| Distilled single LoRA | heldout | 96.00% | 96/100 | 3.02 | 6.21% | 1.66% | 1618.72 |

The final result is a single LoRA adapter distilled from train-split routed-success trajectories plus gold repairs. Validation Pass@1 is 95% and heldout Pass@1 is 96%, which reduces the likelihood that the heldout result is a one-off fluctuation.

## Method Overview

```text
task execution
  -> public / hidden verifier
  -> trajectory logging
  -> failure attribution
  -> clean SFT / curated RLVR-style data
  -> family-level routing analysis
  -> routed-success distillation
  -> single-LoRA evaluation
```

## Repository Structure

```text
agents/          Tool-use rollout agent and action protocol.
environments/    Code execution tools and sandbox/verifier interfaces.
verifier/        Public and hidden test verification entrypoints.
evaluation/      Rollout metric aggregation and family routing analysis.
training/        SFT, clean-SFT, curated data, and distillation data builders.
scripts/         Benchmark construction and evaluation helpers.
reports/         Experiment reports and final interpretation.
results/metrics/ Small JSON metric summaries for the final runs.
```

## Tool Protocol

The stable rollout protocol uses a deliberately small action space:

- `view_file`
- `replace_file`
- `run_test`
- `final`

This made tool behavior easier to analyze for a 3B model and reduced invalid tool calls compared with broader patch/action spaces.

## Distillation Data

The final distillation data uses only train split sources:

| Sample type | Count |
| --- | ---: |
| Gold repair | 600 |
| Routed-success distillation | 537 |
| Total | 1137 |

Failed routed trajectories were excluded from imitation targets. This is why the single adapter can outperform the routed teacher on heldout: it is not a pure clone of teacher outputs, but a filtered and supervised retraining stage over successful behaviors plus gold repairs.

## Minimal Verification

```bash
pip install -r requirements.txt
python evaluation/analyze_rollouts.py --help
python training/build_sft_data.py --help
```

Full reproduction requires local model checkpoints, generated workspaces, hidden tests, and GPU runtime configuration, which are not bundled in this public release.

## Public Release Note

This repository is a curated code-and-report release for portfolio review. It is designed to show the system design, verifier/data pipeline, training-data construction logic, and final metrics without publishing large artifacts or local machine paths.
