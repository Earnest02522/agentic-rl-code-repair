# Agentic RL Code Repair

This repository contains a lightweight public release of a closed-loop Agentic RL project for tool-augmented code repair. The system builds a controlled Python repair benchmark, runs tool-use rollouts, verifies success with public and hidden tests, attributes failure modes, regenerates SFT/RLVR-style data, and distills successful behavior into a single deployable LoRA adapter.

Large model weights, LoRA checkpoints, full rollout traces, hidden tests, generated workspaces, and server-specific launchers are intentionally excluded from this public release.

## Highlights

- Built `Controlled-PyRepair-800`, a deterministic 800-task Python repair benchmark with public tests, hidden verifiers, 20 bug families, and a 600/100/100 train/validation/heldout split.
- Designed a compact tool protocol for small code agents: `view_file`, `replace_file`, `run_test`, and `final`.
- Converted verified rollouts into clean SFT, curated RLVR-style data, family-routed teacher behavior, and a final single-LoRA distilled policy.
- Improved heldout Pass@1 from 5.33% base rollout to 96.00% with the distilled single LoRA, while reducing repeated tool calls from 62.13% to 1.66%.

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

## Benchmark Design

Each task is a small Python repository with:

- A natural-language repair instruction.
- One allowed source file that the agent may edit.
- Public tests inside the workspace for feedback during rollout.
- Hidden tests outside the workspace for final verification.
- Metadata for split, difficulty, bug family, and verifier command.

The heldout split is never used to build SFT, RLVR-style, routing, or distillation data. The final single-LoRA model is trained from train-split gold repairs plus train-split routed-success trajectories.

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

The key lesson from the project is that simply imitating raw rollouts is fragile for a 3B code agent. The naive SFT model inherited repeated `run_test` loops and low success. Filtering imitation targets to clean successful repairs, then using verifier-driven family attribution and distillation, produced a much more stable and deployable policy.

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

## Environment

The public code is intended for Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original experiments used a local `Qwen2.5-Coder-3B-Instruct` checkpoint plus LoRA adapters. Full training and evaluation require a CUDA GPU, local model weights, generated workspaces, and hidden tests.

## Data Preparation

Build the controlled benchmark:

```bash
python scripts/build_controlled_py_v2_800.py
```

This creates task specifications, public-test workspaces, hidden verifier tests, and a build report under the project tree. In this public release, large generated workspaces and hidden tests are excluded, but the benchmark generator is included so the structure is auditable.

Materialize workspaces from task specifications when needed:

```bash
python scripts/materialize_workspaces.py --help
```

## Distillation Data

The final distillation data uses only train split sources:

| Sample type | Count |
| --- | ---: |
| Gold repair | 600 |
| Routed-success distillation | 537 |
| Total | 1137 |

Failed routed trajectories were excluded from imitation targets. This is why the single adapter can outperform the routed teacher on heldout: it is not a pure clone of teacher outputs, but a filtered and supervised retraining stage over successful behaviors plus gold repairs.

## Training Data Builders

The repository keeps the data-generation stages separate so that failure modes can be inspected instead of hidden inside one monolithic script.

```bash
python training/build_sft_data.py --help
python training/build_clean_sft_data.py --help
python training/build_rlvr_data.py --help
python training/build_r1c_replace_only_data.py --help
python training/build_distill_from_rollouts.py --help
```

Typical progression:

1. Build naive SFT data from raw trajectories.
2. Build clean SFT data from gold repairs and successful trajectories.
3. Build curated RLVR-style data by keeping verifier-positive or repairable examples.
4. Analyze weak bug families and build routed-success distillation data.
5. Train a single LoRA adapter from gold repairs plus routed-success trajectories.

## LoRA Training

Example clean-SFT training command:

```bash
python training/train_lora_sft.py \
  --model /path/to/Qwen2.5-Coder-3B-Instruct \
  --train-data data/sft/controlled_py_v2_clean_sft_train.jsonl \
  --output-dir outputs/sft/qwen2p5_coder3b_clean_sft \
  --max-length 4096 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4
```

Example distillation training command initialized from a previous adapter:

```bash
python training/train_lora_sft.py \
  --model /path/to/Qwen2.5-Coder-3B-Instruct \
  --init-adapter outputs/sft/qwen2p5_coder3b_clean_sft \
  --train-data data/sft/controlled_py_v2_family_routed_distill_train.jsonl \
  --output-dir outputs/sft/qwen2p5_coder3b_family_routed_distill \
  --max-length 4096 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4
```

## Rollout Evaluation

Run constrained agent rollouts:

```bash
python agents/code_repair_rollout.py \
  --tasks data/tasks/controlled_py_v2_800.jsonl \
  --model /path/to/Qwen2.5-Coder-3B-Instruct \
  --adapter outputs/sft/qwen2p5_coder3b_family_routed_distill \
  --out results/rollouts/distilled_single_lora_heldout.jsonl \
  --run-workspace-root workspaces/rollout_runs/distilled_single_lora \
  --max-turns 6 \
  --max-new-tokens 900 \
  --temperature 0.0 \
  --tool-mode replace_only
```

For multi-GPU evaluation, shard the task file:

```bash
python agents/code_repair_rollout.py \
  --tasks data/tasks/controlled_py_v2_800.jsonl \
  --model /path/to/Qwen2.5-Coder-3B-Instruct \
  --adapter outputs/sft/qwen2p5_coder3b_family_routed_distill \
  --out results/rollouts/shard_0.jsonl \
  --shard-id 0 \
  --num-shards 8 \
  --tool-mode replace_only
```

Aggregate rollout metrics:

```bash
python evaluation/analyze_rollouts.py \
  --input results/rollouts/distilled_single_lora_heldout.jsonl \
  --out results/metrics/distilled_single_lora_heldout_metrics.json \
  --md-out results/metrics/distilled_single_lora_heldout_metrics.md
```

## Minimal Verification

```bash
pip install -r requirements.txt
python evaluation/analyze_rollouts.py --help
python training/build_sft_data.py --help
python agents/code_repair_rollout.py --help
```

Full reproduction requires local model checkpoints, generated workspaces, hidden tests, and GPU runtime configuration, which are not bundled in this public release.

## Scope And Limitations

- This is a controlled code-repair benchmark, not a claim of broad SWE-bench-style real-repository generalization.
- The hidden tests and full rollout traces are not published to avoid turning the benchmark into a memorization target and to keep the repository lightweight.
- The reported `Cost / success` is measured in rollout token usage, not API billing cost.
- The term `RLVR-style` here refers to verifier-curated training data and reward-style selection logic. The public release focuses on the data/verifier pipeline and LoRA distillation artifact, not on shipping a full distributed RL training stack.

## Public Release Note

This repository is a curated code-and-report release for portfolio review. It is designed to show the system design, verifier/data pipeline, training-data construction logic, and final metrics without publishing large artifacts or local machine paths.
