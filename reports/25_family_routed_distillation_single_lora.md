# Family-Routed Policy Distillation to a Single LoRA

## Goal

Compress the verifier-driven family-routed policy into one deployable LoRA adapter.

Previous results:

- `v2 R1c strict`: 70% heldout Pass@1 with very clean tool usage.
- `family-routed R1c`: 90% heldout Pass@1 by routing selected families to the older v1 R1c adapter.

This experiment distills the routed behavior into a single adapter so the final result is not dependent on runtime adapter routing.

## Distillation Data

The distillation source was built only from the v2 train split, not from val or heldout.

Source rollouts:

- Default teacher: `r1c_v2` on all 600 train tasks.
- Override teacher: `r1c_v1` on four train families:
  - `config_merge`
  - `csv_parsing`
  - `state_update`
  - `string_transformation`

Routed train source:

- 480 tasks routed to `r1c_v2`.
- 120 tasks routed to `r1c_v1`.
- Successful routed trajectories used for distillation: 537/600.

Final distillation SFT data:

| Sample type | Count |
|---|---:|
| gold_repair | 600 |
| routed_success_distill | 537 |
| total | 1137 |

Routed-success sources:

| Source | Count |
|---|---:|
| r1c_v2 | 420 |
| r1c_v1 | 117 |

Failed routed trajectories were not used as imitation targets.

## Training

- Base model: `Qwen2.5-Coder-3B-Instruct`
- Init adapter: `qwen2p5_coder3b_lora_controlled_py_v2_clean`
- Output adapter: `qwen2p5_coder3b_lora_family_routed_distill_v2`
- Epochs: 1
- Steps: 285
- Learning rate: `5e-5`
- Final train loss: `0.01801`

## Results

| Policy | Split | Pass@1 | Success | Avg Tool Calls | Invalid Rate | Repeated Rate | Avg Tokens | Cost/Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2 R1c strict | heldout | 70.0 | 70/100 | 3.00 | 0.00 | 0.00 | 1573.79 | 2248.27 |
| family-routed R1c | heldout | 90.0 | 90/100 | 3.06 | 0.00 | 0.98 | 1492.40 | 1658.22 |
| distilled single LoRA | val | 95.0 | 95/100 | 3.00 | 7.69 | 1.67 | 1580.70 | 1663.89 |
| distilled single LoRA | heldout | 96.0 | 96/100 | 3.02 | 6.21 | 1.66 | 1553.97 | 1618.72 |

## Heldout Family Results

The distilled single LoRA solved nearly all heldout families:

- 100% on all families except `parsing`.
- `parsing`: 1/5.
- `path_handling`: improved to 10/10.

This is better than the routed policy because distillation transferred the routed behavior while also improving some default-policy failures.

## Interpretation

This is the strongest current result. It turns the earlier family-routed system into a single deployable adapter and reaches 96% heldout Pass@1 on the 800-task benchmark.

The only caveat is that invalid call rate is no longer exactly zero, because some generated outputs include parse/tool-format issues even though final task success is high. Repeated calls remain low at 1.66%, and cost per success is the best so far at about 1.62k tokens.

## Resume Version

Recommended resume wording:

Built a closed-loop Agentic RL code-repair system on Qwen2.5-Coder-3B with an 800-task controlled benchmark, public/hidden verifiers, full tool-trajectory logging, failure attribution, SFT/R1c data regeneration, and LoRA training. Improved heldout Pass@1 from 5.3% base to 70% with R1c, used verifier-driven family-level attribution to build a 90% policy-routed teacher, then distilled routed successful trajectories into a single LoRA adapter reaching 96% heldout Pass@1 with 3.02 average tool calls and about 1.62k tokens per success.

Interview caveat:

- 90% was the explicit routed teacher policy.
- 96% is the distilled single-LoRA student evaluated on heldout.
- Distillation used only train trajectories plus gold repairs; val/heldout were evaluation only.
