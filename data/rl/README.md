# RL Data

This directory stores RL prompt/reward data for GRPO/RLVR.

The initial target is veRL-compatible prompt data plus reward metadata. The reward should be computed by the verifier and cost-aware penalties.

Reward components:

- task success
- repair success
- invalid tool-call penalty
- repeated call penalty
- excessive turn penalty
- ignored observation penalty
- optional visual/DOM/check-specific subrewards

