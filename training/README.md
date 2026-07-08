# Training

Training should reuse the existing workspace infrastructure where possible:

- `ms-swift/` for SFT or LoRA SFT.
- `verl/` for GRPO/RLVR.
- Existing model and environment conventions from project one where they are useful.

## SFT Stage

Input:

- successful trajectories
- failed-then-repaired trajectories

Output:

- a tool-use/code-agent SFT adapter
- evaluation against the same task suite

## RL Stage

Input:

- task prompts
- environment rollout traces
- verifier rewards

Reward:

```text
reward = success_reward
       + repair_bonus
       - invalid_tool_penalty
       - repeated_call_penalty
       - excessive_turn_penalty
       - ignored_observation_penalty
```

Start with GRPO. Add PPO only if GRPO is unstable or the rollout/reward contract requires it.

