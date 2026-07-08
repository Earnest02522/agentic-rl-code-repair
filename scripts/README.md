# Scripts

Future runnable scripts should go here.

Suggested order:

1. `run_code_agent_rollout.py`
2. `verify_trajectories.py`
3. `attribute_failures.py`
4. `build_sft_data.py`
5. `build_rl_data.py`
6. `run_sft.sh`
7. `run_grpo.sh`
8. `evaluate_agent.py`

Prefer CPU-only scripts first. Start GPU training only after the environment, verifier, and trajectory schema are stable.

