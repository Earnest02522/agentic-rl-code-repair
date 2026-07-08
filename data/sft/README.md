# SFT Data

This directory stores supervised fine-tuning data generated from:

- successful trajectories
- failed trajectories followed by successful repairs
- distilled correction traces

Recommended output format should follow chat-message JSONL so it can later be converted to `ms-swift` or another SFT runner.

Each row should keep provenance fields:

```json
{
  "id": "sft_d2c_000001",
  "source_trajectory_ids": ["traj_000001", "traj_000002"],
  "task_id": "d2c_button_0001",
  "data_type": "success_trace",
  "messages": []
}
```

