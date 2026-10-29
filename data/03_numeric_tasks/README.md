# Numerical tasks

Fig. 4a: seven models receive 400 numerical combinations of own accuracy `p`, expert accuracy `a_H` and cost `w`, whose expected scores are `p` and `a_H - w`.

```bash
python data/03_numeric_tasks/recompute_numeric_tasks.py
```

Model runs use [run_explicit_numbers.py](../../scripts/run_explicit_numbers.py); its `build` subcommand regenerates the combinations on a CPU.

## Files and fields

| File | Contents |
|---|---|
| `trials.parquet` | Every combination and prompt version: `task`, `label_order`, `logprob_1`, `logprob_2`, `margin`, `choice` and `gate_pass` |
| `conditions.csv` | Stated values, stratum and the higher-scoring action |
| `repeat_readings.parquet` | Two readings of 300 prompts for Qwen3-14B and Qwen3-32B |
| `accuracy_summary.csv` | Accuracy, 95% interval and threshold for the six quantities in Fig. 4a |

Tasks are `identification`, `comparison`, `action_selection` and `action_selection_with_scores`. A reading is valid when the two labels carry at least half of the next-token probability (`gate_pass`). Ties count as incorrect.
