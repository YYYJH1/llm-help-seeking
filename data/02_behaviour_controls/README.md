# Cost controls

Fig. 3a–e: the seven models and 2,291 questions of the main experiment under 43 control conditions, with expert accuracy 0.80 and no fixed fee. The script also reads [01_main_experiment](../01_main_experiment/README.md).

```bash
python data/02_behaviour_controls/recompute_behaviour_controls.py
```

Model runs use [run_mechanism_confirm.py](../../scripts/run_mechanism_confirm.py).

## Files and fields

`choices.parquet` has the fields of the [main experiment](../01_main_experiment/README.md#files-and-fields) plus `variant` and `cost_level` (`x`, 0 to 1.5 points). `variants.csv` describes the variants:

| `variant` | Change | Baseline threshold |
|---|---|---|
| `deduct_consultation` | Deduct `x` for consultation | `0.8 - x` |
| `deduct_consultation_waiting_wording` | Same deduction described as waiting cost | `0.8 - x` |
| `deduct_independent_answer` | Deduct `x` for independent answering | `0.8 + x` |
| `deduct_incorrect_independent_answer` | Deduct `x` for incorrect independent answers | `(0.8 + x) / (1 + x)` |
| `wait_no_deduction` | State a wait of `100x` seconds without deduction | `0.8` |
| `no_deduction_no_wait` | No deduction or wait | `0.8` |
| `expected_score_stated` | State the expected consultation score | `0.8 - x` |
| `goal_reminder` | Ask the model to maximize expected score | `0.8 - x` |
