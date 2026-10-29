# Main experiment

Figs. 1 and 2: seven models answer 2,291 MMLU-Redux questions under 54 combinations of expert accuracy, fixed fee and waiting cost.

```bash
python data/01_main_experiment/recompute_main_experiment.py
```

Add `--no-coefficient-bootstrap` for a quick run. Model runs use [run_calib.py](../../scripts/run_calib.py) and [run_c1_confirm.py](../../scripts/run_c1_confirm.py) (see the [main README](../../README.md#rerun-the-experiments)).

## Files and fields

| File | Contents |
|---|---|
| `models.csv` | Models, precision and calibration temperature |
| `conditions.csv` | The 54 conditions and their expected consultation scores |
| `confidence.parquet` | Answer-only prompt: `logprob_1`…`logprob_5`, calibrated `confidence` and `answer_correct` |
| `choices.parquet` | Consultation prompt for every model, question and condition: `logprob_1`…`logprob_5`, `margin`, `ask` and `answer_correct` |

Digits 1–4 are the answer options and digit 5 is consultation. `margin = logprob_5 - max(logprob_1…4)` and `ask = margin > 0`. A readout is valid when `digit_mass = sum(exp(logprob_1…5)) >= 0.5`. Every model has at least 99% valid readouts in each condition, and invalid readouts remain in the analyses. `confidence = max(softmax(logprob_1…4 / calibration_temperature))`. Question keys link to [mmlu_redux_items.csv](../mmlu_redux_items.csv).
