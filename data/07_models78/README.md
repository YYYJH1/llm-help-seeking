# Cross-model experiment

Extended Data Fig. 1: 46 models from nine developers, open-weight or accessed through an API, on MMLU-Redux questions under six waiting costs, mostly at expert accuracy 0.80 and no fixed fee. The archive keeps all 78 tested models, and `shown_in_figure` marks the 46 in the paper.

```bash
python data/07_models78/recompute_models78.py
```

Model runs use [run_local_decisions.py](../../scripts/run_local_decisions.py) and [run_api_decisions.py](../../scripts/run_api_decisions.py).

## Files and fields

| File | Contents |
|---|---|
| `models.csv` | Model, developer, source (`local` or `api`), reasoning setting, question count and figure panel |
| `choices_all_models.parquet` | Decision (`ask`, `answer`, `format_failure` or `missing`) per model, expert accuracy, waiting cost and question |
| `choices_alternative_request_formats.parquet` | Decisions under JSON or constrained-choice requests at expert accuracy 0.80 |
| `bootstrap_item_order.csv` | Question order used for the bootstrap intervals |
