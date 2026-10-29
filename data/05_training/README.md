# Training evaluation

Fig. 5: Qwen3-4B, Gemma-4-E4B-it and SmolLM3-3B before and after fine-tuning, evaluated on 2,000 held-out MMLU-Redux questions.

```bash
python data/05_training/recompute_training.py
```

Model runs use [train_part3_b.py](../../scripts/train_part3_b.py), [train_part3_b_qr.py](../../scripts/train_part3_b_qr.py) and [run_part3_b_eval.py](../../scripts/run_part3_b_eval.py). Training reads base-model confidence from your own `run_calib.py` outputs (`W5_SOURCES` in `scripts/_part3_b_prompts.py`), and the evaluation also needs the unreleased inputs of [04_arc_patching](../04_arc_patching/README.md).

## Files and fields

| File | Contents |
|---|---|
| `training_runs.csv` | Model, objective, seed, learning rate, KL weight and checkpoint step of each run |
| `trials_<model>.parquet` | One row per evaluation trial: set, question, stated terms, expected action scores and, for each checkpoint, `ask__<tag>`, `margin__<tag>` and `valid__<tag>` |
| `answer_only_<model>.parquet` | Answer-only correctness and confidence for each checkpoint |

A decision is valid when the allowed response digits carry a total probability of at least 0.5. Checkpoint tags are `base`, `capt_s0`…`s2`, `sft_s0`…`s2`, `softtarget_s0`…`s2` and, for Qwen3-4B, `fixedcost_s0` and `rotated_s0`.

| `set` | Trials per question | Measure |
|---|---:|---|
| `in_dist` | 4 | Regret within the training range |
| `equal_net` | 6 | Decision inconsistency (two matched-score levels, three expressions each) |
| `sweep` | 8 | Penalized threshold error and threshold-rule disagreement |
| `unseen_combo` | 4 | Unseen accuracy and cost combinations |
| `deduct_position` | 4 | Costs charged to independent or incorrect independent answers |
| `waiting` | 4 | Waiting-cost wording |
| `price_extrap` | 4 | Waiting costs 0.41–0.70 |
