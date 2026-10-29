# Expert-model consultation

Fig. 6: eight assistant models answer TriviaQA questions or call `ask_edge_expert` for a stored Qwen3-32B answer, on 1,000 development and 2,000 held-out questions under 18 conditions.

```bash
python data/06_triviaqa_expert/recompute_triviaqa_expert.py
```

Model runs use [run_scenario_confirm.py](../../scripts/run_scenario_confirm.py) (see the [main README](../../README.md#rerun-the-experiments)).

## Files and fields

| File | Contents |
|---|---|
| `items.csv` | Question keys with the TriviaQA `question_id` (`rc.nocontext` validation split) |
| `assistant_answers_development.parquet`, `assistant_answers_heldout.parquet` | Independent-answer correctness and 0–9 confidence readings |
| `expert_answers_heldout.parquet` | Correctness of the stored expert answer |
| `consultation_choices_heldout.parquet` | Decisions (`offload`, `self_answer` or `format_failure`) per model, question, presentation and level |
| `confidence_calibration.csv` | Calibration `sigmoid(a + b*logit(c))` fitted on the development questions |

A reply is correct when its normalized first line contains a normalized reference alias. Levels are seconds of waiting or hundredths of a point of battery cost (`cost_points = level/100`), and format failures score zero. `item_key` is the SHA-256 of the normalized question and first reference answer.
