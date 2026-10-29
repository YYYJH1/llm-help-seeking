# Additional robustness checks

These analyses use the saved main-experiment and TriviaQA outputs. They add no model calls or training.

```bash
python data/08_robustness/recompute_robustness.py --out build/robustness_check
```

The script reproduces all eight original Figure 6 gain estimates and intervals before computing the additional comparisons. It reads `01_main_experiment`, `06_triviaqa_expert` and the inputs here.

| File | Contents |
|---|---|
| `scoring_flags.parquet` | Original expert/assistant correctness under four scoring rules, raw confidence and hashed question keys. `alias` identifies the original table; decimal model names use `p` in these aliases. No question or answer text. |
| `sensitivity_calibration.csv` | Archived development-set logistic fits for each model and scoring rule. `original_gate_valid` retains the original diagnostic; all eight displayed models remain in the comparison. |
| `published_primary_gains.csv` | Original Figure 6 estimates and 95% interval endpoints for the reproducibility check. |
| `source_hashes.json` | Original input hashes and archived-analysis provenance. |
| `cost_direction_magnitude.csv` | Original cost-response categories, separate direction labels and interval containment within five percentage points. Values are proportions. |
| `calibration_diagnostics.csv`, `calibration_bins.csv` | Ten-bin calibration error and bin-level confidence/accuracy on the 2,291 main questions per model. |
| `triviaqa_scoring_sensitivity.csv` | Fixed-answer routing gains under primary, exact-alias, negation and short-alias-boundary scores. |
| `triviaqa_cost_only.csv` | Gains when question-level confidence is replaced by development-set answer accuracy, and the additional gain from question-level confidence. |

Intervals use 2,000 paired question resamples with seed 0. Routing averages 18 conditions within each question. Each alternative scoring rule uses its archived development fit, with stated expert accuracy and costs unchanged. The fixed-action control uses development-set accuracy and selects the same action for every question within a condition. Numerical summaries are in `analysis_results.json`.
