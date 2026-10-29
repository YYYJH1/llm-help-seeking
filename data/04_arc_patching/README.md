# Activation patching and prompt interventions

Figs. 3f and 4b–f and Extended Data Fig. 2: three models on 512 ARC-Challenge test questions, with patching sites selected on 128 MMLU-Redux calibration questions.

```bash
python data/04_arc_patching/recompute_arc.py
```

Model runs use [select_mechanism_sites.py](../../scripts/select_mechanism_sites.py), [run_mechanism_internal.py](../../scripts/run_mechanism_internal.py) and [run_mechanism_rescue.py](../../scripts/run_mechanism_rescue.py). They need frozen study inputs that are not released; `patching_sites.csv` records the selected sites.

## Files and fields

| File | Contents |
|---|---|
| `items.csv` | Question keys with their ARC-Challenge or MMLU-Redux source rows |
| `conditions.csv` | The original and new settings, 36 conditions each |
| `contrasts.csv` | Condition pairs that define the behavioral and patching contrasts |
| `confidence_answers.parquet` | Answer-only logits, correctness and calibrated confidence `confidence_z` |
| `behaviour_test.parquet` | Consultation decisions: `margin`, `legal_mass` and `ask` |
| `prompt_intervention_test.parquet` | Decisions, correctness and expected scores before and after each prompt addition |
| `patching_test.parquet` | Patching and random-control outputs at the selected sites |
| `patching_sites.csv` | Selected layers and positions with their development-stage signs |
| `patching_scan_development.parquet` | Layer and position scans on the development questions |

`margin` is the consultation-label logit minus the largest answer-label logit. A decision is valid when the five labels carry `legal_mass >= 0.5`; otherwise `ask` and its expected score are missing. A patching effect is `(swap - swap_ref) - (random - random_ref)`, where `_ref` holds unmodified outputs from the same batch.
