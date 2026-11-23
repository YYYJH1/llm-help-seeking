# LLM help-seeking

Code and data for the associated article.

- `main` holds the model outputs of every experiment and the CPU scripts that recompute the reported statistics ([data/](data/README.md)).
- The tag [v1.0.0](https://github.com/YYYJH1/llm-help-seeking/tree/v1.0.0) also holds the full experiment code. It is the version archived at Zenodo ([10.5281/zenodo.23766964](https://doi.org/10.5281/zenodo.23766964)).

## Recompute the reported statistics

A CPU and Python 3.12 are enough. Install the pinned packages (numpy 2.3.5, pandas 3.0.6, pyarrow 25.0.1 and scipy 1.18.1) and run the scripts from the repository root:

```bash
python -m pip install -r requirements.txt
python data/01_main_experiment/recompute_main_experiment.py
python data/02_behaviour_controls/recompute_behaviour_controls.py
python data/03_numeric_tasks/recompute_numeric_tasks.py
python data/04_arc_patching/recompute_arc.py
python data/05_training/recompute_training.py
python data/06_triviaqa_expert/recompute_triviaqa_expert.py
python data/07_models78/recompute_models78.py
python data/08_robustness/recompute_robustness.py --out build/robustness_check
```

The first seven scripts recompute the statistics of the figures and the Results text from the stored model outputs, print each recomputed value beside the value reported in the article and end with the number of checks that agree (186 in total). The last script reproduces the original Figure 6 gain estimates, computes the additional robustness checks and writes their tables to `--out`. [data/README.md](data/README.md) lists the figures and data dictionary of each script.

## Rerun the experiments

The experiments need a GPU, model weights (or API access) and the public datasets at the revisions listed in [data/README.md](data/README.md#datasets-and-identifiers). The code for them is not on `main`. The tag [v1.0.0](https://github.com/YYYJH1/llm-help-seeking/tree/v1.0.0) holds it together with the [configurations](https://github.com/YYYJH1/llm-help-seeking/tree/v1.0.0/configs), the frozen question pools in [data/splits/](https://github.com/YYYJH1/llm-help-seeking/tree/v1.0.0/data/splits) and the GPU requirements in [requirements.txt](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/requirements.txt), and the Zenodo archive holds the same files. Replace the `/path/to/...` entries in the configurations before running. Some entry points also need study inputs that are not released, as noted in the experiment READMEs.

| Experiment | Entry points at v1.0.0 |
|---|---|
| [Main experiment](data/01_main_experiment/README.md) | [run_calib.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_calib.py) (calibration temperature), [run_c1_confirm.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_c1_confirm.py) |
| [Cost controls](data/02_behaviour_controls/README.md) | [run_mechanism_confirm.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_mechanism_confirm.py) |
| [Numerical tasks](data/03_numeric_tasks/README.md) | [run_explicit_numbers.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_explicit_numbers.py) |
| [Activation patching and prompt interventions](data/04_arc_patching/README.md) | [select_mechanism_sites.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/select_mechanism_sites.py) (site selection), [run_mechanism_internal.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_mechanism_internal.py) (baseline and swaps), [run_mechanism_rescue.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_mechanism_rescue.py) (prompt interventions) |
| [Training](data/05_training/README.md) | [train_part3_b.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/train_part3_b.py) (CAPT, SFT and the two controls), [train_part3_b_qr.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/train_part3_b_qr.py) (soft-target training), [run_part3_b_eval.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_part3_b_eval.py) (evaluation) |
| [Expert-model consultation](data/06_triviaqa_expert/README.md) | [run_scenario_confirm.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_scenario_confirm.py) |
| [Cross-model experiment](data/07_models78/README.md) | [run_local_decisions.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_local_decisions.py) (local models), [run_api_decisions.py](https://github.com/YYYJH1/llm-help-seeking/blob/v1.0.0/scripts/run_api_decisions.py) (API models) |

Code: [MIT](LICENSE). Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation: [CITATION.md](CITATION.md).
