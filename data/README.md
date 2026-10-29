# Data and analyses

Each directory holds the model outputs of one experiment, a data dictionary and a CPU script that recomputes the reported statistics. Run the commands from the repository root.

| Experiment and data dictionary | Figures | Command |
|---|---|---|
| [Main experiment](01_main_experiment/README.md) | Figs. 1 and 2 | `python data/01_main_experiment/recompute_main_experiment.py` |
| [Cost controls](02_behaviour_controls/README.md) | Fig. 3a–e | `python data/02_behaviour_controls/recompute_behaviour_controls.py` |
| [Numerical tasks](03_numeric_tasks/README.md) | Fig. 4a | `python data/03_numeric_tasks/recompute_numeric_tasks.py` |
| [Activation patching and prompt interventions](04_arc_patching/README.md) | Figs. 3f and 4b–f, ED2 | `python data/04_arc_patching/recompute_arc.py` |
| [Training](05_training/README.md) | Fig. 5 | `python data/05_training/recompute_training.py` |
| [Expert-model consultation](06_triviaqa_expert/README.md) | Fig. 6 | `python data/06_triviaqa_expert/recompute_triviaqa_expert.py` |
| [Cross-model experiment (46 paper models)](07_models78/README.md) | ED1 | `python data/07_models78/recompute_models78.py` |
| [Additional robustness checks](08_robustness/README.md) | Main-text checks for Figs. 1, 2 and 6 | `python data/08_robustness/recompute_robustness.py` |

## Shared conventions

Parquet files are read with `pandas.read_parquet`. Probabilities are on a 0–1 scale, scores are points per question and rate differences are percentage points. Unless stated otherwise, 95% intervals are percentile intervals from 2,000 paired bootstrap resamples of questions, sorted by `item_key`, using `numpy.random.default_rng(0)`.

## Datasets and identifiers

Question text is not redistributed. Questions are identified by `item_key`, the SHA-256 of the normalized question and correct answer, and `mmlu_redux_items.csv` and the `items.csv` files map the keys to dataset rows. `splits/` holds the frozen question pools that the experiment code reads.

| Dataset | Revision | Licence/source |
|---|---|---|
| [MMLU-Redux 2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) | `372ea425445d51e1ba1188c56e5e893f8138621f` | CC BY 4.0, based on MMLU (MIT) |
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | `210d026faf9955653af8916fad021475a3f00453` | CC BY-SA 4.0 |
| [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` | See the [dataset authors' license and copyright statement](https://github.com/mandarjoshi90/triviaqa) |
