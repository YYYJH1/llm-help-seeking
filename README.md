# LLM help-seeking

Code and data for the associated article.

| Directory | Contents |
|---|---|
| [data/](data/README.md) | Model outputs of each experiment and CPU scripts that recompute the reported statistics |
| [nmi/](nmi/), [scripts/](scripts/), [configs/](configs/) | Code used to run the experiments and the fine-tuning |

Recompute the results with Python 3.12 on a CPU:

```bash
python -m pip install numpy==2.3.5 pandas==3.0.6 pyarrow==25.0.1 scipy==1.18.1
python data/01_main_experiment/recompute_main_experiment.py
```

## Rerun the experiments

The entry points linked in each experiment README need a GPU, model weights, the public datasets at the revisions listed in [data/README.md](data/README.md#datasets-and-identifiers) and the packages in [requirements.txt](requirements.txt). Replace the `/path/to/...` entries in `configs/` before running. Runs on the main-experiment and TriviaQA evaluation pools go through a consumption ledger (`nmi/data/holdout.py`), and some entry points also need study inputs that are not released, as noted in their READMEs.

Code: [MIT](LICENSE). Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation: [CITATION.md](CITATION.md).
