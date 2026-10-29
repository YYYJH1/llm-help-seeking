"""Task adapter of the scenario. The infrastructure is written against a task adapter, so that a later task such as GSM8K needs only a new data adapter and a new score.

A task gives the pools of questions and scores a reply against an item. The protocol, the entry and the calibration know nothing else about the task. ``TriviaQATask`` is the only one so far:
the pools of ``nmi.data.triviaqa`` and the alias scoring of ``nmi.scenario.scoring`` (primary score and the three sensitivity variants).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nmi.data import triviaqa
from nmi.scenario import scoring


class TriviaQATask:
    """TriviaQA rc.nocontext, validation: the development and the sealed pool of the registry ``data/splits/s1_triviaqa_v1``."""

    name = "triviaqa"
    dev_pool = triviaqa.DEV_POOL
    sealed_pool = triviaqa.SEALED_POOL
    split_version = triviaqa.DEFAULT_SPLIT_VERSION

    @staticmethod
    def load_split(root: str | Path | None = None) -> triviaqa.TriviaSplit:
        return triviaqa.TriviaSplit.load(root)

    @staticmethod
    def load_pool(split: triviaqa.TriviaSplit, pool: str, source: str | Path, *, results_root: str | Path | None = None, holdout_access: Any = None) -> list[triviaqa.TriviaItem]:
        return triviaqa.load_pool(split, pool, source, results_root=results_root, holdout_access=holdout_access)

    @staticmethod
    def score(reply: str, item: triviaqa.TriviaItem) -> scoring.Scores:
        return scoring.score(reply, item.answers)


TASKS = {TriviaQATask.name: TriviaQATask()}
