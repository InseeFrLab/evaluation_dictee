"""Pousse les templates chat de `pipeline/prompts.py` vers le Prompt Registry MLflow.

Nouvelle version promue en production (alias `production`) à chaque appel.
La logique conditionnelle (schéma, flags) reste dans le code : un seul prompt
versionné par fonction, pas une version par combinaison de flags.

Usage :
    uv run add-mlflow-prompt
"""

from __future__ import annotations

import os

import mlflow
import mlflow.genai

from evaluation_dictee.pipeline.prompts import PROMPT_TEMPLATES
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking_mlflow import DEFAULT_EXPERIMENT, EXPERIMENT_ENV, PROMPT_ALIAS

logger = get_logger(__name__)


def push_prompts() -> None:
    """Crée une version de chaque template dans MLflow et la promeut en production."""
    mlflow.set_experiment(os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT))
    for name, messages in PROMPT_TEMPLATES.items():
        prompt = mlflow.genai.register_prompt(
            name=name,
            template=[dict(m) for m in messages],
            commit_message="push depuis pipeline/prompts.py",
        )
        mlflow.genai.set_prompt_alias(name, PROMPT_ALIAS, prompt.version)
        logger.info(
            "Prompt « %s » poussé (version %s), alias : %s", name, prompt.version, PROMPT_ALIAS
        )


if __name__ == "__main__":
    push_prompts()
