"""Pousse les templates chat de `pipeline/prompts.py` vers Langfuse.

Nouvelle version promue en production à chaque appel.
La logique conditionnelle (schéma, flags) reste dans le code : un seul prompt
versionné par fonction, pas une version par combinaison de flags.

Usage :
    uv run add-langfuse-prompt
"""

from __future__ import annotations

from langfuse import get_client

from evaluation_dictee.pipeline.prompts import PROMPT_TEMPLATES
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def push_prompts() -> None:
    """Crée/met à jour chaque template dans Langfuse et le promeut en production."""
    langfuse = get_client()

    for name, messages in PROMPT_TEMPLATES.items():
        prompt = langfuse.create_prompt(
            name=name,
            type="chat",
            prompt=list(messages),
            labels=["production"],
        )
        logger.info(
            "Prompt « %s » poussé (version %s), labels : %s",
            prompt.name,
            prompt.version,
            prompt.labels,
        )

    # Flush explicite : Langfuse envoie en asynchrone, sinon les créations sont perdues.
    langfuse.flush()


if __name__ == "__main__":
    push_prompts()
