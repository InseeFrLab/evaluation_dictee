"""Fabrique de scorer : construit le bon `Scorer` (approche 1 ou 2) selon la configuration."""

from __future__ import annotations

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.models.base import Scorer
from evaluation_dictee.models.two_stage import TwoStageScorer
from evaluation_dictee.models.vlm import VLMScorer


def build_scorer(
    config: ExperimentConfig,
    grid_items: list[GridItem],
    base_url: str,
    api_key: str,
) -> Scorer:
    """Construit le scorer correspondant à l'approche déclarée dans la config.

    Args:
        config: Configuration de l'expérience (dont `approach`, `model`, `prompt`, `grid`).
        grid_items: Items de la grille de codage.
        base_url: URL de base de l'API compatible OpenAI.
        api_key: Clé d'API.

    Returns:
        Un `TwoStageScorer` si `approach == "two_stage"`, sinon un `VLMScorer`.
    """
    if config.approach == "two_stage":
        return TwoStageScorer(
            model_config=config.model,
            prompt_config=config.prompt,
            base_url=base_url,
            api_key=api_key,
            grid_items=grid_items,
            scheme=config.grid.scheme,
            model_config_stage2=config.model_stage2,
        )
    return VLMScorer(
        model_config=config.model,
        prompt_config=config.prompt,
        base_url=base_url,
        api_key=api_key,
        grid_items=grid_items,
        scheme=config.grid.scheme,
    )
