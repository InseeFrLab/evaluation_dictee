"""Mesure des tokens et du coût d'une copie, indépendante de l'outil de suivi.

Langfuse et MLflow comptent chacun les tokens à leur façon (et pas au niveau du run).
Pour que les deux reçoivent les MÊMES métriques, les scorers déclarent ici chaque appel
LLM ; le pipeline lit le total par copie puis l'agrège pour le run.
Le compteur vit dans une `ContextVar` : chaque copie, traitée dans son propre contexte
(thread worker), a le sien.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from evaluation_dictee.utils.model_prices import cout_appel


@dataclass
class CopyUsage:
    """Consommation cumulée d'une copie (tous appels et essais confondus)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0  # 0 pour un modèle sans prix défini (voir `model_prices`)
    n_calls: int = 0


_METER: ContextVar[CopyUsage | None] = ContextVar("usage_meter", default=None)


def start_meter() -> CopyUsage:
    """Ouvre un compteur neuf pour la copie en cours et le rend courant.

    Returns:
        Le compteur, rempli au fil des appels par `record_usage`.
    """
    meter = CopyUsage()
    _METER.set(meter)
    return meter


def record_usage(model: str, response: Any) -> None:
    """Ajoute l'usage d'une réponse LLM au compteur courant (no-op sans compteur).

    Args:
        model: Nom du modèle appelé (sert à retrouver son prix).
        response: Réponse du SDK OpenAI ; son champ `usage` peut être absent.
    """
    meter = _METER.get()
    usage = getattr(response, "usage", None)
    if meter is None or usage is None:
        return
    entree = int(getattr(usage, "prompt_tokens", 0) or 0)
    sortie = int(getattr(usage, "completion_tokens", 0) or 0)
    meter.input_tokens += entree
    meter.output_tokens += sortie
    meter.n_calls += 1
    if (cout := cout_appel(model, entree, sortie)) is not None:
        meter.cost_eur += cout["total_cost"]


def usage_metrics(usages: list[CopyUsage]) -> dict[str, float]:
    """Métriques de consommation agrégées d'un run (identiques pour Langfuse et MLflow).

    Args:
        usages: Consommation de chaque copie traitée par le lancement.

    Returns:
        Totaux de tokens et de coût, et moyennes par copie (zéros si aucune copie).
    """
    n = len(usages)
    entree = sum(u.input_tokens for u in usages)
    sortie = sum(u.output_tokens for u in usages)
    cout = sum(u.cost_eur for u in usages)
    return {
        "n_copies_llm": float(n),
        "total_input_tokens": float(entree),
        "total_output_tokens": float(sortie),
        "total_cost_eur": cout,
        "mean_cost_eur_per_copy": cout / n if n else 0.0,
        "mean_output_tokens_per_copy": sortie / n if n else 0.0,
    }
