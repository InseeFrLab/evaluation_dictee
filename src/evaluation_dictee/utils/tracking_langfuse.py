"""Backend Langfuse du suivi d'expériences (voir `utils/tracking.py`).

Suivi d'expériences Langfuse : session (= un lancement) → trace (= une copie) → generation.

Le `session_id` est UNIQUE PAR LANCEMENT (nom du run + horodatage) : relancer la
même config ouvre une nouvelle session ; le tag `run:<nom>` regroupe les lancements.
Pas de trace englobante — chaque copie est une trace autonome, inspectable seule.
Tout est no-op si Langfuse est indisponible (ne bloque pas le dev local).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langfuse import get_client, propagate_attributes

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking_common import (
    _launch_session_id,
    _run_metadata,
    _run_tags,
    _user_id,
)

if TYPE_CHECKING:
    from langfuse._client.span import LangfuseSpan

    from evaluation_dictee.data.loaders import Copy

logger = get_logger(__name__)


@contextmanager
def experiment_run(config: ExperimentConfig) -> Iterator[str]:
    """Ouvre le contexte Langfuse d'un run (session neuve, attributs propagés aux traces).

    Propage user_id, session_id, tags et metadata à toutes les traces créées dans
    le bloc. Si Langfuse est indisponible, le contexte reste ouvert sans tracer.

    Args:
        config: Configuration de l'expérience à tracer.

    Yields:
        L'identifiant de session généré pour ce lancement.
    """
    session_id = _launch_session_id(config)
    user_id = _user_id()
    logger.info("Langfuse : nouvelle session « %s » (user_id=%s).", session_id, user_id)
    metadata = {**_run_metadata(config), "run": config.name}
    try:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=_run_tags(config),
            metadata=metadata,
        ):
            yield session_id
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        logger.warning("Langfuse indisponible : le run ne sera pas tracé.", exc_info=True)
        yield session_id


@contextmanager
def copy_trace(copy: Copy) -> Iterator[LangfuseSpan | None]:
    """Ouvre une trace Langfuse pour une copie (les générations LLM s'y imbriquent).

    Args:
        copy: Copie à tracer (fournit `copy_id` et `item_ids`).

    Yields:
        Le span Langfuse de la copie, ou None si Langfuse est indisponible.
    """
    langfuse = get_client()
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"copy:{copy.copy_id}",
            input={"copy_id": copy.copy_id, "n_items": len(copy.item_ids)},
        ) as span:
            yield span
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        logger.warning("Trace de la copie %s non créée (Langfuse indisponible).", copy.copy_id)
        yield None


def log_metrics(config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    """Enregistre les métriques agrégées du run (metadata + Scores Langfuse). No-op si absent.

    Les valeurs numériques deviennent des Scores Langfuse (type NUMERIC) sur la
    trace de synthèse ; toutes les métriques sont aussi jointes en metadata.

    Args:
        config: Configuration de l'expérience (paramètres joints en metadata).
        metrics: Métriques agrégées du run ; seules les valeurs int/float sont
            converties en Scores.
    """
    langfuse = get_client()
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    metadata = {**_run_metadata(config), **{k: str(v) for k, v in metrics.items()}}
    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"run-summary:{config.name}",
            output=numeric,
            metadata=metadata,
        ):
            for name, value in numeric.items():
                langfuse.score_current_trace(name=name, value=value, data_type="NUMERIC")
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        logger.warning("Métriques non enregistrées dans Langfuse (indisponible).", exc_info=True)


def flush() -> None:
    """Vide la file d'envoi asynchrone de Langfuse (à appeler avant de quitter)."""
    get_client().flush()


def openai_client_class() -> type:
    """Client OpenAI instrumenté Langfuse : ses appels deviennent des générations.

    Returns:
        La classe `OpenAI` de `langfuse.openai`.
    """
    from langfuse.openai import OpenAI

    return OpenAI


def load_chat_prompt(name: str, fallback: list[Any], variables: dict[str, object]) -> list[Any]:
    """Récupère et compile un prompt chat Langfuse (le repli local est géré par l'appelant).

    Args:
        name: Nom du prompt versionné dans Langfuse.
        fallback: Template local renvoyé par Langfuse s'il est injoignable.
        variables: Valeurs à injecter dans les `{{...}}` du template.

    Returns:
        Les messages compilés.
    """
    return get_client().get_prompt(name, type="chat", fallback=fallback).compile(**variables)


def llm_trace_kwargs(name: str, fallback: list[Any]) -> dict[str, Any]:
    """Arguments à passer à l'appel LLM pour lier la génération à la version du prompt.

    Args:
        name: Nom du prompt versionné dans Langfuse.
        fallback: Template local utilisé si Langfuse est injoignable.

    Returns:
        `{"langfuse_prompt": <prompt>}`, ou {} si Langfuse est indisponible (repli local).
    """
    try:
        prompt = get_client().get_prompt(name, type="chat", fallback=fallback)
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        return {}
    return {} if prompt.is_fallback else {"langfuse_prompt": prompt}
