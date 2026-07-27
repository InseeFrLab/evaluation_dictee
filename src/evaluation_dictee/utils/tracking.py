"""Suivi d'expériences Langfuse : session (= un lancement) → trace (= une copie) → generation.

Le `session_id` est UNIQUE PAR LANCEMENT (nom du run + horodatage) : relancer la
même config ouvre une nouvelle session ; le tag `run:<nom>` regroupe les lancements.
Pas de trace englobante — chaque copie est une trace autonome, inspectable seule.
Tout est no-op si Langfuse est indisponible (ne bloque pas le dev local).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langfuse import get_client, propagate_attributes

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.utils.logging import get_logger

if TYPE_CHECKING:
    from langfuse._client.span import LangfuseSpan

    from evaluation_dictee.data.loaders import Copy

logger = get_logger(__name__)


def _run_metadata(config: ExperimentConfig) -> dict[str, str]:
    """Paramètres de l'expérience en chaînes (contrainte metadata Langfuse).

    Args:
        config: Configuration de l'expérience à décrire.

    Returns:
        Dictionnaire {modèle, méthode, schéma, n_few_shot, corpus} en chaînes.
    """
    return {
        "model": config.model.name,
        "method": config.prompt.method,
        "scheme": config.grid.scheme,
        "n_few_shot": str(config.prompt.n_few_shot),
        "corpus": config.data.corpus,
    }


def _run_tags(config: ExperimentConfig) -> list[str]:
    """Tags de filtrage/regroupement dans l'UI Langfuse.

    Args:
        config: Configuration de l'expérience.

    Returns:
        Liste de tags (méthode, corpus, `scheme:…`, `model:…`, `run:<nom>`).
    """
    return [
        config.prompt.method,
        config.data.corpus,
        f"scheme:{config.grid.scheme}",
        f"model:{config.model.name}",
        # session_id unique par lancement : ce tag regroupe les lancements d'une config
        f"run:{config.name}",
    ]


def _user_id() -> str | None:
    """Identifiant utilisateur Langfuse = namespace Kubernetes (KUBERNETES_NAMESPACE), ou None.

    Returns:
        La valeur de `KUBERNETES_NAMESPACE`, ou None si absente ou vide.
    """
    return os.environ.get("KUBERNETES_NAMESPACE") or None


def _launch_session_id(config: ExperimentConfig) -> str:
    """Session Langfuse UNIQUE PAR LANCEMENT : « <nom>-AAAAMMJJ-HHMMSS » (triable dans le temps).

    Args:
        config: Configuration de l'expérience (fournit `config.name`).

    Returns:
        L'identifiant de session, nom du run suffixé de l'horodatage courant.
    """
    return f"{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"


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
