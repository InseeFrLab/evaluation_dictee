"""Suivi d'expériences : façade qui choisit le backend (Langfuse ou MLflow).

Le backend se choisit avec la variable d'environnement `TRACKING_BACKEND`
(`langfuse`, par défaut, ou `mlflow`). Les deux reçoivent les mêmes informations :
session (= un lancement), user, tags, trace par copie, scores par copie et métriques
agrégées du run. Voir `docs/comparatif_langfuse_mlflow.md` pour les différences.

Interface commune : `experiment_run`, `copy_trace`, `log_metrics`, `flush`,
`openai_client_class`, `load_chat_prompt`, `llm_trace_kwargs`.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any

from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking_common import (  # noqa: F401 — réexport pour les tests
    _run_metadata,
    _run_tags,
)

if TYPE_CHECKING:
    from evaluation_dictee.config import ExperimentConfig
    from evaluation_dictee.data.loaders import Copy

logger = get_logger(__name__)

BACKEND_ENV = "TRACKING_BACKEND"
BACKENDS = ("langfuse", "mlflow")


def get_backend() -> ModuleType:
    """Renvoie le module du backend choisi par `TRACKING_BACKEND` (import à la demande).

    Returns:
        Le module `tracking_langfuse` ou `tracking_mlflow`.

    Raises:
        ValueError: Si la variable contient une autre valeur que `langfuse` ou `mlflow`.
    """
    nom = os.environ.get(BACKEND_ENV, "langfuse").strip().lower()
    if nom == "langfuse":
        from evaluation_dictee.utils import tracking_langfuse

        return tracking_langfuse
    if nom == "mlflow":
        from evaluation_dictee.utils import tracking_mlflow

        return tracking_mlflow
    raise ValueError(f"{BACKEND_ENV}={nom!r} invalide : attendu {' ou '.join(BACKENDS)}.")


# ── Temps passé dans le suivi ────────────────────────────────────────────────
# Mesuré ICI, une seule fois, pour que Langfuse et MLflow soient chronométrés pareil.
# L'envoi réseau est asynchrone dans les deux outils : « chemin chaud » = ce que paie le
# pipeline pendant le run ; « flush » = la queue qu'il reste à vider à la fin.
_TIMINGS: dict[str, float] = defaultdict(float)
_COUNTS: dict[str, int] = defaultdict(int)
_TIMINGS_LOCK = threading.Lock()


@contextmanager
def _timed(label: str) -> Iterator[None]:
    """Ajoute la durée du bloc au poste `label` (thread-safe)."""
    debut = time.perf_counter()
    try:
        yield
    finally:
        with _TIMINGS_LOCK:
            _TIMINGS[label] += time.perf_counter() - debut
            _COUNTS[label] += 1


def reset_timings() -> None:
    """Remet à zéro les compteurs de temps (début d'un run)."""
    with _TIMINGS_LOCK:
        _TIMINGS.clear()
        _COUNTS.clear()


def timing_summary() -> dict[str, float]:
    """Temps cumulé (s) par poste, plus `total_s` = chemin chaud + flush.

    Returns:
        {poste: secondes}, postes : `run_open_close`, `trace_open`, `trace_close`,
        `trace_update`, `trace_score`, `log_metrics`, `flush`. Les durées de `trace_*`
        sont cumulées sur tous les threads (somme, pas temps mural).
    """
    with _TIMINGS_LOCK:
        resume = dict(_TIMINGS)
    resume["total_s"] = sum(resume.values())
    return resume


class _TimedTrace:
    """Enveloppe d'une trace de copie qui chronomètre `update` et `score_trace`."""

    def __init__(self, trace: Any) -> None:
        """Mémorise la trace du backend."""
        self._trace = trace

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Met à jour la trace (chronométré)."""
        with _timed("trace_update"):
            return self._trace.update(*args, **kwargs)

    def score_trace(self, *args: Any, **kwargs: Any) -> Any:
        """Attache un score à la trace (chronométré)."""
        with _timed("trace_score"):
            return self._trace.score_trace(*args, **kwargs)

    def __getattr__(self, nom: str) -> Any:
        """Délègue le reste au span du backend."""
        return getattr(self._trace, nom)


@contextmanager
def experiment_run(config: ExperimentConfig) -> Iterator[str]:
    """Ouvre le contexte de suivi d'un lancement (voir le backend actif)."""
    reset_timings()
    manager = get_backend().experiment_run(config)
    with _timed("run_open_close"):
        session_id = manager.__enter__()
    try:
        yield session_id
    except BaseException as exc:
        with _timed("run_open_close"):
            if not manager.__exit__(type(exc), exc, exc.__traceback__):
                raise
    else:
        with _timed("run_open_close"):
            manager.__exit__(None, None, None)


@contextmanager
def copy_trace(copy: Copy) -> Iterator[Any]:
    """Ouvre la trace d'une copie (voir le backend actif) ; ouverture et fermeture chronométrées."""
    manager = get_backend().copy_trace(copy)
    with _timed("trace_open"):
        trace = manager.__enter__()
    wrapped = _TimedTrace(trace) if trace is not None else None
    try:
        yield wrapped
    except BaseException as exc:
        with _timed("trace_close"):
            if not manager.__exit__(type(exc), exc, exc.__traceback__):
                raise
    else:
        with _timed("trace_close"):
            manager.__exit__(None, None, None)


def log_metrics(config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    """Enregistre les métriques agrégées du run (voir le backend actif)."""
    logger.info(
        "Métriques envoyées à %s : %s",
        get_backend().__name__.rsplit("_", 1)[-1],
        ", ".join(sorted(metrics)),
    )
    with _timed("log_metrics"):
        get_backend().log_metrics(config, metrics)


def flush() -> None:
    """Vide la file d'envoi asynchrone du backend actif (à appeler avant de quitter)."""
    with _timed("flush"):
        get_backend().flush()


def openai_client_class() -> type:
    """Classe de client OpenAI à utiliser (instrumentée ou non selon le backend)."""
    return get_backend().openai_client_class()


def load_chat_prompt(name: str, fallback: list[Any], variables: dict[str, object]) -> list[Any]:
    """Charge et compile un prompt versionné du backend actif."""
    return get_backend().load_chat_prompt(name, fallback, variables)


def llm_trace_kwargs(name: str, fallback: list[Any]) -> dict[str, Any]:
    """Arguments d'appel LLM liant la génération à la version du prompt (selon le backend)."""
    return get_backend().llm_trace_kwargs(name, fallback)
