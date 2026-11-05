"""Backend MLflow du suivi d'expériences (voir `utils/tracking.py`).

Suivi d'expériences MLflow : run (= un lancement) → traces (= une copie) → spans LLM.

Équivalent MLflow de l'ancien suivi Langfuse. Correspondances :
- session Langfuse → `session_id` de trace MLflow (métadonnée `mlflow.trace.session`) ET
  run MLflow (porte les paramètres et les métriques agrégées du lancement) ;
- trace par copie → trace MLflow autonome (span racine `copy:<id>`) ;
- score de trace → assessment « feedback » MLflow ;
- user_id → `user` de trace MLflow.

Le `session_id` est UNIQUE PAR LANCEMENT (nom du run + horodatage). MLflow n'a pas
d'équivalent de `propagate_attributes` : le contexte du lancement est donc porté par une
`ContextVar` (copiée dans les threads workers) et réappliqué par `copy_trace`.
Tout est no-op si MLflow est indisponible (ne bloque pas le dev local).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.genai
from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanStatusCode
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.model_prices import cout_appel
from evaluation_dictee.utils.tracking_common import (
    _launch_session_id,
    _run_metadata,
    _trace_tags,
    _user_id,
)

if TYPE_CHECKING:
    from evaluation_dictee.data.loaders import Copy

logger = get_logger(__name__)

# Alias du registre servi aux runs (équivalent du label Langfuse « production »).
PROMPT_ALIAS = "production"

# Expérience MLflow cible ; surchargeable (ex. « tracking-test » lors des essais).
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
DEFAULT_EXPERIMENT = "evaluation_dictee"


@dataclass(frozen=True)
class _RunContext:
    """Attributs du lancement courant, à réappliquer sur chaque trace de copie."""

    session_id: str
    user_id: str | None
    run_id: str | None
    tags: dict[str, str] = field(default_factory=dict)


# Pas de propagation automatique côté MLflow : contexte explicite, copié dans les workers
# par `contextvars.copy_context()` (déjà fait dans `pipeline/benchmark.py`).
_RUN_CONTEXT: ContextVar[_RunContext | None] = ContextVar("run_context", default=None)


def _ajouter_cout(span: Any) -> None:
    """Pose `mlflow.llm.cost` sur un span LLM dont le modèle a un prix défini.

    MLflow ne chiffre que les modèles de son catalogue public : nos modèles `llm.lab` y
    sont absents (coût None). Ce processeur, appelé à la fin de chaque span, applique les
    prix de `utils/model_prices.py` ; MLflow agrège ensuite le coût au niveau de la trace.

    Args:
        span: Span MLflow en cours de finalisation.
    """
    if span.get_attribute(SpanAttributeKey.LLM_COST) is not None:
        return
    model = span.get_attribute(SpanAttributeKey.MODEL)
    usage = span.get_attribute(SpanAttributeKey.CHAT_USAGE)
    if not model or not usage:
        return
    cout = cout_appel(
        model,
        usage.get(TokenUsageKey.INPUT_TOKENS, 0),
        usage.get(TokenUsageKey.OUTPUT_TOKENS, 0),
    )
    if cout is not None:
        span.set_attribute(SpanAttributeKey.LLM_COST, cout)


@contextmanager
def experiment_run(config: ExperimentConfig) -> Iterator[str]:
    """Ouvre le contexte MLflow d'un lancement (run neuf, attributs réappliqués aux traces).

    Active l'autolog OpenAI (chaque appel LLM devient un span), ouvre un run MLflow
    nommé d'après la session et publie le contexte lu par `copy_trace`. Si MLflow est
    indisponible, le contexte reste ouvert sans tracer.

    Args:
        config: Configuration de l'expérience à tracer.

    Yields:
        L'identifiant de session généré pour ce lancement.
    """
    session_id = _launch_session_id(config)
    user_id = _user_id()
    logger.info("MLflow : nouvelle session « %s » (user=%s).", session_id, user_id)
    stack = ExitStack()
    token = None
    try:
        mlflow.set_experiment(os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT))
        mlflow.openai.autolog()
        mlflow.tracing.configure(span_processors=[_ajouter_cout])
        run = stack.enter_context(mlflow.start_run(run_name=session_id))
        tags = _trace_tags(config)
        mlflow.log_params(_run_metadata(config))
        mlflow.set_tags({**tags, "session_id": session_id})
        token = _RUN_CONTEXT.set(
            _RunContext(session_id, user_id, run.info.run_id, tags=tags),
        )
    except Exception:  # pragma: no cover - repli défensif si MLflow indisponible
        logger.warning("MLflow indisponible : le run ne sera pas tracé.", exc_info=True)
    try:
        yield session_id
    finally:
        if token is not None:
            _RUN_CONTEXT.reset(token)
        try:
            stack.close()
        except Exception:  # pragma: no cover
            logger.warning("Fermeture du run MLflow en échec.", exc_info=True)


class CopyTrace:
    """Adaptateur d'un span MLflow vers l'interface de trace attendue par le benchmark.

    Expose `update(...)` et `score_trace(...)` comme le span Langfuse, pour que
    `pipeline/benchmark.py` n'ait pas à changer.
    """

    def __init__(self, span: Any) -> None:
        """Mémorise le span racine de la copie."""
        self._span = span

    def update(
        self,
        output: Any = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Met à jour la sortie et/ou le statut du span (`level="ERROR"` → statut ERROR)."""
        if output is not None:
            self._span.set_outputs(output)
        if level == "ERROR":
            self._span.set_status(SpanStatusCode.ERROR)
            if status_message:
                self._span.set_attribute("error_message", status_message)

    def score_trace(self, name: str, value: float) -> None:
        """Attache un score numérique à la trace (assessment « feedback » MLflow)."""
        mlflow.log_feedback(
            trace_id=self._span.trace_id,
            name=name,
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.CODE, source_id="evaluation_dictee"
            ),
        )


@contextmanager
def copy_trace(copy: Copy) -> Iterator[CopyTrace | None]:
    """Ouvre une trace MLflow pour une copie (les appels LLM s'y imbriquent).

    Args:
        copy: Copie à tracer (fournit `copy_id` et `item_ids`).

    Yields:
        L'adaptateur de la trace de la copie, ou None si MLflow est indisponible.
    """
    ctx = _RUN_CONTEXT.get()
    stack = ExitStack()
    trace: CopyTrace | None = None
    try:
        span = stack.enter_context(
            mlflow.start_span(
                name=f"copy:{copy.copy_id}",
                span_type="CHAIN",
                run_id=ctx.run_id if ctx else None,
            )
        )
        span.set_inputs({"copy_id": copy.copy_id, "n_items": len(copy.item_ids)})
        if ctx is not None:
            mlflow.update_current_trace(session_id=ctx.session_id, user=ctx.user_id, tags=ctx.tags)
        trace = CopyTrace(span)
    except Exception:  # pragma: no cover - repli défensif si MLflow indisponible
        logger.warning("Trace de la copie %s non créée (MLflow indisponible).", copy.copy_id)
    with stack:
        yield trace


def log_metrics(config: ExperimentConfig, metrics: dict[str, Any]) -> None:
    """Enregistre les métriques agrégées sur le run MLflow courant. No-op si absent.

    Les valeurs numériques finies deviennent des métriques MLflow ; les autres sont
    jointes en tags du run, pour garder une trace lisible de tout ce qui a été calculé.

    Args:
        config: Configuration de l'expérience (non utilisée : params déjà au run).
        metrics: Métriques agrégées du run.
    """
    numeric = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    }
    try:
        mlflow.log_metrics(numeric)
        autres = {f"metric.{k}": str(v) for k, v in metrics.items() if k not in numeric}
        if autres:
            mlflow.set_tags(autres)
    except Exception:  # pragma: no cover - repli défensif si MLflow indisponible
        logger.warning("Métriques non enregistrées dans MLflow (indisponible).", exc_info=True)


def flush() -> None:
    """Vide la file d'envoi asynchrone des traces MLflow (à appeler avant de quitter)."""
    try:
        mlflow.flush_trace_async_logging()
    except Exception:  # pragma: no cover
        logger.warning("Flush MLflow en échec.", exc_info=True)


def openai_client_class() -> type:
    """Client OpenAI standard : `mlflow.openai.autolog()` l'instrumente (pas de wrapper).

    Returns:
        La classe `OpenAI` du SDK officiel.
    """
    from openai import OpenAI

    return OpenAI


def load_chat_prompt(name: str, fallback: list[Any], variables: dict[str, object]) -> list[Any]:
    """Charge et compile un prompt chat du registre MLflow (alias `production`).

    Chargé dans la trace de la copie, MLflow lie automatiquement la version à la trace
    (tag `mlflow.linkedPrompts`).

    Args:
        name: Nom du prompt dans le registre MLflow.
        fallback: Non utilisé (le repli local est géré par l'appelant).
        variables: Valeurs à injecter dans les `{{...}}` du template.

    Returns:
        Les messages compilés.
    """
    return mlflow.genai.load_prompt(f"prompts:/{name}@{PROMPT_ALIAS}").format(**variables)


def llm_trace_kwargs(name: str, fallback: list[Any]) -> dict[str, Any]:
    """Aucun argument : le lien prompt ↔ trace est posé par `load_chat_prompt`.

    Args:
        name: Non utilisé.
        fallback: Non utilisé.

    Returns:
        Un dictionnaire vide.
    """
    return {}
