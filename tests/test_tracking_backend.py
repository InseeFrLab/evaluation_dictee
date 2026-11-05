"""Choix du backend de suivi et mesure de consommation (métriques communes aux deux outils)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation_dictee.utils import tracking
from evaluation_dictee.utils.usage import record_usage, start_meter, usage_metrics


def test_backend_par_defaut_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tracking.BACKEND_ENV, raising=False)
    assert tracking.get_backend().__name__.endswith("tracking_langfuse")


def test_backend_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(tracking.BACKEND_ENV, "MLflow")
    assert tracking.get_backend().__name__.endswith("tracking_mlflow")


def test_backend_inconnu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(tracking.BACKEND_ENV, "wandb")
    with pytest.raises(ValueError, match="invalide"):
        tracking.get_backend()


def _reponse(entree: int, sortie: int) -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=entree, completion_tokens=sortie))


def test_usage_cumule_les_essais_d_une_copie() -> None:
    meter = start_meter()
    record_usage("gemma4-26b-moe", _reponse(3000, 2000))
    record_usage("gemma4-26b-moe", _reponse(3000, 500))  # retry
    assert (meter.input_tokens, meter.output_tokens, meter.n_calls) == (6000, 2500, 2)
    assert meter.cost_eur > 0


def test_usage_modele_sans_prix_compte_les_tokens_sans_cout() -> None:
    meter = start_meter()
    record_usage("modele-inconnu", _reponse(10, 5))
    assert meter.input_tokens == 10
    assert meter.cost_eur == 0.0


def test_usage_sans_champ_usage_ignore() -> None:
    meter = start_meter()
    record_usage("gemma4-26b-moe", SimpleNamespace(usage=None))
    assert meter.n_calls == 0


def test_usage_metrics_vide_et_agrege() -> None:
    assert usage_metrics([])["mean_cost_eur_per_copy"] == 0.0
    a, b = start_meter(), start_meter()
    a.input_tokens, a.output_tokens, a.cost_eur = 100, 50, 0.002
    b.input_tokens, b.output_tokens, b.cost_eur = 300, 150, 0.004
    m = usage_metrics([a, b])
    assert m["total_input_tokens"] == 400
    assert m["total_cost_eur"] == pytest.approx(0.006)
    assert m["mean_output_tokens_per_copy"] == 100
