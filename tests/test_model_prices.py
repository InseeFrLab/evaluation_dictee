"""Prix par token des modèles llm.lab (suivi de coût Langfuse et MLflow)."""

from __future__ import annotations

import pytest

from evaluation_dictee.utils.model_prices import cout_appel


def test_cout_appel_modele_connu() -> None:
    cout = cout_appel("gemma4-26b-moe", 3541, 2747)
    assert cout is not None
    assert cout["total_cost"] == pytest.approx(0.001008958, rel=1e-6)
    assert cout["total_cost"] == pytest.approx(cout["input_cost"] + cout["output_cost"])


def test_cout_appel_insensible_a_la_casse() -> None:
    assert cout_appel("GEMMA4-26B-MOE", 10, 10) == cout_appel("gemma4-26b-moe", 10, 10)


def test_cout_appel_modele_inconnu() -> None:
    assert cout_appel("modele-inconnu", 10, 10) is None
