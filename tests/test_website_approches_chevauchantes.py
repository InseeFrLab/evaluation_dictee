"""Tests de `modeles_exportes()` quand une approche préfixe une autre approche.

`dictee_end2end_comptage_exemples` (end-to-end en une étape, mais avec deux options
de prompt retenues) commence par `dictee_end2end`, le nom de l'approche de référence.
Sans priorité à la base la plus spécifique, un run de la première approche matcherait
AUSSI la seconde, produisant un modèle fantôme (`comptage_exemples_<modele>`) en plus
du vrai modèle correctement déduit.
"""

import importlib
import sys
from pathlib import Path

import pytest

WEBSITE_DIR = Path(__file__).resolve().parents[1] / "website"


@pytest.fixture
def analyse(monkeypatch):
    """Importe `_analyse` avec `_noms_exportes` gelée sur une liste contrôlée."""
    sys.path.insert(0, str(WEBSITE_DIR))
    try:
        mod = importlib.import_module("_analyse")
        importlib.reload(mod)

        def _figer(noms: list[str]) -> None:
            monkeypatch.setattr(mod, "_EXPORTES", sorted(noms))

        mod._figer_noms_exportes_pour_test = _figer
        yield mod
    finally:
        sys.path.remove(str(WEBSITE_DIR))
        sys.modules.pop("_analyse", None)


def test_approche_qui_en_prefixe_une_autre_ne_cree_pas_de_fantome(analyse) -> None:
    analyse.APPROCHES = {
        "end-to-end": "dictee_end2end",
        "end-to-end (comptage+exemples)": "dictee_end2end_comptage_exemples",
    }
    analyse._figer_noms_exportes_pour_test(
        [
            "dictee_end2end_qwen3-6-35b-moe",
            "dictee_end2end_qwen3-8-27b",
            "dictee_end2end_comptage_exemples_qwen3-8-27b",
        ]
    )
    assert analyse.modeles_exportes() == ["qwen3-6-35b-moe", "qwen3-8-27b"]


def test_approches_disponibles_par_modele_reste_correct(analyse) -> None:
    analyse.APPROCHES = {
        "end-to-end": "dictee_end2end",
        "end-to-end (comptage+exemples)": "dictee_end2end_comptage_exemples",
    }
    analyse._figer_noms_exportes_pour_test(
        [
            "dictee_end2end_qwen3-6-35b-moe",
            "dictee_end2end_qwen3-8-27b",
            "dictee_end2end_comptage_exemples_qwen3-8-27b",
        ]
    )
    assert analyse.approches_du_modele("qwen3-8-27b") == [
        "end-to-end",
        "end-to-end (comptage+exemples)",
    ]
    # Non promu sur cette approche : ne doit PAS apparaître, ni fausser le modèle.
    assert analyse.approches_du_modele("qwen3-6-35b-moe") == ["end-to-end"]


def test_base_la_plus_specifique_prend_la_plus_longue(analyse) -> None:
    analyse.APPROCHES = {
        "court": "dictee_end2end",
        "long": "dictee_end2end_comptage_exemples",
    }
    assert (
        analyse._base_la_plus_specifique("dictee_end2end_comptage_exemples_qwen3-8-27b")
        == "dictee_end2end_comptage_exemples"
    )
    assert analyse._base_la_plus_specifique("dictee_end2end_qwen3-8-27b") == "dictee_end2end"
    assert analyse._base_la_plus_specifique("dictee_autre_chose") is None
