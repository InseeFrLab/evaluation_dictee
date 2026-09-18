"""Tests de la promotion manuelle d'un modèle vers son meilleur prompt (MODELES_PROMUS).

Ancien design (retiré) : deviner automatiquement une approche supplémentaire depuis un
nom de fichier, avec priorité au préfixe le plus long en cas de chevauchement. Trop
difficile à expliquer, et le vrai problème reste entier de toute façon : un fichier
promu placé à la racine de `predictions/` commence par `dictee_end2end_` (comme un
run de référence) DÈS QU'IL EXISTE PHYSIQUEMENT, indépendamment de tout mécanisme de
découverte automatique ou manuel. Le nouveau design : `APPROCHES` ne bouge jamais (2
entrées, end-to-end / two-stage), la promotion est déclarée à la main dans
`MODELES_PROMUS`, et les noms de run réservés à une promotion (`LIBELLES_PROMOTION`)
sont explicitement exclus de la découverte de modèles.
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


def test_un_fichier_promu_a_la_racine_ne_cree_pas_de_fantome(analyse) -> None:
    """Le cas concret qui a fait échouer la première tentative de correctif.

    `dictee_end2end_comptage_exemples_qwen3-8-27b` existe à la racine de
    predictions/ (obligatoire pour alimenter le site) et commence par
    `dictee_end2end_` : sans l'exclusion explicite, il serait lu comme un modèle
    fantôme `comptage_exemples_qwen3-8-27b`.
    """
    analyse._figer_noms_exportes_pour_test(
        [
            "dictee_end2end_qwen3-8-27b",
            "dictee_end2end_comptage_exemples_qwen3-8-27b",
        ]
    )
    assert analyse.modeles_exportes() == ["qwen3-8-27b"]


def test_promotion_non_declaree_ne_change_rien(analyse) -> None:
    """MODELES_PROMUS vide : approches_du_modele reste identique à avant ce mécanisme."""
    analyse.MODELES_PROMUS = {}
    analyse._figer_noms_exportes_pour_test(["dictee_end2end_qwen3-8-27b"])
    assert analyse.approches_du_modele("qwen3-8-27b") == ["end-to-end"]
    assert analyse.approche_promue("qwen3-8-27b") is None


def test_promotion_declaree_et_exportee(analyse) -> None:
    analyse.MODELES_PROMUS = {"qwen3-8-27b": "dictee_end2end_comptage_exemples"}
    analyse._figer_noms_exportes_pour_test(
        [
            "dictee_end2end_qwen3-8-27b",
            "dictee_end2end_comptage_exemples_qwen3-8-27b",
        ]
    )
    promo = analyse.approche_promue("qwen3-8-27b")
    assert promo == (
        "end-to-end (comptage+exemples)",
        "dictee_end2end_comptage_exemples_qwen3-8-27b",
    )
    # La promotion reste hors d'approches_du_modele : elle ne doit jamais peser sur
    # le calcul de complétude end-to-end / two-stage.
    assert analyse.approches_du_modele("qwen3-8-27b") == ["end-to-end"]


def test_promotion_declaree_mais_pas_encore_exportee(analyse) -> None:
    """Le run n'existe pas encore sur S3 : la promotion ne doit rien afficher de faux."""
    analyse.MODELES_PROMUS = {"gemma4-26b-moe": "dictee_end2end_exemples"}
    analyse._figer_noms_exportes_pour_test(["dictee_end2end_gemma4-26b-moe"])
    assert analyse.approche_promue("gemma4-26b-moe") is None


def test_promotion_n_affecte_pas_la_completude(analyse) -> None:
    """Un modèle non promu ne doit JAMAIS être signalé « incomplet » pour autant.

    Sans cette isolation, aucun modèle ne serait jamais complet dès qu'un seul est
    promu (le prompt gagnant diffère par modèle, décision D10) : `qwen3-8-27b` n'aura
    jamais « exemples seul », gemma4-26b-moe n'aura jamais « comptage+exemples ».

    `MODELES_COMPLETS` est une constante figée AU CHARGEMENT du module (avant que ce
    test ne déclare sa promotion) : en usage réel `MODELES_PROMUS` est écrit dans le
    code source avant tout rendu, donc figé au bon moment ; ici on revérifie la même
    formule après coup plutôt que de relire la constante déjà calculée.
    """
    analyse.MODELES_PROMUS = {"qwen3-8-27b": "dictee_end2end_comptage_exemples"}
    analyse._figer_noms_exportes_pour_test(
        [
            "dictee_end2end_qwen3-8-27b",
            "dictee_two_stage_qwen3-8-27b",
            "dictee_end2end_comptage_exemples_qwen3-8-27b",
        ]
    )
    complets = [
        m
        for m in analyse.modeles_exportes()
        if len(analyse.approches_du_modele(m)) == len(analyse.APPROCHES)
    ]
    assert complets == ["qwen3-8-27b"]


def test_deux_promotions_distinctes_ne_se_confondent_pas(analyse) -> None:
    analyse.MODELES_PROMUS = {
        "qwen3-8-27b": "dictee_end2end_comptage_exemples",
        "gemma4-26b-moe": "dictee_end2end_exemples",
    }
    analyse._figer_noms_exportes_pour_test(
        [
            "dictee_end2end_qwen3-8-27b",
            "dictee_end2end_gemma4-26b-moe",
            "dictee_end2end_comptage_exemples_qwen3-8-27b",
            "dictee_end2end_exemples_gemma4-26b-moe",
        ]
    )
    assert analyse.modeles_exportes() == ["gemma4-26b-moe", "qwen3-8-27b"]
    assert analyse.approche_promue("qwen3-8-27b")[0] == "end-to-end (comptage+exemples)"
    assert analyse.approche_promue("gemma4-26b-moe")[0] == "end-to-end (exemples)"


def test_couleur_promotion_ne_collisionne_pas_avec_end_to_end(analyse) -> None:
    """Une approche promue doit avoir sa propre teinte, pas celle de end-to-end."""
    c1 = analyse.couleur_run("end-to-end", "qwen3-8-27b", modeles=["qwen3-8-27b"])
    c2 = analyse.couleur_run(
        "end-to-end (comptage+exemples)", "qwen3-8-27b", modeles=["qwen3-8-27b"]
    )
    assert c1 != c2
