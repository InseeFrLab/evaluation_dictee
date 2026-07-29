"""Tests des outils statistiques et du rapport par item."""

import math

import pandas as pd
import pytest

from evaluation_dictee.evaluation.calibration import (
    expected_calibration_error,
    reliability_bins,
)
from evaluation_dictee.evaluation.report import (
    disagreement_decomposition,
    per_copy_metrics,
    per_item_metrics,
)
from evaluation_dictee.evaluation.statistics import (
    cluster_bootstrap,
    design_effect,
    kappa_interval,
    kappa_interval_clustered,
    wilson_interval,
)


# ── Wilson ────────────────────────────────────────────────────────────────────
def test_wilson_proportion_mediane() -> None:
    ci = wilson_interval(50, 100)
    assert ci.estimate == 0.5
    assert ci.lower < 0.5 < ci.upper
    assert 0.39 < ci.lower < 0.41  # valeur connue ≈ 0.404


def test_wilson_bornes_dans_0_1() -> None:
    ci = wilson_interval(100, 100)
    assert ci.upper <= 1.0
    ci0 = wilson_interval(0, 100)
    assert ci0.lower >= 0.0


def test_wilson_n_nul_leve_erreur() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


# ── Kappa et son IC analytique ────────────────────────────────────────────────
def test_kappa_interval_encadre_estimation() -> None:
    y_true = ["1"] * 60 + ["9"] * 40
    y_pred = ["1"] * 55 + ["9"] * 5 + ["9"] * 35 + ["1"] * 5
    ci = kappa_interval(y_true, y_pred)
    assert 0 < ci.estimate < 1
    assert ci.lower < ci.estimate < ci.upper


def test_kappa_interval_accord_parfait_interval_serre() -> None:
    y = ["1"] * 50 + ["9"] * 50
    ci = kappa_interval(y, y)
    assert ci.estimate == pytest.approx(1.0)
    assert ci.upper - ci.lower < 1e-9


def test_kappa_interval_une_seule_categorie_degenere() -> None:
    # p_e = 1 : la variance n'est pas définie et sklearn renvoie un kappa NaN.
    ci = kappa_interval(["1"] * 50, ["1"] * 50)
    assert math.isnan(ci.estimate)
    assert math.isnan(ci.lower)
    assert math.isnan(ci.upper)


def test_kappa_interval_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        kappa_interval(["1", "9"], ["1"])


def test_kappa_interval_series_vides() -> None:
    with pytest.raises(ValueError):
        kappa_interval([], [])


# ── Design effect ─────────────────────────────────────────────────────────────
def test_design_effect_grappes_homogenes_est_maximal() -> None:
    # Deux copies parfaitement homogènes : toute la variance est inter-copies,
    # l'ICC vaut 1 et le design effect atteint la taille de grappe.
    grappes = ["A"] * 10 + ["B"] * 10
    indicatrice = [0] * 10 + [1] * 10
    assert design_effect(pd.Series(indicatrice), pd.Series(grappes)) == pytest.approx(
        10.0, rel=1e-6
    )


def test_design_effect_sans_variance_vaut_un() -> None:
    grappes = ["A"] * 5 + ["B"] * 5
    assert design_effect(pd.Series([1] * 10), pd.Series(grappes)) == 1.0


def test_design_effect_minore_a_un() -> None:
    # Grappes volontairement mélangées : l'ICC estimé est négatif, on borne à 1.
    grappes = ["A", "A", "B", "B", "C", "C"]
    assert design_effect(pd.Series([0, 1, 0, 1, 0, 1]), pd.Series(grappes)) == 1.0


def test_design_effect_propre_a_chaque_indicatrice() -> None:
    # Deux indicatrices sur les MÊMES grappes n'ont pas le même design effect :
    # la première est parfaitement groupée, la seconde alterne dans chaque copie.
    grappes = pd.Series(["A"] * 4 + ["B"] * 4)
    groupee = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    alternee = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    assert design_effect(groupee, grappes) > design_effect(alternee, grappes)
    assert design_effect(alternee, grappes) == 1.0


def test_design_effect_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        design_effect(pd.Series([0, 1]), pd.Series(["A"]))


# ── Wilson corrigé du design effect ───────────────────────────────────────────
def test_wilson_deff_elargit_intervalle() -> None:
    brut = wilson_interval(500, 1000)
    corrige = wilson_interval(500, 1000, deff=4.0)
    assert corrige.estimate == brut.estimate  # la valeur ponctuelle ne change pas
    largeur_brute = brut.upper - brut.lower
    largeur_corrigee = corrige.upper - corrige.lower
    # deff = 4 → intervalle élargi d'un facteur ≈ √4 = 2.
    assert largeur_corrigee / largeur_brute == pytest.approx(2.0, rel=0.02)


def test_wilson_deff_neutre_a_un() -> None:
    assert wilson_interval(300, 1000, deff=1.0) == wilson_interval(300, 1000)


def test_wilson_deff_invalide() -> None:
    with pytest.raises(ValueError):
        wilson_interval(500, 1000, deff=0.5)


# ── Kappa par bootstrap de grappes ────────────────────────────────────────────
def _df_kappa_groupe() -> pd.DataFrame:
    """20 copies de 10 items : la moitié bien codées, l'autre systématiquement ratée."""
    lignes = []
    for c in range(20):
        for i in range(10):
            vrai = "1" if i % 2 else "9"
            predit = vrai if c % 2 == 0 else ("9" if vrai == "1" else "1")
            lignes.append({"copy_id": f"c{c}", "y_true": vrai, "y_pred": predit})
    return pd.DataFrame(lignes)


def test_kappa_clustered_encadre_estimation() -> None:
    df = _df_kappa_groupe()
    ci = kappa_interval_clustered(df["y_true"], df["y_pred"], df["copy_id"], n_boot=200)
    assert ci.lower <= ci.estimate <= ci.upper


def test_kappa_clustered_plus_large_que_delta_method() -> None:
    # Les items étant fortement corrélés dans chaque copie, le bootstrap par
    # grappes doit produire un intervalle plus large que la delta method, qui
    # suppose l'indépendance.
    df = _df_kappa_groupe()
    grappes = kappa_interval_clustered(df["y_true"], df["y_pred"], df["copy_id"], n_boot=400)
    naif = kappa_interval(df["y_true"], df["y_pred"])
    assert (grappes.upper - grappes.lower) > (naif.upper - naif.lower)


def test_kappa_clustered_reproductible() -> None:
    df = _df_kappa_groupe()
    args = (df["y_true"], df["y_pred"], df["copy_id"])
    a = kappa_interval_clustered(*args, n_boot=100, seed=7)
    b = kappa_interval_clustered(*args, n_boot=100, seed=7)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_kappa_clustered_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        kappa_interval_clustered(["1", "9"], ["1", "9"], ["c1"])


# ── Bootstrap par grappes ─────────────────────────────────────────────────────
def _df_jouet() -> pd.DataFrame:
    rows = []
    for copie in ["c1", "c2", "c3", "c4"]:
        for i in range(10):
            ok = i < 9  # 90 % d'accord
            rows.append(
                {
                    "copy_id": copie,
                    "item_id": f"it{i}",
                    "y_true": "1",
                    "y_pred": "1" if ok else "9",
                    "confidence": 0.9 if ok else 0.4,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_contient_estimation() -> None:
    df = _df_jouet()
    ci = cluster_bootstrap(df, lambda d: float((d["y_true"] == d["y_pred"]).mean()), n_boot=200)
    assert ci.lower <= ci.estimate <= ci.upper
    assert ci.estimate == 0.9


def test_cluster_bootstrap_reproductible() -> None:
    df = _df_jouet()
    fn = lambda d: float((d["y_true"] == d["y_pred"]).mean())  # noqa: E731
    ci1 = cluster_bootstrap(df, fn, n_boot=100, seed=7)
    ci2 = cluster_bootstrap(df, fn, n_boot=100, seed=7)
    assert (ci1.lower, ci1.upper) == (ci2.lower, ci2.upper)


# ── Rapport par item / copie ──────────────────────────────────────────────────
def test_per_item_metrics_colonnes() -> None:
    stats = per_item_metrics(_df_jouet())
    for col in ["accord", "accord_lo", "accord_hi", "rappel_erreur", "n_sur_correction"]:
        assert col in stats.columns
    assert (stats["accord_lo"] <= stats["accord"]).all()
    assert (stats["accord"] <= stats["accord_hi"]).all()


def test_sur_correction_et_sur_detection() -> None:
    df = pd.DataFrame(
        {
            "copy_id": ["c"] * 4,
            "item_id": ["i1", "i1", "i1", "i1"],
            # expert: erreur, modèle: correct → sur-correction (1 cas)
            # expert: correct, modèle: erreur → sur-détection (1 cas)
            "y_true": ["9", "1", "1", "9"],
            "y_pred": ["1", "9", "1", "9"],
            "confidence": [0.8] * 4,
        }
    )
    stats = per_item_metrics(df)
    assert stats.loc["i1", "n_sur_correction"] == 1
    assert stats.loc["i1", "n_sur_detection"] == 1
    # 2 erreurs expert, 1 retrouvée → rappel 0.5 ; 2 erreurs modèle, 1 confirmée → précision 0.5
    assert stats.loc["i1", "rappel_erreur"] == 0.5
    assert stats.loc["i1", "precision_erreur"] == 0.5


def test_per_copy_metrics() -> None:
    stats = per_copy_metrics(_df_jouet())
    assert len(stats) == 4
    assert (stats["accord"] == 0.9).all()


def test_disagreement_decomposition() -> None:
    deco = disagreement_decomposition(_df_jouet())
    assert deco["n"].sum() == 4  # un désaccord par copie
    assert deco["pct_desaccords"].sum() == pytest.approx(100.0)


# ── Calibration ───────────────────────────────────────────────────────────────
def test_ece_modele_parfaitement_calibre() -> None:
    # confiance 1.0 et toujours juste → ECE = 0
    ece = expected_calibration_error(["1"] * 10, ["1"] * 10, [1.0] * 10)
    assert ece == pytest.approx(0.0)


def test_ece_modele_surconfiant() -> None:
    # confiance 0.95 mais 50 % d'accord seulement → ECE ≈ 0.45
    y_true = ["1"] * 5 + ["9"] * 5
    y_pred = ["1"] * 10
    ece = expected_calibration_error(y_true, y_pred, [0.95] * 10)
    assert ece == pytest.approx(0.45, abs=0.01)


def test_reliability_bins_couvrent_les_donnees() -> None:
    bins = reliability_bins(["1", "1"], ["1", "9"], [0.2, 0.99], n_bins=10)
    assert sum(b.n for b in bins) == 2
