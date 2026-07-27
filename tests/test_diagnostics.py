"""Tests du module de diagnostic des écarts modèle/annotateur."""

import pandas as pd

from evaluation_dictee.evaluation.diagnostics import (
    accuracy_by_position,
    disagreement_hotspots,
    longest_disagreement_run,
    overconfident_errors,
    per_copy_position_effect,
    position_accuracy_correlation,
    simulate_word_omission_shift,
)

ITEMS = [f"it{i:02d}" for i in range(10)]


def _df(copy_id: str, faux_positions: set[int], conf: float = 1.0) -> pd.DataFrame:
    """Construit une copie où les positions de `faux_positions` sont en désaccord."""
    rows = []
    for i, iid in enumerate(ITEMS):
        faux = i in faux_positions
        rows.append(
            {
                "copy_id": copy_id,
                "item_id": iid,
                "y_true": "9" if faux else "1",
                "y_pred": "1",  # le modèle dit toujours correct
                "confidence": conf,
            }
        )
    return pd.DataFrame(rows)


def test_accuracy_by_position_detecte_degradation() -> None:
    # Désaccords concentrés en fin → accord doit chuter avec la position
    df = _df("c1", faux_positions={6, 7, 8, 9})
    tab = accuracy_by_position(df, ITEMS, n_bins=5)
    assert tab.iloc[-1]["accord"] < tab.iloc[0]["accord"]


def test_correlation_negative_si_fin_degradee() -> None:
    df = _df("c1", faux_positions={6, 7, 8, 9})
    corr = position_accuracy_correlation(df, ITEMS)
    assert corr < 0


def test_longest_run_detecte_sequence_consecutive() -> None:
    # 4 désaccords consécutifs (positions 5..8) vs erreurs isolées
    df = _df("c1", faux_positions={5, 6, 7, 8})
    runs = longest_disagreement_run(df, ITEMS)
    assert runs.loc["c1", "plus_longue_sequence"] == 4
    assert runs.loc["c1", "position_debut_sequence"] == 6  # positions 1-based


def test_erreurs_isolees_donnent_petite_sequence() -> None:
    df = _df("c1", faux_positions={1, 4, 8})  # dispersées
    runs = longest_disagreement_run(df, ITEMS)
    assert runs.loc["c1", "plus_longue_sequence"] == 1


def test_per_copy_position_effect_chute() -> None:
    df = _df("c1", faux_positions={5, 6, 7, 8, 9})  # tout en 2de moitié
    eff = per_copy_position_effect(df, ITEMS)
    assert eff.loc["c1", "accord_moitie1"] == 1.0
    assert eff.loc["c1", "accord_moitie2"] < 0.5
    assert eff.loc["c1", "chute"] > 0


def test_overconfident_errors() -> None:
    df = _df("c1", faux_positions={2, 3}, conf=0.99)
    surconf = overconfident_errors(df, seuil_confiance=0.9)
    assert len(surconf) == 2  # 2 erreurs, toutes à confiance 0.99
    df_bas = _df("c2", faux_positions={2, 3}, conf=0.2)
    assert len(overconfident_errors(df_bas, seuil_confiance=0.9)) == 0


def test_hotspots_renvoie_les_palmares() -> None:
    df = pd.concat(
        [
            _df("bonne", faux_positions=set()),
            _df("mauvaise", faux_positions={3, 4, 5, 6, 7}),
        ]
    )
    hot = disagreement_hotspots(df, ITEMS, top=2)
    assert set(hot.keys()) == {
        "pire_accord",
        "plus_longue_sequence",
        "plus_forte_chute",
        "sur_confiance",
    }
    assert hot["pire_accord"].index[0] == "mauvaise"


# ── Le test clé : démontrer l'effet d'un mot oublié ──────────────────────────
def test_omission_decale_et_detruit_la_fin() -> None:
    # Codage expert d'une copie : alternance correct / erreur sur 20 items
    expert = ["1", "9"] * 10
    # Un mot oublié au tout début (position 1) décale presque tout
    res_debut = simulate_word_omission_shift(expert, omission_position=1)
    # Un mot oublié à la fin (position 18) ne décale presque rien
    res_fin = simulate_word_omission_shift(expert, omission_position=18)

    assert res_debut["accord_sans_decalage"] == 1.0
    assert res_debut["accord_avec_decalage"] < 0.6
    assert res_fin["accord_avec_decalage"] > res_debut["accord_avec_decalage"]


def test_omission_position_tardive_preserve_le_debut() -> None:
    expert = ["1"] * 10 + ["9"] * 10
    res = simulate_word_omission_shift(expert, omission_position=15)
    # Les 15 premiers items restent alignés → accord encore élevé
    assert res["accord_avec_decalage"] >= 0.7
