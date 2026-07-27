"""Tests du ré-alignement et des nouveaux outils de diagnostic par copie."""

import pandas as pd

from evaluation_dictee.evaluation.report import copies_by_disagreement
from evaluation_dictee.pipeline.alignment import (
    needs_realignment,
    realign,
)


# ── Ré-alignement ─────────────────────────────────────────────────────────────
def test_pas_de_decalage_pas_de_realignement() -> None:
    attendus = ["Le", "soir", "tombait"]
    trans = ["Le", "soir", "tombait"]
    assert needs_realignment(attendus, trans) is False


def test_mot_scinde_detecte_comme_decalage() -> None:
    # L'élève a écrit « re trouver » → modèle lit deux tokens, tout décale
    attendus = ["nous", "retrouver", "leur", "chemin"]
    trans = ["nous", "re", "trouver", "leur"]  # décalé d'un cran après « re »
    assert needs_realignment(attendus, trans) is True


def test_realign_recolle_sur_mots_attendus() -> None:
    attendus = ["nous", "retrouver", "leur", "chemin"]
    # modèle : a scindé « retrouver » en « re »+« trouver », décalant la suite
    codes = ["1", "1", "1", "1"]
    trans = ["nous", "re", "trouver", "leur"]
    conf = [0.9, 0.9, 0.9, 0.9]
    aligned = realign(attendus, codes, trans, conf)
    assert len(aligned) == len(attendus)
    assert aligned[0].transcription == "nous"


def test_realign_longueur_toujours_egale_aux_attendus() -> None:
    attendus = ["a", "b", "c", "d", "e"]
    codes = ["1", "9", "1"]  # modèle a rendu moins d'items
    trans = ["a", "b", "c"]
    conf = [1.0, 1.0, 1.0]
    aligned = realign(attendus, codes, trans, conf)
    assert len(aligned) == 5
    assert aligned[-1].code in {"0", "?"}


def test_mots_colles_realignement() -> None:
    # L'élève colle « nous les » → modèle lit un seul token « nousles »
    attendus = ["nous", "les", "verrons"]
    codes = ["9", "1"]
    trans = ["nousles", "verrons"]
    conf = [0.8, 0.9]
    aligned = realign(attendus, codes, trans, conf)
    assert len(aligned) == 3


# ── Table de désaccord par copie ──────────────────────────────────────────────
def _df_multi() -> pd.DataFrame:
    rows = []
    # copie A : 1 désaccord / 4 ; copie B : 3 désaccords / 4
    for cid, faux in [("A", {3}), ("B", {1, 2, 3})]:
        for i in range(4):
            rows.append(
                {
                    "copy_id": cid,
                    "item_id": f"it{i}",
                    "y_true": "1",
                    "y_pred": "9" if i in faux else "1",
                    "confidence": 0.9,
                }
            )
    return pd.DataFrame(rows)


def test_copies_by_disagreement_tri() -> None:
    tab = copies_by_disagreement(_df_multi())
    # B (75 % désaccord) doit être en tête, A (25 %) ensuite
    assert tab.index[0] == "B"
    assert tab.index[1] == "A"
    assert tab.loc["B", "pct_desaccord"] == 75.0
    assert tab.loc["A", "n_desaccords"] == 1


def test_copies_by_disagreement_colonnes() -> None:
    tab = copies_by_disagreement(_df_multi())
    for col in ["n_items", "n_desaccords", "pct_desaccord", "accord"]:
        assert col in tab.columns


# ── Stratégies d'alignement multiples ─────────────────────────────────────────
def test_realign_anchored_utilise_les_ancres() -> None:
    from evaluation_dictee.pipeline.alignment import realign_anchored

    # "Martine" est une ancre unique des deux côtés
    attendus = ["téléphoner", "à", "Martine", "?"]
    codes = ["1", "1", "1", "1", "1"]
    trans = ["telephoner", "x", "a", "Martine", "?"]  # un token parasite avant
    conf = [0.9] * 5
    aligned = realign_anchored(attendus, codes, trans, conf)
    assert len(aligned) == 4
    i_martine = attendus.index("Martine")
    assert aligned[i_martine].transcription == "Martine"


def test_best_realignment_choisit_le_meilleur() -> None:
    from evaluation_dictee.pipeline.alignment import best_realignment

    attendus = ["nous", "retrouver", "leur", "chemin"]
    codes = ["1", "1", "1", "1"]
    trans = ["nous", "re", "trouver", "leur"]
    conf = [0.9] * 4
    aligned = best_realignment(attendus, codes, trans, conf)
    assert len(aligned) == len(attendus)
    assert aligned[0].transcription == "nous"
