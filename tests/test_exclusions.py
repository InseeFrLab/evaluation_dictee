"""Tests de l'exclusion des copies vierges et illisibles des métriques.

Une copie illisible ou vierge n'est pas un désaccord de jugement : l'expert n'a rendu
aucun code auquel comparer le modèle. Les compter imputerait au modèle un défaut
d'annotation ou de numérisation. Elles sont donc écartées des métriques, conservées
dans le JSONL, et listées pour vérification humaine.
"""

from evaluation_dictee.data import reference


def test_code_illisible_non_evaluable() -> None:
    assert reference.est_evaluable("i") is False
    assert reference.est_evaluable("I") is False  # tolère la casse du CSV expert
    assert reference.est_evaluable(" i ") is False


def test_code_vide_non_evaluable() -> None:
    """Une cellule vide du CSV expert n'est pas davantage un jugement."""
    assert reference.est_evaluable("") is False
    assert reference.est_evaluable("   ") is False


def test_codes_de_la_grille_evaluables() -> None:
    for code in ["1", "9", "0", "3", "4", "5"]:
        assert reference.est_evaluable(code) is True


def test_illisible_reste_hors_de_la_grille() -> None:
    """`i` ne doit jamais être confondu avec un code de la grille après normalisation."""
    assert reference.normalize("i", "simplifiee") not in reference.allowed_codes("simplifiee")


# ── Filtrage appliqué par le site et les rapports ────────────────────────────


def test_marquage_reconstitue_la_colonne_absente() -> None:
    """Les JSONL écrits avant D8 n'ont pas de colonne `exclusion` : elle est déduite."""
    import pandas as pd

    from evaluation_dictee.evaluation.report import marquer_exclusions

    df = pd.DataFrame(
        [
            {"copy_id": "c1", "item_id": "i1", "y_true": "1", "blank": False},
            {"copy_id": "c2", "item_id": "i1", "y_true": "0", "blank": True},
            {"copy_id": "c3", "item_id": "i1", "y_true": "i", "blank": False},
        ]
    )
    assert list(marquer_exclusions(df)["exclusion"]) == [None, "vierge", "illisible"]


def test_filtrage_retire_et_compte_les_copies() -> None:
    import pandas as pd

    from evaluation_dictee.evaluation.report import filtrer_evaluables

    df = pd.DataFrame(
        [
            {"copy_id": "c1", "item_id": "i1", "y_true": "1", "blank": False},
            {"copy_id": "c2", "item_id": "i1", "y_true": "0", "blank": True},
            {"copy_id": "c3", "item_id": "i1", "y_true": "i", "blank": False},
        ]
    )
    reste, retirees = filtrer_evaluables(df)
    assert list(reste["copy_id"]) == ["c1"]
    assert retirees == {"illisible": 1, "vierge": 1}


def test_filtrage_sans_exclusion_ne_retire_rien() -> None:
    import pandas as pd

    from evaluation_dictee.evaluation.report import filtrer_evaluables

    df = pd.DataFrame([{"copy_id": "c1", "item_id": "i1", "y_true": "9", "blank": False}])
    reste, retirees = filtrer_evaluables(df)
    assert len(reste) == 1
    assert retirees == {}
