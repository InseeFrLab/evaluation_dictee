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
