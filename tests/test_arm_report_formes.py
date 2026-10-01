"""Garde-fou : autant de formes de marqueur que de bras par défaut.

`charger_series` attribue une forme à chaque bras par `FORMES[j % len(FORMES)]`
(arm_report.py). Si `BRAS_PAR_DEFAUT` compte plus d'entrées que `FORMES`, deux bras
finissent avec la MÊME forme — pour un même modèle (même couleur), ils deviennent
indiscernables sur les figures sans qu'aucune erreur ne le signale.
"""

from evaluation_dictee.evaluation.arm_report import BRAS_PAR_DEFAUT, FORMES, _marqueur


def test_assez_de_formes_pour_tous_les_bras() -> None:
    assert len(FORMES) >= len(BRAS_PAR_DEFAUT), (
        f"{len(BRAS_PAR_DEFAUT)} bras pour seulement {len(FORMES)} formes : "
        "deux bras partageraient la même forme pour un même modèle. "
        "Ajouter une forme dans _marqueur() et FORMES avant d'ajouter un bras."
    )


def test_chaque_forme_se_rend_sans_lever() -> None:
    """Chaque nom de FORMES doit être géré explicitement par `_marqueur`."""
    for forme in FORMES:
        svg = _marqueur(forme, 10.0, 10.0, "#2a78d6", "test")
        assert svg.startswith(("<circle", "<rect", "<polygon"))


def test_les_formes_sont_uniques() -> None:
    assert len(FORMES) == len(set(FORMES))
