"""Vérifie qu'après normalisation, modèle et experts codent dans le MÊME alphabet.

Attrape l'erreur où le prétraitement des codes experts aurait été oublié (ex. codes
3/4/5 experts comparés à des prédictions modèle 1/9/0).
"""

import pytest

from evaluation_dictee.data import reference


def test_alphabets_attendus() -> None:
    assert reference.allowed_codes("simplifiee") == {"1", "9", "0"}
    assert reference.allowed_codes("complete") == {"1", "3", "4", "5", "9", "0"}


def test_schema_inconnu_leve_erreur() -> None:
    with pytest.raises(ValueError):
        reference.allowed_codes("inexistant")


@pytest.mark.parametrize("scheme", ["simplifiee", "complete"])
def test_codes_experts_normalises_dans_alphabet(scheme: str) -> None:
    # Tous les codes bruts possibles d'un expert (grille complète + ponctuation)
    codes_experts_bruts = ["1", "3", "4", "5", "9", "0"]
    attendus = reference.allowed_codes(scheme)
    for code in codes_experts_bruts:
        norm = reference.normalize(code, scheme)
        assert norm in attendus, f"Code expert {code!r} → {norm!r} hors de {attendus} ({scheme})"


@pytest.mark.parametrize("scheme", ["simplifiee", "complete"])
def test_codes_modele_normalises_dans_alphabet(scheme: str) -> None:
    # Codes que le modèle peut produire selon le schéma demandé dans le prompt
    codes_modele = ["1", "9", "0"] if scheme == "simplifiee" else ["1", "3", "4", "5", "9", "0"]
    attendus = reference.allowed_codes(scheme)
    for code in codes_modele:
        assert reference.normalize(code, scheme) in attendus


def test_modele_et_experts_meme_alphabet_apres_normalisation() -> None:
    """Cœur du test : sur les deux schémas, les deux sources convergent."""
    for scheme in ("simplifiee", "complete"):
        experts = {reference.normalize(c, scheme) for c in ["1", "3", "4", "5", "9", "0"]}
        modele_brut = ["1", "9", "0"] if scheme == "simplifiee" else ["1", "3", "4", "5", "9", "0"]
        modele = {reference.normalize(c, scheme) for c in modele_brut}
        assert modele <= experts
        assert experts <= reference.allowed_codes(scheme)
