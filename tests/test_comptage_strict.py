"""Tests des deux consignes ajoutées au bras comptage : cohérence et voisinage.

Elles sont VOLONTAIREMENT inconditionnelles : conditionner la vérification à un manque
déclaré serait inopérant sur un modèle qui annonce 83 items sur 49 copies sur 50 alors
que 27 d'entre elles ont des mots manquants (mesuré le 11/09/2026).
"""

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.prompts import build_dictation_prompt

ITEMS = [GridItem("i1", "Le", "mot", [], [], []), GridItem("i2", "soir", "mot", [], [], [])]


def _texte(messages: list[dict]) -> str:
    return "\n\n".join(str(m["content"]) for m in messages)


def _prompt(**kwargs: object) -> str:
    return _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(**kwargs)))  # type: ignore[arg-type]


def test_consignes_absentes_par_defaut() -> None:
    """Le bras comptage simple ne doit pas changer : il a déjà été mesuré."""
    prompt = _prompt(count_items=True)
    assert "FAIS COÏNCIDER" not in prompt
    assert "VÉRIFICATION DE VOISINAGE" not in prompt


def test_coherence_du_comptage() -> None:
    prompt = _prompt(count_items=True, enforce_count=True)
    assert "FAIS COÏNCIDER TON CODAGE AVEC TON COMPTAGE" in prompt
    assert "n_items_lus" in prompt


def test_coherence_sans_comptage_est_sans_objet() -> None:
    """Sans comptage déclaré, il n'y a rien à faire coïncider : la consigne est omise."""
    assert "FAIS COÏNCIDER" not in _prompt(count_items=False, enforce_count=True)


def test_voisinage_est_inconditionnel() -> None:
    """La vérification s'applique à chaque item, indépendamment du comptage."""
    prompt = _prompt(check_neighbours=True)
    assert "VÉRIFICATION DE VOISINAGE" in prompt
    assert "N-1" in prompt and "N+1" in prompt
    assert "pas seulement en cas de doute" in prompt


def test_les_deux_consignes_sont_independantes() -> None:
    """Deux drapeaux distincts : on pourra isoler laquelle des deux agit."""
    assert "VÉRIFICATION DE VOISINAGE" not in _prompt(count_items=True, enforce_count=True)
    assert "FAIS COÏNCIDER" not in _prompt(check_neighbours=True)
    complet = _prompt(count_items=True, enforce_count=True, check_neighbours=True)
    assert "FAIS COÏNCIDER" in complet and "VÉRIFICATION DE VOISINAGE" in complet
