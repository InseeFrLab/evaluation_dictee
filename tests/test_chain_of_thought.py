"""Tests de l'option chain_of_thought (chain-of-thought)."""

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.prompts import build_dictation_prompt

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
]


def _texte(messages: list[dict]) -> str:
    """Concatène le contenu de tous les messages pour les assertions."""
    return "\n\n".join(str(m["content"]) for m in messages)


def test_cot_desactive_pas_de_champ_comparaison() -> None:
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=False)))
    assert "comparaison" not in prompt
    assert '"code": "1"' in prompt


def test_cot_active_ajoute_champ_comparaison() -> None:
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=True)))
    assert "comparaison" in prompt
    assert '"comparaison"' in prompt
    assert "inquiets" in prompt.lower()  # mot issu de l'exemple pédagogique du prompt


def test_cot_ordre_du_raisonnement_est_impose() -> None:
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=True)))
    assert "AVANT de choisir le code" in prompt
