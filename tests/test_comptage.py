"""Tests de l'option count_items : compter les items écrits avant de coder.

Le biais visé est mesuré : le modèle a le texte de référence sous les yeux et code
« présent » des mots que l'élève n'a pas écrits. Compter d'abord doit l'en empêcher.
"""

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.models.vlm import _items_json_schema
from evaluation_dictee.pipeline.prompts import build_dictation_prompt

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
]


def _texte(messages: list[dict]) -> str:
    return "\n\n".join(str(m["content"]) for m in messages)


def test_comptage_absent_par_defaut() -> None:
    """Sans l'option, ni le prompt ni le schéma ne changent (runs publiés préservés)."""
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig()))
    assert "n_items_lus" not in prompt
    assert list(_items_json_schema(False, False)["properties"]) == ["items"]


def test_comptage_ajoute_le_champ_en_premier() -> None:
    """`n_items_lus` précède `items` : le comptage doit contraindre le codage."""
    assert list(_items_json_schema(False, True)["properties"]) == ["n_items_lus", "items"]


def test_comptage_est_obligatoire_dans_le_schema() -> None:
    assert _items_json_schema(False, True)["required"] == ["n_items_lus", "items"]


def test_prompt_demande_le_comptage_et_la_confrontation() -> None:
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(count_items=True)))
    assert "COMMENCE PAR COMPTER" in prompt
    assert "n_items_lus" in prompt
    # La consigne doit relier le comptage aux items absents, sinon elle ne sert à rien.
    assert "absent" in prompt


def test_comptage_et_cot_cumulables() -> None:
    """Les deux options coexistent, même si les configs du projet les séparent."""
    schema = _items_json_schema(True, True)
    assert list(schema["properties"]) == ["n_items_lus", "items"]
    item = schema["properties"]["items"]["items"]["properties"]
    assert list(item).index("comparaison") < list(item).index("code")
