"""Tests de l'option chain_of_thought (chain-of-thought)."""

import json

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.models.vlm import _items_json_schema, comparaison_avant_code
from evaluation_dictee.pipeline.alignment import best_realignment
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


# ── Correctifs : ordre des clés, champ fantôme, survie au ré-alignement ──────


def test_schema_declare_comparaison_avant_code() -> None:
    """Le schéma demande la comparaison AVANT le code : après, elle ne raisonne rien."""
    props = _items_json_schema(chain_of_thought=True)["properties"]["items"]["items"]["properties"]
    cles = list(props)
    assert cles.index("comparaison") < cles.index("code")


def test_schema_sans_cot_inchange() -> None:
    """Sans CoT, seul le champ `comparaison` diffère (retiré : `confidence`, voir D)."""
    props = _items_json_schema(chain_of_thought=False)["properties"]["items"]["items"]["properties"]
    assert list(props) == ["item_id", "transcription", "code"]


def test_ordre_reel_detecte() -> None:
    """`json.loads` conserve l'ordre du document : on peut contrôler ce qu'a fait le modèle."""
    avant = json.loads('{"item_id": "i1", "comparaison": "identique", "code": "1"}')
    apres = json.loads('{"item_id": "i1", "code": "1", "comparaison": "identique"}')
    assert comparaison_avant_code(avant) is True
    assert comparaison_avant_code(apres) is False


def test_ordre_indetermine_sans_cot() -> None:
    """Sans champ « comparaison », la question n'a pas de sens : None, pas False."""
    assert comparaison_avant_code({"item_id": "i1", "code": "1"}) is None


def test_champ_reason_retire_du_prompt_cot() -> None:
    """« reason » était demandé au prompt mais absent du schéma : impossible à produire."""
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=True)))
    assert "reason" not in prompt


def test_comparaison_survit_au_realignement() -> None:
    """La comparaison suit son token : sinon elle disparaît sur les copies décalées."""
    aligned = best_realignment(
        expected_words=["Le", "soir"],
        codes=["1", "9"],
        transcriptions=["Le", "soirs"],
        confidences=[0.9, 0.8],
        comparaisons=["identique", "un 's' en trop"],
    )
    assert [a.comparaison for a in aligned] == ["identique", "un 's' en trop"]


def test_realignement_sans_cot_ne_casse_pas() -> None:
    """Sans comparaisons fournies, le ré-alignement fonctionne comme avant."""
    aligned = best_realignment(["Le", "soir"], ["1", "1"], ["Le", "soir"], [0.9, 0.9])
    assert [a.comparaison for a in aligned] == [None, None]


def test_cot_prevoit_le_cas_du_mot_absent() -> None:
    """Sans branche « absent », le modèle recopie le mot attendu et code 1 (mesuré)."""
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=True)))
    assert "absent" in prompt
    assert "N'invente JAMAIS une transcription à partir du texte de référence" in prompt
