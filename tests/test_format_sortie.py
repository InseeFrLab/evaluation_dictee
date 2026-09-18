"""Tests des corrections A1 (format dupliqué), A2 (champ « reason ») et D (confiance).

Avant ces corrections, `_format_sortie` concaténait un texte pré-écrit par option :
combiner `count_items` à autre chose produisait DEUX blocs « Réponds UNIQUEMENT... »
successifs avec deux exemples JSON divergents (le second oubliait `n_items_lus`), et
`_FORMAT_ITEMS_SIMPLE` demandait un champ « reason » absent du schéma, donc impossible
à produire sous décodage contraint. Le score de confiance auto-déclaré est retiré
partout (portée D confirmée : uniquement le champ prompt/schéma, pas l'infrastructure
de calibration — `evaluation/calibration.py` reste en place pour un futur signal).
"""

import json

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.models.vlm import _items_json_schema
from evaluation_dictee.pipeline.prompts import _format_sortie, build_dictation_prompt

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
]


def _texte(messages: list[dict]) -> str:
    return "\n\n".join(str(m["content"]) for m in messages)


# ── A1 : un seul bloc de format, quelle que soit la combinaison d'options ────────


def test_un_seul_bloc_format_meme_combinaisons() -> None:
    """Aucune combinaison ne doit produire deux « Réponds UNIQUEMENT... »."""
    for cot in (False, True):
        for count in (False, True):
            texte = _format_sortie(cot, count)
            assert texte.count("Réponds UNIQUEMENT") == 1, (cot, count, texte)


def _extraire_json(texte: str) -> dict:
    """Isole le premier objet JSON équilibré d'un texte (accolades comptées)."""
    debut = texte.index("{")
    profondeur = 0
    for i, c in enumerate(texte[debut:], start=debut):
        if c == "{":
            profondeur += 1
        elif c == "}":
            profondeur -= 1
            if profondeur == 0:
                return json.loads(texte[debut : i + 1])
    raise AssertionError("accolade jamais refermée")


def test_exemple_json_du_format_est_valide_et_complet() -> None:
    """L'exemple JSON du format doit rester du JSON valide, avec TOUS les champs actifs."""
    exemple = _extraire_json(_format_sortie(chain_of_thought=True, count_items=True))
    assert "n_items_lus" in exemple
    item = exemple["items"][0]
    assert list(item) == ["item_id", "transcription", "comparaison", "code"]


def test_n_items_lus_present_une_seule_fois() -> None:
    """Régression directe : l'ancien bug faisait apparaître le rappel puis l'oubliait."""
    texte = _format_sortie(chain_of_thought=False, count_items=True)
    assert texte.count("n_items_lus") == 2  # une fois dans l'exemple, une fois rappelé en clair


# ── A2 : le champ « reason » a disparu, CoT ou non ───────────────────────────────


def test_reason_absent_du_format_dans_tous_les_cas() -> None:
    for cot in (False, True):
        for count in (False, True):
            assert "reason" not in _format_sortie(cot, count)


def test_reason_absent_du_prompt_complet_sans_cot() -> None:
    """Le prompt de référence (sans aucune option) ne demande plus « reason »."""
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig()))
    assert "reason" not in prompt


# ── D : le score de confiance auto-déclaré a disparu du prompt et du schéma ──────


def test_confidence_absente_du_schema_dans_tous_les_cas() -> None:
    for cot in (False, True):
        for count in (False, True):
            props = _items_json_schema(cot, count)["properties"]["items"]["items"]["properties"]
            assert "confidence" not in props


def test_confidence_absente_du_prompt_de_reference() -> None:
    prompt = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig()))
    assert "confidence" not in prompt
    assert "score de confiance" not in prompt


def test_confidence_absente_du_format_sortie() -> None:
    for cot in (False, True):
        for count in (False, True):
            assert "confidence" not in _format_sortie(cot, count)
