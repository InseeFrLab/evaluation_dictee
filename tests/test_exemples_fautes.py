"""Tests de l'option show_error_examples : les fautes déjà observées dans le prompt.

Elle vise la cause dominante mesurée sur 500 copies : 90 % des fautes manquées sont des
items que le modèle transcrit à l'identique du mot attendu — il ne VOIT pas la
différence. Lui montrer les confusions réellement relevées sur cet item précis est une
aide à la lecture, pas au raisonnement.
"""

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.prompts import build_dictation_prompt

ITEMS = [
    GridItem("i1", "soir", "mot", ["soire"], ["soirs"], []),
    GridItem("i2", ".", "ponctuation", [], [], []),
    GridItem("i3", "mit", "mot", [], ["mis"], ["mies"]),
]


def _prompt(**kwargs: object) -> str:
    cfg = PromptConfig(**kwargs)  # type: ignore[arg-type]
    return "\n\n".join(str(m["content"]) for m in build_dictation_prompt("Le soir", ITEMS, cfg))


def test_absent_par_defaut() -> None:
    """Sans l'option, le prompt reste celui des runs déjà mesurés."""
    prompt = _prompt()
    assert "fautes déjà observées" not in prompt
    assert "soire" not in prompt


def test_exemples_accoles_a_leur_item() -> None:
    prompt = _prompt(show_error_examples=True)
    assert "« soir » (mot)  [fautes déjà observées : « soire », « soirs »]" in prompt


def test_item_sans_faute_connue_reste_nu() -> None:
    """28 items sur 83 n'ont aucune faute recensée : pas de crochets vides."""
    assert "« . » (ponctuation)  [" not in _prompt(show_error_examples=True)


def test_grille_simplifiee_fusionne_les_familles() -> None:
    """En 1/9/0, lexical et grammatical reçoivent le même code : les séparer n'aide pas."""
    prompt = _prompt(show_error_examples=True)
    assert "« mis », « mies »" in prompt
    assert "grammaticales" not in prompt


def test_grille_complete_distingue_les_familles() -> None:
    """En grille complète, la famille de la faute est justement ce qu'il faut trancher."""
    cfg = PromptConfig(show_error_examples=True)
    prompt = "\n\n".join(
        str(m["content"]) for m in build_dictation_prompt("Le soir", ITEMS, cfg, scheme="complete")
    )
    assert "grammaticales : « mis »" in prompt
    assert "lexicales ET grammaticales : « mies »" in prompt


def test_mise_en_garde_contre_la_suggestion() -> None:
    """Le risque propre à l'option : aller vers une faute connue par suggestion."""
    prompt = _prompt(show_error_examples=True)
    assert "n'est PAS exhaustive" in prompt
    assert "Tu transcris ce que tu VOIS" in prompt
