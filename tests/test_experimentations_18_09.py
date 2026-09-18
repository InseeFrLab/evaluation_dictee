"""Tests des 4 expérimentations du 18/09/2026, construites sur comptage+exemples.

- E1 (reference_items_only) : le texte de référence n'apparaît plus en phrase continue.
- E2 (garde-fou de comptage) : voir tests/test_comptage_incoherent.py — pipeline pur,
  pas d'option de prompt dédiée (actif dès que count_items est vrai).
- E3 (check_punctuation_presence) : consigne dédiée à la ponctuation.
- E4 (contrastive_examples) : contraste le mot attendu aux fautes connues.
"""

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.prompts import build_dictation_prompt

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", ["soire"], [], []),
    GridItem("i3", ".", "ponctuation", [], [], []),
]

BASE_COMPTAGE_EXEMPLES = {
    "count_items": True,
    "enforce_count": True,
    "check_neighbours": True,
    "show_error_examples": True,
}


def _texte(**kwargs: object) -> str:
    cfg = PromptConfig(**{**BASE_COMPTAGE_EXEMPLES, **kwargs})  # type: ignore[arg-type]
    return "\n\n".join(str(m["content"]) for m in build_dictation_prompt("Le soir.", ITEMS, cfg))


# ── E1 : texte de référence en items isolés uniquement ──────────────────────────


def test_e1_absent_par_defaut_phrase_visible() -> None:
    prompt = _texte()
    assert "Le soir." in prompt
    assert "Texte de référence" in prompt


def test_e1_actif_retire_la_phrase_continue() -> None:
    prompt = _texte(reference_items_only=True)
    assert "« Le soir. »" not in prompt
    assert "Texte de référence (ce que l'élève devait écrire)" not in prompt
    # Le texte reste intégralement présent, mais UNIQUEMENT dans la liste d'items.
    assert "« Le »" in prompt
    assert "« soir »" in prompt


def test_e1_explique_pourquoi_au_modele() -> None:
    prompt = _texte(reference_items_only=True)
    assert "volontaire" in prompt.lower()


# ── E3 : consigne dédiée à la ponctuation ────────────────────────────────────────


def test_e3_absente_par_defaut() -> None:
    assert "CAS PARTICULIER DE LA PONCTUATION" not in _texte()


def test_e3_active_la_consigne() -> None:
    prompt = _texte(check_punctuation_presence=True)
    assert "CAS PARTICULIER DE LA PONCTUATION" in prompt
    assert "AVANT de juger s'il est du bon type" in prompt


# ── E4 : contraste attendu / fautes connues ──────────────────────────────────────


def test_e4_defaut_liste_les_fautes_seules() -> None:
    prompt = _texte()
    assert "fautes déjà observées : « soire »" in prompt
    assert "confusions fréquentes" not in prompt


def test_e4_actif_contraste_avec_le_mot_attendu() -> None:
    prompt = _texte(contrastive_examples=True)
    assert "attendu : « soir » — confusions fréquentes : « soire »" in prompt


def test_e4_sans_effet_si_exemples_desactives() -> None:
    """contrastive_examples seul, sans show_error_examples, ne doit rien changer."""
    cfg = PromptConfig(show_error_examples=False, contrastive_examples=True)
    prompt = "\n\n".join(str(m["content"]) for m in build_dictation_prompt("Le soir.", ITEMS, cfg))
    assert "confusions fréquentes" not in prompt
    assert "fautes déjà observées" not in prompt


# ── Les innovations de comptage+exemples restent actives dans les 4 bras ────────


def test_innovations_comptage_exemples_toujours_presentes() -> None:
    for kwargs in (
        {"reference_items_only": True},
        {"check_punctuation_presence": True},
        {"contrastive_examples": True},
    ):
        prompt = _texte(**kwargs)
        assert "COMMENCE PAR COMPTER" in prompt
        assert "FAIS COÏNCIDER TON CODAGE" in prompt
        assert "VÉRIFICATION DE VOISINAGE" in prompt
        assert "n_items_lus" in prompt
