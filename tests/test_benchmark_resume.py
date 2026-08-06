"""Tests de la reprise du benchmark : copies déjà acquises et copies à refaire.

La reprise est préparée par `pipeline/purge.preparer_reprise`, qui retire du fichier
les copies sans aucun code exploitable avant de renvoyer celles à sauter.
"""

import json
from pathlib import Path

import pytest

from evaluation_dictee.models.base import CODE_NON_PARSE
from evaluation_dictee.pipeline.purge import preparer_reprise


def _ecrire(path: Path, lignes: list[dict]) -> None:
    """Écrit un JSONL de prédictions à partir d'enregistrements bruts."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in lignes:
            f.write(json.dumps(rec) + "\n")


def test_fichier_absent(tmp_path: Path) -> None:
    """Fichier absent = aucune copie déjà traitée (premier lancement)."""
    assert preparer_reprise(tmp_path / "nope.jsonl") == set()


def test_extrait_les_copy_ids(tmp_path: Path) -> None:
    """Lit correctement les copy_id depuis un JSONL existant."""
    path = tmp_path / "p.jsonl"
    _ecrire(
        path,
        [
            {"copy_id": "c1.png", "item_id": "i1", "y_pred": "1"},
            {"copy_id": "c1.png", "item_id": "i2", "y_pred": "9"},
            {"copy_id": "c2.png", "item_id": "i1", "y_pred": "1"},
        ],
    )
    assert preparer_reprise(path) == {"c1.png", "c2.png"}


def test_ignore_la_derniere_ligne_tronquee(tmp_path: Path) -> None:
    """Une ligne tronquée par un crash à mi-écriture ne casse pas la reprise."""
    path = tmp_path / "p.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i1", "y_pred": "1"}) + "\n")
        f.write('{"copy_id": "c2.png", "item_id":')  # coupée par le crash
    assert preparer_reprise(path) == {"c1.png"}


def test_ignore_les_lignes_vides(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i1", "y_pred": "1"}) + "\n")
        f.write("\n")
    assert preparer_reprise(path) == {"c1.png"}


def test_ignore_les_enregistrements_sans_copy_id(tmp_path: Path) -> None:
    """Ligne JSON valide mais sans copy_id : ignorée silencieusement."""
    path = tmp_path / "p.jsonl"
    _ecrire(
        path,
        [
            {"copy_id": "c1.png", "item_id": "i1", "y_pred": "1"},
            {"unrelated": "junk"},
            {"copy_id": "c2.png", "item_id": "i1", "y_pred": "1"},
        ],
    )
    assert preparer_reprise(path) == {"c1.png", "c2.png"}


def test_copie_entierement_non_parsee_est_retiree_et_refaite(tmp_path: Path) -> None:
    """Une copie 100 % non parsée n'a rien produit : à refaire, et retirée du fichier."""
    path = tmp_path / "p.jsonl"
    _ecrire(
        path,
        [
            {"copy_id": "ok.png", "item_id": "i1", "y_pred": "1"},
            {"copy_id": "ok.png", "item_id": "i2", "y_pred": "9"},
            {"copy_id": "ratee.png", "item_id": "i1", "y_pred": CODE_NON_PARSE},
            {"copy_id": "ratee.png", "item_id": "i2", "y_pred": CODE_NON_PARSE},
        ],
    )
    assert preparer_reprise(path) == {"ok.png"}

    restants = {json.loads(li)["copy_id"] for li in path.read_text(encoding="utf-8").splitlines()}
    assert restants == {"ok.png"}  # les lignes de la copie ratée ont disparu
    assert path.with_suffix(".jsonl.bak").exists()  # sauvegarde avant réécriture
    assert not path.with_suffix(".jsonl.tmp").exists()  # temporaire renommé


def test_copie_partiellement_non_parsee_est_conservee(tmp_path: Path) -> None:
    """Un item non parsé isolé n'invalide pas la copie.

    La refaire à chaque relance serait sans fin : un modèle qui omet régulièrement
    un item ferait boucler indéfiniment le même run.
    """
    path = tmp_path / "p.jsonl"
    _ecrire(
        path,
        [
            {"copy_id": "c.png", "item_id": "i1", "y_pred": "1"},
            {"copy_id": "c.png", "item_id": "i2", "y_pred": CODE_NON_PARSE},
        ],
    )
    assert preparer_reprise(path) == {"c.png"}
    assert not path.with_suffix(".jsonl.bak").exists()  # rien à purger, rien à réécrire


def test_seuil_permet_de_refaire_les_copies_partielles(tmp_path: Path) -> None:
    """Un seuil plus bas refait aussi les copies partiellement ratées."""
    path = tmp_path / "p.jsonl"
    _ecrire(
        path,
        [
            {"copy_id": "saine.png", "item_id": "i1", "y_pred": "1"},
            {"copy_id": "partielle.png", "item_id": "i1", "y_pred": "1"},
            {"copy_id": "partielle.png", "item_id": "i2", "y_pred": CODE_NON_PARSE},
        ],
    )
    assert preparer_reprise(path, seuil=0.0) == {"saine.png"}


@pytest.mark.parametrize("seuil", [0.0, 0.5, 1.0])
def test_copie_saine_jamais_retiree(tmp_path: Path, seuil: float) -> None:
    """Quel que soit le seuil, une copie sans item non parsé est conservée."""
    path = tmp_path / "p.jsonl"
    _ecrire(path, [{"copy_id": "c.png", "item_id": "i1", "y_pred": "1"}])
    assert preparer_reprise(path, seuil=seuil) == {"c.png"}
