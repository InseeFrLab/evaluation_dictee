"""Tests de l'export des prédictions (utils/s3_export).

On utilise un répertoire local comme « destination » : fsspec traite un chemin
sans schéma comme du LocalFileSystem, ce qui teste la même logique de copie que
vers s3:// sans dépendre du réseau.
"""

import json
from pathlib import Path

import pytest

from evaluation_dictee.utils.s3_export import (
    COPIES_ECARTEES_SUFFIX,
    EXPERIMENTATIONS_SUBDIR,
    HTR_SUFFIX,
    SCORING_SUFFIX,
    export_run,
    resolve_dest_prefix,
    upload_predictions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_upload_predictions_copies_content(tmp_path: Path) -> None:
    """Le fichier est copié à l'identique sous le préfixe de destination."""
    src = tmp_path / "run_predictions.jsonl"
    _write_jsonl(src, [{"copy_id": "c1", "item_id": "i1", "y_pred": "1"}])
    dest_dir = tmp_path / "predictions"

    dest = upload_predictions(src, dest_dir)

    assert dest.endswith("predictions/run_predictions.jsonl")
    assert Path(dest).read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_upload_predictions_missing_file_raises(tmp_path: Path) -> None:
    """Un fichier absent lève une erreur claire plutôt que d'écrire du vide."""
    with pytest.raises(FileNotFoundError):
        upload_predictions(tmp_path / "absent.jsonl", tmp_path / "predictions")


def test_export_run_scoring_and_htr_naming(tmp_path: Path) -> None:
    """export_run choisit le bon suffixe (scoring vs HTR) selon `htr`."""
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    _write_jsonl(source_dir / f"dictee_x{SCORING_SUFFIX}", [{"a": 1}])
    _write_jsonl(source_dir / f"dictee_x{HTR_SUFFIX}", [{"b": 2}])
    dest_dir = tmp_path / "predictions"

    scoring_dest = export_run("dictee_x", dest_dir, source_dir=source_dir)
    htr_dest = export_run("dictee_x", dest_dir, source_dir=source_dir, htr=True)

    assert scoring_dest.endswith(f"dictee_x{SCORING_SUFFIX}")
    assert htr_dest.endswith(f"dictee_x{HTR_SUFFIX}")


def test_export_run_trailing_slash_prefix(tmp_path: Path) -> None:
    """Un préfixe avec slash final ne produit pas de double slash."""
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    _write_jsonl(source_dir / f"run{SCORING_SUFFIX}", [{"a": 1}])

    dest = export_run("run", str(tmp_path / "predictions") + "/", source_dir=source_dir)

    assert "//run" not in dest.replace("://", "")


def test_export_run_inclut_les_copies_ecartees(tmp_path: Path) -> None:
    """Le CSV des copies écartées (D8) suit les prédictions quand il existe."""
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    _write_jsonl(source_dir / f"run{SCORING_SUFFIX}", [{"a": 1}])
    ecartees = source_dir / f"run{COPIES_ECARTEES_SUFFIX}"
    ecartees.write_text(
        "copy_id;motif;items_concernes;items_total\nc1;vierge;83;83\n", encoding="utf-8"
    )
    dest_dir = tmp_path / "predictions"

    export_run("run", dest_dir, source_dir=source_dir)

    dest_ecartees = dest_dir / f"run{COPIES_ECARTEES_SUFFIX}"
    assert dest_ecartees.read_text(encoding="utf-8") == ecartees.read_text(encoding="utf-8")


def test_export_run_sans_copies_ecartees_ne_cree_rien(tmp_path: Path) -> None:
    """Un run qui n'a rien écarté n'exporte pas de CSV vide ou fantôme."""
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    _write_jsonl(source_dir / f"run{SCORING_SUFFIX}", [{"a": 1}])
    dest_dir = tmp_path / "predictions"

    export_run("run", dest_dir, source_dir=source_dir)

    assert not (dest_dir / f"run{COPIES_ECARTEES_SUFFIX}").exists()


def test_export_run_htr_ignore_les_copies_ecartees(tmp_path: Path) -> None:
    """Le pipeline HTR ne produit pas ce fichier : l'export HTR ne doit pas le chercher."""
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    _write_jsonl(source_dir / f"run{HTR_SUFFIX}", [{"a": 1}])
    dest_dir = tmp_path / "predictions"

    export_run("run", dest_dir, source_dir=source_dir, htr=True)

    assert not (dest_dir / f"run{COPIES_ECARTEES_SUFFIX}").exists()


# ── Routage predictions/ vs predictions/experimentations/ ───────────────────


def test_resolve_dest_prefix_reference_va_a_la_racine() -> None:
    assert resolve_dest_prefix("s3://bucket/predictions", experimentation=False) == (
        "s3://bucket/predictions"
    )


def test_resolve_dest_prefix_experimentation_va_dans_le_sous_dossier() -> None:
    assert resolve_dest_prefix("s3://bucket/predictions", experimentation=True) == (
        f"s3://bucket/predictions/{EXPERIMENTATIONS_SUBDIR}"
    )


def test_resolve_dest_prefix_tolere_le_slash_final() -> None:
    assert resolve_dest_prefix("s3://bucket/predictions/", experimentation=True) == (
        f"s3://bucket/predictions/{EXPERIMENTATIONS_SUBDIR}"
    )
