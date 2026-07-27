"""Tests de l'export des prédictions (utils/s3_export).

On utilise un répertoire local comme « destination » : fsspec traite un chemin
sans schéma comme du LocalFileSystem, ce qui teste la même logique de copie que
vers s3:// sans dépendre du réseau.
"""

import json
from pathlib import Path

import pytest

from evaluation_dictee.utils.s3_export import (
    HTR_SUFFIX,
    SCORING_SUFFIX,
    export_run,
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
