"""Tests de `_localiser` (arm_report.py) : local, puis predictions/, puis son sous-dossier.

`Secrets.s3_predictions_prefix` est lu depuis l'environnement à chaque instanciation
(`pydantic_settings`) : le pointer vers un dossier local permet de tester la même
logique de repli que vers S3, sans réseau (fsspec traite un chemin sans schéma comme
du LocalFileSystem — même principe que `tests/test_s3_export.py`).
"""

from pathlib import Path

from evaluation_dictee.evaluation.arm_report import _localiser


def _toucher(chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("{}\n", encoding="utf-8")


def test_prefere_le_fichier_local(tmp_path: Path, monkeypatch) -> None:
    local_dir = tmp_path / "local"
    _toucher(local_dir / "run_modele_predictions.jsonl")
    monkeypatch.setenv("S3_PREDICTIONS_PREFIX", str(tmp_path / "s3"))

    assert _localiser(local_dir, "run", "modele") == str(local_dir / "run_modele_predictions.jsonl")


def test_repli_sur_la_racine_predictions(tmp_path: Path, monkeypatch) -> None:
    """Un run de référence, absent en local, est retrouvé à la racine de predictions/."""
    s3 = tmp_path / "s3" / "predictions"
    _toucher(s3 / "dictee_end2end_qwen3-6-35b-moe_predictions.jsonl")
    monkeypatch.setenv("S3_PREDICTIONS_PREFIX", str(s3))

    trouve = _localiser(tmp_path / "vide", "dictee_end2end", "qwen3-6-35b-moe")
    assert trouve == str(s3 / "dictee_end2end_qwen3-6-35b-moe_predictions.jsonl")


def test_repli_sur_le_sous_dossier_experimentations(tmp_path: Path, monkeypatch) -> None:
    """Un bras d'expérience, absent en local ET à la racine, est cherché dans son sous-dossier."""
    s3 = tmp_path / "s3" / "predictions"
    _toucher(s3 / "experimentations" / "dictee_end2end_cot_qwen3-6-35b-moe_predictions.jsonl")
    monkeypatch.setenv("S3_PREDICTIONS_PREFIX", str(s3))

    trouve = _localiser(tmp_path / "vide", "dictee_end2end_cot", "qwen3-6-35b-moe")
    assert trouve == str(
        s3 / "experimentations" / "dictee_end2end_cot_qwen3-6-35b-moe_predictions.jsonl"
    )


def test_run_introuvable_nulle_part(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("S3_PREDICTIONS_PREFIX", str(tmp_path / "s3" / "predictions"))
    assert _localiser(tmp_path / "vide", "dictee_end2end", "modele-absent") is None
