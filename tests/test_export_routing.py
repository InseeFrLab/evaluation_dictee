"""Tests du routage automatique de l'export (scripts/export_predictions.py).

Un run alimente le site UNIQUEMENT s'il est exporté au niveau racine de
`predictions/` ; un bras testé sur un échantillon va dans `predictions/
experimentations/`, où le site ne descend jamais (cf. `website/_analyse.py`).

La détection compare l'effectif RÉELLEMENT présent dans les prédictions au corpus,
plutôt que `data.limit` du YAML : celui-ci ne reflète pas un `--limit` passé en
ligne de commande à `run_benchmark.py`, qui est la façon dont tous les bras
d'expérience de ce projet ont été lancés (`data.limit` y vaut `null` dans le YAML).
"""

import csv
import json
from pathlib import Path

from scripts.export_predictions import SEUIL_COMPLETUDE, _est_run_complet


def _config(labels_path: str):
    from evaluation_dictee.config import DataConfig, ExperimentConfig, ModelConfig

    return ExperimentConfig(
        name="test",
        model=ModelConfig(name="m"),
        data=DataConfig(images_path="img/", labels_path=labels_path),
    )


def _ecrire_labels(path: Path, n_copies: int) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["idx", "copy_id", "i1"])
        for i in range(n_copies):
            w.writerow([i, f"copy_{i}.png", "1"])


def _ecrire_predictions(path: Path, copy_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for cid in copy_ids:
            f.write(json.dumps({"copy_id": cid, "item_id": "i1", "y_pred": "1"}) + "\n")


def test_run_complet_reconnu_malgre_quelques_copies_manquantes(tmp_path: Path) -> None:
    """Quelques copies perdues (échec API) ne doivent pas déclasser un run complet."""
    labels = tmp_path / "labels.csv"
    _ecrire_labels(labels, n_copies=100)
    preds = tmp_path / "run_predictions.jsonl"
    _ecrire_predictions(preds, [f"copy_{i}.png" for i in range(96)])  # 96/100 = 96 %

    assert SEUIL_COMPLETUDE <= 0.96
    assert _est_run_complet(_config(str(labels)), preds) is True


def test_echantillon_partiel_detecte_comme_experimentation(tmp_path: Path) -> None:
    """C'est le cas réel de ce projet : --limit 500 en CLI sur un corpus de 3469."""
    labels = tmp_path / "labels.csv"
    _ecrire_labels(labels, n_copies=3469)
    preds = tmp_path / "run_predictions.jsonl"
    _ecrire_predictions(preds, [f"copy_{i}.png" for i in range(500)])

    assert _est_run_complet(_config(str(labels)), preds) is False


def test_fichier_de_predictions_absent(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _ecrire_labels(labels, n_copies=10)
    assert _est_run_complet(_config(str(labels)), tmp_path / "absent.jsonl") is False
