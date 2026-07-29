"""Tests de l'évaluation concurrente du benchmark (parallélisme + parité séquentielle).

On mocke les I/O lourdes (chargement dataset/grille) et la trace Langfuse, pour
exercer uniquement l'orchestration : scoring parallèle, écriture mono-thread,
gestion des échecs/non-transcrits et reprise.
"""

import contextlib
import json
import threading
import time
from pathlib import Path

import pytest

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data.loaders import Copy
from evaluation_dictee.models.base import CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.pipeline import benchmark as bench


class _FakeGrid:
    reference_text = "texte de référence"
    items: list = []


class FakeScorer(Scorer):
    """Scorer déterministe (prédit = code expert) qui mesure le parallélisme observé."""

    def __init__(
        self,
        delay: float = 0.0,
        fail: set[str] | None = None,
        non_transcribed: set[str] | None = None,
    ) -> None:
        self.delay = delay
        self.fail = fail or set()
        self.non_transcribed = non_transcribed or set()
        self.scored: list[str] = []
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.scored.append(copy.copy_id)
        try:
            if self.delay:
                time.sleep(self.delay)
            if copy.copy_id in self.fail:
                raise RuntimeError("échec simulé")
            if copy.copy_id in self.non_transcribed:
                return CopyPrediction(copy_id=copy.copy_id, items=[], transcribed=False)
            items = [
                ItemPrediction(item_id=i, code=c, confidence=0.9, transcription="x")
                for i, c in zip(copy.item_ids, copy.expert_codes, strict=True)
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items)
        finally:
            with self._lock:
                self._active -= 1


def _copies(n: int) -> list[Copy]:
    return [
        Copy(
            copy_id=f"c{k:03d}.png",
            image_path=f"s3://x/c{k:03d}.png",
            expert_codes=["1", "9", "0"],
            item_ids=[f"i{k}_1", f"i{k}_2", f"i{k}_3"],
        )
        for k in range(n)
    ]


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """Neutralise dataset/grille/trace pour isoler l'orchestration."""

    @contextlib.contextmanager
    def _no_trace(copy):
        yield None

    monkeypatch.setattr(bench, "load_grid", lambda _p: _FakeGrid())
    monkeypatch.setattr(bench, "copy_trace", _no_trace)
    # Neutralise le chargement image + la détection copie vierge (non vierge par défaut).
    monkeypatch.setattr(bench, "load_image", lambda _p: object())
    monkeypatch.setattr(bench, "ink_ratio", lambda _img: 0.5)


def _config(n: int) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test_run",
            "model": {"name": "fake"},
            "data": {"images_path": "x", "labels_path": "y"},
        }
    )


def _read_lines(path: Path) -> list[str]:
    return sorted(line for line in path.read_text(encoding="utf-8").splitlines() if line)


def test_concurrent_matches_sequential(patched, monkeypatch, tmp_path: Path) -> None:
    """Même sortie JSONL (à l'ordre près) en séquentiel et en concurrent."""
    copies = _copies(12)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)

    seq_dir, par_dir = tmp_path / "seq", tmp_path / "par"
    bench.run_benchmark(_config(12), FakeScorer(), output_dir=seq_dir, concurrency=1)
    bench.run_benchmark(_config(12), FakeScorer(), output_dir=par_dir, concurrency=8)

    assert _read_lines(seq_dir / "test_run_predictions.jsonl") == _read_lines(
        par_dir / "test_run_predictions.jsonl"
    )


def test_concurrency_actually_parallel(patched, monkeypatch, tmp_path: Path) -> None:
    """Avec un délai par copie, plusieurs scorings tournent simultanément."""
    copies = _copies(16)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)
    scorer = FakeScorer(delay=0.05)

    bench.run_benchmark(_config(16), scorer, output_dir=tmp_path, concurrency=8)

    assert scorer.max_active >= 2  # preuve de parallélisme réel


def test_failures_and_non_transcribed(patched, monkeypatch, tmp_path: Path) -> None:
    """Les échecs vont dans failed_copies.txt ; les non-transcrites sont exclues."""
    copies = _copies(6)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)
    scorer = FakeScorer(fail={"c001.png"}, non_transcribed={"c002.png"})

    result = bench.run_benchmark(_config(6), scorer, output_dir=tmp_path, concurrency=4)

    assert "c002.png" in result.non_transcribed
    assert (tmp_path / "test_run_failed_copies.txt").exists()
    written = {
        json.loads(line)["copy_id"] for line in _read_lines(tmp_path / "test_run_predictions.jsonl")
    }
    assert "c001.png" not in written  # échec → non écrit
    assert "c002.png" not in written  # non transcrite → non écrit
    assert "c000.png" in written


def test_copie_vierge_auto_codee_zero(patched, monkeypatch, tmp_path: Path) -> None:
    """Une copie vierge est codée '0' sans appel modèle, identiquement pour toute méthode."""
    copies = _copies(4)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)
    # c001 est sous le seuil d'encre → vierge ; les autres au-dessus.
    monkeypatch.setattr(bench, "load_image", lambda path: path)  # identité : garde le chemin
    monkeypatch.setattr(bench, "ink_ratio", lambda path: 0.01 if "c001" in path else 0.5)
    scorer = FakeScorer()

    result = bench.run_benchmark(_config(4), scorer, output_dir=tmp_path, concurrency=1)

    assert result.blank_copies == ["c001.png"]
    assert "c001.png" not in scorer.scored  # aucune inférence sur une copie vierge
    recs = [json.loads(line) for line in _read_lines(tmp_path / "test_run_predictions.jsonl")]
    vierge = [r for r in recs if r["copy_id"] == "c001.png"]
    assert len(vierge) == 3
    assert all(r["y_pred"] == "0" for r in vierge)
    assert all(r["blank"] is True for r in vierge)
    assert all(r["confidence"] == 1.0 for r in vierge)
    # Une copie non vierge reste scorée normalement et marquée blank=False.
    non_vierge = [r for r in recs if r["copy_id"] == "c000.png"]
    assert all(r["blank"] is False for r in non_vierge)


def test_resume_skips_processed(patched, monkeypatch, tmp_path: Path) -> None:
    """Une copie déjà présente dans le JSONL n'est pas re-scorée."""
    copies = _copies(5)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)
    out = tmp_path / "test_run_predictions.jsonl"
    out.write_text(
        json.dumps({"copy_id": "c000.png", "item_id": "i0_1", "y_true": "1", "y_pred": "1"}) + "\n",
        encoding="utf-8",
    )
    scorer = FakeScorer()

    bench.run_benchmark(_config(5), scorer, output_dir=tmp_path, concurrency=4)

    assert "c000.png" not in scorer.scored  # sautée à la reprise
    assert len(scorer.scored) == 4
