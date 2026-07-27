"""Tests de la transcription HTR concurrente (non-régression vs séquentiel).

Un faux transcripteur remplace l'appel VLM : on vérifie que le parallélisme ne
change ni le contenu, ni l'ORDRE des enregistrements, ni la liste des échecs.
"""

import json
import threading
import time
from pathlib import Path

from evaluation_dictee.transcription.htr_benchmark import run_htr_benchmark
from evaluation_dictee.transcription.scoledit import ScoledtSample


class FakeTranscriber:
    """Transcripteur déterministe qui mesure le parallélisme observé."""

    def __init__(self, delay: float = 0.0, empty_for: set[str] | None = None) -> None:
        self.delay = delay
        self.empty_for = empty_for or set()
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def transcribe(self, image_path: str) -> str:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            # Transcription déterministe dérivée du chemin ; vide si demandé (échec).
            return "" if image_path in self.empty_for else f"texte {image_path}"
        finally:
            with self._lock:
                self._active -= 1


def _samples(n: int) -> list[ScoledtSample]:
    return [
        ScoledtSample(
            scan=f"s{k:03d}",
            level="CE1",
            student_id=k,
            image_path=f"img/{k:03d}.png",
            reference=f"reference du scan {k}",
            tei_raw="",
        )
        for k in range(n)
    ]


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_parallel_identical_to_sequential(tmp_path: Path) -> None:
    """Même JSONL, MÊME ORDRE, mêmes moyennes en séquentiel et en concurrent."""
    samples = _samples(12)

    seq = run_htr_benchmark(
        samples, FakeTranscriber(), run_name="seq", output_dir=tmp_path, concurrency=1
    )
    par = run_htr_benchmark(
        samples, FakeTranscriber(), run_name="par", output_dir=tmp_path, concurrency=8
    )

    # Ordre + contenu identiques (on ignore la seule colonne "scan"→run-agnostique).
    seq_lines = _lines(tmp_path / "seq_htr_predictions.jsonl")
    par_lines = _lines(tmp_path / "par_htr_predictions.jsonl")
    assert [json.loads(x)["scan"] for x in seq_lines] == [json.loads(x)["scan"] for x in par_lines]
    assert [json.loads(x)["scan"] for x in par_lines] == [s.scan for s in samples]  # ordre préservé
    assert seq.mean_cer == par.mean_cer
    assert seq.mean_wer == par.mean_wer


def test_concurrency_actually_parallel(tmp_path: Path) -> None:
    """Avec un délai, plusieurs transcriptions tournent en même temps."""
    transcriber = FakeTranscriber(delay=0.05)
    run_htr_benchmark(_samples(16), transcriber, run_name="p", output_dir=tmp_path, concurrency=8)
    assert transcriber.max_active >= 2


def test_failures_listed_in_order(tmp_path: Path) -> None:
    """Les transcriptions vides sont comptées comme échecs, dans l'ordre des samples."""
    samples = _samples(6)
    transcriber = FakeTranscriber(empty_for={"img/001.png", "img/004.png"})

    result = run_htr_benchmark(
        samples, transcriber, run_name="f", output_dir=tmp_path, concurrency=4
    )

    assert result.n_echecs == 2
    assert result.failed_scans == ["s001", "s004"]  # ordre samples, pas ordre d'achèvement
