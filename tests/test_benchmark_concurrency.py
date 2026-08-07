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
from evaluation_dictee.models.base import (
    CODE_NON_PARSE,
    CopyPrediction,
    ItemPrediction,
    Scorer,
)
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

    assert _read_lines(seq_dir / "test_run_fake_predictions.jsonl") == _read_lines(
        par_dir / "test_run_fake_predictions.jsonl"
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
    assert (tmp_path / "test_run_fake_failed_copies.txt").exists()
    written = {
        json.loads(line)["copy_id"]
        for line in _read_lines(tmp_path / "test_run_fake_predictions.jsonl")
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
    recs = [json.loads(line) for line in _read_lines(tmp_path / "test_run_fake_predictions.jsonl")]
    vierge = [r for r in recs if r["copy_id"] == "c001.png"]
    assert len(vierge) == 3
    assert all(r["y_pred"] == "0" for r in vierge)
    assert all(r["blank"] is True for r in vierge)
    assert all(r["confidence"] == 1.0 for r in vierge)
    # Une copie non vierge reste scorée normalement et marquée blank=False.
    non_vierge = [r for r in recs if r["copy_id"] == "c000.png"]
    assert all(r["blank"] is False for r in non_vierge)


def test_nom_fichier_inclut_toujours_le_modele(patched, monkeypatch, tmp_path: Path) -> None:
    """Le fichier de sortie porte le modèle, et le suffixe n'est ajouté qu'une fois.

    Régression : `benchmark` ajoutait `model.name` à un `name` que
    `override_model_names` avait déjà suffixé, d'où deux fichiers distincts pour un
    même run selon son mode de lancement (et donc un checkpoint jamais retrouvé).
    """
    copies = _copies(2)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)

    # Cas 1 : modèle lu dans le YAML (`name` non suffixé).
    bench.run_benchmark(_config(2), FakeScorer(), output_dir=tmp_path, concurrency=1)
    assert (tmp_path / "test_run_fake_predictions.jsonl").exists()

    # Cas 2 : lancement CLI, où `name` porte déjà le suffixe. Même fichier attendu.
    deja_suffixe = _config(2).model_copy(update={"name": "test_run_fake"})
    bench.run_benchmark(deja_suffixe, FakeScorer(), output_dir=tmp_path, concurrency=1)

    produits = sorted(p.name for p in tmp_path.glob("*_predictions.jsonl"))
    assert produits == ["test_run_fake_predictions.jsonl"]


def test_nom_fichier_slugifie_le_modele(patched, monkeypatch, tmp_path: Path) -> None:
    """Un nom de modèle contenant « / » ne crée pas de sous-dossier fantôme."""
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: _copies(1))
    config = _config(1).model_copy(
        update={"model": _config(1).model.model_copy(update={"name": "Qwen/Qwen2.5-VL-7B"})}
    )

    bench.run_benchmark(config, FakeScorer(), output_dir=tmp_path, concurrency=1)

    assert (tmp_path / "test_run_Qwen-Qwen2.5-VL-7B_predictions.jsonl").exists()


def test_modele_inscrit_dans_chaque_ligne(patched, monkeypatch, tmp_path: Path) -> None:
    """Chaque ligne du JSONL porte le modèle, y compris celui de l'étape 2."""
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: _copies(2))
    config = ExperimentConfig.model_validate(
        {
            "name": "test_run",
            "approach": "two_stage",
            "model": {"name": "vlm-etape1"},
            "model_stage2": {"name": "llm-etape2", "kind": "llm"},
            "data": {"images_path": "x", "labels_path": "y"},
        }
    )

    bench.run_benchmark(config, FakeScorer(), output_dir=tmp_path, concurrency=1)

    out = tmp_path / "test_run_vlm-etape1_llm-etape2_predictions.jsonl"
    recs = [json.loads(line) for line in _read_lines(out)]
    assert recs
    assert all(r["model"] == "vlm-etape1" for r in recs)
    assert all(r["model_stage2"] == "llm-etape2" for r in recs)


def test_model_stage2_absent_en_end_to_end(patched, monkeypatch, tmp_path: Path) -> None:
    """En end_to_end, `model_stage2` est explicitement None (et non absent)."""
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: _copies(1))

    bench.run_benchmark(_config(1), FakeScorer(), output_dir=tmp_path, concurrency=1)

    recs = [json.loads(line) for line in _read_lines(tmp_path / "test_run_fake_predictions.jsonl")]
    assert all(r["model"] == "fake" and r["model_stage2"] is None for r in recs)


def test_verrou_bloque_un_second_run_sur_le_meme_fichier(tmp_path: Path) -> None:
    """Deux runs visant le même JSONL : le second échoue au lieu d'y dupliquer des copies."""
    out = tmp_path / "test_run_fake_predictions.jsonl"

    # Les contextes sont entrés de gauche à droite : la seconde prise de verrou lève,
    # et `pytest.raises`, déjà actif, l'intercepte.
    with (
        bench._single_writer(out),
        pytest.raises(RuntimeError, match="Un autre run écrit déjà"),
        bench._single_writer(out),
    ):
        pass

    # Verrou relâché à la sortie du bloc : un run suivant repasse.
    with bench._single_writer(out):
        pass


def test_lignes_dupliquees_exclues_des_metriques(patched, monkeypatch, tmp_path: Path) -> None:
    """Une copie écrite deux fois (runs concurrents) ne pèse qu'une fois dans les métriques."""
    copies = _copies(5)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)

    # c000 pré-écrite EN DOUBLE, comme l'aurait fait un second run concurrent.
    out = tmp_path / "test_run_fake_predictions.jsonl"
    doublons = [
        json.dumps({"copy_id": "c000.png", "item_id": f"i0_{k}", "y_true": "1", "y_pred": "1"})
        for k in (1, 2, 3)
    ] * 2
    out.write_text("\n".join(doublons) + "\n", encoding="utf-8")

    result = bench.run_benchmark(_config(5), FakeScorer(), output_dir=tmp_path, concurrency=1)

    # 5 copies × 3 items = 15 items distincts, malgré les 6 lignes écrites pour c000.
    assert result.metrics.n_items == 15
    assert len(result.y_true) == 15


def test_resume_skips_processed(patched, monkeypatch, tmp_path: Path) -> None:
    """Une copie déjà présente dans le JSONL n'est pas re-scorée."""
    copies = _copies(5)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)
    out = tmp_path / "test_run_fake_predictions.jsonl"
    out.write_text(
        json.dumps({"copy_id": "c000.png", "item_id": "i0_1", "y_true": "1", "y_pred": "1"}) + "\n",
        encoding="utf-8",
    )
    scorer = FakeScorer()

    bench.run_benchmark(_config(5), scorer, output_dir=tmp_path, concurrency=4)

    assert "c000.png" not in scorer.scored  # sautée à la reprise
    assert len(scorer.scored) == 4


def test_reprise_refait_les_copies_non_exploitables(patched, monkeypatch, tmp_path: Path) -> None:
    """Bout en bout : un run relancé recode les copies dont aucun item n'était parsé.

    Le premier run écrit tout ; on abîme ensuite une copie comme l'aurait fait un
    échec d'appel (tous les items non parsés). Le second run doit la refaire — et
    elle seule.
    """
    copies = _copies(4)
    monkeypatch.setattr(bench, "load_dataset", lambda **_k: copies)
    out = tmp_path / "test_run_fake_predictions.jsonl"

    bench.run_benchmark(_config(4), FakeScorer(), output_dir=tmp_path, concurrency=2)

    abimee = "c002.png"
    lignes = []
    for ligne in out.read_text(encoding="utf-8").splitlines():
        rec = json.loads(ligne)
        if rec["copy_id"] == abimee:
            rec["y_pred"] = CODE_NON_PARSE
        lignes.append(json.dumps(rec))
    out.write_text("\n".join(lignes) + "\n", encoding="utf-8")

    scorer = FakeScorer()
    bench.run_benchmark(_config(4), scorer, output_dir=tmp_path, concurrency=2)

    assert scorer.scored == [abimee]  # seule la copie abîmée est recodée
    recodee = [
        json.loads(li)
        for li in out.read_text(encoding="utf-8").splitlines()
        if json.loads(li)["copy_id"] == abimee
    ]
    assert len(recodee) == 3  # les anciennes lignes ont été retirées, pas dupliquées
    assert all(r["y_pred"] != CODE_NON_PARSE for r in recodee)
