"""Échantillonnage aléatoire reproductible des copies (`sample_seed`)."""

from __future__ import annotations

from pathlib import Path

from evaluation_dictee.data.loaders import load_dataset


def _jeu(tmp_path: Path, n: int = 30) -> tuple[str, str]:
    images = tmp_path / "img"
    images.mkdir()
    lignes = ["idx;nom;i1"]
    for i in range(n):
        nom = f"c_{i:03d}.png"
        (images / nom).write_bytes(b"x")
        lignes.append(f"{i};{nom};1")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(lignes), encoding="utf-8")
    return str(images), str(labels)


def test_sans_graine_on_prend_les_premieres(tmp_path: Path) -> None:
    images, labels = _jeu(tmp_path)
    ids = [c.copy_id for c in load_dataset(images, labels, limit=5)]
    assert ids == [f"c_{i:03d}.png" for i in range(5)]


def test_avec_graine_tirage_reproductible_et_trie(tmp_path: Path) -> None:
    images, labels = _jeu(tmp_path)
    a = [c.copy_id for c in load_dataset(images, labels, limit=5, sample_seed=7)]
    b = [c.copy_id for c in load_dataset(images, labels, limit=5, sample_seed=7)]
    autre = [c.copy_id for c in load_dataset(images, labels, limit=5, sample_seed=8)]
    assert a == b == sorted(a)
    assert a != [f"c_{i:03d}.png" for i in range(5)]
    assert a != autre
