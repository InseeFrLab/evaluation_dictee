"""Chargement du corpus Scoledit (HTR) : appariement scans + annotations TEI.

La référence TEI est nettoyée en texte brut en PRÉSERVANT les fautes de l'élève,
qui sont l'objet même de l'évaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import fsspec


@dataclass
class ScoledtSample:
    """Un échantillon Scoledit : image + transcription de référence."""

    scan: str
    level: str
    student_id: int
    image_path: str
    reference: str
    tei_raw: str


def tei_to_text(tei: str) -> str:
    """Convertit une transcription TEI légère Scoledit en texte brut (fautes préservées).

    Args:
        tei: Transcription au format TEI léger.

    Returns:
        Texte brut, balises retirées et espaces normalisés.
    """
    txt = tei.replace("<lb/>", " ")
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def _join(base: str, name: str) -> str:
    """Concatène un chemin de base et un nom avec un unique séparateur.

    Args:
        base: Chemin de base.
        name: Nom à ajouter.

    Returns:
        Chemin joint.
    """
    return base.rstrip("/") + "/" + name


def load_scoledit_dataset(
    scans_dir: str,
    annotations_dir: str,
    limit: int | None = None,
) -> list[ScoledtSample]:
    """Charge les échantillons Scoledit en appariant scans et annotations.

    Args:
        scans_dir: Dossier des images numérisées.
        annotations_dir: Dossier des annotations JSON (champs scan, level, tei...).
        limit: Nombre maximal d'échantillons à charger (tous si None).

    Returns:
        Liste des échantillons appariés.
    """
    fs, _, paths = fsspec.get_fs_token_paths(annotations_dir.rstrip("/") + "/*")
    json_files = sorted(p for p in fs.glob(annotations_dir.rstrip("/") + "/*.json"))

    samples: list[ScoledtSample] = []
    for jpath in json_files:
        with fsspec.open(_scheme_prefix(annotations_dir, jpath), "rt", encoding="utf-8") as f:
            meta = json.load(f)
        scan = meta.get("scan", "")
        image_path = _join(scans_dir, f"{scan}.jpg")
        samples.append(
            ScoledtSample(
                scan=scan,
                level=meta.get("level", ""),
                student_id=int(meta.get("student_id", -1)),
                image_path=image_path,
                reference=tei_to_text(meta.get("tei", "")),
                tei_raw=meta.get("tei", ""),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _scheme_prefix(reference_dir: str, path: str) -> str:
    """Restaure le préfixe s3:// perdu par fs.glob quand la source est S3.

    Args:
        reference_dir: Dossier de référence indiquant le schéma d'origine.
        path: Chemin retourné par glob, éventuellement sans préfixe.

    Returns:
        Chemin préfixé de s3:// si nécessaire, sinon inchangé.
    """
    if reference_dir.startswith("s3://") and not path.startswith("s3://"):
        return "s3://" + path
    return path
