"""Export des prédictions vers S3 (répertoire `predictions/` du bucket projet).

Le pipeline écrit les prédictions en local — `data/processed/<name>_predictions.jsonl` —
en mode append + fsync par copie, pour la reprise sur crash. Ce mode est propre au
disque local : S3 ne supporte ni l'append ni le fsync. On sépare donc l'écriture
(locale, incrémentale) de l'export (S3, une fois le run terminé).

But : pouvoir relancer les notebooks et le site Quarto sans réexécuter le pipeline.
Rien n'est jamais commité dans Git — les transcriptions sont des données d'élèves
mineurs, hébergées sur le SSP Cloud.

L'accès S3 suit le même pattern que `data/loaders.py` : `fsspec` lit les identifiants
et l'endpoint MinIO depuis l'environnement (variables injectées par Onyxia).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import fsspec

from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

# Suffixes des fichiers produits par les deux pipelines. Le scoring (end_to_end ET
# two_stage) partage un seul format : c'est le `name` du run qui distingue les runs.
SCORING_SUFFIX = "_predictions.jsonl"
HTR_SUFFIX = "_htr_predictions.jsonl"


def _join_s3(prefix: str, name: str) -> str:
    """Concatène un préfixe (S3 ou local) et un nom de fichier (un seul slash)."""
    return prefix.rstrip("/") + "/" + name


def upload_predictions(local_path: str | Path, dest_prefix: str | Path) -> str:
    """Copie un fichier de prédictions local vers un préfixe S3.

    Args:
        local_path: Chemin local du fichier JSONL de prédictions.
        dest_prefix: Préfixe de destination (ex. s3://bucket/predictions).

    Returns:
        L'URI de destination du fichier écrit.

    Raises:
        FileNotFoundError: Si le fichier local est absent (run non lancé).
    """
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Fichier de prédictions introuvable : {local_path}. "
            "Lancer d'abord le benchmark (scripts/run_benchmark.py) pour le produire."
        )

    dest = _join_s3(str(dest_prefix), local_path.name)
    # Copie binaire par flux (pas de réencodage, robuste aux gros fichiers).
    with open(local_path, "rb") as src, fsspec.open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)

    logger.info("Prédictions exportées : %s → %s", local_path, dest)
    return dest


def export_run(
    run_name: str,
    dest_prefix: str | Path,
    source_dir: str | Path = "data/processed",
    htr: bool = False,
) -> str:
    """Exporte vers S3 le fichier de prédictions d'un run donné.

    Args:
        run_name: Nom du run (préfixe des fichiers de sortie).
        dest_prefix: Préfixe S3 de destination.
        source_dir: Dossier local des prédictions. [défaut : data/processed]
        htr: Si True, exporte `<name>_htr_predictions.jsonl` (transcription seule)
            au lieu du fichier de scoring.

    Returns:
        L'URI S3 du fichier écrit.
    """
    suffix = HTR_SUFFIX if htr else SCORING_SUFFIX
    local_path = Path(source_dir) / f"{run_name}{suffix}"
    return upload_predictions(local_path, dest_prefix)
