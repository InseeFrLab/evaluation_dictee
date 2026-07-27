"""Exporte les prédictions d'un run vers S3 (répertoire predictions/).

Le benchmark écrit les prédictions en local (data/processed/<name>_predictions.jsonl).
Ce script les pousse vers S3 pour pouvoir relancer les notebooks et le site Quarto
sans réexécuter le pipeline. Aucune donnée n'est commitée dans Git.

Usage :
    # scoring — end_to_end OU two_stage (même format, c'est le `name` qui distingue) :
    uv run scripts/export_predictions.py --config configs/scoring/dictee_REFERENCE.yaml

    # transcription HTR seule :
    uv run scripts/export_predictions.py --config configs/htr/htr_REFERENCE.yaml --htr

    # ou directement par nom de run :
    uv run scripts/export_predictions.py --run-name dictee_gemma4_zeroshot
"""

from __future__ import annotations

import argparse

import yaml

from evaluation_dictee.config import Secrets
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.s3_export import export_run

logger = get_logger(__name__)


def _run_name_from_config(config_path: str) -> str:
    """Lit le champ `name` d'un YAML de run (scoring ou HTR)."""
    with open(config_path, encoding="utf-8") as f:
        return str(yaml.safe_load(f)["name"])


def main() -> None:
    """Résout le nom du run et exporte son fichier de prédictions vers S3."""
    parser = argparse.ArgumentParser(
        description="Exporte les prédictions d'un run vers S3 (répertoire predictions/)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML du run (le nom est lu dans le champ `name`).")
    source.add_argument("--run-name", help="Nom du run (préfixe du fichier de prédictions).")
    parser.add_argument(
        "--htr",
        action="store_true",
        help="Exporte le fichier HTR (<name>_htr_predictions.jsonl) au lieu du scoring.",
    )
    parser.add_argument(
        "--source-dir",
        default="data/processed",
        help="Dossier local des prédictions. [défaut : data/processed]",
    )
    parser.add_argument(
        "--dest-prefix",
        default=None,
        help="Préfixe S3 de destination. [défaut : S3_PREDICTIONS_PREFIX / config]",
    )
    args = parser.parse_args()

    run_name = args.run_name or _run_name_from_config(args.config)
    dest_prefix = args.dest_prefix or Secrets().s3_predictions_prefix

    dest = export_run(run_name, dest_prefix, source_dir=args.source_dir, htr=args.htr)
    logger.info("OK — disponible sur S3 : %s", dest)


if __name__ == "__main__":
    main()
