"""Génère le rapport HTML de comparaison des bras d'expérience.

Un « bras » est une variante de prompt testée contre la référence (chain-of-thought,
comptage des items, exemples de fautes…). Ce rapport dit si la variante fait mieux que
la référence, sur le même modèle et les mêmes copies, et pourquoi — ce que ni Langfuse
(qui mesure des runs isolés) ni le site (qui publie ce qui est retenu) ne font.

Usage :
    uv run scripts/rapport_bras.py
    uv run scripts/rapport_bras.py --modeles qwen3-8-27b gemma4-26b-moe
    uv run scripts/rapport_bras.py --bras référence=dictee_end2end exemples=dictee_end2end_exemples
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_dictee.evaluation.arm_report import BRAS_PAR_DEFAUT, construire_rapport
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

MODELES_PAR_DEFAUT = ["qwen3-6-35b-moe", "qwen3-8-27b", "gemma4-26b-moe"]


def main() -> None:
    """Construit le rapport et l'écrit sur disque."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modeles",
        nargs="+",
        default=MODELES_PAR_DEFAUT,
        help="Modèles à comparer (défaut : les trois modèles testés).",
    )
    parser.add_argument(
        "--bras",
        nargs="+",
        default=None,
        metavar="LIBELLÉ=PRÉFIXE",
        help=(
            "Bras à inclure, sous la forme `libellé=préfixe_du_run`. "
            f"Défaut : {', '.join(f'{k}={v}' for k, v in BRAS_PAR_DEFAUT.items())}"
        ),
    )
    parser.add_argument(
        "--reference",
        default="référence",
        help="Libellé du bras servant d'étalon aux écarts. [défaut : référence]",
    )
    parser.add_argument("--dossier", default="data/processed", help="Dossier des prédictions.")
    parser.add_argument(
        "--sortie",
        default="data/processed/rapport_bras.html",
        help="Fichier HTML produit. Sous data/, donc hors de Git.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=400,
        help="Tirages du bootstrap des intervalles de confiance. [défaut : 400]",
    )
    args = parser.parse_args()

    bras = None
    if args.bras:
        bras = dict(paire.split("=", 1) for paire in args.bras)

    html, rapport = construire_rapport(
        modeles=args.modeles,
        bras=bras,
        dossier=args.dossier,
        reference=args.reference,
        n_boot=args.n_boot,
    )
    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(html, encoding="utf-8")

    logger.info(
        "%d série(s) sur %d copies communes : %s",
        len(rapport.series),
        rapport.n_copies,
        ", ".join(sorted({s.bras for s in rapport.series})),
    )
    logger.info("Rapport écrit : %s", sortie)


if __name__ == "__main__":
    main()
