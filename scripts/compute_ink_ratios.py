"""Mesure la densité d'encre de chaque copie et exporte la distribution.

À quoi ça sert : le benchmark déclare une copie VIERGE quand sa densité d'encre
passe sous `data.blank_ink_threshold`, et code alors tous ses items « 0 » sans
appeler le modèle. Ce script calcule la même mesure pour **tout le corpus**, sans
appel modèle ni GPU, afin de :

- documenter le seuil (le site trace la distribution obtenue, voir
  `website/ecarts.qmd`) ;
- rejuger ce seuil en quelques minutes, sans relancer un benchmark de 30 h.

La sortie ne contient **aucune donnée d'élève** : un identifiant de copie et un
scalaire par ligne. Elle vit malgré tout dans `data/processed/` (ignoré par Git)
et sur S3, comme les prédictions.

Usage :
    # Depuis le YAML d'un run (reprend son corpus et son seuil) :
    uv run scripts/compute_ink_ratios.py --config configs/scoring/dictee_end2end.yaml

    # Ou en désignant directement les données :
    uv run scripts/compute_ink_ratios.py \
        --images-path s3://projet-production-ecrits-depp/dictee_2015/ \
        --labels-path s3://projet-production-ecrits-depp/resultat_dictee_2015.csv

    # Puis pousser sur S3 pour que le site le relise :
    uv run scripts/compute_ink_ratios.py --config configs/... --export
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec

from evaluation_dictee.config import ExperimentConfig, Secrets, load_config
from evaluation_dictee.data.loaders import Copy, ink_ratio, load_dataset, load_image
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

#: Nom du fichier produit, dérivé du corpus (et non du run) : la densité d'encre
#: ne dépend ni du modèle ni de l'approche.
NOM_FICHIER = "{corpus}_ink_ratios.csv"


def _mesurer(copy: Copy) -> tuple[str, float | None]:
    """Densité d'encre d'une copie ; None si l'image est illisible."""
    try:
        return copy.copy_id, ink_ratio(load_image(copy.image_path))
    except Exception as exc:  # noqa: BLE001 — une image cassée ne doit pas tout arrêter
        logger.warning("%s illisible (%s) : ignorée.", copy.copy_id, type(exc).__name__)
        return copy.copy_id, None


def mesurer_corpus(copies: list[Copy], workers: int = 16) -> list[tuple[str, float]]:
    """Mesure la densité d'encre de toutes les copies, en parallèle.

    Args:
        copies: copies à mesurer.
        workers: nombre de threads (le coût est dominé par les entrées/sorties S3).

    Returns:
        La liste des couples (copy_id, densité), triée par identifiant, privée des
        copies dont l'image n'a pas pu être lue.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resultats = list(pool.map(_mesurer, copies))
    return sorted((cid, r) for cid, r in resultats if r is not None)


def ecrire_csv(mesures: list[tuple[str, float]], chemin: str | Path, seuil: float) -> Path:
    """Écrit les mesures en CSV (`copy_id;ink_ratio;blank`).

    Args:
        mesures: couples (copy_id, densité d'encre).
        chemin: fichier de destination.
        seuil: seuil au-dessous duquel la copie est marquée vierge.

    Returns:
        Le chemin écrit.
    """
    sortie = Path(chemin)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["copy_id", "ink_ratio", "blank"])
        for copy_id, densite in mesures:
            writer.writerow([copy_id, f"{densite:.6f}", int(densite < seuil)])
    return sortie


def _resoudre_source(args: argparse.Namespace) -> tuple[str, str, str, float, int | None]:
    """Résout (images, labels, corpus, seuil, limite) depuis le YAML ou les options."""
    if args.config:
        config: ExperimentConfig = load_config(args.config)
        return (
            config.data.images_path,
            config.data.labels_path,
            config.data.corpus,
            config.data.blank_ink_threshold,
            args.limit if args.limit is not None else config.data.limit,
        )
    if not (args.images_path and args.labels_path):
        raise SystemExit("Sans --config, --images-path ET --labels-path sont requis.")
    return (args.images_path, args.labels_path, args.corpus, args.threshold, args.limit)


def main() -> None:
    """Mesure la densité d'encre du corpus, écrit le CSV et l'exporte si demandé."""
    parser = argparse.ArgumentParser(
        description="Mesure la densité d'encre de chaque copie (détection des copies vierges)."
    )
    parser.add_argument("--config", help="YAML d'un run : corpus, chemins et seuil en sont lus.")
    parser.add_argument("--images-path", help="Dossier des images (local ou s3://).")
    parser.add_argument("--labels-path", help="CSV des codes experts (local ou s3://).")
    parser.add_argument("--corpus", default="dictee", help="Nom du corpus. [défaut : dictee]")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.025,
        help="Seuil de copie vierge, si absent du YAML. [défaut : 0.025]",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de copies.")
    parser.add_argument("--workers", type=int, default=16, help="Threads de lecture. [défaut : 16]")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Dossier local de sortie. [défaut : data/processed]",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Pousse aussi le CSV sur S3, où le site Quarto le relit.",
    )
    args = parser.parse_args()

    images, labels, corpus, seuil, limite = _resoudre_source(args)
    copies = load_dataset(images, labels, limit=limite)
    logger.info("%d copies à mesurer (seuil de copie vierge : %.4f).", len(copies), seuil)

    mesures = mesurer_corpus(copies, workers=args.workers)
    n_vierges = sum(1 for _, densite in mesures if densite < seuil)
    logger.info(
        "%d copies mesurées, dont %d vierges (%.2f %%).",
        len(mesures),
        n_vierges,
        n_vierges / len(mesures) * 100 if mesures else 0.0,
    )

    nom = NOM_FICHIER.format(corpus=corpus)
    local = ecrire_csv(mesures, Path(args.output_dir) / nom, seuil)
    logger.info("Écrit : %s", local)

    if args.export:
        dest = Secrets().s3_predictions_prefix.rstrip("/") + "/" + nom
        with (
            local.open("rb") as src,
            fsspec.open(dest, "wb") as dst,
        ):
            dst.write(src.read())
        logger.info("OK — disponible sur S3 : %s", dest)


if __name__ == "__main__":
    main()
