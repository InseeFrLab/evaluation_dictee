"""Exporte les prédictions d'un run vers S3 (répertoire predictions/).

Le benchmark écrit les prédictions en local (data/processed/<name>_predictions.jsonl).
Ce script les pousse vers S3 pour pouvoir relancer les notebooks et le site Quarto
sans réexécuter le pipeline. La liste des copies écartées des métriques (vierges ou
illisibles, décision D8) est exportée avec, si elle existe : elle doit rester
vérifiable par quiconque relit le run, pas seulement sur la machine qui l'a produit.
Aucune donnée n'est commitée dans Git.

Deux destinations possibles, jamais mélangées :
- `predictions/`                  — runs de référence, corpus COMPLET.
  C'est le seul dossier lu par le site Quarto.
- `predictions/experimentations/` — bras testés sur un ÉCHANTILLON (chain-of-thought,
  comptage, exemples de fautes...). Le site liste `predictions/` SANS y descendre : un
  run posé ici n'apparaît donc jamais comme un modèle, par construction, sans aucune
  liste de noms à maintenir. Comparer les bras entre eux se fait avec
  `scripts/rapport_bras.py`, pas sur le site.

La destination est déduite en comparant l'effectif RÉELLEMENT exporté au nombre de
copies du corpus (CSV des labels de la config) — pas `data.limit` du YAML, qui ne
reflète pas un `--limit` passé en ligne de commande à `run_benchmark.py` : c'est
pourtant ainsi que tous les bras d'expérience de ce projet ont été lancés. Avec
`--run-name` seul, aucun YAML n'existe pour retrouver `labels_path` : un choix
explicite est alors obligatoire.

Usage :
    # scoring — end_to_end OU two_stage (même format, c'est le `name` qui distingue) :
    uv run scripts/export_predictions.py --config configs/scoring/dictee_REFERENCE.yaml

    # un bras testé sur un échantillon (destination déduite automatiquement) :
    uv run scripts/export_predictions.py --config configs/scoring/dictee_end2end_cot.yaml \
        --model-name qwen3-6-35b-moe

    # transcription HTR seule :
    uv run scripts/export_predictions.py --config configs/htr/htr_REFERENCE.yaml --htr

    # par nom de run, sans YAML : préciser la destination explicitement
    uv run scripts/export_predictions.py --run-name dictee_end2end_cot_qwen3-6-35b-moe \
        --experimentation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_dictee.config import Secrets, load_config, override_model_names, run_output_name
from evaluation_dictee.data.loaders import load_labels
from evaluation_dictee.evaluation.report import load_predictions
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.s3_export import (
    SCORING_SUFFIX,
    export_run,
    resolve_dest_prefix,
    resolve_run_name,
)

logger = get_logger(__name__)

#: Marge sous 100 % du corpus qui compte encore comme un run COMPLET : quelques
#: copies perdues (échec API, retries épuisés) ne doivent pas reclasser un run
#: quasi complet en expérimentation.
SEUIL_COMPLETUDE = 0.9


def _est_run_complet(config, chemin_predictions: Path) -> bool:
    """Le run couvre-t-il tout le corpus, ou seulement un échantillon ?

    Compare l'effectif RÉELLEMENT présent dans les prédictions au nombre de copies
    du corpus (CSV des labels), plutôt que `data.limit` du YAML — voir le docstring
    du module pour la raison.

    Args:
        config: configuration de l'expérience (fournit `data.labels_path`).
        chemin_predictions: fichier de prédictions à exporter.

    Returns:
        True si le run couvre au moins `SEUIL_COMPLETUDE` du corpus.
    """
    if not chemin_predictions.is_file():
        # Fichier absent : `export_run` le signalera clairement juste après, avec un
        # message actionnable. Ici on ne fait que refuser de trancher sur du vide.
        return False
    df = load_predictions(str(chemin_predictions))
    if df.empty or "copy_id" not in df.columns:
        return False
    n_reel = df["copy_id"].nunique()
    n_total = len(load_labels(config.data.labels_path))
    return n_total > 0 and n_reel >= SEUIL_COMPLETUDE * n_total


def main() -> None:
    """Résout le nom du run et la destination, puis exporte vers S3."""
    parser = argparse.ArgumentParser(
        description="Exporte les prédictions d'un run vers S3 (répertoire predictions/)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML du run (le nom est lu dans le champ `name`).")
    source.add_argument("--run-name", help="Nom du run (préfixe du fichier de prédictions).")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Surcharge le modèle de la config (comme run_benchmark.py --model-name).",
    )
    parser.add_argument(
        "--htr",
        action="store_true",
        help="Exporte le fichier HTR (<name>_htr_predictions.jsonl) au lieu du scoring.",
    )
    parser.add_argument(
        "--experimentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Route vers predictions/experimentations/ plutôt que predictions/. Déduit "
            "avec --config en comparant l'effectif réel des prédictions au corpus "
            "complet ; OBLIGATOIRE avec --run-name, faute de YAML pour le déduire."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default="data/processed",
        help="Dossier local des prédictions. [défaut : data/processed]",
    )
    parser.add_argument(
        "--dest-prefix",
        default=None,
        help="Préfixe S3 de base. [défaut : S3_PREDICTIONS_PREFIX]",
    )
    args = parser.parse_args()

    experimentation = args.experimentation
    if args.config and not args.htr:
        # Un seul chargement : le nom du run ET la destination en dépendent tous deux.
        config = override_model_names(load_config(args.config), args.model_name)
        run_name = run_output_name(config)
        if experimentation is None:
            chemin = Path(args.source_dir) / f"{run_name}{SCORING_SUFFIX}"
            complet = _est_run_complet(config, chemin)
            experimentation = not complet
            logger.info(
                "Destination déduite de l'effectif réel du run (%s le corpus) : %s",
                "couvre" if complet else "ne couvre pas",
                "predictions/experimentations/" if experimentation else "predictions/",
            )
    elif args.config:  # --htr : pas de notion d'expérimentation, nommage par `name` seul
        run_name = resolve_run_name(args.config, htr=True)
    else:
        run_name = args.run_name
        if experimentation is None:
            parser.error(
                "--experimentation (ou --no-experimentation) est requis avec --run-name : "
                "sans YAML, la destination ne peut pas être déduite."
            )

    if experimentation is None:
        # HTR sans réglage explicite : pas d'ambiguïté à lever pour le site, qui ne
        # lit jamais les runs HTR ; on garde l'emplacement historique par défaut.
        experimentation = False

    base_prefix = args.dest_prefix or Secrets().s3_predictions_prefix
    dest_prefix = resolve_dest_prefix(base_prefix, experimentation)

    dest = export_run(run_name, dest_prefix, source_dir=args.source_dir, htr=args.htr)
    logger.info("OK — prédictions disponibles sur S3 : %s", dest)


if __name__ == "__main__":
    main()
