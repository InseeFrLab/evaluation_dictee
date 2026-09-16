"""Reconstitue `<run>_copies_ecartees.csv` pour un run antérieur à la décision D8.

Le fichier de prédictions d'un tel run ne porte pas le champ `exclusion` — le code qui
l'écrit n'existait pas encore — et `run_benchmark` n'a donc jamais produit sa liste de
copies écartées. Ce script rejoue exactement la même règle (copie écartée en entier si
majoritairement inexploitable, motif = « vierge » ou « illisible ») sur un JSONL déjà
écrit, sans rien réévaluer auprès du modèle.

Usage :
    uv run scripts/backfill_copies_ecartees.py --run-name dictee_end2end_cot_qwen3-6-35b-moe
    uv run scripts/backfill_copies_ecartees.py --tous   # tous les runs sans sidecar
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from evaluation_dictee.data import reference
from evaluation_dictee.evaluation.report import load_predictions
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def calculer_copies_ecartees(chemin: Path) -> dict[str, tuple[str, int, int]]:
    """Applique la règle de majorité du pipeline à un JSONL de prédictions déjà écrit.

    Args:
        chemin: fichier `<run>_predictions.jsonl`.

    Returns:
        Par copie écartée : (motif, items concernés, items au total).
    """
    df = load_predictions(chemin)
    if df.empty:
        return {}

    def motif(rec: dict) -> str | None:
        if rec.get("exclusion") is not None:
            return str(rec["exclusion"])
        if rec.get("blank"):
            return "vierge"
        if not reference.est_evaluable(str(rec.get("y_true", ""))):
            return "illisible"
        return None

    par_copie: dict[str, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    for rec in df.to_dict("records"):
        total[rec["copy_id"]] += 1
        m = motif(rec)
        if m is not None:
            par_copie[rec["copy_id"]][m] += 1

    ecartees = {}
    for copy_id, compte in par_copie.items():
        m, n = compte.most_common(1)[0]
        if n * 2 > total[copy_id]:
            ecartees[copy_id] = (m, n, total[copy_id])
    return ecartees


def ecrire_csv(ecartees: dict[str, tuple[str, int, int]], destination: Path) -> bool:
    """Écrit le CSV s'il y a quelque chose à écarter. Renvoie False sinon (rien fait)."""
    if not ecartees:
        return False
    with open(destination, "w", encoding="utf-8") as f:
        f.write("copy_id;motif;items_concernes;items_total\n")
        for copy_id, (motif, n_concernes, n_total) in sorted(ecartees.items()):
            f.write(f"{copy_id};{motif};{n_concernes};{n_total}\n")
    return True


def main() -> None:
    """Reconstitue le sidecar d'un run, ou de tous ceux qui n'en ont pas."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", help="Nom du run (préfixe du fichier de prédictions).")
    parser.add_argument(
        "--tous",
        action="store_true",
        help="Traite tous les runs de --dossier dont le sidecar est absent.",
    )
    parser.add_argument("--dossier", default="data/processed", help="Dossier des prédictions.")
    args = parser.parse_args()
    if not args.run_name and not args.tous:
        parser.error("préciser --run-name ou --tous")

    dossier = Path(args.dossier)
    if args.tous:
        runs = [
            p.name.removesuffix("_predictions.jsonl")
            for p in sorted(dossier.glob("*_predictions.jsonl"))
            if not (
                dossier / f"{p.name.removesuffix('_predictions.jsonl')}_copies_ecartees.csv"
            ).exists()
        ]
    else:
        runs = [args.run_name]

    for run in runs:
        chemin = dossier / f"{run}_predictions.jsonl"
        if not chemin.exists():
            logger.warning("Introuvable, sauté : %s", chemin)
            continue
        ecartees = calculer_copies_ecartees(chemin)
        destination = dossier / f"{run}_copies_ecartees.csv"
        if ecrire_csv(ecartees, destination):
            logger.info("%s : %d copie(s) écartée(s) -> %s", run, len(ecartees), destination)
        else:
            logger.info("%s : aucune copie à écarter, rien écrit.", run)


if __name__ == "__main__":
    main()
