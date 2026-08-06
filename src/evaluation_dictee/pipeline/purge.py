"""Préparation de la reprise d'un run : retrait des copies non exploitables.

Un item codé `CODE_NON_PARSE` n'a pas de valeur d'évaluation : il signale que la
réponse du modèle n'a pu être ni parsée ni alignée (réponse vide, JSON cassé, item
absent de la réponse). Il compte pourtant en désaccord dans toutes les métriques,
et dégrade donc le modèle pour une raison purement technique.

Le risque vient de la reprise : le benchmark saute les copies déjà présentes dans le
JSONL, sans regarder leur contenu. Une copie ratée y reste figée, même après
correction de la cause — c'est ce qui est arrivé au run two_stage du 5 août 2026, où
84 copies sont restées entièrement non parsées alors que les options d'appel de
l'étape 2 avaient été corrigées entre-temps : la relance sautait exactement les
copies à refaire.

`preparer_reprise` est donc appelée au démarrage de chaque run (`run_benchmark`) :
elle retire du fichier les copies sans aucun code exploitable et renvoie les copies
réellement acquises. Aucun point d'entrée CLI : ce nettoyage n'a de sens que dans le
lancement du pipeline, dont il conditionne la liste des copies à traiter.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

from evaluation_dictee.models.base import CODE_NON_PARSE
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

#: Part d'items non parsés à partir de laquelle une copie est retirée du fichier.
#: 1.0 = seulement les copies dont AUCUN item n'est exploitable. Une valeur plus
#: basse refait aussi les copies partiellement ratées, au prix de recoder des items
#: valides — et, à chaque relance, de refaire les copies dont le modèle omet
#: régulièrement un item.
SEUIL_PURGE = 1.0


def compter_non_parses(chemin: str | Path) -> tuple[Counter[str], Counter[str]]:
    """Compte, par copie, les items présents et les items non parsés.

    Args:
        chemin: fichier JSONL de prédictions.

    Returns:
        Le couple (items par copie, items non parsés par copie). Les lignes
        illisibles — dernière ligne tronquée par un crash, par exemple — sont
        ignorées.
    """
    n_items: Counter[str] = Counter()
    n_non_parses: Counter[str] = Counter()
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                rec = json.loads(ligne)
                copy_id = rec["copy_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            n_items[copy_id] += 1
            if str(rec.get("y_pred")) == CODE_NON_PARSE:
                n_non_parses[copy_id] += 1
    return n_items, n_non_parses


def copies_a_purger(
    n_items: Counter[str], n_non_parses: Counter[str], seuil: float = SEUIL_PURGE
) -> set[str]:
    """Copies dont la part d'items non parsés atteint le seuil.

    Args:
        n_items: nombre d'items par copie.
        n_non_parses: nombre d'items non parsés par copie.
        seuil: part minimale d'items non parsés pour retirer la copie. Une copie
            sans aucun item non parsé n'est jamais retenue, même à seuil nul.

    Returns:
        L'ensemble des copy_id à retirer du fichier.
    """
    return {
        copy_id
        for copy_id, total in n_items.items()
        if n_non_parses.get(copy_id, 0) and n_non_parses[copy_id] / total >= seuil
    }


def purger(chemin: str | Path, a_purger: set[str]) -> int:
    """Réécrit le fichier de prédictions sans les copies indiquées.

    Le fichier d'origine est d'abord copié en `.bak`, et la réécriture passe par un
    temporaire suivi d'un `os.replace` : le fichier de prédictions n'est jamais
    partiel, même si le process meurt en cours de route. C'est le même niveau de
    garantie que l'écriture incrémentale du benchmark.

    Args:
        chemin: fichier JSONL de prédictions.
        a_purger: copy_id à retirer.

    Returns:
        Le nombre de lignes supprimées.
    """
    chemin = Path(chemin)
    shutil.copy2(chemin, chemin.with_suffix(chemin.suffix + ".bak"))
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    supprimees = 0
    with (
        open(chemin, encoding="utf-8") as source,
        open(temporaire, "w", encoding="utf-8") as cible,
    ):
        for ligne in source:
            nue = ligne.strip()
            if nue:
                try:
                    if json.loads(nue)["copy_id"] in a_purger:
                        supprimees += 1
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass
            cible.write(ligne)
        cible.flush()
        os.fsync(cible.fileno())
    os.replace(temporaire, chemin)
    return supprimees


def preparer_reprise(chemin: str | Path, seuil: float = SEUIL_PURGE) -> set[str]:
    """Nettoie le fichier de prédictions et renvoie les copies déjà acquises.

    Appelée au démarrage d'un run. Sans fichier (premier lancement), ne fait rien.
    Sans copie ratée — le cas normal —, ne réécrit rien : le fichier n'est que relu.

    Args:
        chemin: fichier JSONL de prédictions du run.
        seuil: part d'items non parsés à partir de laquelle refaire une copie.

    Returns:
        Les copy_id à considérer comme traités, donc à sauter.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        return set()

    n_items, n_non_parses = compter_non_parses(chemin)
    a_purger = copies_a_purger(n_items, n_non_parses, seuil)
    if not a_purger:
        return set(n_items)

    items_perdus = sum(n_non_parses[c] for c in a_purger)
    logger.warning(
        "%d copies déjà présentes n'ont aucun code exploitable (%d items « %s ») : "
        "elles sont retirées du fichier et vont être refaites. Sauvegarde : %s.bak",
        len(a_purger),
        items_perdus,
        CODE_NON_PARSE,
        chemin.name,
    )
    supprimees = purger(chemin, a_purger)
    logger.info("%d lignes retirées de %s.", supprimees, chemin.name)
    return set(n_items) - a_purger
