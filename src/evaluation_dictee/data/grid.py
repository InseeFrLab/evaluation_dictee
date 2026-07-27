"""Chargement de la grille de codage de la dictée (JSON versionné dans configs/).

Le texte de référence est désormais dérivé de la grille.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class GridItem:
    """Un item de la grille : mot attendu, type et fautes connues."""

    item_id: str
    attendu: str
    type: str  # "mot" ou "ponctuation"
    ex_lexicale: list[str]
    ex_grammaticale: list[str]
    ex_les_deux: list[str]


@dataclass
class DictationGrid:
    """Grille complète de la dictée."""

    reference_text: str
    items: list[GridItem]


def load_grid(path: str) -> DictationGrid:
    """Charge la grille de codage depuis un fichier JSON.

    Args:
        path: Chemin du fichier JSON de la grille.

    Returns:
        La grille désérialisée : texte de référence et liste des items.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = [
        GridItem(
            item_id=it["id"],
            attendu=it["attendu"],
            type=it["type"],
            ex_lexicale=it.get("ex_lexicale", []),
            ex_grammaticale=it.get("ex_grammaticale", []),
            ex_les_deux=it.get("ex_les_deux", []),
        )
        for it in raw["items"]
    ]
    return DictationGrid(reference_text=raw["reference_text"], items=items)
