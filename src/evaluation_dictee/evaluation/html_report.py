"""Exporte un HTML autonome du notebook d'analyse en ne gardant que les sections choisies.

Les sections sont repérées par un tag de cellule `section:<nom>`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor
from traitlets.config import Config

_SECTION_PREFIX = "section:"


@dataclass
class SectionInfo:
    """Description d'une section repérable du notebook."""

    tag: str
    title: str


def _find_first_title(cell: nbformat.NotebookNode) -> str:
    """Extrait le premier titre markdown non vide de la cellule.

    Args:
        cell: cellule de notebook à inspecter.

    Returns:
        Le premier titre markdown (sans les `#`), ou à défaut la première ligne
        tronquée à 80 caractères.
    """
    src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
    for line in src.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return src.strip().split("\n", 1)[0][:80]


def list_sections(notebook_path: str | Path) -> list[SectionInfo]:
    """Liste les sections d'un notebook (première cellule taguée), dans l'ordre d'apparition.

    Args:
        notebook_path: chemin du notebook `.ipynb`.

    Returns:
        Les sections repérées (tag `section:<nom>` et titre), sans doublon.
    """
    nb = nbformat.read(str(notebook_path), as_version=4)
    sections: list[SectionInfo] = []
    seen: set[str] = set()
    for cell in nb.cells:
        for tag in cell.metadata.get("tags", []):
            if tag.startswith(_SECTION_PREFIX) and tag not in seen:
                sections.append(SectionInfo(tag=tag, title=_find_first_title(cell)))
                seen.add(tag)
    return sections


def _filter_by_sections(
    nb: nbformat.NotebookNode, selected_tags: list[str]
) -> nbformat.NotebookNode:
    """Ne garde que les cellules des sections sélectionnées.

    Chaque cellule appartient à la dernière section rencontrée. Les cellules avant
    la 1re section (imports, paramètres) sont toujours conservées.

    Args:
        nb: notebook à filtrer.
        selected_tags: tags de section à conserver.

    Returns:
        Une copie du notebook ne contenant que l'en-tête et les sections choisies.
    """
    filtered = copy.deepcopy(nb)
    kept_cells: list[nbformat.NotebookNode] = []
    current_section: str | None = None
    seen_first_section = False

    for cell in filtered.cells:
        tags = cell.metadata.get("tags", [])
        section_tags = [t for t in tags if t.startswith(_SECTION_PREFIX)]
        if section_tags:
            current_section = section_tags[0]
            seen_first_section = True

        # En-tête (avant la 1re section) toujours gardé ; sinon selon la sélection
        if not seen_first_section or (current_section in selected_tags):
            kept_cells.append(cell)

    filtered.cells = kept_cells
    return filtered


def _run_notebook(nb: nbformat.NotebookNode, notebook_dir: Path) -> nbformat.NotebookNode:
    """Exécute le notebook (nécessaire pour que les sorties apparaissent en HTML).

    Args:
        nb: notebook à exécuter.
        notebook_dir: répertoire de travail du kernel (chemins relatifs).

    Returns:
        Le notebook exécuté, sorties de cellules incluses.
    """
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(notebook_dir)}})
    return nb


def _export_html(
    nb: nbformat.NotebookNode,
    hide_code: bool = True,
    template_name: str = "lab",
) -> str:
    """Convertit le notebook exécuté en HTML autonome (assets inlinés).

    Args:
        nb: notebook (idéalement déjà exécuté) à convertir.
        hide_code: si True, masque les cellules de code et les invites.
        template_name: template nbconvert à utiliser.

    Returns:
        Le corps HTML du notebook.
    """
    c = Config()
    # Retire les cellules taguées "hide"/"remove_cell" (cellules techniques)
    c.TagRemovePreprocessor.remove_cell_tags = {"hide", "remove_cell"}
    if hide_code:
        c.TagRemovePreprocessor.remove_input_tags = {"hide_input"}
    c.TagRemovePreprocessor.enabled = True

    exporter = HTMLExporter(config=c, template_name=template_name)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)
    if hide_code:
        exporter.exclude_input = True
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True

    body, _ = exporter.from_notebook_node(nb)
    return body


def build_html_report(
    notebook_path: str | Path,
    selected_tags: list[str],
    output_path: str | Path,
    hide_code: bool = True,
    execute: bool = True,
) -> Path:
    """Exécute le notebook et exporte les sections choisies en HTML.

    Args:
        notebook_path: chemin du notebook `.ipynb`.
        selected_tags: tags de section à conserver.
        output_path: chemin du fichier HTML à écrire.
        hide_code: si True, masque les cellules de code dans la sortie.
        execute: si True, exécute le notebook avant l'export.

    Returns:
        Le chemin du fichier HTML écrit.
    """
    notebook_path = Path(notebook_path)
    nb = nbformat.read(str(notebook_path), as_version=4)

    if execute:
        nb = _run_notebook(nb, notebook_path.parent)

    filtered = _filter_by_sections(nb, selected_tags)
    html = _export_html(filtered, hide_code=hide_code)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def bootstrap_prevalence_ci(
    df,
    item_col: str = "item_id",
    label_col: str = "y_true",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
):
    """Bootstrap groupé par copie de la prévalence d'erreur par item.

    Ré-échantillonne les COPIES (pas les items) pour respecter la structure
    hiérarchique de la donnée.

    Args:
        df: prédictions à l'item (colonne copy_id requise).
        item_col: colonne identifiant l'item.
        label_col: colonne des codes ; « erreur » = code != "1".
        n_boot: nombre de tirages bootstrap.
        level: niveau de confiance de l'intervalle.
        seed: graine du générateur aléatoire (reproductibilité).

    Returns:
        Un DataFrame indexé par item : prévalence d'erreur estimée (%) et bornes
        basse/haute de l'intervalle bootstrap.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    copies = df["copy_id"].unique()
    items = sorted(df[item_col].unique())

    est = df.groupby(item_col)[label_col].apply(lambda s: (s != "1").mean() * 100)

    samples = np.zeros((n_boot, len(items)))
    for b in range(n_boot):
        sampled_copies = rng.choice(copies, size=len(copies), replace=True)
        sub = pd.concat([df[df["copy_id"] == c] for c in sampled_copies], ignore_index=True)
        grouped = sub.groupby(item_col)[label_col].apply(lambda s: (s != "1").mean() * 100)
        for j, it in enumerate(items):
            samples[b, j] = grouped.get(it, np.nan)

    alpha = (1 - level) / 2
    lo = np.nanpercentile(samples, alpha * 100, axis=0)
    hi = np.nanpercentile(samples, (1 - alpha) * 100, axis=0)

    return pd.DataFrame(
        {"estimate": [est.get(i, np.nan) for i in items], "lo": lo, "hi": hi},
        index=pd.Index(items, name=item_col),
    )
