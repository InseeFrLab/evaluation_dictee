"""Chargement des imagettes et des labels experts (local ou S3).

Les fichiers DEPP sont des TIFF 1 bit à extension trompeuse ".png" : on les
normalise systématiquement en niveaux de gris.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import fsspec
import numpy as np
from PIL import Image


@dataclass
class Copy:
    """Une copie d'élève : son identifiant, son image et les codes experts."""

    copy_id: str
    image_path: str
    expert_codes: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)


def _join(base: str, name: str) -> str:
    """Concatène un dossier (local ou s3://) et un nom de fichier.

    Args:
        base: Dossier de base, avec ou sans slash final.
        name: Nom du fichier à ajouter.

    Returns:
        Le chemin complet `base/name` (un seul slash de séparation).
    """
    return base.rstrip("/") + "/" + name


def load_image(path: str) -> Image.Image:
    """Charge une image (locale ou S3) en niveaux de gris (gère les TIFF 1 bit déguisés en .png).

    Args:
        path: Chemin local ou S3 de l'image.

    Returns:
        L'image convertie en niveaux de gris (mode "L").
    """
    with fsspec.open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
    return img.convert("L")


def ink_ratio(image: Image.Image) -> float:
    """Proportion de pixels sombres (encre) d'une image en niveaux de gris.

    Sert à repérer les formulaires vierges (l'élève n'a rien écrit) : seul le
    pré-imprimé (titre, lignes, cadre, pied de page) marque quelques pour cent.

    Args:
        image: Image PIL (convertie en "L" si besoin).

    Returns:
        Fraction de pixels d'intensité < 128, entre 0.0 et 1.0.
    """
    arr = np.asarray(image.convert("L"))
    return float((arr < 128).mean())


def load_labels(csv_path: str) -> dict[str, dict[str, str]]:
    """Charge les codes de l'annotateur depuis le CSV de correction (séparateur ';', local ou S3).

    Structure : colonne d'index, colonne nom d'image, puis une colonne par item.

    Args:
        csv_path: Chemin local ou S3 du CSV de correction.

    Returns:
        Un dictionnaire `copy_id -> {item_id: code}`, valeurs nettoyées des espaces.
    """
    labels: dict[str, dict[str, str]] = {}
    with fsspec.open(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        item_ids = header[2:]
        for row in reader:
            if not row:
                continue
            copy_id = row[1].strip()
            labels[copy_id] = {item: row[i + 2].strip() for i, item in enumerate(item_ids)}
    return labels


def _exists(path: str) -> bool:
    """Teste l'existence d'un fichier local ou S3.

    Args:
        path: Chemin local ou S3 à vérifier.

    Returns:
        True si le fichier existe, False sinon.
    """
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return bool(paths) and fs.exists(paths[0])


def load_dataset(
    images_dir: str,
    labels_csv: str,
    limit: int | None = None,
) -> list[Copy]:
    """Construit la liste des copies à partir des images et du CSV de labels.

    Ne retient que les copies dont l'image existe réellement dans `images_dir`.

    Args:
        images_dir: Dossier (local ou S3) contenant les imagettes.
        labels_csv: Chemin du CSV des codes experts.
        limit: Nombre maximal de copies à charger (toutes si None).

    Returns:
        La liste des copies trouvées, triées par identifiant.
    """
    labels = load_labels(labels_csv)

    copies: list[Copy] = []
    for copy_id in sorted(labels):
        image_path = _join(images_dir, copy_id)
        if not _exists(image_path):
            continue
        item_codes = labels[copy_id]
        copies.append(
            Copy(
                copy_id=copy_id,
                image_path=image_path,
                item_ids=list(item_codes.keys()),
                expert_codes=list(item_codes.values()),
            )
        )
        if limit is not None and len(copies) >= limit:
            break
    return copies
