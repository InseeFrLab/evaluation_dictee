"""Segmentation en lignes par Doc-UFCN (Teklia), pour la sonde PyLaia.

Script AUTONOME, à lancer avec l'interpréteur d'un environnement dédié : Doc-UFCN
épingle torch 2.x, PyLaia torch 1.13, et aucun des deux n'entre dans le venv du
projet. Il n'importe donc rien du paquet `evaluation_dictee`.

`scripts/probe_pylaia_htr.py --segmenteur doc-ufcn` l'appelle en sous-processus ; on
peut aussi le lancer seul pour inspecter les polygones.

Prérequis :

    uv venv --python 3.10 /chemin/ufcn-env
    VIRTUAL_ENV=/chemin/ufcn-env uv pip install doc-ufcn

Usage :
    /chemin/ufcn-env/bin/python scripts/segment_doc_ufcn.py PAGES_DIR SORTIE.json \
        --modele generic-historical-line
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from doc_ufcn import models
from doc_ufcn.main import DocUFCN


def main() -> None:
    """Détecte les lignes de chaque page d'un dossier et écrit leurs polygones en JSON."""
    parser = argparse.ArgumentParser(description="Segmentation en lignes par Doc-UFCN.")
    parser.add_argument("pages_dir", type=Path, help="Dossier des pages (*.png).")
    parser.add_argument("sortie", type=Path, help="Fichier JSON {page: [polygones]}.")
    parser.add_argument(
        "--modele",
        default="generic-historical-line",
        help="Modèle Teklia/doc-ufcn-<nom> sur HuggingFace.",
    )
    parser.add_argument(
        "--min-cc", type=int, default=None, help="Aire minimale d'une ligne (défaut : modèle)."
    )
    args = parser.parse_args()

    chemin, params = models.download_model(args.modele)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    modele = DocUFCN(len(params["classes"]), params["input_size"], appareil)
    modele.load(chemin, params["mean"], params["std"])
    min_cc = args.min_cc if args.min_cc is not None else params.get("min_cc", 50)

    resultats: dict[str, list[dict]] = {}
    for page in sorted(args.pages_dir.glob("*.png")):
        image = cv2.cvtColor(cv2.imread(str(page)), cv2.COLOR_BGR2RGB)
        polygones, *_ = modele.predict(image, min_cc=min_cc)
        # Classe 0 = fond ; toutes les autres sont des lignes (horizontales, verticales…).
        resultats[page.stem] = [
            {
                "classe": params["classes"][canal],
                "confiance": float(p["confidence"]),
                "polygone": [[int(x), int(y)] for x, y in p["polygon"]],
            }
            for canal in range(1, len(params["classes"]))
            for p in polygones.get(canal, [])
        ]
        print(f"{page.stem} : {len(resultats[page.stem])} lignes", flush=True)

    args.sortie.write_text(json.dumps(resultats), encoding="utf-8")


if __name__ == "__main__":
    main()
