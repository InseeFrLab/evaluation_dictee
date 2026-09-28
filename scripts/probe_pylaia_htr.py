"""Sonde exploratoire : PyLaia (Teklia) sur le corpus Scoledit.

Script AUTONOME, hors pipeline. Rien ne l'appelle, il n'appelle aucun scorer et ne
modifie aucun fichier de `src/`. Il emprunte seulement, en LECTURE SEULE, le chargeur
Scoledit et le calcul CER/WER du paquet, pour que ses chiffres soient directement
comparables à ceux des runs VLM.

Question à laquelle il répond : un HTR classique français lit-il l'écriture d'élèves
assez bien pour qu'il vaille la peine de le brancher dans l'approche two-step ?

Chaîne : Scoledit → segmentation en lignes → `pylaia-htr-decode-ctc` → réassemblage
→ CER/WER.

⚠ La segmentation en lignes est une projection horizontale d'encre, volontairement
rudimentaire : PyLaia travaille à la ligne, pas à la page, et le projet n'a pas de
segmenteur. C'est le maillon faible de la chaîne. Un CER mauvais peut venir d'elle
autant que du modèle — TOUJOURS regarder les imagettes de lignes écrites dans
`<out-dir>/lignes/` avant de conclure quoi que ce soit sur PyLaia.

Prérequis (aucun n'est installé par `uv sync`, c'est voulu : rien n'entre dans les
dépendances du projet pour une sonde) :

    uv pip install pylaia
    git clone https://huggingface.co/Teklia/pylaia-rimes    # modèle français (RIMES)

Usage :
    uv run scripts/probe_pylaia_htr.py --model-dir pylaia-rimes --limit 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation_dictee.data.loaders import load_image
from evaluation_dictee.transcription.htr_metrics import compute_transcription_metrics
from evaluation_dictee.transcription.scoledit import ScoledtSample, load_scoledit_dataset
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

#: Hauteur à laquelle Teklia redimensionne ses lignes à l'entraînement (ratio préservé).
HAUTEUR_LIGNE = 128

#: Fichiers attendus dans un dépôt de modèle PyLaia publié par Teklia.
FICHIERS_MODELE = ("syms.txt", "model", "weights.ckpt")

#: Interligne Seyès (px) séparant les deux résolutions de Scoledit (≈ 47 px / ≈ 171 px).
SEUIL_HAUTE_RESOLUTION = 100

#: Réglages du dérèglage et de la segmentation, par résolution de scan. « basse » :
#: les valeurs par défaut des fonctions, mises au point sur les scans à ≈ 47 px
#: d'interligne. « haute » : même dérèglage — agrandir `longueur_v` en proportion
#: (40 → 145) laisse survivre les réglures verticales, que l'inclinaison du scan
#: découpe en tronçons courts, et fusionne la page en 1 à 2 bandes —, mais
#: segmentation agrandie (≈ ×2) pour ne plus éclater une ligne en plusieurs bandes.
REGLAGES: dict[str, dict[str, dict[str, int]]] = {
    "basse": {
        "reglures": {"longueur_v": 40, "longueur_h": 30, "epaisseur_max": 6},
        "segmentation": {"hauteur_min": 20, "marge": 6, "lissage": 5},
    },
    "haute": {
        "reglures": {"longueur_v": 40, "longueur_h": 30, "epaisseur_max": 6},
        "segmentation": {"hauteur_min": 40, "marge": 12, "lissage": 10},
    },
}


@dataclass
class ResultatCopie:
    """Résultat de la sonde pour une copie Scoledit."""

    scan: str
    level: str
    n_lignes: int
    cer: float
    wer: float
    cer_normalise: float
    wer_normalise: float
    reference: str
    hypothese: str


# ───────────────────────── Prétraitement : réglures Seyès ─────────────────────────


def seuil_otsu(arr: np.ndarray, plafond: int = 250) -> int:
    """Seuil d'Otsu séparant l'encre sombre de l'encre claire, sur une page donnée.

    Calculé sur les seuls pixels non blancs (< `plafond`) : la question n'est pas
    « encre ou papier » mais « crayon ou réglure », les deux étant de l'encre.

    Un seuil fixe ne convient pas : mesuré sur neuf copies, l'optimum varie de 148
    à 202 selon que la copie est écrite au crayon appuyé ou à peine marquée. Un
    seuil trop haut conserve les réglures d'une page sombre, trop bas il efface
    l'écriture d'une page pâle.

    Args:
        arr: Page en niveaux de gris.
        plafond: Au-delà, le pixel est du papier et n'entre pas dans le calcul.

    Returns:
        Le seuil, borné à [140, 210] pour écarter les pages dégénérées.
    """
    valeurs = arr[arr < plafond]
    if valeurs.size == 0:
        return 170
    hist = np.histogram(valeurs, bins=256, range=(0, 256))[0].astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.clip(np.nanargmax(variance), 140, 210))


def supprimer_reglures(
    image: Image.Image,
    seuil: int | None = None,
    longueur_v: int = 40,
    longueur_h: int = 30,
    epaisseur_max: int = 6,
) -> Image.Image:
    """Retire les réglures imprimées d'une page de cahier Seyès.

    Deux leviers, parce qu'aucun ne suffit seul :

    1. **Le seuil.** Les réglures HORIZONTALES sont nettement plus claires que le
       crayon (sur 103a : l'encre passe de 8,1 % à 4,6 % des pixels à seuil 170).
       C'est ce qui les élimine, et non la morphologie : la page étant légèrement
       inclinée au scan, une réglure horizontale dérive en ordonnée et ne forme
       jamais un segment continu sur une rangée de pixels. Une ouverture
       horizontale n'en détectait littéralement aucune (0,0000).
    2. **La morphologie.** Les réglures VERTICALES, elles, sont sombres et
       survivent au seuil, mais l'ouverture les capture bien. Un trait d'écriture
       qui croise une réglure est localement épais : on ne blanchit donc un pixel
       que si l'épaisseur perpendiculaire y est faible, ce qui préserve les
       traversées.

    Les scans Scoledit sont en RGB mais neutres (B−R ≈ 0) : aucune séparation par
    la couleur n'est possible, seule la forme et l'intensité discriminent.

    Appliquer ce nettoyage AVANT la segmentation : les réglures faussent la
    détection des lignes autant qu'elles gênent la lecture.

    Args:
        image: Page en niveaux de gris.
        seuil: En deçà, le pixel est tenu pour de l'encre ; au-delà il est blanchi.
            `None` (défaut) le calcule par Otsu sur la page — voir `seuil_otsu`.
        longueur_v: Longueur minimale, en pixels, d'une réglure verticale.
        longueur_h: Longueur minimale, en pixels, d'une réglure horizontale résiduelle.
        epaisseur_max: Épaisseur au-delà de laquelle on tient le trait pour de l'écriture.

    Returns:
        La page nettoyée, en niveaux de gris.
    """
    from scipy import ndimage  # via scikit-learn ; import local, la sonde reste optionnelle

    arr = np.asarray(image.convert("L"))
    encre = arr < (seuil_otsu(arr) if seuil is None else seuil)

    longues_v = ndimage.binary_opening(encre, structure=np.ones((longueur_v, 1)))
    longues_h = ndimage.binary_opening(encre, structure=np.ones((1, longueur_h)))
    epais_v = ndimage.binary_opening(encre, structure=np.ones((epaisseur_max, 1)))
    epais_h = ndimage.binary_opening(encre, structure=np.ones((1, epaisseur_max)))

    reglures = (longues_v & ~epais_h) | (longues_h & ~epais_v)
    return Image.fromarray(np.where(encre & ~reglures, arr, 255).astype(np.uint8))


# ───────────────────────── Normalisation de l'échelle ─────────────────────────


def estimer_interligne(image: Image.Image, mini: int = 15, maxi: int = 250) -> int:
    """Estime l'interligne Seyès d'une page, en pixels, par autocorrélation.

    Scoledit mélange deux résolutions de scan (≈ 550 px et ≈ 2 100 px de large, soit
    un interligne de ≈ 47 px contre ≈ 171 px). Les réglures imprimées, régulières,
    donnent au profil vertical de l'encre claire une période nette : le premier pic
    de son autocorrélation. On la mesure sur la page BRUTE, avant dérèglage.

    Args:
        image: Page en niveaux de gris, réglures comprises.
        mini: Plus petite période cherchée, en pixels.
        maxi: Plus grande période cherchée, en pixels.

    Returns:
        L'interligne estimé, en pixels.
    """
    arr = np.asarray(image.convert("L"))
    profil = (arr < 200).sum(axis=1).astype(float)
    profil -= profil.mean()
    auto = np.correlate(profil, profil, mode="full")[len(profil) - 1 :]
    maxi = min(maxi, len(auto) - 1)
    return int(np.argmax(auto[mini:maxi]) + mini)


def normaliser_echelle(image: Image.Image, interligne_cible: int) -> Image.Image:
    """Remet une page à l'échelle pour que son interligne Seyès vaille `interligne_cible`.

    Tous les réglages du prétraitement (longueur des réglures, épaisseur d'un trait,
    hauteur minimale d'une bande…) sont en pixels absolus. Sans normalisation, ils ne
    conviennent qu'à une seule résolution : sur les scans haute résolution, chaque
    ligne d'écriture était éclatée en plusieurs bandes et le CER dépassait 75 %.

    Args:
        image: Page en niveaux de gris, réglures comprises.
        interligne_cible: Interligne visé, en pixels.

    Returns:
        La page redimensionnée (inchangée si l'écart est inférieur à 10 %).
    """
    facteur = interligne_cible / estimer_interligne(image)
    if abs(facteur - 1) < 0.1:
        return image
    taille = (max(1, round(image.width * facteur)), max(1, round(image.height * facteur)))
    return image.resize(taille, Image.Resampling.LANCZOS)


# ───────────────────────── Segmentation en lignes ─────────────────────────


def _plages_vraies(masque: np.ndarray) -> list[tuple[int, int]]:
    """Repère les plages continues de `True` dans un masque booléen 1D.

    Args:
        masque: Tableau booléen.

    Returns:
        Liste de couples (début inclus, fin exclue).
    """
    if not masque.any():
        return []
    bords = np.diff(masque.astype(np.int8))
    debuts = [int(i) + 1 for i in np.flatnonzero(bords == 1)]
    fins = [int(i) + 1 for i in np.flatnonzero(bords == -1)]
    if masque[0]:
        debuts.insert(0, 0)
    if masque[-1]:
        fins.append(len(masque))
    return list(zip(debuts, fins, strict=True))


def segmenter_lignes(
    image: Image.Image,
    hauteur_min: int = 20,
    marge: int = 6,
    lissage: int = 5,
    fraction_seuil: float = 0.08,
) -> list[Image.Image]:
    """Découpe une page en imagettes de lignes par projection horizontale de l'encre.

    Chaque ligne du scan donne un compte de pixels sombres ; les bandes où ce compte
    dépasse un seuil relatif au maximum de la page sont considérées comme du texte.

    Args:
        image: Page en niveaux de gris.
        hauteur_min: Hauteur minimale d'une bande retenue, en pixels (filtre le bruit).
        marge: Marge ajoutée en haut et en bas de chaque bande, en pixels.
        lissage: Largeur de la moyenne glissante appliquée au profil.
        fraction_seuil: Seuil de détection, en fraction du maximum du profil.

    Returns:
        Les imagettes de lignes, de haut en bas, redimensionnées à `HAUTEUR_LIGNE`.
    """
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    profil = (arr < 128).sum(axis=1).astype(float)
    if lissage > 1:
        profil = np.convolve(profil, np.ones(lissage) / lissage, mode="same")

    seuil = max(profil.max() * fraction_seuil, 1.0)
    hauteur, largeur = arr.shape

    lignes: list[Image.Image] = []
    for y0, y1 in _plages_vraies(profil > seuil):
        if y1 - y0 < hauteur_min:
            continue
        haut = max(0, y0 - marge)
        bas = min(hauteur, y1 + marge)

        # Rogner aussi les marges blanches latérales : sans cela l'imagette fait
        # toute la largeur de la page et, une fois mise à 128 px de haut, devient
        # démesurément large — PyLaia décode alors surtout du vide.
        colonnes = np.flatnonzero((arr[haut:bas] < 128).any(axis=0))
        if colonnes.size == 0:
            continue
        gauche = max(0, int(colonnes[0]) - marge)
        droite = min(largeur, int(colonnes[-1]) + 1 + marge)

        bande = image.crop((gauche, haut, droite, bas))
        ratio = HAUTEUR_LIGNE / bande.height
        lignes.append(
            bande.resize(
                (max(1, int(bande.width * ratio)), HAUTEUR_LIGNE), Image.Resampling.LANCZOS
            )
        )
    return lignes


# ───────────────────────── Appel de PyLaia ─────────────────────────


def verifier_prerequis(model_dir: Path, pylaia_bin: str) -> str:
    """Vérifie que la CLI PyLaia et les fichiers du modèle sont présents.

    Args:
        model_dir: Dossier du modèle PyLaia (dépôt HuggingFace cloné).
        pylaia_bin: Nom ou chemin de l'exécutable de décodage.

    Returns:
        Le chemin résolu de l'exécutable.

    Raises:
        SystemExit: Si la CLI ou un fichier du modèle manque, avec la marche à suivre.
    """
    resolu = pylaia_bin if Path(pylaia_bin).exists() else shutil.which(pylaia_bin)
    if resolu is None:
        raise SystemExit(
            f"{pylaia_bin} introuvable.\n"
            "PyLaia épingle torch>=1.13,<1.14 ; ses torchvision/torchaudio n'ont pas de "
            "wheel au-delà de Python 3.10. Il ne s'installe donc PAS dans le venv du "
            "projet (3.13) — il lui faut un environnement séparé, en 3.10 :\n"
            "    uv venv --python 3.10 /chemin/pylaia-env\n"
            "    VIRTUAL_ENV=/chemin/pylaia-env uv pip install pylaia\n"
            "    uv run scripts/probe_pylaia_htr.py --pylaia-bin "
            "/chemin/pylaia-env/bin/pylaia-htr-decode-ctc ..."
        )
    manquants = [f for f in FICHIERS_MODELE if not (model_dir / f).exists()]
    if manquants:
        raise SystemExit(
            f"Fichiers absents de {model_dir} : {', '.join(manquants)}.\n"
            "Récupérer le modèle français :\n"
            "    git clone https://huggingface.co/Teklia/pylaia-rimes"
        )
    return resolu


def _recomposer(tokens: str) -> str:
    """Recompose un texte à partir de la sortie symbole par symbole de PyLaia.

    PyLaia décode en CTC et émet, selon les versions et les options, soit du texte
    déjà assemblé, soit une suite de symboles séparés par des espaces où le blanc est
    noté `<space>`. On gère les deux plutôt que de parier sur une version.

    Args:
        tokens: Partie « transcription » d'une ligne de sortie PyLaia.

    Returns:
        Le texte reconstitué.
    """
    morceaux = tokens.split()
    if not morceaux:
        return ""
    symboliques = sum(1 for m in morceaux if len(m) == 1 or m == "<space>")
    if symboliques < 0.8 * len(morceaux):
        return tokens.strip()  # déjà du texte assemblé
    return "".join(" " if m == "<space>" else m for m in morceaux).strip()


def decoder(
    pylaia_bin: str, model_dir: Path, lignes_dir: Path, liste: Path, extra: list[str]
) -> dict[str, str]:
    """Lance `pylaia-htr-decode-ctc` sur un lot d'imagettes et lit sa sortie.

    Args:
        pylaia_bin: Chemin de l'exécutable de décodage PyLaia.
        model_dir: Dossier du modèle PyLaia.
        lignes_dir: Dossier contenant les imagettes de lignes.
        liste: Fichier listant les identifiants d'imagettes, un par ligne.
        extra: Arguments supplémentaires passés tels quels à PyLaia.

    Returns:
        Dictionnaire {identifiant d'imagette: transcription}.

    Raises:
        SystemExit: Si PyLaia sort en erreur.
    """
    commande = [
        pylaia_bin,
        "--common.experiment_dirname",
        str(model_dir),
        "--common.model_filename",
        str(model_dir / "model"),
        # `img_dirs` (pluriel) attend une LISTE au format jsonargparse, d'où les crochets.
        "--img_dirs",
        f"[{lignes_dir}]",
        *extra,
        str(model_dir / "syms.txt"),
        str(liste),
    ]
    logger.info("Décodage : %s", " ".join(commande))
    proc = subprocess.run(commande, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"PyLaia a échoué (code {proc.returncode}) :\n{proc.stderr[-4000:]}")

    transcriptions: dict[str, str] = {}
    for ligne in proc.stdout.splitlines():
        identifiant, _, reste = ligne.partition(" ")
        if identifiant:
            transcriptions[identifiant] = _recomposer(reste)
    return transcriptions


# ───────────────────────── Sonde ─────────────────────────


def _decouper_polygone(page: Image.Image, polygone: list[list[int]], marge: int = 6) -> Image.Image:
    """Extrait une ligne délimitée par un polygone et la met à `HAUTEUR_LIGNE`.

    Le rectangle englobant d'une ligne penchée mord sur ses voisines : on blanchit
    donc tout ce qui est hors du polygone (dilaté de `marge`) avant de rogner.

    Args:
        page: Page en niveaux de gris.
        polygone: Sommets (x, y) du contour de la ligne, en pixels de la page.
        marge: Dilatation du polygone, en pixels, pour ne pas rogner les jambages.

    Returns:
        L'imagette de ligne, redimensionnée à `HAUTEUR_LIGNE` de haut.
    """
    from PIL import ImageDraw, ImageFilter

    masque = Image.new("L", page.size, 0)
    ImageDraw.Draw(masque).polygon([tuple(pt) for pt in polygone], fill=255)
    if marge > 0:
        masque = masque.filter(ImageFilter.MaxFilter(2 * marge + 1))
    fond = Image.new("L", page.size, 255)
    ligne = Image.composite(page.convert("L"), fond, masque).crop(masque.getbbox())
    ratio = HAUTEUR_LIGNE / ligne.height
    return ligne.resize((max(1, int(ligne.width * ratio)), HAUTEUR_LIGNE), Image.Resampling.LANCZOS)


def segmenter_doc_ufcn(
    pages: dict[str, Image.Image], travail_dir: Path, ufcn_python: str, modele: str
) -> dict[str, list[list[list[int]]]]:
    """Détecte les lignes de toutes les pages avec Doc-UFCN, en un seul sous-processus.

    Doc-UFCN tourne dans son propre environnement (torch 2.x, incompatible avec
    PyLaia) : on lui passe les pages sur disque et on relit ses polygones en JSON.

    Args:
        pages: {scan: page en niveaux de gris}, telle que Doc-UFCN doit la voir.
        travail_dir: Dossier où écrire les pages et le JSON de sortie.
        ufcn_python: Interpréteur de l'environnement Doc-UFCN.
        modele: Nom du modèle `Teklia/doc-ufcn-<nom>`.

    Returns:
        {scan: polygones des lignes, triés de haut en bas}.

    Raises:
        SystemExit: Si Doc-UFCN sort en erreur.
    """
    pages_dir = travail_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for scan, page in pages.items():
        page.convert("L").save(pages_dir / f"{scan}.png")
    sortie = travail_dir / "polygones_doc_ufcn.json"
    script = Path(__file__).with_name("segment_doc_ufcn.py")
    commande = [ufcn_python, str(script), str(pages_dir), str(sortie), "--modele", modele]
    logger.info("Segmentation : %s", " ".join(commande))
    proc = subprocess.run(commande, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Doc-UFCN a échoué (code {proc.returncode}) :\n{proc.stderr[-4000:]}")

    brut = json.loads(sortie.read_text(encoding="utf-8"))
    return {
        scan: [
            ligne["polygone"]
            for ligne in sorted(
                lignes, key=lambda li: float(np.mean([y for _, y in li["polygone"]]))
            )
        ]
        for scan, lignes in brut.items()
    }


def preparer_lignes(
    echantillons: list[ScoledtSample],
    lignes_dir: Path,
    max_lignes: int,
    deregler: bool = True,
    interligne_cible: int | None = None,
    segmenteur: str = "projection",
    ufcn_python: str = "python",
    ufcn_modele: str = "generic-historical-line",
    ufcn_page_nette: bool = False,
    reglages_par_resolution: bool = False,
) -> dict[str, list[str]]:
    """Segmente chaque copie et écrit les imagettes de lignes sur disque.

    Args:
        echantillons: Échantillons Scoledit à traiter.
        lignes_dir: Dossier de destination des imagettes.
        max_lignes: Nombre maximal de lignes conservées par copie (garde-fou).
        deregler: Retirer les réglures Seyès. Avec Doc-UFCN, la détection se fait
            sur la page brute et seul le découpage des lignes lit la page nettoyée.
        interligne_cible: Remettre chaque page à cet interligne avant tout
            traitement (voir `normaliser_echelle`). `None` : pas de normalisation.
        segmenteur: `projection` (profil d'encre) ou `doc-ufcn` (réseau Teklia).
        ufcn_python: Interpréteur de l'environnement Doc-UFCN.
        ufcn_modele: Modèle Doc-UFCN.
        ufcn_page_nette: Donner à Doc-UFCN la page dérèglée plutôt que la page brute.
        reglages_par_resolution: Choisir les réglages `REGLAGES` « basse » ou
            « haute » selon l'interligne mesuré de chaque page (voir
            `SEUIL_HAUTE_RESOLUTION`). Sinon, réglages « basse » partout.

    Returns:
        Dictionnaire {scan: identifiants des imagettes, dans l'ordre de lecture}.
    """
    lignes_dir.mkdir(parents=True, exist_ok=True)
    par_copie: dict[str, list[str]] = {}
    brutes: dict[str, Image.Image] = {}
    nettes: dict[str, Image.Image] = {}
    profils: dict[str, str] = {}
    for ech in echantillons:
        try:
            page = load_image(ech.image_path)
        except Exception as err:  # noqa: BLE001 - une copie illisible ne doit pas tout arrêter
            logger.warning("Image illisible (%s) : %s", ech.scan, err)
            par_copie[ech.scan] = []
            continue
        if interligne_cible is not None:
            page = normaliser_echelle(page, interligne_cible)
        profil = "basse"
        if reglages_par_resolution:
            interligne = estimer_interligne(page)
            profil = "haute" if interligne > SEUIL_HAUTE_RESOLUTION else "basse"
            logger.info("%s : interligne %d px → réglages « %s »", ech.scan, interligne, profil)
        profils[ech.scan] = profil
        brutes[ech.scan] = page
        nettes[ech.scan] = (
            supprimer_reglures(page, **REGLAGES[profil]["reglures"]) if deregler else page
        )

    if segmenteur == "doc-ufcn":
        polygones = segmenter_doc_ufcn(
            nettes if ufcn_page_nette else brutes, lignes_dir.parent, ufcn_python, ufcn_modele
        )
        lignes_par_copie = {
            scan: [_decouper_polygone(nettes[scan], p) for p in polygones.get(scan, [])]
            for scan in nettes
        }
    else:
        lignes_par_copie = {
            scan: segmenter_lignes(page, **REGLAGES[profils[scan]]["segmentation"])
            for scan, page in nettes.items()
        }

    for scan, lignes in lignes_par_copie.items():
        identifiants = []
        for i, ligne in enumerate(lignes[:max_lignes]):
            identifiant = f"{scan}__{i:03d}"
            ligne.convert("L").save(lignes_dir / f"{identifiant}.jpg", quality=95)
            identifiants.append(identifiant)
        par_copie[scan] = identifiants
        logger.info("%s : %d lignes", scan, len(identifiants))
    return par_copie


def ecrire_rapport(resultats: list[ResultatCopie], out_dir: Path) -> None:
    """Écrit le rapport JSON et la comparaison texte lisible à l'œil.

    Args:
        resultats: Résultats par copie.
        out_dir: Dossier de sortie.
    """
    cers = [r.cer for r in resultats]
    wers = [r.wer for r in resultats]
    synthese = {
        "n_copies": len(resultats),
        "cer_moyen": float(np.mean(cers)) if cers else None,
        "wer_moyen": float(np.mean(wers)) if wers else None,
        "cer_median": float(np.median(cers)) if cers else None,
        "copies": [asdict(r) for r in resultats],
    }
    (out_dir / "resultats.json").write_text(
        json.dumps(synthese, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    blocs = [
        f"── {r.scan} ({r.level}) — {r.n_lignes} lignes — "
        f"CER {r.cer:.1%} / WER {r.wer:.1%}\n"
        f"  RÉFÉRENCE : {r.reference}\n"
        f"  PYLAIA    : {r.hypothese}\n"
        for r in sorted(resultats, key=lambda r: r.cer)
    ]
    (out_dir / "comparaison.txt").write_text("\n".join(blocs), encoding="utf-8")


def main() -> None:
    """Segmente Scoledit, décode avec PyLaia et compare aux transcriptions de référence."""
    parser = argparse.ArgumentParser(
        description="Sonde exploratoire PyLaia sur Scoledit (hors pipeline).",
        epilog="Les sorties contiennent de l'écriture d'élèves : ne jamais les committer.",
    )
    parser.add_argument("--model-dir", required=True, type=Path, help="Dépôt PyLaia cloné.")
    parser.add_argument(
        "--pylaia-bin",
        default="pylaia-htr-decode-ctc",
        help="Exécutable de décodage PyLaia (souvent dans un venv Python 3.10 dédié).",
    )
    parser.add_argument("--limit", type=int, default=10, help="Nombre de copies (défaut 10).")
    parser.add_argument(
        "--scans-dir",
        default="s3://projet-production-ecrits-depp/scoledit/scans/CE1/",
        help="Dossier des scans Scoledit.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="s3://projet-production-ecrits-depp/scoledit/annotation/CE1/",
        help="Dossier des annotations Scoledit.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/probe_pylaia"),
        help="Dossier de sortie (sous data/, ignoré par Git).",
    )
    parser.add_argument(
        "--max-lignes", type=int, default=40, help="Lignes maximum par copie (défaut 40)."
    )
    parser.add_argument(
        "--garder-reglures",
        action="store_true",
        help="Ne pas retirer les réglures Seyès (retirées par défaut : segmentation "
        "nettement plus stable, CER moyen 74,6%% → 71,5%% sur 5 copies CE1).",
    )
    parser.add_argument(
        "--interligne-cible",
        type=int,
        default=None,
        help="Remettre chaque page à cet interligne Seyès (px) avant traitement. "
        "Par défaut : pas de normalisation.",
    )
    parser.add_argument(
        "--segmenteur",
        choices=("projection", "doc-ufcn"),
        default="projection",
        help="Découpage en lignes : profil d'encre (défaut) ou réseau Doc-UFCN de Teklia.",
    )
    parser.add_argument(
        "--ufcn-python",
        default="python",
        help="Interpréteur de l'environnement Doc-UFCN (venv Python 3.10 dédié).",
    )
    parser.add_argument(
        "--ufcn-modele",
        default="generic-historical-line",
        help="Modèle Teklia/doc-ufcn-<nom> (generic-historical-line, norhand-v1-line).",
    )
    parser.add_argument(
        "--ufcn-page-nette",
        action="store_true",
        help="Donner à Doc-UFCN la page dérèglée (défaut : page brute).",
    )
    parser.add_argument(
        "--reglages-par-resolution",
        action="store_true",
        help="Adapter dérèglage et segmentation à la résolution de chaque scan "
        "(deux jeux de réglages, voir REGLAGES). Par défaut : réglages basse résolution.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passés tels quels à pylaia-htr-decode-ctc (à mettre en dernier).",
    )
    args = parser.parse_args()

    pylaia_bin = verifier_prerequis(args.model_dir, args.pylaia_bin)

    echantillons = load_scoledit_dataset(
        scans_dir=args.scans_dir, annotations_dir=args.annotations_dir, limit=args.limit
    )
    if not echantillons:
        raise SystemExit("Aucun échantillon chargé. Vérifier les chemins et l'accès S3.")
    logger.info("%d échantillons Scoledit chargés.", len(echantillons))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lignes_dir = args.out_dir / "lignes"
    par_copie = preparer_lignes(
        echantillons,
        lignes_dir,
        args.max_lignes,
        deregler=not args.garder_reglures,
        interligne_cible=args.interligne_cible,
        segmenteur=args.segmenteur,
        ufcn_python=args.ufcn_python,
        ufcn_modele=args.ufcn_modele,
        ufcn_page_nette=args.ufcn_page_nette,
        reglages_par_resolution=args.reglages_par_resolution,
    )

    tous = [ident for idents in par_copie.values() for ident in idents]
    if not tous:
        raise SystemExit(
            "Aucune ligne segmentée. Régler --max-lignes ou les seuils de segmenter_lignes()."
        )
    liste = args.out_dir / "img_list.txt"
    liste.write_text("\n".join(tous) + "\n", encoding="utf-8")
    logger.info("%d lignes à décoder.", len(tous))

    transcriptions = decoder(pylaia_bin, args.model_dir, lignes_dir, liste, args.extra)
    logger.info("%d / %d lignes décodées.", len(transcriptions), len(tous))

    resultats = []
    for ech in echantillons:
        identifiants = par_copie.get(ech.scan, [])
        hypothese = " ".join(transcriptions.get(i, "") for i in identifiants).strip()
        m = compute_transcription_metrics(ech.reference, hypothese)
        resultats.append(
            ResultatCopie(
                scan=ech.scan,
                level=ech.level,
                n_lignes=len(identifiants),
                cer=m.cer,
                wer=m.wer,
                cer_normalise=m.cer_normalise,
                wer_normalise=m.wer_normalise,
                reference=ech.reference,
                hypothese=hypothese,
            )
        )

    ecrire_rapport(resultats, args.out_dir)

    logger.info("── Synthèse (%d copies) ──", len(resultats))
    logger.info("CER moyen : %.1f%%", float(np.mean([r.cer for r in resultats])) * 100)
    logger.info("WER moyen : %.1f%%", float(np.mean([r.wer for r in resultats])) * 100)
    logger.info("Rapport   : %s", args.out_dir / "resultats.json")
    logger.info("À l'œil   : %s", args.out_dir / "comparaison.txt")
    logger.info("Lignes découpées : %s — les regarder AVANT de conclure.", lignes_dir)


if __name__ == "__main__":
    main()
