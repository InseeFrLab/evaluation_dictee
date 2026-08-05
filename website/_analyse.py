"""Chargement et calculs partagés par les pages « Résultats » et « Écarts ».

Le site **relit les prédictions déjà exportées** (S3 par défaut, ou un dossier
local via `S3_PREDICTIONS_PREFIX`) et recalcule chiffres et figures au rendu :
aucun pipeline n'est réexécuté, aucune figure n'est figée. Toute défaillance
(paquet non importable, run non exporté, S3 injoignable) est capturée dans
`NOTES` et laisse la page rendable, avec des « — » à la place des chiffres.

Ce module est préfixé par `_` : Quarto ne le rend pas comme une page, il est
seulement importé par les `.qmd` (dont le répertoire de travail est `website/`).

Le contenu reprend l'analyse de `notebooks/03_analyse_resultats.ipynb`. Les
briques génériques (Wilson, IC du kappa, design effect, métriques par item et
par copie, comparaison multi-modèles) viennent du paquet `evaluation_dictee` ;
seule la mise en forme propre au site vit ici.
"""

from __future__ import annotations

import html
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ── Import du paquet du projet ────────────────────────────────────────────────
# Le site peut être rendu hors `uv run` (Quarto lancé avec le Python système) :
# on ajoute alors `src/` au chemin d'import en remontant depuis le dossier courant.
REPO_ROOT: Path | None = None
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "src" / "evaluation_dictee").is_dir():
        REPO_ROOT = _parent
        sys.path.insert(0, str(_parent / "src"))
        break

#: Incidents rencontrés au chargement, affichés en fin de page par `bloc_notes()`.
NOTES: list[str] = []

try:
    from evaluation_dictee.data.grid import GridItem, load_grid
    from evaluation_dictee.evaluation.multi_model import (
        agreement_per_item,
        confidence_score,
        disagreement_type_summary,
        referral_curve_with_ci,
    )
    from evaluation_dictee.evaluation.report import (
        copies_by_disagreement,
        disagreement_decomposition,
        load_predictions,
        per_copy_metrics,
        per_item_metrics,
    )
    from evaluation_dictee.evaluation.statistics import (
        design_effect,
        kappa_interval_clustered,
        wilson_interval,
    )

    PAQUET_OK = True
except ImportError as exc:  # noqa: BLE001 — la page doit rester rendable
    PAQUET_OK = False
    NOTES.append(
        f"Paquet `evaluation_dictee` non importable ({exc}) : aucun chiffre n'a pu "
        "être calculé. Rendre le site depuis l'environnement du projet "
        "(`uv sync --extra website` puis `uv run quarto render website`)."
    )

# ── Paramètres du site ────────────────────────────────────────────────────────
#: Racine des prédictions exportées. Pointer sur un dossier local pour un rendu
#: hors ligne : `export S3_PREDICTIONS_PREFIX=/chemin/vers/predictions`.
PREFIX = os.environ.get("S3_PREDICTIONS_PREFIX", "s3://projet-production-ecrits-depp/predictions")

#: Nom de base des runs comparés côte à côte, dans l'ordre d'affichage :
#: libellé → champ `name` du YAML du run.
RUNS_BASE: dict[str, str] = {
    "end-to-end": "dictee_end2end",
    "two-stage": "dictee_two_stage",
}

#: Modèle dont on affiche les résultats. Le benchmark suffixe TOUJOURS ses sorties
#: par le nom du modèle (`<name>_<modele>_predictions.jsonl`, cf.
#: `config.run_output_name`) : le site doit donc nommer le modèle pour retrouver
#: le fichier. À défaut de correspondance exacte, `_resoudre_run` retombe sur le
#: fichier réellement exporté pour ce run (voir plus bas).
MODELE = os.environ.get("RESULTATS_MODELE", "gemma4-26b-moe")

#: Runs comparés côte à côte : libellé → nom de run (base + modèle).
#: Surchargeables pour comparer d'autres runs sans toucher aux pages.
RUNS: dict[str, str] = {
    "end-to-end": os.environ.get("RESULTATS_RUN_END_TO_END", f"dictee_end2end_{MODELE}"),
    "two-stage": os.environ.get("RESULTATS_RUN_TWO_STAGE", f"dictee_two_stage_{MODELE}"),
}

#: Run servant de référence quand une analyse en exige un seul (classement des
#: pires copies, courbe de renvoi). Par défaut l'approche privilégiée du projet.
RUN_REFERENCE = os.environ.get("RESULTATS_RUN_REFERENCE", "end-to-end")

#: Chemin de la grille de codage (mots attendus, ordre de la dictée).
GRID_PATH = os.environ.get("RESULTATS_GRID_PATH", "configs/grille_dictee_2015.json")

#: Codes attendus dans la grille simplifiée. `y_true` contient une minorité de
#: codes hérités de la saisie AGATE (vides, « i », « v ») : on les compte comme
#: des erreurs (convention « erreur = code != 1 » du projet) mais on les signale.
CODES_VALIDES = frozenset({"1", "9", "0"})

#: Libellé lisible de chaque code.
LIBELLE_CODE = {"1": "correct", "9": "erreur", "0": "absent", "?": "non lu"}

# ── Copies vierges, écartées des classements d'écart ──────────────────────────
# Le critère est celui du pipeline, et lui seul. `pipeline/benchmark.py` mesure la
# densité d'encre de l'image (`data/loaders.ink_ratio`) et, sous
# `config.data.blank_ink_threshold` (2,5 % par défaut — seul le pré-imprimé
# marque), déclare la copie VIERGE : elle est codée « 0 » sur tous les items sans
# appel modèle, et chaque ligne du JSONL est estampillée `blank: true`. Le site se
# contente de relire ce marqueur : aucune heuristique concurrente ici, pour que
# site et pipeline ne puissent pas diverger dans leur définition.
#
# Un run produit AVANT ce garde-fou n'a pas la colonne : le site ne peut alors
# écarter aucune copie et le dit explicitement, plutôt que de deviner.
COLONNE_VIERGE = "blank"

#: Seuil d'encre du pipeline, repris pour l'affichage quand la source de données
#: ne le porte pas. La valeur qui s'applique au run est celle de son YAML
#: (`data.blank_ink_threshold`).
SEUIL_ENCRE_PIPELINE = 0.025

#: Colonne portant la densité d'encre mesurée, quand elle est disponible.
COLONNE_ENCRE = "ink_ratio"

#: Fichier de distribution des densités d'encre, produit par
#: `scripts/compute_ink_ratios.py` et cherché à côté des prédictions.
FICHIER_ENCRE = os.environ.get("RESULTATS_FICHIER_ENCRE", "dictee_ink_ratios.csv")

# ── Palette ───────────────────────────────────────────────────────────────────
C_EXPERT = "#1f4e79"
C_MODELE = "#c44536"
C_OK, C_MOYEN, C_PB = "#2e7d32", "#ef6c00", "#c62828"
#: Une couleur par run, dans l'ordre de `RUNS`.
COULEURS_RUN = ["#c44536", "#7b2d8b", "#0b7285", "#946200"]
Z95 = 1.96


def init_matplotlib() -> None:
    """Applique le style de figures du projet (à appeler une fois par page)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9.5,
        }
    )


# ── Formatage ─────────────────────────────────────────────────────────────────
def _absent(x: object) -> bool:
    """Vrai si la valeur est manquante (None ou NaN) et doit s'afficher « — »."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x: object, d: int = 1) -> str:
    """Formate une proportion de [0, 1] en pourcentage, « — » si absente."""
    return "—" if _absent(x) else f"{float(x):.{d}%}".replace("%", " %")


def pts(x: object, d: int = 1) -> str:
    """Formate un pourcentage déjà exprimé en points (0-100), « — » si absent."""
    return "—" if _absent(x) else f"{float(x):.{d}f} %"


def num(x: object, d: int = 3) -> str:
    """Formate un nombre décimal, « — » si absent."""
    return "—" if _absent(x) else f"{float(x):.{d}f}"


def milliers(n: object) -> str:
    """Formate un entier avec une espace fine comme séparateur de milliers."""
    return "—" if _absent(n) else f"{int(n):,}".replace(",", " ")


def formateur_unite(unite: str):
    """Renvoie le formateur associé à une unité (« pct » pour %, « num » sinon)."""
    return pct if unite == "pct" else num


def ic(valeur: object, bas: object, haut: object, formateur=pct) -> str:
    """Formate « estimation [bas ; haut] », « — » si l'estimation est absente."""
    if _absent(valeur):
        return "—"
    if _absent(bas) or _absent(haut):
        return formateur(valeur)
    return f"{formateur(valeur)} <small>[{formateur(bas)} ; {formateur(haut)}]</small>"


def table_markdown(entetes: list[str], lignes: list[list[str]], legende: str = "") -> str:
    """Assemble un tableau Markdown (aligné à droite sauf la 1re colonne).

    Args:
        entetes: libellés de colonnes.
        lignes: valeurs, déjà formatées en chaînes.
        legende: légende Quarto, ajoutée sous le tableau si non vide.

    Returns:
        Le tableau au format Markdown, prêt à `print()` dans une cellule
        `output: asis`.
    """
    aligns = ["---"] + [":---:"] * (len(entetes) - 1)
    out = ["| " + " | ".join(entetes) + " |", "| " + " | ".join(aligns) + " |"]
    out += ["| " + " | ".join(ligne) + " |" for ligne in lignes]
    if legende:
        out += ["", f": {legende} {{.striped .hover}}"]
    return "\n".join(out)


def bloc_notes() -> str:
    """Callout replié listant les incidents de chargement (vide si aucun)."""
    if not NOTES:
        return ""
    lignes = [
        '::: {.callout-warning appearance="simple" collapse="true" '
        'title="Chiffres non calculés ou incomplets"}'
    ]
    lignes += [f"- {n}" for n in NOTES]
    lignes.append(":::")
    return "\n".join(lignes)


# ── Chargement ────────────────────────────────────────────────────────────────
@dataclass
class Run:
    """Un run de prédictions et ses agrégats, prêts à être tracés."""

    label: str
    nom: str
    couleur: str
    df: pd.DataFrame
    copies: pd.DataFrame = field(repr=False)
    items: pd.DataFrame = field(repr=False)

    @property
    def n_copies(self) -> int:
        """Nombre de copies évaluées."""
        return int(self.df["copy_id"].nunique())

    @property
    def n_items(self) -> int:
        """Nombre de lignes item × copie."""
        return len(self.df)

    @property
    def modele(self) -> str:
        """Modèle(s) du run, lu dans le suffixe de son nom (« — » si non nommé).

        Le nom du modèle ne figure PAS dans les lignes du JSONL : la seule trace
        portée par les prédictions exportées est ce suffixe, posé par
        `config.run_output_name`. Un run two_stage à deux modèles distincts en
        porte deux, séparés par `_` (étape 1 puis étape 2).
        """
        base = RUNS_BASE.get(self.label, "")
        if base and self.nom.startswith(f"{base}_"):
            return self.nom[len(base) + 1 :]
        return "—"


#: Suffixes des fichiers exportés (cf. `utils/s3_export.py`). Le suffixe HTR se
#: termine par celui du scoring : ne jamais tester l'un sans écarter l'autre.
SUFFIXE_SCORING = "_predictions.jsonl"
SUFFIXE_HTR = "_htr_predictions.jsonl"


def _chemin(nom_run: str) -> str:
    """URI du JSONL de prédictions d'un run."""
    return PREFIX.rstrip("/") + "/" + nom_run + SUFFIXE_SCORING


#: Nom des runs de scoring exportés à côté de `PREFIX`, listés une seule fois.
_EXPORTES: list[str] | None = None


def _noms_exportes() -> list[str]:
    """Noms des runs de scoring exportés à côté de `PREFIX`, triés.

    On liste le RÉPERTOIRE, jamais un motif `<base>_*` : sur S3, un glob par
    préfixe fait mettre en cache par s3fs une vue *partielle* du répertoire,
    après quoi les autres fichiers deviennent invisibles — y compris pour
    `load_predictions`, qui échouerait alors sur un fichier bien présent.

    Returns:
        Les noms de runs (suffixe de modèle compris, `_predictions.jsonl` ôté),
        hors runs HTR. Liste vide si le préfixe est injoignable.
    """
    global _EXPORTES
    if _EXPORTES is None:
        import fsspec

        try:
            fs, _, _ = fsspec.get_fs_token_paths(PREFIX)
            entrees = [str(e) for e in fs.ls(PREFIX.rstrip("/"), detail=False)]
        except Exception:  # noqa: BLE001 — la page doit rester rendable
            entrees = []
        _EXPORTES = sorted(
            Path(e).name.removesuffix(SUFFIXE_SCORING)
            for e in entrees
            if e.endswith(SUFFIXE_SCORING) and not e.endswith(SUFFIXE_HTR)
        )
    return _EXPORTES


def _runs_exportes(base: str) -> list[str]:
    """Runs exportés pour un run de base, tous modèles confondus.

    Args:
        base: champ `name` du run (ex. `dictee_end2end`).

    Returns:
        Les noms de runs exportés qui portent ce `name`, triés.
    """
    return [nom for nom in _noms_exportes() if nom == base or nom.startswith(f"{base}_")]


def _resoudre_run(label: str, nom: str) -> str:
    """Nom de run réellement exporté, à défaut de celui attendu.

    Le modèle fait partie du nom de fichier : un site configuré pour
    `gemma4-26b-moe` ne trouve rien si le run exporté est un `qwen3-6-35b-moe`.
    Plutôt que d'afficher une page vide, on cherche ce qui a été exporté pour ce
    run et on signale la substitution dans `NOTES`.

    Args:
        label: libellé du run (clé de `RUNS`).
        nom: nom de run attendu (base + modèle).

    Returns:
        Le nom attendu s'il est exporté, sinon le seul (ou le premier) nom
        exporté pour ce run de base, sinon le nom attendu inchangé.
    """
    base = RUNS_BASE.get(label, "")
    if not base:
        return nom
    exportes = _runs_exportes(base)
    if not exportes or nom in exportes:
        return nom
    choisi = exportes[0]
    detail = (
        "" if len(exportes) == 1 else f" ({len(exportes)} runs exportés : {', '.join(exportes)})"
    )
    NOTES.append(
        f"`{nom}` n'est pas exporté : les chiffres « {label} » portent sur "
        f"`{choisi}`{detail}. Fixer `RESULTATS_MODELE` ou "
        f"`RESULTATS_RUN_{'END_TO_END' if label == 'end-to-end' else 'TWO_STAGE'}` "
        "pour choisir explicitement."
    )
    return choisi


def charger_runs() -> dict[str, Run]:
    """Charge tous les runs de `RUNS` et calcule leurs agrégats par item et copie.

    Returns:
        Les runs chargés, indexés par libellé, dans l'ordre de `RUNS`. Un run
        absent ou illisible est omis et l'incident consigné dans `NOTES`.
    """
    if not PAQUET_OK:
        return {}
    runs: dict[str, Run] = {}
    for i, (label, nom) in enumerate(RUNS.items()):
        if not nom:
            continue
        nom = _resoudre_run(label, nom)
        try:
            df = load_predictions(_chemin(nom))
        except Exception as exc:  # noqa: BLE001 — la page doit rester rendable
            NOTES.append(f"`{nom}` indisponible ({type(exc).__name__}) : colonne non calculée.")
            continue
        if df.empty:
            NOTES.append(f"`{nom}` est vide : colonne non calculée.")
            continue
        runs[label] = Run(
            label=label,
            nom=nom,
            couleur=COULEURS_RUN[i % len(COULEURS_RUN)],
            df=df,
            copies=per_copy_metrics(df),
            items=per_item_metrics(df),
        )
    if not runs:
        NOTES.append(
            "Aucun run n'a pu être chargé. Vérifier `S3_PREDICTIONS_PREFIX` et "
            "l'export (`uv run scripts/export_predictions.py --config …`)."
        )
    return runs


def charger_grille() -> tuple[list[GridItem], dict[str, str], dict[str, int]]:
    """Charge la grille de codage : items ordonnés, mot attendu et rang par item.

    Returns:
        Le triplet (items de la grille, item_id → mot attendu, item_id → rang
        dans la dictée). Vide si la grille est introuvable (incident consigné).
    """
    if not PAQUET_OK or REPO_ROOT is None:
        return [], {}, {}
    try:
        grille = load_grid(str(REPO_ROOT / GRID_PATH))
    except Exception as exc:  # noqa: BLE001 — la page doit rester rendable
        NOTES.append(f"Grille `{GRID_PATH}` illisible ({type(exc).__name__}) : items non libellés.")
        return [], {}, {}
    mots = {it.item_id: it.attendu for it in grille.items}
    rangs = {it.item_id: i + 1 for i, it in enumerate(grille.items)}
    return grille.items, mots, rangs


def libelle_item(item_id: str, mots: dict[str, str], rangs: dict[str, int]) -> str:
    """Libellé d'item au format `NN · mot`, qui lève l'ambiguïté des mots répétés."""
    return f"{rangs.get(item_id, 0):02d} · {mots.get(item_id, item_id)}"


def note_codes_non_standard(runs: dict[str, Run]) -> str:
    """Phrase signalant les codes experts hors grille simplifiée (ou chaîne vide)."""
    if not runs:
        return ""
    ref = next(iter(runs.values())).df
    hors = ref[~ref["y_true"].isin(CODES_VALIDES)]
    if hors.empty:
        return ""
    detail = ", ".join(
        f"« {code or 'vide'} » ({milliers(n)})" for code, n in hors["y_true"].value_counts().items()
    )
    return (
        f"{milliers(len(hors))} items sur {milliers(len(ref))} "
        f"({len(hors) / len(ref):.2%}) portent un code expert hors grille simplifiée : "
        f"{detail}. Ils sont conservés et comptés comme des erreurs (convention "
        "« erreur = code ≠ 1 »), ce qui pèse de façon négligeable sur les agrégats."
    )


# ── Synthèse globale ──────────────────────────────────────────────────────────
def _proportion_groupee(indicatrice: pd.Series, grappes: pd.Series) -> tuple[object, float]:
    """Proportion et IC de Wilson corrigés du design effect propre à l'indicatrice.

    Args:
        indicatrice: indicatrice binaire dont on estime la moyenne.
        grappes: identifiant de copie de chaque observation.

    Returns:
        Le couple (intervalle de confiance, design effect appliqué). L'intervalle
        vaut None si l'indicatrice est vide.
    """
    if not len(indicatrice):
        return None, float("nan")
    deff = design_effect(indicatrice, grappes)
    return wilson_interval(int(indicatrice.sum()), len(indicatrice), deff=deff), deff


def synthese_globale(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Accord, kappa, rappel et précision sur l'erreur, avec IC à 95 % par grappes.

    Les 83 items d'une copie ne sont pas indépendants : les IC sont donc corrigés
    du **design effect de Kish**, calculé pour **chaque** métrique sur sa propre
    indicatrice (celle de l'accord et celle du rappel n'ont pas le même). Le kappa
    n'étant la moyenne d'aucune indicatrice, son IC vient d'un bootstrap par
    grappes, qui ne suppose rien sur la structure de corrélation.

    Args:
        df: prédictions à l'item (colonnes copy_id, y_true, y_pred).
        n_boot: nombre de tirages du bootstrap par grappes (kappa).

    Returns:
        Un DataFrame indexé par métrique : valeur, borne basse, borne haute,
        `unite` (« pct » pour une proportion de [0, 1], « num » pour le kappa),
        `deff` (design effect appliqué, NaN pour le kappa qui passe par le
        bootstrap), `methode` et `lecture` (sens de la métrique).
    """
    grappes = df["copy_id"]
    err_exp = df["y_true"] != "1"
    err_mod = df["y_pred"] != "1"

    accord, deff_accord = _proportion_groupee(df["y_true"] == df["y_pred"], grappes)
    # Rappel et précision se mesurent sur des SOUS-ENSEMBLES d'items : la grappe
    # doit être restreinte de la même façon, sinon le design effect est faussé.
    rappel, deff_rappel = _proportion_groupee(err_mod[err_exp], grappes[err_exp])
    precision, deff_precision = _proportion_groupee(err_exp[err_mod], grappes[err_mod])
    # Sur-correction : l'expert voit une erreur, le modèle non (biais VLM connu).
    surcorr, deff_surcorr = _proportion_groupee(~err_mod[err_exp], grappes[err_exp])
    surdet, deff_surdet = _proportion_groupee(err_mod[~err_exp], grappes[~err_exp])
    kappa = kappa_interval_clustered(df["y_true"], df["y_pred"], grappes, n_boot=n_boot)

    methode_wilson = "Wilson corrigé du design effect"
    lignes = [
        (
            "Accord brut",
            accord,
            "pct",
            deff_accord,
            methode_wilson,
            "part des items codés à l'identique : plus haut = mieux",
        ),
        (
            "Kappa de Cohen",
            kappa,
            "num",
            float("nan"),
            f"bootstrap par grappes ({n_boot} tirages)",
            "accord corrigé du hasard : plus haut = mieux",
        ),
        (
            "Rappel des erreurs (sensibilité)",
            rappel,
            "pct",
            deff_rappel,
            methode_wilson,
            "part des erreurs de l'élève retrouvées : plus haut = moins de fautes ratées",
        ),
        (
            "Précision sur les erreurs",
            precision,
            "pct",
            deff_precision,
            methode_wilson,
            "part des erreurs signalées qui en sont vraiment : plus haut = moins de "
            "fausses alertes",
        ),
        (
            "Taux de sur-correction",
            surcorr,
            "pct",
            deff_surcorr,
            methode_wilson,
            "erreurs de l'élève que le modèle valide à tort : plus **bas** = mieux",
        ),
        (
            "Taux de sur-détection",
            surdet,
            "pct",
            deff_surdet,
            methode_wilson,
            "items corrects que le modèle sanctionne à tort : plus **bas** = mieux",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "metrique": nom,
                "valeur": float("nan") if v is None else v.estimate,
                "bas": float("nan") if v is None else v.lower,
                "haut": float("nan") if v is None else v.upper,
                "unite": unite,
                "deff": deff,
                "methode": methode,
                "lecture": lecture,
            }
            for nom, v, unite, deff, methode, lecture in lignes
        ]
    ).set_index("metrique")


# ── Prévalence par item ───────────────────────────────────────────────────────
def prevalence_par_item(
    df: pd.DataFrame, mots: dict[str, str], rangs: dict[str, int]
) -> pd.DataFrame:
    """Prévalence d'erreur par item, expert vs modèle, avec IC de Wilson à 95 %.

    **Aucune correction de clustering ici, et c'est voulu** : pour un item donné il
    y a exactement *une* observation par copie, chacune venant d'un élève différent.
    Il n'y a donc pas de corrélation intra-copie à corriger et le design effect vaut
    1. L'appliquer élargirait les intervalles d'un facteur ≈ 3 sans justification.
    La correction concerne les métriques agrégées sur les 83 items d'une copie,
    dans `synthese_globale`.

    Args:
        df: prédictions à l'item.
        mots: item_id → mot attendu.
        rangs: item_id → rang dans la dictée.

    Returns:
        Un DataFrame indexé par item_id, trié par rang : prévalences d'erreur
        expert et modèle avec leurs IC, et l'écart modèle − expert en points de
        pourcentage.
    """
    lignes = []
    for item_id, grp in df.groupby("item_id"):
        n = len(grp)
        k_exp = int((grp["y_true"] != "1").sum())
        k_mod = int((grp["y_pred"] != "1").sum())
        ci_exp = wilson_interval(k_exp, n)
        ci_mod = wilson_interval(k_mod, n)
        lignes.append(
            {
                "item_id": item_id,
                "position": rangs.get(item_id, 0),
                "mot_attendu": mots.get(item_id, item_id),
                "pct_expert": k_exp / n * 100,
                "ic_expert_bas": ci_exp.lower * 100,
                "ic_expert_haut": ci_exp.upper * 100,
                "pct_modele": k_mod / n * 100,
                "ic_modele_bas": ci_mod.lower * 100,
                "ic_modele_haut": ci_mod.upper * 100,
                "ecart": (k_mod - k_exp) / n * 100,
            }
        )
    return pd.DataFrame(lignes).set_index("item_id").sort_values("position")


# ── Métriques selon la difficulté de la copie ─────────────────────────────────
def metriques_par_difficulte(
    df: pd.DataFrame, copies: pd.DataFrame, col_tri: str, n_quantiles: int = 8
) -> pd.DataFrame:
    """Accord, kappa, rappel et précision par quantile de difficulté de copie.

    Args:
        df: prédictions à l'item.
        copies: métriques par copie (index = copy_id), source de la difficulté.
        col_tri: colonne de `copies` définissant la difficulté (nb d'erreurs).
        n_quantiles: nombre de groupes de difficulté.

    Comme dans `synthese_globale`, les IC sont corrigés du design effect propre à
    chaque indicatrice : ces métriques agrègent les 83 items corrélés d'une copie.

    Returns:
        Un DataFrame, une ligne par quantile, trié par difficulté croissante :
        difficulté moyenne (`x`), effectif, accord et son IC Wilson, kappa,
        rappel et son IC, précision.
    """
    from sklearn.metrics import cohen_kappa_score

    try:
        groupes = pd.qcut(copies[col_tri], q=n_quantiles, duplicates="drop")
    except ValueError:  # trop peu de valeurs distinctes pour des quantiles
        groupes = pd.cut(copies[col_tri], bins=n_quantiles)

    copies = copies.assign(_groupe=groupes)
    milieu = copies.groupby("_groupe", observed=True)[col_tri].mean()

    lignes = []
    for etiquette, copies_grp in copies.groupby("_groupe", observed=True):
        sub = df[df["copy_id"].isin(copies_grp.index)]
        if len(sub) < 10:
            continue
        grappes = sub["copy_id"]
        err_exp = sub["y_true"] != "1"
        err_mod = sub["y_pred"] != "1"
        n_me = int(err_mod.sum())
        n_vp = int((err_exp & err_mod).sum())
        accord, _ = _proportion_groupee(sub["y_true"] == sub["y_pred"], grappes)
        rappel, _ = _proportion_groupee(err_mod[err_exp], grappes[err_exp])
        try:
            kappa = float(cohen_kappa_score(sub["y_true"], sub["y_pred"]))
        except ValueError:
            kappa = float("nan")
        lignes.append(
            {
                "x": milieu[etiquette],
                "n_copies": len(copies_grp),
                "accord": accord.estimate,
                "accord_bas": accord.lower,
                "accord_haut": accord.upper,
                "kappa": kappa,
                "rappel": rappel.estimate if rappel else float("nan"),
                "rappel_bas": rappel.lower if rappel else float("nan"),
                "rappel_haut": rappel.upper if rappel else float("nan"),
                "precision": n_vp / n_me if n_me else float("nan"),
            }
        )
    return pd.DataFrame(lignes).sort_values("x")


# ── Comparaison multi-runs ────────────────────────────────────────────────────
def joindre_runs(runs: dict[str, Run]) -> pd.DataFrame | None:
    """Joint les runs sur (copy_id, item_id) au format attendu par `multi_model`.

    Équivalent de `multi_model.load_multi_runs`, mais à partir des DataFrames
    déjà en mémoire : le site ne relit pas les JSONL une seconde fois.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        Le DataFrame joint (jointure interne : seuls les items présents dans
        tous les runs), avec une colonne `y_pred__<libellé>` par run. None si
        moins de deux runs sont disponibles ou si l'intersection est vide.
    """
    if len(runs) < 2:
        return None
    premier = next(iter(runs.values()))
    joint = premier.df[["copy_id", "item_id", "y_true"]].copy()
    for label, run in runs.items():
        cols = run.df[["copy_id", "item_id", "y_pred", "confidence"]].rename(
            columns={"y_pred": f"y_pred__{label}", "confidence": f"conf__{label}"}
        )
        joint = joint.merge(cols, on=["copy_id", "item_id"], how="inner")
    if joint.empty:
        NOTES.append("Aucun item commun aux runs comparés : la section multi-modèles est omise.")
        return None
    return joint


# ── Écarts par copie ──────────────────────────────────────────────────────────
def classement_ecarts(df: pd.DataFrame) -> pd.DataFrame:
    """Copies triées par taux de désaccord décroissant, enrichi du détail utile.

    Args:
        df: prédictions à l'item d'un run.

    Returns:
        Un DataFrame indexé par copy_id, trié des pires copies aux meilleures :
        effectif, nombre et part de désaccords, accord, nombre d'erreurs vues par
        l'expert et par le modèle, écart signé sur ce nombre, parts d'items codés
        « 0 » (mot absent) et hors grille par l'expert, et — si le run porte le
        marqueur du pipeline — la colonne booléenne `vierge`. À part égale de
        désaccord, les copies sont triées par identifiant, pour que le classement
        soit reproductible.
    """
    classement = copies_by_disagreement(df)
    copies = per_copy_metrics(df)
    out = classement.join(copies[["n_erreurs_expert", "n_erreurs_modele", "confiance_moyenne"]])
    out["ecart_n_erreurs"] = out["n_erreurs_modele"] - out["n_erreurs_expert"]
    # Colonnes descriptives : elles éclairent la lecture des fiches mais ne
    # servent PAS à écarter des copies (seul le marqueur du pipeline le fait).
    out["pct_absent_expert"] = (df["y_true"] == "0").groupby(df["copy_id"]).mean() * 100
    out["pct_hors_grille_expert"] = (~df["y_true"].isin(CODES_VALIDES)).groupby(
        df["copy_id"]
    ).mean() * 100
    if COLONNE_VIERGE in df.columns:
        # Le marqueur est identique sur toutes les lignes d'une copie ; `max`
        # ramène simplement le booléen au niveau copie.
        out["vierge"] = df[COLONNE_VIERGE].fillna(False).groupby(df["copy_id"]).max().astype(bool)
    # Tri stable sur un index déjà ordonné : à taux de désaccord égal, l'ordre
    # reste celui des identifiants de copie.
    return out.sort_index().sort_values("pct_desaccord", ascending=False, kind="mergesort")


def couverture_copies(runs: dict[str, Run]) -> pd.DataFrame:
    """Couverture de chaque run : copies évaluées et copies écartées du corpus.

    Une copie manquante n'est pas neutre : le pipeline two-stage écarte les copies
    qu'il ne parvient pas à transcrire à l'étape 1, alors que l'end-to-end produit
    un codage pour toutes. Comparer les runs à effectif égal exige donc de savoir
    combien de copies chacun a laissées de côté.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        Un DataFrame indexé par libellé de run : nombre de copies évaluées,
        nombre et part de copies du corpus (union des runs) non couvertes.
    """
    corpus = set()
    for run in runs.values():
        corpus |= set(run.df["copy_id"].unique())
    lignes = []
    for label, run in runs.items():
        vues = set(run.df["copy_id"].unique())
        lignes.append(
            {
                "run": label,
                "n_copies": len(vues),
                "n_ecartees": len(corpus - vues),
                "pct_ecartees": len(corpus - vues) / len(corpus) * 100 if corpus else 0.0,
            }
        )
    return pd.DataFrame(lignes).set_index("run")


def seuil_encre() -> float:
    """Seuil de copie vierge du projet, lu dans la config du paquet.

    Returns:
        La valeur par défaut de `DataConfig.blank_ink_threshold`, pour que la page
        ne puisse pas afficher un seuil périmé si celui du code change. Retombe sur
        `SEUIL_ENCRE_PIPELINE` si la config n'est pas importable.
    """
    try:
        from evaluation_dictee.config import DataConfig

        return float(DataConfig.model_fields["blank_ink_threshold"].default)
    except Exception:  # noqa: BLE001 — la page doit rester rendable
        return SEUIL_ENCRE_PIPELINE


def charger_densites_encre(runs: dict[str, Run]) -> tuple[pd.DataFrame | None, str]:
    """Densité d'encre par copie, et provenance de la mesure.

    Deux sources, par ordre de préférence :

    1. la colonne `ink_ratio` du JSONL, écrite par le benchmark en même temps
       qu'il applique le seuil — c'est la mesure qui a réellement décidé ;
    2. le CSV produit par `scripts/compute_ink_ratios.py`, qui mesure le corpus
       sans appel modèle (utile avant d'avoir relancé le benchmark).

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        Le couple (DataFrame indexé par copy_id avec la colonne `ink_ratio`,
        libellé de provenance). Le DataFrame vaut None si aucune source n'est
        disponible ; l'incident est alors consigné dans `NOTES`.
    """
    for label, run in runs.items():
        if COLONNE_ENCRE in run.df.columns:
            serie = run.df.groupby("copy_id")[COLONNE_ENCRE].first().dropna()
            if len(serie):
                return serie.to_frame(COLONNE_ENCRE), f"prédictions du run {label}"

    # Import local : `fsspec` arrive avec s3fs, mais la page doit rester rendable
    # même dans un environnement où le paquet du projet n'est pas installé.
    chemin = PREFIX.rstrip("/") + "/" + FICHIER_ENCRE
    try:
        import fsspec

        with fsspec.open(chemin, "rt", encoding="utf-8") as f:
            mesures = pd.read_csv(f, sep=";")
    except Exception as exc:  # noqa: BLE001 — la page doit rester rendable
        NOTES.append(
            f"Distribution des densités d'encre indisponible ({type(exc).__name__}) : "
            f"section omise. La produire avec `uv run scripts/compute_ink_ratios.py "
            f"--config configs/scoring/dictee_end2end.yaml --export`."
        )
        return None, ""
    if COLONNE_ENCRE not in mesures.columns or "copy_id" not in mesures.columns:
        NOTES.append(f"`{FICHIER_ENCRE}` n'a pas les colonnes attendues : section omise.")
        return None, ""
    return mesures.set_index("copy_id")[[COLONNE_ENCRE]], f"`{FICHIER_ENCRE}`"


def marqueur_vierge_disponible(runs: dict[str, Run]) -> bool:
    """Vrai si tous les runs portent le marqueur de copie vierge du pipeline.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        True seulement si chaque run a la colonne `blank`. Les runs exportés avant
        l'ajout du filtre de densité d'encre ne l'ont pas : aucune copie ne peut
        alors être écartée, et la page doit le signaler au lieu de deviner.
    """
    return bool(runs) and all(COLONNE_VIERGE in run.df.columns for run in runs.values())


def separer_copies_vierges(classement: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les copies vierges du reste du classement, selon le pipeline.

    Le critère est le marqueur `blank` posé par `pipeline/benchmark.py` d'après la
    densité d'encre de l'image. En son absence (run antérieur au garde-fou), rien
    n'est écarté : mieux vaut un classement bruité et signalé comme tel qu'un
    filtrage improvisé qui divergerait du pipeline.

    Args:
        classement: sortie de `classement_ecarts`.

    Returns:
        Le couple (copies vierges, copies exploitables). L'ordre du classement est
        conservé de part et d'autre ; le premier terme est vide si le run ne porte
        pas le marqueur.
    """
    if "vierge" not in classement.columns:
        return classement.iloc[:0], classement
    vierge = classement["vierge"]
    return classement[vierge], classement[~vierge]


def nature_copie(ligne: pd.Series) -> str:
    """Qualifie une copie d'après ce que l'annotateur a pu y coder.

    Sert à annoter les fiches : une copie que l'expert n'a pas pu coder sur une
    large part de ses items s'interprète autrement qu'une copie ordinaire, même
    quand la densité d'encre ne la classe pas comme vierge (écriture illisible).

    Args:
        ligne: une ligne de `classement_ecarts`.

    Returns:
        Un libellé court, ou une chaîne vide pour une copie ordinaire.
    """
    if bool(ligne.get("vierge", False)):
        return "copie vierge selon le pipeline — densité d'encre sous le seuil"
    absent = float(ligne["pct_absent_expert"])
    hors = float(ligne["pct_hors_grille_expert"])
    total = absent + hors
    if total < 25:
        return ""
    motif = "illisibles" if hors > absent else "absents"
    return (
        f"copie largement incodable — l'expert code {total:.0f} % des items comme "
        f"{motif}, alors que la densité d'encre ne la classe pas vierge"
    )


def transcription_modele(sub: pd.DataFrame) -> tuple[str, str]:
    """Transcription du modèle pour une copie, et sa provenance.

    Args:
        sub: lignes de prédiction d'une seule copie, dans l'ordre de la dictée.

    Returns:
        Le couple (texte, provenance). La transcription brute de l'étape 1 (HTR)
        est prioritaire ; à défaut le texte est recollé item par item.
    """
    if "raw_transcription" in sub.columns:
        brutes = sub["raw_transcription"].dropna()
        brutes = brutes[brutes.astype(str).str.strip() != ""]
        if len(brutes):
            return str(brutes.iloc[0]).strip(), "étape 1 (HTR), lecture brute du modèle"
    if "transcription" in sub.columns:
        mots = [str(m) for m in sub["transcription"].fillna("").tolist() if str(m).strip()]
        if mots:
            return " ".join(mots), "recollée item par item"
    return "", "non fournie par ce run"


def decomposition_copie(sub: pd.DataFrame) -> str:
    """Résume les transitions expert → modèle d'une copie (« 1→9 : 12 ; … »)."""
    dis = sub[sub["y_true"] != sub["y_pred"]]
    if dis.empty:
        return "aucun désaccord"
    compte = dis.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False)
    return " ; ".join(f"{t or '∅'}→{p} : {n}" for (t, p), n in compte.items())


# ── Rendu HTML des codages ────────────────────────────────────────────────────
def _puce(mot: str, code: str, classe: str, infobulle: str, detail: str = "") -> str:
    """Puce HTML d'un item : mot attendu au-dessus, code en dessous."""
    sup = f"<span class='puce-detail'>{html.escape(detail)}</span>" if detail else ""
    return (
        f"<span class='puce {classe}' title='{html.escape(infobulle)}'>"
        f"<span class='puce-mot'>{html.escape(mot)}</span>"
        f"<span class='puce-code'>{html.escape(code)}</span>{sup}</span>"
    )


def codage_expert_html(
    items: list[GridItem], codes_expert: dict[str, str], codes_modele: dict[str, str]
) -> str:
    """Bandeau des codes de l'annotateur expert, nuancé par code.

    Args:
        items: items de la grille, dans l'ordre de la dictée.
        codes_expert: item_id → code expert.
        codes_modele: item_id → code modèle (sert à marquer les items divergents).

    Returns:
        Le fragment HTML du bandeau de puces.
    """
    puces = []
    for it in items:
        code = codes_expert.get(it.item_id, "?")
        divergent = codes_modele.get(it.item_id, "?") != code
        classe = f"puce-code-{code if code in CODES_VALIDES else 'autre'}"
        if divergent:
            classe += " puce-divergente"
        libelle = LIBELLE_CODE.get(code, f"code « {code or 'vide'} »")
        puces.append(_puce(it.attendu, code or "∅", classe, f"expert : {libelle}"))
    return f"<div class='bandeau-puces'>{''.join(puces)}</div>"


def codage_modele_html(
    items: list[GridItem],
    codes_expert: dict[str, str],
    codes_modele: dict[str, str],
    lectures: dict[str, str] | None = None,
) -> str:
    """Bandeau des codes du modèle : **vert** si accord avec l'expert, **rouge** sinon.

    Args:
        items: items de la grille, dans l'ordre de la dictée.
        codes_expert: item_id → code expert.
        codes_modele: item_id → code modèle.
        lectures: item_id → mot lu par le modèle ; affiché sous la puce quand il
            diffère du mot attendu (c'est là que se logent les erreurs de lecture).

    Returns:
        Le fragment HTML du bandeau de puces.
    """
    lectures = lectures or {}
    puces = []
    for it in items:
        code_m = codes_modele.get(it.item_id, "?")
        code_e = codes_expert.get(it.item_id, "?")
        accord = code_m == code_e
        classe = "puce-accord" if accord else "puce-desaccord"
        infobulle = (
            f"modèle : {LIBELLE_CODE.get(code_m, code_m)} — accord avec l'expert"
            if accord
            else f"modèle : {LIBELLE_CODE.get(code_m, code_m)} / "
            f"expert : {LIBELLE_CODE.get(code_e, code_e or 'vide')}"
        )
        lu = str(lectures.get(it.item_id, "") or "").strip()
        detail = ""
        if not accord:
            detail = f"exp. {code_e or '∅'}"
        if lu and lu != it.attendu:
            detail = f"{detail} · lu « {lu} »" if detail else f"lu « {lu} »"
        puces.append(_puce(it.attendu, code_m, classe, infobulle, detail))
    return f"<div class='bandeau-puces'>{''.join(puces)}</div>"


def legende_puces() -> str:
    """Légende des couleurs des bandeaux de puces."""
    return (
        "<div class='legende-puces'>"
        "<span class='puce puce-accord'><span class='puce-mot'>accord</span></span>"
        "<span class='puce puce-desaccord'><span class='puce-mot'>désaccord</span></span>"
        "<span class='puce puce-code-1'><span class='puce-mot'>1 correct</span></span>"
        "<span class='puce puce-code-9'><span class='puce-mot'>9 erreur</span></span>"
        "<span class='puce puce-code-0'><span class='puce-mot'>0 absent</span></span>"
        "</div>"
    )


# Fonctions du paquet ré-exportées telles quelles : les pages passent par ce
# module pour y accéder, afin qu'un import raté n'empêche pas le rendu (le bloc
# try/except ci-dessus est alors le seul point de défaillance).
__all__ = [
    "agreement_per_item",
    "confidence_score",
    "disagreement_decomposition",
    "disagreement_type_summary",
    "referral_curve_with_ci",
    "wilson_interval",
]
