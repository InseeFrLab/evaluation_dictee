"""Analyse des résultats par item et par copie, avec intervalles de confiance.

Convention : « erreur » = code != "1". Deux désaccords aux conséquences opposées :
SUR-CORRECTION (expert erreur, modèle correct — biais VLM connu) et SUR-DÉTECTION
(expert correct, modèle erreur).
"""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from evaluation_dictee.evaluation.statistics import wilson_interval


def load_predictions(predictions_path: str | Path) -> pd.DataFrame:
    """Charge les prédictions sauvegardées par le benchmark (JSON Lines).

    Accepte un chemin local OU un URI S3 (s3://...) : même accès fsspec que
    `data/loaders.py`, ce qui permet de lire les prédictions exportées sur S3
    sans réexécuter le pipeline.

    Args:
        predictions_path: chemin local ou S3 du JSONL (une prédiction par ligne).

    Returns:
        Un DataFrame, une ligne par item, avec les colonnes du JSONL.
    """
    records = []
    with fsspec.open(str(predictions_path), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def per_item_metrics(df: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Métriques par item, avec intervalle de Wilson sur l'accord et la prévalence.

    Args:
        df: prédictions à l'item (colonnes item_id, y_true, y_pred).
        level: niveau de confiance des intervalles de Wilson.

    Returns:
        Un DataFrame indexé par item_id : accord et son IC, kappa, prévalence
        d'erreur experte et modèle avec IC, rappel/précision sur l'erreur, et
        comptes de sur-correction et de sur-détection.
    """
    rows = []
    for item_id, grp in df.groupby("item_id"):
        y_true = grp["y_true"]
        y_pred = grp["y_pred"]
        n = len(grp)
        n_accord = int((y_true == y_pred).sum())
        ci = wilson_interval(n_accord, n, level)

        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except ValueError:
            kappa = float("nan")

        exp_err = y_true != "1"
        mod_err = y_pred != "1"
        n_exp_err = int(exp_err.sum())
        n_mod_err = int(mod_err.sum())
        vrais_pos = int((exp_err & mod_err).sum())
        sur_corr = int((exp_err & ~mod_err).sum())
        sur_det = int((~exp_err & mod_err).sum())

        ci_exp = wilson_interval(n_exp_err, n, level)
        ci_mod = wilson_interval(n_mod_err, n, level)

        rows.append(
            {
                "item_id": item_id,
                "n": n,
                "accord": ci.estimate,
                "accord_lo": ci.lower,
                "accord_hi": ci.upper,
                "kappa": kappa,
                "pct_erreur_expert": n_exp_err / n * 100,
                "pct_erreur_expert_lo": ci_exp.lower * 100,
                "pct_erreur_expert_hi": ci_exp.upper * 100,
                "pct_erreur_modele": n_mod_err / n * 100,
                "pct_erreur_modele_lo": ci_mod.lower * 100,
                "pct_erreur_modele_hi": ci_mod.upper * 100,
                "rappel_erreur": vrais_pos / n_exp_err if n_exp_err else float("nan"),
                "precision_erreur": vrais_pos / n_mod_err if n_mod_err else float("nan"),
                "n_sur_correction": sur_corr,
                "n_sur_detection": sur_det,
            }
        )
    return pd.DataFrame(rows).set_index("item_id")


def per_copy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Métriques par copie : repère les élèves/copies difficiles pour le modèle.

    Args:
        df: prédictions à l'item (colonnes copy_id, y_true, y_pred, confidence).

    Returns:
        Un DataFrame indexé par copy_id : effectif, accord, comptes d'erreurs,
        de fautes (code "9") et de manquants (code "0") côté expert et modèle,
        prévalences d'erreur et confiance moyenne.
    """
    rows = []
    for copy_id, grp in df.groupby("copy_id"):
        n = len(grp)
        y_true, y_pred = grp["y_true"], grp["y_pred"]
        n_err_exp = int((y_true != "1").sum())
        n_err_mod = int((y_pred != "1").sum())
        n_fautes_exp = int((y_true == "9").sum())
        n_fautes_mod = int((y_pred == "9").sum())
        n_manq_exp = int((y_true == "0").sum())
        n_manq_mod = int((y_pred == "0").sum())
        rows.append(
            {
                "copy_id": copy_id,
                "n_items": n,
                "accord": float((y_true == y_pred).mean()),
                "n_erreurs_expert": n_err_exp,
                "n_erreurs_modele": n_err_mod,
                "n_fautes_expert": n_fautes_exp,
                "n_fautes_modele": n_fautes_mod,
                "n_manquants_expert": n_manq_exp,
                "n_manquants_modele": n_manq_mod,
                "pct_erreur_expert": n_err_exp / n * 100 if n else 0.0,
                "pct_erreur_modele": n_err_mod / n * 100 if n else 0.0,
                "confiance_moyenne": float(grp["confidence"].dropna().mean())
                if grp["confidence"].notna().any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("copy_id")


def disagreement_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Décompose chaque type de désaccord (transition expert→modèle), globalement.

    Args:
        df: prédictions à l'item (colonnes y_true, y_pred).

    Returns:
        Un DataFrame trié par fréquence : libellé de la transition, effectif et
        part parmi les désaccords. Vide s'il n'y a aucun désaccord.
    """
    dis = df[df["y_true"] != df["y_pred"]]
    if len(dis) == 0:
        return pd.DataFrame(columns=["transition", "n", "pct_desaccords"])
    counts = (
        dis.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    counts["transition"] = "expert:" + counts["y_true"] + " → modèle:" + counts["y_pred"]
    counts["pct_desaccords"] = counts["n"] / len(dis) * 100
    return counts[["transition", "n", "pct_desaccords"]].reset_index(drop=True)


def confusion_df(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """Matrice de confusion globale expert × modèle.

    Args:
        df: prédictions à l'item (colonnes y_true, y_pred).
        normalize: si True, normalise chaque ligne (expert) pour sommer à 1.

    Returns:
        Un DataFrame carré, lignes préfixées « expert: » et colonnes « modèle: ».
    """
    labels = sorted(set(df["y_true"]) | set(df["y_pred"]))
    matrix = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
    out = pd.DataFrame(
        matrix,
        index=[f"expert:{c}" for c in labels],
        columns=[f"modèle:{c}" for c in labels],
    )
    if normalize:
        out = out.div(out.sum(axis=1).replace(0, 1), axis=0)
    return out


def copies_by_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """Copies triées par taux de désaccord brut décroissant (les pires en tête).

    Args:
        df: prédictions à l'item (colonnes copy_id, y_true, y_pred).

    Returns:
        Un DataFrame indexé par copy_id : effectif, nombre et pourcentage de
        désaccords, accord, trié par pourcentage de désaccord décroissant.
    """
    rows = []
    for copy_id, grp in df.groupby("copy_id"):
        n = len(grp)
        n_dis = int((grp["y_true"] != grp["y_pred"]).sum())
        rows.append(
            {
                "copy_id": copy_id,
                "n_items": n,
                "n_desaccords": n_dis,
                "pct_desaccord": n_dis / n * 100 if n else 0.0,
                "accord": (n - n_dis) / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("copy_id").sort_values("pct_desaccord", ascending=False)
