"""Rapport HTML de comparaison de BRAS d'expérience (variantes de prompt).

Un « bras » est une variante testée contre la référence : chain-of-thought, comptage
des items, exemples de fautes… Ce rapport répond à une question que ni Langfuse ni le
site ne traitent : **cette variante fait-elle mieux que la référence, sur le même
modèle et les mêmes copies, et pourquoi**.

- Langfuse enregistre chaque run isolément : il n'a ni notion d'écart contre une
  référence, ni intervalle de confiance.
- Le site publie ce qui est retenu, pour l'extérieur. Il n'a pas vocation à porter
  toutes les expérimentations menées.

Le rapport est écrit sous `data/processed/`, hors de Git : il ne contient que des
agrégats, mais il dérive d'écrits d'élèves mineurs et suit donc le même régime que le
reste des données du projet.
"""

from __future__ import annotations

import html
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from evaluation_dictee.data.grid import GridItem, load_grid
from evaluation_dictee.evaluation.metrics import compute_scoring_metrics
from evaluation_dictee.evaluation.report import filtrer_evaluables, load_predictions

#: Bras connus : libellé affiché -> préfixe du run (`<prefixe>_<modele>_predictions.jsonl`).
#: La référence vient en premier, c'est elle qui sert d'étalon à tous les écarts.
BRAS_PAR_DEFAUT: dict[str, str] = {
    "référence": "dictee_end2end",
    "CoT": "dictee_end2end_cot",
    "comptage": "dictee_end2end_comptage",
    "comptage+": "dictee_end2end_comptage_strict",
    "exemples": "dictee_end2end_exemples",
}

#: Slots 1 à 3 de la palette catégorielle : les seuls qui passent le contrôle « toutes
#: paires » du validateur, requis ici puisqu'un nuage de points n'a pas d'ordre adjacent.
#: Au-delà de trois modèles, il faudrait facetter plutôt qu'ajouter des teintes.
COULEURS_MODELES: list[str] = ["#2a78d6", "#eb6834", "#1baf7a"]

#: La FORME porte le bras, la COULEUR le modèle : douze séries distinctes par la seule
#: couleur ne seraient ni lisibles ni sûres pour les daltonismes.
FORMES = ["cercle", "carre", "triangle", "losange", "croix"]

_MARGE_G, _MARGE_D, _MARGE_H, _MARGE_B = 200, 24, 54, 34
_LARGEUR_TRACE = 620
_HAUTEUR_LIGNE = 16
_RAYON = 4.2


@dataclass
class Serie:
    """Un couple (modèle, bras) et ses prédictions, restreintes au corpus commun."""

    modele: str
    bras: str
    records: list[dict]
    couleur: str = ""
    forme: str = ""

    @property
    def cle(self) -> str:
        """Identifiant de la série, utilisé comme clé des agrégats."""
        return f"{self.modele}|{self.bras}"


@dataclass
class Rapport:
    """Données agrégées du rapport, prêtes pour le rendu."""

    series: list[Serie]
    items: list[GridItem]
    n_copies: int
    par_item: dict[str, dict[str, dict]] = field(default_factory=dict)
    err_humain: dict[str, float] = field(default_factory=dict)
    par_type: dict[str, dict[str, dict]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────


def _localiser(dossier: Path, prefixe: str, modele: str) -> str | None:
    """Chemin des prédictions d'un run : d'abord en local, sinon sur S3.

    Les runs de référence sont exportés sur S3 et ne sont pas forcément présents sur
    la machine qui produit le rapport : sans ce repli, la colonne « écart » resterait
    vide alors que la donnée existe.

    Args:
        dossier: dossier local des prédictions.
        prefixe: préfixe du run (champ `name` de la config).
        modele: nom du modèle.

    Returns:
        Le chemin local ou l'URI S3, ou None si le run est introuvable.
    """
    fichier = f"{prefixe}_{modele}_predictions.jsonl"
    local = dossier / fichier
    if local.exists():
        return str(local)
    try:
        import fsspec

        from evaluation_dictee.config import Secrets

        uri = f"{Secrets().s3_predictions_prefix.rstrip('/')}/{fichier}"
        fs, chemin = fsspec.core.url_to_fs(uri)
        return uri if fs.exists(chemin) else None
    except Exception:  # noqa: BLE001 — S3 indisponible : le run est simplement omis
        return None


def charger_series(
    modeles: list[str],
    bras: dict[str, str],
    dossier: str | Path = "data/processed",
) -> list[Serie]:
    """Charge chaque couple (modèle, bras) disponible, exclusions D8 appliquées.

    Args:
        modeles: modèles à comparer.
        bras: libellé de bras -> préfixe de run.
        dossier: dossier des prédictions.

    Returns:
        Les séries trouvées ; un fichier absent est simplement omis.
    """
    dossier = Path(dossier)
    series: list[Serie] = []
    for i, modele in enumerate(modeles):
        for j, (libelle, prefixe) in enumerate(bras.items()):
            chemin = _localiser(dossier, prefixe, modele)
            if chemin is None:
                continue
            df, _ = filtrer_evaluables(load_predictions(chemin))
            if df.empty:
                continue
            series.append(
                Serie(
                    modele=modele,
                    bras=libelle,
                    records=df.to_dict("records"),
                    couleur=COULEURS_MODELES[i % len(COULEURS_MODELES)],
                    forme=FORMES[j % len(FORMES)],
                )
            )
    return series


def restreindre_corpus_commun(series: list[Serie]) -> tuple[list[Serie], int]:
    """Restreint toutes les séries aux copies qu'elles ont TOUTES traitées.

    Comparer deux bras sur des corpus différents confond l'effet du bras avec celui
    de la composition de l'échantillon : les copies difficiles ne se répartissent pas
    au hasard.

    Args:
        series: séries chargées.

    Returns:
        Le couple (séries restreintes, nombre de copies communes).
    """
    if not series:
        return series, 0
    communes = set.intersection(*[{r["copy_id"] for r in s.records} for s in series])
    for s in series:
        s.records = [r for r in s.records if r["copy_id"] in communes]
    return series, len(communes)


# ─────────────────────────────────────────────────────────────────────────────
# Agrégats
# ─────────────────────────────────────────────────────────────────────────────


def _kappa(records: list[dict]) -> float:
    return compute_scoring_metrics(
        [r["y_true"] for r in records], [r["y_pred"] for r in records]
    ).cohen_kappa


def _par_copie(records: list[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        d[r["copy_id"]].append(r)
    return d


def intervalle_ecart(
    bras: list[dict], reference: list[dict], n_boot: int = 400, seed: int = 42
) -> tuple[float, float]:
    """IC 95 % de l'écart de kappa, par bootstrap sur les COPIES.

    Les items d'une même copie ne sont pas indépendants (même élève, même écriture,
    et un décalage d'alignement fausse d'un coup toute la suite) : rééchantillonner
    les items surestimerait nettement la précision.

    Args:
        bras: prédictions du bras testé.
        reference: prédictions de la référence, mêmes copies.
        n_boot: nombre de tirages.
        seed: graine, pour que deux rendus du rapport donnent le même intervalle.

    Returns:
        Les bornes basse et haute de l'écart de kappa.
    """
    alea = random.Random(seed)
    cb, cr = _par_copie(bras), _par_copie(reference)
    copies = sorted(set(cb) & set(cr))
    ecarts = []
    for _ in range(n_boot):
        tirage = [alea.choice(copies) for _ in copies]
        ecarts.append(
            _kappa([r for c in tirage for r in cb[c]]) - _kappa([r for c in tirage for r in cr[c]])
        )
    ecarts.sort()
    return ecarts[int(0.025 * n_boot)], ecarts[int(0.975 * n_boot)]


def agreger(series: list[Serie], grille_path: str | Path) -> Rapport:
    """Calcule tous les agrégats du rapport : par item, par nature, taux de faute.

    Args:
        series: séries restreintes au corpus commun.
        grille_path: chemin de la grille de codage.

    Returns:
        Le rapport prêt pour le rendu.
    """
    grille = load_grid(str(grille_path))
    nature = {it.item_id: it.type for it in grille.items}
    series, n_copies = restreindre_corpus_commun(series)
    rapport = Rapport(series=series, items=grille.items, n_copies=n_copies)

    for s in series:
        groupes: dict[str, list[dict]] = defaultdict(list)
        blocs: dict[str, list[dict]] = defaultdict(list)
        for r in s.records:
            groupes[r["item_id"]].append(r)
            blocs[nature.get(r["item_id"], "mot")].append(r)

        valeurs = {}
        for item_id, rs in groupes.items():
            m = compute_scoring_metrics([r["y_true"] for r in rs], [r["y_pred"] for r in rs])
            kappa = None if m.cohen_kappa is None or math.isnan(m.cohen_kappa) else m.cohen_kappa
            valeurs[item_id] = {
                "accord": m.raw_agreement,
                "kappa": kappa,
                # Taux de faute attribué aux élèves : part des copies codées « 9 ».
                # Les absences en sont exclues, un mot non écrit n'étant pas une faute.
                "taux_erreur": sum(1 for r in rs if r["y_pred"] == "9") / len(rs),
            }
        rapport.par_item[s.cle] = valeurs

        rapport.par_type[s.cle] = {}
        for t, rs in blocs.items():
            m = compute_scoring_metrics([r["y_true"] for r in rs], [r["y_pred"] for r in rs])
            kappa = None if m.cohen_kappa is None or math.isnan(m.cohen_kappa) else m.cohen_kappa
            rapport.par_type[s.cle][t] = {
                "accord": m.raw_agreement,
                "kappa": kappa,
                "n_items": m.n_items,
                "codes": dict(Counter(r["y_pred"] for r in rs)),
            }

    # Taux de faute mesuré par l'expert : identique dans toutes les séries.
    if series:
        groupes = defaultdict(list)
        for r in series[0].records:
            groupes[r["item_id"]].append(r)
        rapport.err_humain = {
            item_id: sum(1 for r in rs if r["y_true"] == "9") / len(rs)
            for item_id, rs in groupes.items()
        }
    return rapport


# ─────────────────────────────────────────────────────────────────────────────
# Rendu HTML
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root{--fg:#1a1a1a;--muted:#666;--bord:#d8d8d8;--fond:#fff;--surface-1:#fff;--zebre:#fafafa;
      --bon:#1a7f37;--mauvais:#b42318;--neutre:#666;--accent:#0b5cad}
*{box-sizing:border-box}
body{margin:0;background:var(--fond);color:var(--fg);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 72px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 6px;padding-top:20px;border-top:1px solid var(--bord)}
h3{font-size:15px;margin:22px 0 6px;color:var(--muted);font-weight:600}
p.sub{color:var(--muted);margin:0 0 28px;font-size:13px}
p.note{color:var(--muted);font-size:13px;margin:6px 0 16px;max-width:78ch}
table{border-collapse:collapse;width:100%;margin:10px 0 4px;font-size:14px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--bord)}
th{font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em;
   color:var(--muted);border-bottom:2px solid var(--bord)}
th:first-child,td:first-child{text-align:left}
tbody tr:nth-child(even){background:var(--zebre)}
.bon{color:var(--bon);font-weight:600}
.mauvais{color:var(--mauvais);font-weight:600}
.neutre{color:var(--neutre)}
.tag{display:inline-block;padding:1px 7px;border-radius:3px;font-size:11px;
     background:#eef2f6;color:var(--accent);font-weight:600}
.scroll{overflow-x:auto}
.enc{background:#f7f9fb;border-left:3px solid var(--accent);padding:12px 16px;
     margin:16px 0;font-size:14px;border-radius:0 3px 3px 0}
.fig{margin:22px 0 8px}
.fig-tit{font-size:15px;font-weight:600;margin:0 0 2px}
.fig-sub{color:var(--muted);font-size:13px;margin:0 0 10px;max-width:78ch}
.leg{display:flex;flex-wrap:wrap;gap:14px 26px;margin:0 0 10px;font-size:12.5px}
.leg-bloc{display:flex;align-items:center;gap:11px;flex-wrap:wrap}
.leg-tit{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;
         font-weight:600}
.leg-item{display:inline-flex;align-items:center;gap:5px}
.axe-grille{stroke:#e6e6e6}
.axe-lbl{font-size:11px;fill:var(--muted)}
.item-lbl{font-size:11px;fill:var(--fg);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.item-val{font-size:10.5px;fill:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bande{fill:#f6f6f5}
.portee{stroke:#d0d0d0;stroke-width:1.2}
.etalon{fill:#0b0b0b}
svg text{font-family:inherit}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--bord);
       color:var(--muted);font-size:12px}
"""


def _pct(x: float | None) -> str:
    return "—" if x is None or math.isnan(x) else f"{x * 100:.1f} %"


def _kap(x: float | None) -> str:
    """Kappa, ou tiret : il est indéfini quand une strate ne contient qu'un seul code."""
    return "—" if x is None or math.isnan(x) else f"{x:.3f}"


def _classe(x: float, seuil: float = 0.0) -> str:
    return "bon" if x > seuil else ("mauvais" if x < -seuil else "neutre")


def _marqueur(forme: str, x: float, y: float, couleur: str, titre: str) -> str:
    """Un marqueur SVG, avec anneau de surface pour rester lisible s'ils se recouvrent."""
    r = _RAYON
    fin = f'fill="{couleur}" stroke="var(--surface-1)" stroke-width="1.4" opacity="0.9">'
    fin += f"<title>{html.escape(titre)}</title>"
    if forme == "cercle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {fin}</circle>'
    if forme == "carre":
        c = r * 0.88
        return (
            f'<rect x="{x - c:.1f}" y="{y - c:.1f}" width="{c * 2:.1f}" '
            f'height="{c * 2:.1f}" rx="1" {fin}</rect>'
        )
    if forme == "triangle":
        c = r * 1.15
        pts = (
            f"{x:.1f},{y - c:.1f} {x + c * 0.92:.1f},{y + c * 0.62:.1f} "
            f"{x - c * 0.92:.1f},{y + c * 0.62:.1f}"
        )
        return f'<polygon points="{pts}" {fin}</polygon>'
    if forme == "croix":
        c, b = r * 1.25, r * 0.42
        pts = " ".join(
            f"{x + dx:.1f},{y + dy:.1f}"
            for dx, dy in [
                (-b, -c),
                (b, -c),
                (b, -b),
                (c, -b),
                (c, b),
                (b, b),
                (b, c),
                (-b, c),
                (-b, b),
                (-c, b),
                (-c, -b),
                (-b, -b),
            ]
        )
        return f'<polygon points="{pts}" {fin}</polygon>'
    c = r * 1.12
    pts = f"{x:.1f},{y - c:.1f} {x + c:.1f},{y:.1f} {x:.1f},{y + c:.1f} {x - c:.1f},{y:.1f}"
    return f'<polygon points="{pts}" {fin}</polygon>'


def _legende(rapport: Rapport, etalon: bool = False) -> str:
    modeles = {s.modele: s.couleur for s in rapport.series}
    bras = {s.bras: s.forme for s in rapport.series}
    parts = ['<div class="leg"><div class="leg-bloc"><span class="leg-tit">Modèle</span>']
    for modele, couleur in modeles.items():
        parts.append(
            f'<span class="leg-item"><svg width="14" height="14" aria-hidden="true">'
            f'<circle cx="7" cy="7" r="5" fill="{couleur}"/></svg>{html.escape(modele)}</span>'
        )
    parts.append('</div><div class="leg-bloc"><span class="leg-tit">Bras</span>')
    for libelle, forme in bras.items():
        parts.append(
            f'<span class="leg-item"><svg width="14" height="14" aria-hidden="true">'
            f"{_marqueur(forme, 7, 7, '#52514e', '')}</svg>{html.escape(libelle)}</span>"
        )
    parts.append("</div>")
    if etalon:
        parts.append(
            "<div class='leg-bloc'><span class='leg-tit'>Étalon</span>"
            "<span class='leg-item'><svg width='14' height='14' aria-hidden='true'>"
            "<rect x='6' y='1' width='2.6' height='12' fill='#0b0b0b'/></svg>"
            "expert (vérité terrain)</span></div>"
        )
    return "".join(parts) + "</div>"


def _section_comparaison(rapport: Rapport, reference: str, n_boot: int) -> str:
    """Section 1 : chaque bras contre la référence du MÊME modèle, mêmes copies."""
    par_modele: dict[str, dict[str, Serie]] = defaultdict(dict)
    for s in rapport.series:
        par_modele[s.modele][s.bras] = s

    lignes = []
    for modele, bras in par_modele.items():
        ref = bras.get(reference)
        premier = True
        for libelle, serie in bras.items():
            m = compute_scoring_metrics(
                [r["y_true"] for r in serie.records], [r["y_pred"] for r in serie.records]
            )
            ecart = ic = ""
            if ref is not None and libelle != reference:
                bas, haut = intervalle_ecart(serie.records, ref.records, n_boot=n_boot)
                delta = m.cohen_kappa - _kappa(ref.records)
                etoile = "" if bas <= 0 <= haut else " ★"
                ecart = f'<span class="{_classe(delta, 0.001)}">{delta:+.3f}{etoile}</span>'
                ic = f'<span class="neutre">[{bas:+.3f} ; {haut:+.3f}]</span>'
            lignes.append(
                f"<tr><td>{html.escape(modele) if premier else ''}</td>"
                f"<td style='text-align:left'><span class='tag'>{html.escape(libelle)}</span></td>"
                f"<td>{m.n_items}</td><td>{_pct(m.raw_agreement)}</td>"
                f"<td><b>{_kap(m.cohen_kappa)}</b></td><td>{ecart}</td><td>{ic}</td></tr>"
            )
            premier = False
    return (
        "<h2>1. Chaque bras contre sa référence</h2>"
        "<p class='note'>Chaque bras est comparé à la référence du <em>même</em> modèle, sur "
        "les <em>mêmes</em> copies. L'intervalle de confiance à 95 % vient d'un bootstrap sur "
        "les copies entières : les items d'une même copie ne sont pas indépendants, et un "
        "décalage d'alignement fausse d'un coup toute la suite. ★ signale un intervalle "
        "excluant zéro.</p>"
        "<div class='scroll'><table><thead><tr><th>Modèle</th><th>Bras</th><th>Items</th>"
        "<th>Accord</th><th>Kappa</th><th>Écart</th><th>IC 95 %</th></tr></thead><tbody>"
        + "".join(lignes)
        + "</tbody></table></div>"
    )


def _nuage(
    rapport: Rapport,
    items: list[GridItem],
    champ: str,
    domaine: tuple[float, float],
    ticks: list[float],
    fmt,
    titre: str,
    sous_titre: str,
    etalon: bool = False,
) -> str:
    """Nuage horizontal : une ligne par item, un marqueur par couple modèle × bras."""
    bas, haut = domaine
    hauteur = _MARGE_H + len(items) * _HAUTEUR_LIGNE + _MARGE_B
    largeur = _MARGE_G + _LARGEUR_TRACE + _MARGE_D
    decalage = 52 if etalon else 10

    def px(v: float) -> float:
        return _MARGE_G + (v - bas) / (haut - bas) * _LARGEUR_TRACE

    out = [
        f'<figure class="fig"><figcaption class="fig-tit">{html.escape(titre)}</figcaption>',
        f"<p class='fig-sub'>{sous_titre}</p>",
        _legende(rapport, etalon),
        f'<div class="scroll"><svg viewBox="0 0 {largeur} {hauteur}" width="{largeur}" '
        f'height="{hauteur}" role="img" aria-label="{html.escape(titre)}">',
    ]
    for t in ticks:
        x = px(t)
        out.append(
            f'<line class="axe-grille" x1="{x:.1f}" y1="{_MARGE_H - 12}" x2="{x:.1f}" '
            f'y2="{hauteur - _MARGE_B + 4}"/>'
            f'<text class="axe-lbl" x="{x:.1f}" y="{_MARGE_H - 20}" text-anchor="middle">'
            f"{fmt(t)}</text>"
        )
    for rang, item in enumerate(items):
        y = _MARGE_H + rang * _HAUTEUR_LIGNE + _HAUTEUR_LIGNE / 2
        if rang % 2 == 0:
            out.append(
                f'<rect class="bande" x="{_MARGE_G}" y="{y - _HAUTEUR_LIGNE / 2:.1f}" '
                f'width="{_LARGEUR_TRACE}" height="{_HAUTEUR_LIGNE}"/>'
            )
        libelle = html.escape(f"{rang_grille(rapport, item):>2}. {item.attendu}")
        out.append(
            f'<text class="item-lbl" x="{_MARGE_G - decalage}" y="{y + 3.5:.1f}" '
            f'text-anchor="end">{libelle}</text>'
        )
        vh = rapport.err_humain.get(item.item_id, 0.0)
        if etalon:
            out.append(
                f'<text class="item-val" x="{_MARGE_G - 10}" y="{y + 3.5:.1f}" '
                f'text-anchor="end">{vh * 100:.0f} %</text>'
            )
            valeurs = [
                rapport.par_item[s.cle][item.item_id][champ]
                for s in rapport.series
                if item.item_id in rapport.par_item[s.cle]
            ]
            valeurs = [v for v in valeurs if v is not None]
            if valeurs:
                lo, hi = min([*valeurs, vh]), max([*valeurs, vh])
                out.append(
                    f'<line class="portee" x1="{px(lo):.1f}" y1="{y:.1f}" '
                    f'x2="{px(hi):.1f}" y2="{y:.1f}"/>'
                )
        for s in rapport.series:
            v = rapport.par_item[s.cle].get(item.item_id, {}).get(champ)
            if v is None:
                continue
            if etalon:
                detail = f"{v * 100:.0f} % de fautes (expert {vh * 100:.0f} %)"
            elif champ == "accord":
                detail = f"accord {v * 100:.0f} %"
            else:
                detail = f"kappa {v:.2f}"
            out.append(
                _marqueur(
                    s.forme,
                    px(max(bas, min(haut, v))),
                    y,
                    s.couleur,
                    f"{item.attendu} — {s.modele} / {s.bras} : {detail}",
                )
            )
        if etalon:
            out.append(
                f'<rect class="etalon" x="{px(vh) - 1.3:.1f}" y="{y - 6:.1f}" width="2.6" '
                f'height="12"><title>expert : {vh * 100:.0f} % de fautes sur '
                f"{html.escape(item.attendu)}</title></rect>"
            )
    out.append("</svg></div></figure>")
    return "".join(out)


def rang_grille(rapport: Rapport, item: GridItem) -> int:
    """Position de l'item dans l'ordre de la dictée (1 à 83)."""
    return rapport.items.index(item) + 1


def _section_par_item(rapport: Rapport) -> str:
    """Section 2 du rapport (numérotée 4 à l'origine) : accord et kappa item par item."""
    moyennes = {
        it.item_id: sum(
            rapport.par_item[s.cle][it.item_id]["accord"]
            for s in rapport.series
            if it.item_id in rapport.par_item[s.cle]
        )
        / max(sum(1 for s in rapport.series if it.item_id in rapport.par_item[s.cle]), 1)
        for it in rapport.items
    }
    # Du plus difficile au plus facile : le nuage se lit comme un classement, et les
    # items où les modèles divergent sautent aux yeux.
    items = sorted(rapport.items, key=lambda it: moyennes.get(it.item_id, 1.0))

    kappas = [
        v["kappa"]
        for s in rapport.series
        for v in rapport.par_item[s.cle].values()
        if v["kappa"] is not None
    ]
    bas = min(-0.2, (min(kappas) // 0.2) * 0.2) if kappas else -0.2
    ticks = [bas + 0.2 * n for n in range(int((1 - bas) / 0.2) + 1)]
    definis = sum(
        1
        for s in rapport.series
        for v in rapport.par_item[s.cle].values()
        if v["kappa"] is not None
    )
    total = sum(len(rapport.par_item[s.cle]) for s in rapport.series)

    return (
        "<h2>2. Performance item par item</h2>"
        "<p class='note'>Chaque ligne est un item de la dictée, chaque marqueur un couple "
        f"modèle × bras agrégé sur les {rapport.n_copies} copies communes. La "
        "<strong>couleur</strong> porte le modèle, la <strong>forme</strong> le bras : autant "
        "de teintes distinctes que de séries ne serait ni lisible ni sûr pour les daltonismes. "
        "Survoler un marqueur en donne la valeur exacte.</p>"
        + _nuage(
            rapport,
            items,
            "accord",
            (0.0, 1.0),
            [0, 0.25, 0.5, 0.75, 1.0],
            lambda t: f"{t * 100:.0f} %",
            "Accord brut par item",
            "Part des copies où le code du modèle coïncide avec celui de l'expert.",
        )
        + _nuage(
            rapport,
            items,
            "kappa",
            (bas, 1.0),
            ticks,
            lambda t: f"{t:.1f}",
            "Kappa de Cohen par item",
            "Accord corrigé du hasard. "
            f"{total - definis} des {total} points sont absents : le kappa est indéfini quand "
            "l'expert attribue le même code à toutes les copies d'un item — il n'y a alors "
            "aucune variance à expliquer, et l'afficher à zéro serait faux.",
        )
    )


def _section_taux_faute(rapport: Rapport) -> str:
    """Section 3 (5 à l'origine) : où tombe chaque modèle face à la difficulté réelle."""
    items = sorted(rapport.items, key=lambda it: -rapport.err_humain.get(it.item_id, 0.0))
    return (
        "<h2>3. Où tombe chaque modèle par rapport à la difficulté réelle des items ?</h2>"
        "<p class='note'>L'abscisse n'est plus une performance : c'est le <strong>taux de "
        "faute attribué aux élèves</strong>. Les items sont classés du plus raté au mieux "
        "réussi d'après l'expert. Un modèle bien calibré pose ses marqueurs sur le repère "
        "foncé ; à gauche il sous-détecte les fautes, à droite il en invente.</p>"
        + _nuage(
            rapport,
            items,
            "taux_erreur",
            (0.0, 1.0),
            [0, 0.25, 0.5, 0.75, 1.0],
            lambda t: f"{t * 100:.0f} %",
            "Taux de faute attribué aux élèves, par item",
            "Part des copies où l'item est jugé fautif (code « 9 »). Les absences (code "
            "« 0 ») sont exclues : un mot non écrit n'est pas une faute d'orthographe.",
            etalon=True,
        )
    )


def _section_par_type(rapport: Rapport) -> str:
    """Section 4 (7 à l'origine) : mots et ponctuation, deux tâches distinctes."""
    n_mots = sum(1 for it in rapport.items if it.type != "ponctuation")
    n_ponct = len(rapport.items) - n_mots

    if rapport.series:
        ref = rapport.series[0].records
        lignes_expert = []
        for t in ["mot", "ponctuation"]:
            ids = {
                it.item_id
                for it in rapport.items
                if (it.type == "ponctuation") == (t == "ponctuation")
            }
            rs = [r for r in ref if r["item_id"] in ids]
            if not rs:
                continue
            c = Counter(r["y_true"] for r in rs)
            tot = sum(c.values())
            lignes_expert.append(
                f"<tr><td>{t}</td><td>{n_ponct if t == 'ponctuation' else n_mots}</td>"
                f"<td>{tot}</td><td>{c['1'] / tot * 100:.1f} %</td>"
                f"<td>{c['9'] / tot * 100:.1f} %</td><td>{c['0'] / tot * 100:.1f} %</td></tr>"
            )
    else:
        lignes_expert = []

    lignes = []
    precedent = None
    for s in rapport.series:
        v = rapport.par_type.get(s.cle, {})
        mot, ponct = v.get("mot"), v.get("ponctuation")
        if not mot or not ponct:
            continue
        ecart = (mot["kappa"] or 0) - (ponct["kappa"] or 0)
        lignes.append(
            f"<tr><td>{html.escape(s.modele) if s.modele != precedent else ''}</td>"
            f"<td style='text-align:left'><span class='tag'>{html.escape(s.bras)}</span></td>"
            f"<td>{_pct(mot['accord'])}</td><td><b>{_kap(mot['kappa'])}</b></td>"
            f"<td>{_pct(ponct['accord'])}</td><td><b>{_kap(ponct['kappa'])}</b></td>"
            f"<td class='{_classe(ecart, 0.001)}'>{ecart:+.3f}</td></tr>"
        )
        precedent = s.modele

    return (
        "<h2>4. Mots ou ponctuation : deux tâches distinctes</h2>"
        f"<p class='note'>La grille compte {n_mots} mots et {n_ponct} signes de ponctuation. "
        "Ce ne sont pas les mêmes jugements : un mot peut être mal orthographié de mille "
        "façons, un signe n'a que trois issues et les élèves l'omettent bien plus souvent. Un "
        "kappa global mélange les deux.</p>"
        "<h3>Ce que l'expert observe</h3>"
        "<div class='scroll'><table><thead><tr><th>Nature</th><th>Items</th><th>Décisions</th>"
        "<th>Correct (1)</th><th>Erreur (9)</th><th>Absent (0)</th></tr></thead><tbody>"
        + "".join(lignes_expert)
        + "</tbody></table></div>"
        "<h3>Performance de chaque série, par nature d'item</h3>"
        "<div class='scroll'><table><thead><tr><th rowspan='2'>Modèle</th><th rowspan='2'>Bras</th>"
        "<th colspan='2'>Mots</th><th colspan='2'>Ponctuation</th>"
        "<th rowspan='2'>Écart de kappa</th></tr>"
        "<tr><th>Accord</th><th>Kappa</th><th>Accord</th><th>Kappa</th></tr></thead><tbody>"
        + "".join(lignes)
        + "</tbody></table></div>"
    )


def rendre_html(rapport: Rapport, reference: str = "référence", n_boot: int = 400) -> str:
    """Assemble le rapport HTML complet, autonome (CSS inclus, aucune dépendance).

    Args:
        rapport: agrégats produits par `agreger`.
        reference: libellé du bras servant d'étalon aux écarts.
        n_boot: nombre de tirages du bootstrap des intervalles.

    Returns:
        Le document HTML.
    """
    horodatage = datetime.now().strftime("%d/%m/%Y à %H:%M")
    modeles = sorted({s.modele for s in rapport.series})
    bras = list(dict.fromkeys(s.bras for s in rapport.series))
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Comparaison de bras — évaluation dictée</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        "<h1>Comparaison de bras d'expérience</h1>"
        f"<p class='sub'>Dictée CM2 · {rapport.n_copies} copies communes · "
        f"{len(modeles)} modèle(s) × {len(bras)} bras · généré le {horodatage}</p>"
        "<div class='enc'>Un <strong>bras</strong> est une variante de prompt testée contre "
        "la référence. Ce rapport répond à une question que ni Langfuse ni le site ne "
        "traitent : cette variante fait-elle mieux que la référence, sur le même modèle et "
        "les mêmes copies, et pourquoi. Les copies vierges et les items illisibles sont "
        "écartés (décision D8).</div>"
        + _section_comparaison(rapport, reference, n_boot)
        + _section_par_item(rapport)
        + _section_taux_faute(rapport)
        + _section_par_type(rapport)
        + "<footer>Agrégats uniquement : ni image, ni transcription, ni donnée identifiante. "
        "Le fichier reste néanmoins sous <code>data/</code>, non versionné.</footer>"
        "</div></body></html>"
    )


def construire_rapport(
    modeles: list[str],
    bras: dict[str, str] | None = None,
    dossier: str | Path = "data/processed",
    grille: str | Path = "configs/grille_dictee_2015.json",
    reference: str = "référence",
    n_boot: int = 400,
) -> tuple[str, Rapport]:
    """Chaîne complète : chargement, agrégation, rendu.

    Args:
        modeles: modèles à comparer.
        bras: libellé -> préfixe de run. [défaut : `BRAS_PAR_DEFAUT`]
        dossier: dossier des prédictions.
        grille: chemin de la grille de codage.
        reference: libellé du bras servant d'étalon.
        n_boot: nombre de tirages du bootstrap.

    Returns:
        Le couple (HTML, agrégats).

    Raises:
        RuntimeError: si aucune série n'a pu être chargée.
    """
    series = charger_series(modeles, bras or BRAS_PAR_DEFAUT, dossier)
    if not series:
        raise RuntimeError(
            f"Aucun run trouvé dans {dossier} pour les modèles {modeles}. "
            "Lancer un benchmark d'abord (voir configs/README.md)."
        )
    rapport = agreger(series, grille)
    return rendre_html(rapport, reference, n_boot), rapport
