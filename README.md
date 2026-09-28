# Évaluation automatique de la dictée

Collaboration **DEPP × SSP Lab (Insee)**, destinée aux équipes qui évaluent
l'opportunité d'assister la correction humaine par l'IA.

[![CI](https://github.com/InseeFrLab/evaluation_dictee/actions/workflows/ci.yml/badge.svg)](https://github.com/InseeFrLab/evaluation_dictee/actions/workflows/ci.yml)
[![Site](https://github.com/InseeFrLab/evaluation_dictee/actions/workflows/site.yml/badge.svg)](https://inseefrlab.github.io/evaluation_dictee/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Documentation** — contexte, décisions méthodologiques et conventions :
> [CLAUDE.md](CLAUDE.md) · [`docs/decisions.md`](docs/decisions.md) ·
> [site Quarto](https://inseefrlab.github.io/evaluation_dictee/)



## Ce que fait le projet

Observer la possibilité et la fiabilité d'un codage automatique des items d'une
dictée manuscrite, recueillie dans le cadre des évaluations
[**CEDRE**](https://www.education.gouv.fr/depp/cycle-des-evaluations-disciplinaires-realisees-sur-echantillon-cedre-en-fin-d-ecole-et-fin-de-2870).

### La brique principale du projet : l'évaluation des dictées

Deux pipelines sont disponibles pour la prédiction des codes d'évaluation d'une copie scannée, comparées sur les mêmes
métriques (dont *accord brut* et *kappa de cohen*) :

| Approche | Principe | Intérêt |
|---|---|---|
| **end2end** *(défaut)* | un modèle multimodal (VLM) lit l'image et attribue les codes en une seule passe | plus simple, aucune perte d'information entre deux étapes |
| **two step** | la copie est d'abord transcrite en texte par un modèle HTR, puis ce texte est codé par un modèle plus léger (LLM Texte) | isole les erreurs de **lecture** de celles de **jugement** |

Le choix se fait dans la configuration de l'expérience (voir
[Configuration](#configuration)).

### Autres briques du projet

- **Évaluation HTR seule** sur le corpus **Scoledit** (transcriptions humaines, fautes
  d'élèves préservées) : mesure la fidélité de lecture indépendamment du codage, et
  permet donc de savoir si un désaccord vient d'une mauvaise lecture ou d'un mauvais
  jugement.
- **Fine-tuning QLoRA** d'un VLM sur Scoledit (CP→CM2) pour spécialiser la lecture de
  l'écriture manuscrite d'élèves : produit un adaptateur léger (~50-200 Mo) et
  demande un **GPU H100**.



## Installation

### Prérequis

| | Version | Note |
|---|---|---|
| **Python** | ≥ 3.11 | installé par `uv` si absent |
| **[uv](https://docs.astral.sh/uv/)** | ≥ 0.4 | **seul** gestionnaire d'environnement du projet (pas de `pip install` ni de `conda`) |
| Accès **SSP Cloud** | — | S3 (MinIO), llm.lab, Langfuse ; voir [Configuration](#configuration) |
| `quarto` | ≥ 1.10 | uniquement pour rendre le site |
| GPU **H100** | — | uniquement pour le fine-tuning |

`uv` n'est pas installé ? `curl -LsSf https://astral.sh/uv/install.sh | sh`
(déjà présent sur les services *vscode-python* du SSP Cloud).

### Installation du repo et des dépendances du projet

```bash
git clone https://github.com/InseeFrLab/evaluation_dictee.git
cd evaluation_dictee
uv sync            # dépendances + groupe dev + mode éditable (imports evaluation_dictee)
```

Extras optionnels, à la demande :

```bash
uv sync --extra notebooks   # JupyterLab + matplotlib (analyse des résultats)
uv sync --extra website     # noyau Jupyter + matplotlib (rendu Quarto)
uv sync --extra gpu         # torch, transformers, vllm (machines GPU seulement)
```

---

## Exemple minimal

Une fois les accès configurés ([section suivante](#configuration)) :

```bash
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml --limit 5
```

Cinq copies sont codées et comparées à l'expert en quelques minutes. Résultat :
`data/processed/<name>_<modele>_predictions.jsonl`, une ligne par item × copie, et
les métriques d'accord affichées en fin de run.

> Le nom du fichier **porte toujours le modèle** de l'étape 1 (plus celui de l'étape 2
> s'il diffère), que le modèle vienne du YAML ou de `--model-name` : deux modèles
> n'écrasent jamais le même checkpoint et peuvent tourner en parallèle.

Tout est journalisé dans **Langfuse** : une *session* par run, une *trace* par copie
(entrée/sortie + score d'accord), les appels LLM en *générations* imbriquées, et les
métriques agrégées du run en Scores et metadata.



## Données

Les données sont **produites et détenues par la DEPP**, qui les met
à disposition du SSP Lab pour ce projet. Elles ne sont **couvertes ni par la licence
de ce dépôt** ([MIT](LICENSE), qui ne porte que sur le code) **ni par aucune licence
ouverte** : tout réemploi hors de ce cadre suppose l'accord de la DEPP.

**Aucune donnée n'est versionnée dans ce dépôt**, et ce n'est pas négociable : les
copies sont des **écritures d'élèves mineurs**; elles ne quittent jamais le SSP Cloud.

### Origine et accès

| Jeu | Emplacement | Contenu |
|---|---|---|
| **Dictée CM2 2015** - imagettes | `s3://projet-production-ecrits-depp/dictee_2015/` | ~3 469 copies scannées |
| **Dictée CM2 2015** - codes experts | `s3://projet-production-ecrits-depp/resultat_dictee_2015.csv` | *gold standard*, 83 items par copie |
| [**Scoledit**](https://e-calm.huma-num.fr/) (HTR) | `s3://projet-production-ecrits-depp/scoledit/{scans,annotation}/<niveau>/` | transcriptions humaines CP→CM2, fautes préservées |
| **Grille de codage** | [`configs/grille_dictee_2015.json`](configs/grille_dictee_2015.json) | *versionné* : mot attendu + fautes connues, par item |
| **Prédictions exportées** | `$S3_PREDICTIONS_PREFIX` (défaut `…/predictions`) | sorties de run, relues par les notebooks et le site |

Les chemins d'entrée vivent **dans le YAML de chaque expérience**, pas dans `.env` :
un run reste ainsi reproductible à partir de sa seule config. L'accès S3 est
transparent sur Onyxia (identifiants injectés), un chemin `s3://…` se lit comme un
chemin local.

### Format attendu

- Les fichiers `.png` fournis sont en réalité des **TIFF bi-level 1 bit** (compression
  G4, 1594×2044). `src/evaluation_dictee/data/loaders.py` normalise le format en
  entrée : ne pas court-circuiter ce chargeur.
- La binarisation 1 bit détruit les nuances de gris, ce qui rend **les accents
  difficiles à lire** : c'est une cause connue d'une part des désaccords modèle/expert.
- Les annotations sont des **codes**, pas des transcriptions mot à mot. Fine-tuner un
  HTR demande un *ground truth* de transcription séparé (d'où **Scoledit**).



## Configuration

### Configuration des secrets

Tout passe par des **variables d'environnement** : **aucun secret ne doit être ajouté dans le code ni
dans les YAML**. Liste complète et commentée des variables d'environnement : [`.env.example`](.env.example).

| Clé | Requis pour |
|-----|-------------|
| `LLM_BASE_URL`, `LLM_API_KEY` | tout run |
| `LANGFUSE_*` (URL + clés publique et secrète) | traçage des runs |
| `S3_PREDICTIONS_PREFIX` | export des prédictions |
| `MLFLOW_TRACKING_URI` | fine-tuning |

- **Sur le SSP Cloud** : il faut stocker ces valeurs dans le **Vault Onyxia**
  (`Mon compte` → `Vault`), puis référencer le secret au lancement du service : Onyxia
  les injecte dans l'environnement. Les identifiants **S3 sont injectés
  automatiquement**, il n'y a rien à configurer pour le stockage.
- **En local** : `cp .env.example .env`, puis compléter. Ce fichier n'est jamais commité sur GitHub.

### Congiguration d'une expérience (une expérience = un YAML)

Un fichier dans `configs/` décrit une expérience reproductible, rangé par famille
(`scoring/`, `htr/`, `finetune/`). Partir de
[`configs/scoring/dictee_REFERENCE.yaml`](configs/scoring/dictee_REFERENCE.yaml),
exhaustivement commenté ; détails dans [`configs/README.md`](configs/README.md).

| Champ | Rôle |
|-------|------|
| `approach` | `end_to_end` ou `two_stage` |
| `model.name` | modèle servi par llm.lab (ex. `gemma4-26b-moe`) |
| `concurrency` | copies évaluées en parallèle (défaut 8) —> **principal levier de temps mural** |
| `data.limit` | nombre de copies ; `null` = tout le corpus |
| `grid.scheme` | `simplifiee` (1/9/0) ou `complete` (1/3/4/5/9/0) |



## Structure du projet

```
evaluation_dictee/
├── src/evaluation_dictee/   ← le paquet : config, data, models, pipeline, evaluation, transcription, utils
├── scripts/                 ← points d'entrée Python (un run = un process)
├── launchers/               ← lanceurs shell des runs longs (voir launchers/README.md)
├── configs/                 ← une expérience = un YAML (scoring/, htr/, finetune/) + la grille de codage
├── notebooks/               ← 03 analyse, 04 diagnostic, 05 transcription HTR
├── website/                 ← site Quarto (archi, résultats, métriques, fine-tuning)
├── tests/                   ← 155 tests unitaires (pytest)
└── docs/                    ← décisions méthodologiques, grille de codage, schéma du pipeline
```

<details>
<summary>Détail des modules et des scripts</summary>

```
src/evaluation_dictee/
├── config.py           ← configs validées (Pydantic) + secrets
├── data/               ← chargement images (S3, TIFF 1 bit), grille, labels
├── models/             ← interface Scorer, scorers end-to-end et two-stage
├── pipeline/           ← prompts, évaluation + checkpointing, ré-alignement (Needleman-Wunsch)
├── evaluation/         ← métriques, statistiques, calibration, diagnostics, rapports HTML
├── transcription/      ← pipeline HTR indépendant (Scoledit) : loader, CER/WER, diffs
└── utils/              ← logging, suivi Langfuse, export S3

scripts/
├── run_benchmark.py            ← scoring dictée (les deux approches)
├── run_htr_benchmark.py        ← évaluation HTR Scoledit
├── export_predictions.py       ← export d'un run terminé vers S3
└── finetune_htr_scoledit.py    ← fine-tuning QLoRA (GPU H100)

launchers/
├── launch_eval.sh              ← run complet : détache du terminal, surveille, relance, exporte le run
└── install_assistant.sh        ← installe Claude Code / openCode, branchés sur llm.lab

.claude/skills/launch/          ← skill Claude Code   ┐ n'appellent que launch_eval.sh,
.opencode/command/              ← commande openCode   ┘ aucune logique dupliquée
```

</details>



## Lancer un run complet

Un benchmark complet (3 469 copies × ~30 s) prend **~15 h** : il ne doit jamais
dépendre de l'onglet du navigateur. 

Il ne faut **pas lancer les copies à la main**, mais passer par
le lanceur, qui est le **point d'entrée unique**. En effet, le lanceur **détache le run
du terminal** (celui-ci continue donc de tourner après la fermeture de la session ou
de l'onglet), le surveille, puis relance les copies en échec avant d'exporter le résultat
vers S3.

```bash
launchers/launch_eval.sh --config configs/scoring/dictee_end2end.yaml            # lancer
launchers/launch_eval.sh --config configs/scoring/dictee_end2end.yaml --status   # avancement + heure de fin
launchers/launch_eval.sh --config configs/scoring/dictee_end2end.yaml --stop     # arrêt propre
launchers/launch_eval.sh --help                                                  # toutes les options
```

> **Toujours préciser le run concerné (option `--config`)** pour toute action spécifique sur un run
> (avec les options `--status` et `stop` par exemple)

Options courantes : 

- `--model-name <modèle>` (surcharge le YAML), 
- `--limit N` (test rapide, sans export S3), 
- `--passes N` (nombre de relances des copies en échec, défaut 3), 
- `--foreground` (débogage).

>Sortie de `--status`, avec un débit **mesuré** sur la passe en cours :
>
>```text
>Run dictee_end2end_qwen3-6-35b-moe
>✔ en cours :
>      10489  01:33  uv run scripts/run_benchmark.py --config …
>  copies      412/3469 (11%)
>  débit       464.5 copies/h (depuis 0h 53min)
>  fin estimée 2026-09-06 09:12 (dans 6h 34min)
>```
>
>Les logs sont horodatés, un par lancement : `logs/<run>_<horodatage>.log`, avec
>`logs/<run>.latest.log` qui pointe vers le dernier. Détails des vérifications
>effectuées au démarrage : [`launchers/README.md`](launchers/README.md).

### Reprise automatique

Le benchmark écrit sur disque après **chaque** copie (`flush + fsync`). 

Après une interruption, relancer la même commande saute les copies déjà faites, seule la copie
en cours est perdue. Pour repartir de zéro, supprimer le `.jsonl` (ou changer
`config.name`).

### Deux règles à ne pas enfreindre

>**Ne jamais arrêter un run par son PID.** `uv run …` lance DEUX processus : le `uv`
visible en tête de `ps`, et le `python3` qui fait réellement le travail. 
>
>Tuer le premier ne tue pas le second : le run paraît arrêté alors qu'il continue d'écrire
dans le JSONL, et un nouveau lancement viendrait s'y ajouter en double. 
>
>-> Utiliser `--stop`, qui arrête les deux ou l'instruction : 
`pkill -f "run_benchmark.py --config <la config>"`.

> **Un seul run par fichier de sortie.** Un second run visant le même fichier s'arrête
> sur le verrou `<sortie>.lock` : deux runs qui appendent le même JSONL dupliquent les
> copies et faussent les métriques.


### Depuis un assistant de code

Les assistants ne font qu'appeler le lanceur, sans dupliquer la moindre logique.

| Assistant | Fichier | Comment |
|---|---|---|
| Claude Code | [`.claude/skills/launch/`](.claude/skills/launch/SKILL.md) | `/launch`, ou « lance l'évaluation complète » |
| openCode | [`.opencode/command/launch.md`](.opencode/command/launch.md) | `/launch`, `/launch status`, `/launch stop` |
| Un autre (Cursor, Codex…) | — | lui faire lire `launchers/README.md`, ou lancer le `.sh` soi-même |

Pas encore d'assistant installé ? `launchers/install_assistant.sh` l'installe et le
branche sur **llm.lab** plutôt que sur une API payante.


## Reproduire les résultats

Le pipeline complet, de l'environnement vierge aux figures publiées.

**1. Environnement et accès** : [Installation](#installation) puis
[Configuration](#configuration) de l'environnement. Pousser une fois les prompts à utiliser et les coûts dans
Langfuse pour garder une trace des évolutions :

```bash
# --env-file .env : Langfuse lit ses clés dans os.environ, que .env n'alimente pas seul.
uv run --env-file .env add-langfuse-prompt       # prompts d'évaluation
uv run --env-file .env add-langfuse-models       # coût théorique /1M tokens
```

**2. Vérifier que tout répond** : S3 (voir [Données](#données)), le modèle, les tests :

```bash
uv run python -c "from openai import OpenAI; from evaluation_dictee.config import Secrets; \
    s=Secrets(); c=OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key); \
    print(c.chat.completions.create(model='gemma4-26b-moe', \
    messages=[{'role':'user','content':'Dis bonjour'}], max_tokens=10).choices[0].message.content)"
uv run pytest -q
```

**3. Relancer les évaluations** : une config = une expérience ; les runs complets
passent par le [lanceur](#lancer-un-run-complet).

```bash
launchers/launch_eval.sh --config configs/scoring/dictee_end2end.yaml    # ~30 h
launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml  # ~30 h
uv run scripts/run_htr_benchmark.py --config configs/htr/htr_REFERENCE.yaml
```

**4. Exporter vers S3** : le pipeline écrit en local (append + fsync, pour la reprise) ;
l'export permet de rejouer notebooks et site sans relancer le pipeline.
`launch_eval.sh` le fait seul ; à la main :

```bash
uv run scripts/export_predictions.py --run-name dictee_two_stage_gemma4-26b-moe
uv run scripts/export_predictions.py --config configs/htr/htr_REFERENCE.yaml --htr
```

**5. Régénérer les analyses** : changer la variable `RUN_NAME` en tête de notebook
suffit pour analyser un autre run.

```bash
uv sync --extra notebooks && uv run jupyter lab
```

| Notebook | Ce qu'il fait | Prérequis |
|----------|---------------|-----------|
| `03_analyse_resultats.ipynb` | métriques globales, prévalence par item (IC bootstrap), distributions, corrélation modèle vs expert, seuils critiques, export HTML DEPP | un run de scoring terminé |
| `04_diagnostic.ipynb` | copies triées par désaccord, HTML des N pires, HTML d'une copie précise (scan + transcription + comparaison expert/modèle) | un run de scoring terminé |
| `05_analyse_transcription_htr.ipynb` | CER/WER, distribution, HTML des N pires transcriptions et de N aléatoires | un run HTR terminé |

Le **rapport pour la DEPP** se génère depuis la section 9 du notebook 03 :
`data/processed/rapport_depp_<RUN>.html`, autonome (assets inlinés), prêt à envoyer.

**6. Republier le site** — les pages « Résultats » et « Écarts » relisent les
prédictions exportées sur S3 au moment du rendu.

```bash
uv sync --extra website
quarto preview website     # aperçu local (rechargement à chaud)
quarto render website      # génère website/_site/
```

La publication sur GitHub Pages est automatique à chaque *push* sur `main`
(`.github/workflows/site.yml`). 



## Contribuer

### Workflow

1. Partir de `main` à jour, créer une branche : `git switch -c <type>-<sujet>`
   (ex. `feat-scorer-multimodal`, `docs-nettoyage-readme`).
2. Développer, puis **vérifier localement avant de pousser** (voir ci-dessous).
3. Ouvrir une *pull request* vers `main`. La [CI](.github/workflows/ci.yml) rejoue
   lint, formatage, typage (non bloquant) et tests.

**Messages de commit** en français, format `type: description` — `feat:`, `fix:`,
`docs:`, `refactor:`, `test:`.

### Vérifications locales

```bash
uv run ruff format src tests scripts        # formatage
uv run ruff check src tests scripts         # lint
uv run mypy src                             # typage
uv run pytest                               # toute la suite (155 tests)
uv run pytest tests/test_alignment.py -v    # un fichier
uv run pytest -k "chain_of_thought"         # par motif
```

### Conventions

Détaillées dans [CLAUDE.md](CLAUDE.md) § 8. L'essentiel :

- **Python ≥ 3.11**, dépendances gérées avec **uv exclusivement**.
- Annoter les fonctions publiques ; docstrings **en français**, style Google, court.
- **Pas de chemin en dur** : tout passe par `configs/*.yaml` et `config.py`.
- **Pas de secret dans le code** : variables d'environnement uniquement.
- **Reproductibilité** : une expérience = une config versionnée + un run tracé dans
  Langfuse. Fixer les graines aléatoires.
- Les `notebooks/` servent à explorer : dès qu'un bout de code devient réutilisable,
  le déplacer dans `src/`.

**⚠️ Ne jamais committer de données sur GitHub, ni les checkpoitns, les logs ou le fichier *.env*.**

### Par où commencer

Lire [CLAUDE.md](CLAUDE.md), puis suivre le fil de `scripts/run_benchmark.py` : il lit
une config, charge les données, appelle un modèle, calcule les métriques. En cas de
doute sur une décision méthodologique, la réponse est dans CLAUDE.md ou
[`docs/decisions.md`](docs/decisions.md).


## Licence

**Le code** de ce dépôt est sous licence **MIT** — voir [`LICENSE`](LICENSE).
Copyright (c) 2026 Insee et DEPP (Ministère de l'Éducation nationale).

**Les données** ne sont pas couvertes par cette licence. Produites et détenues par la
DEPP, elles ne sont pas versionnées ici et restent soumises au cadre d'accès de la
DEPP et du SSP Cloud — voir [Données](#données).




## Aide-mémoire

```bash
# ─────────── Installation & configuration (une seule fois) ───────────
uv sync                                          # environnement Python (dev inclus)
cp .env.example .env && $EDITOR .env             # hors Onyxia seulement (sinon : Vault)
uv run --env-file .env add-langfuse-prompt       # pousse les prompts d'évaluation
uv run --env-file .env add-langfuse-models       # coût théorique /1M tokens
launchers/install_assistant.sh --check           # (option) assistant de code : tester l'endpoint
launchers/install_assistant.sh --endpoint "$LLM_BASE_URL"   # installer + brancher sur llm.lab

# ─────────── Runs ───────────
# Run COMPLET (~3469 copies) — la voie normale. Répéter --config (et --model-name)
# à l'identique sur les trois commandes : elles identifient le run.
CFG=configs/scoring/dictee_end2end.yaml
launchers/launch_eval.sh --config $CFG            # lancer
launchers/launch_eval.sh --config $CFG --status   # avancement + heure de fin estimée
launchers/launch_eval.sh --config $CFG --stop     # arrêt propre
launchers/launch_eval.sh --config $CFG --limit 5  # test rapide de bout en bout, sans export S3

uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml   # un passage, au premier plan
uv run scripts/run_htr_benchmark.py --config configs/htr/htr_REFERENCE.yaml      # transcription seule
uv run scripts/finetune_htr_scoledit.py --config configs/finetune/finetune_REFERENCE.yaml  # GPU H100

# ─────────── Suivre un run en cours ───────────
tail -f logs/<run>.latest.log                    # log du dernier lancement
cat data/processed/<run>_failed_copies.txt       # copies en échec du dernier passage

# ─────────── Export des prédictions vers S3 ───────────
uv run scripts/export_predictions.py --run-name dictee_two_stage_gemma4-26b-moe
uv run scripts/export_predictions.py --config configs/htr/htr_REFERENCE.yaml --htr
eval-ecrit export configs/scoring/dictee_REFERENCE.yaml     # équivalent via la CLI installée
# Destination : $S3_PREDICTIONS_PREFIX/<name>_<modele>_predictions.jsonl
# (défaut s3://projet-production-ecrits-depp/predictions, surchargeable par --dest-prefix)

# ─────────── Analyse & documentation ───────────
uv sync --extra notebooks && uv run jupyter lab  # notebooks 03 / 04 / 05
quarto preview website                           # aperçu local (rechargement à chaud)
quarto render website                            # génère website/_site/
```

---

🤖 Generated with [Claude Code](https://claude.com/fr/product/claude-code)
