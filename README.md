# Évaluation automatique de la dictée des élèves

Évaluer automatiquement la dictée manuscrite d'élèves (CEDRE, école primaire) à
l'aide de modèles multimodaux open weight, et **comparer rigoureusement** le codage
automatique à celui d'un correcteur expert. Collaboration **DEPP × SSP Lab (INSEE)**.

> Contexte, décisions méthodologiques et conventions : **[CLAUDE.md](CLAUDE.md)** et
> [`docs/decisions.md`](docs/decisions.md).

---

## Ce que fait le projet

1. **Charge** les imagettes de dictée (TIFF 1 bit, depuis S3) et les codes de
   l'annotateur expert (gold standard).
2. **Demande à un modèle multimodal** (gemma4 ou qwen3.6 sur llm.lab) de coder chaque
   mot — correct / erreur / absent — directement depuis l'image et le texte de
   référence, sans étape d'OCR séparée.
3. **Compare** les codes du modèle à ceux de l'expert.
4. **Mesure** la fiabilité (kappa, rappel des fautes, sur-correction) et la
   **calibration de la confiance**, pour décider quels items renvoyer à un humain.

La cible est la **grille simplifiée** — `1` correct / `9` erreur / `0` absent
(décision D2 dans [`docs/decisions.md`](docs/decisions.md)).

### Deux approches comparables

Deux architectures derrière la même interface `Scorer`, choisies par le champ
`approach` du YAML :

| Approche | Principe | Intérêt |
|---|---|---|
| `end_to_end` *(défaut)* | un VLM lit l'image ET code en une passe | plus simple, moins de perte d'information |
| `two_stage` | étape 1 transcription HTR, étape 2 codage du texte (`model_stage2`, éventuellement un modèle texte plus léger) | isole les erreurs de **lecture** de celles de **jugement** |

### Autres briques

- **Évaluation HTR seule** sur le corpus **Scoledit** (transcriptions humaines, fautes
  préservées) : mesure la fidélité de lecture indépendamment du codage.
  `configs/htr/`, `scripts/run_htr_benchmark.py`.
- **Fine-tuning QLoRA** d'un VLM sur Scoledit (CP→CM2) pour spécialiser la lecture :
  adaptateur léger (~50-200 Mo), **GPU H100 requis**, suivi via **MLflow** (pas
  Langfuse). `configs/finetune/`, `scripts/finetune_htr_scoledit.py` — voir l'en-tête
  du script pour les prérequis (`unsloth`, `trl`, `bitsandbytes`).
- **Site Quarto** ([`website/`](website/)) : architecture, résultats, métriques et
  fine-tuning expliqués pour un public statisticien novice en IA.

---

## Démarrage rapide (SSP Cloud / VSCode)

### 1. Installer

```bash
git clone <url-du-depot> evaluation_dictee
cd evaluation_dictee
uv sync            # dépendances + groupe dev + mode éditable (imports evaluation_dictee)
```

### 2. Configurer les accès

Toute la configuration sensible passe par des **variables d'environnement**, lues via
`Secrets` (Pydantic, `src/evaluation_dictee/config.py`). **Aucun secret dans le code
ni dans les YAML.**

**Sur le SSP Cloud (recommandé) : le Vault Onyxia.** Stocker les secrets une fois
(`Mon compte` → `Vault`), puis référencer ce secret au lancement d'un service
(section `Vault`) : Onyxia les injecte comme variables d'environnement.

| Clé | Valeur |
|-----|--------|
| `LLM_BASE_URL` | `https://llm.lab.sspcloud.fr/api/v1` |
| `LLM_API_KEY` | ton token llm.lab |
| `LANGFUSE_BASE_URL` | `https://langfuse.lab.sspcloud.fr` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | depuis l'UI Langfuse |
| `MLFLOW_TRACKING_URI` | `https://mlflow.lab.sspcloud.fr` (fine-tuning seulement) |

Les services Onyxia ont déjà `VAULT_ADDR` et `VAULT_TOKEN`, donc le CLI fonctionne
directement : `vault kv get <chemin>/evaluation_dictee`.

> Les identifiants **S3** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
> `AWS_SESSION_TOKEN`, `AWS_S3_ENDPOINT`) sont **injectés automatiquement** par
> Onyxia : rien à configurer pour accéder au stockage.

**En local (hors Onyxia) :** repli sur un fichier `.env`, jamais commité —
`cp .env.example .env`, puis renseigner `LLM_API_KEY`, `LANGFUSE_*`, et si besoin
`AWS_*` / `MLFLOW_*`. Le code ne fait aucune différence entre une variable venue du
Vault et une variable venue de `.env` : dans les deux cas elle arrive par
l'environnement.

**Vérifier que S3 et le modèle répondent** : les deux commandes de la section
« Vérifier que tout marche » de l'[aide-mémoire](#aide-mémoire).

### 3. Un premier run, court

```bash
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml --limit 5
```

Cela produit `data/processed/<name>_<modele>_predictions.jsonl`, une ligne par
item × copie. **Le nom du fichier porte toujours le modèle** de l'étape 1 (plus celui
de l'étape 2 s'il diffère), que le modèle vienne du YAML ou de `--model-name` : deux
modèles n'écrasent donc jamais le même checkpoint, et peuvent tourner en parallèle.

Tout est journalisé dans **Langfuse** : une *session* par run, une *trace* par copie
(entrée/sortie + score d'accord), les appels LLM en *générations* imbriquées, et les
métriques agrégées du run en Scores et metadata.

Deux propriétés à connaître :

- **Reprise automatique.** Le benchmark écrit sur disque après CHAQUE copie
  (`flush + fsync`). Après une interruption, relancer la même commande saute les
  copies déjà faites — seule la copie en cours est perdue.
- **Parallélisme.** Le champ `concurrency` du YAML (défaut 8) règle le nombre de
  copies évaluées simultanément : c'est le principal levier de temps mural. Monter
  (16, 32…) si l'endpoint suit, redescendre en cas de timeouts / erreurs 429.

### 4. Le run complet

**Ne pas lancer les ~3469 copies à la main** : `launchers/launch_eval.sh` s'en charge
— voir [Runs longs](#runs-longs--le-lanceur-launcherslaunch_evalsh) ci-dessous.

### 5. Analyser les résultats

```bash
uv sync --extra notebooks            # une seule fois (JupyterLab + matplotlib)
uv run jupyter lab
```

| Notebook | Ce qu'il fait | Prérequis |
|----------|---------------|-----------|
| `03_analyse_resultats.ipynb` | métriques globales, prévalence par item (IC bootstrap), distributions, corrélation modèle vs expert, seuils critiques, export HTML DEPP | un run de scoring terminé |
| `04_diagnostic.ipynb` | copies triées par désaccord, HTML des N pires, HTML d'une copie précise (scan + transcription + comparaison expert/modèle) | un run de scoring terminé |
| `05_analyse_transcription_htr.ipynb` | CER/WER, distribution, HTML des N pires transcriptions et de N aléatoires | un run HTR terminé |

Changer la variable `RUN_NAME` en tête de notebook suffit pour analyser un autre run.
Le **rapport pour la DEPP** se génère depuis la section 9 du notebook 03 :
`data/processed/rapport_depp_<RUN>.html`, autonome (assets inlinés), prêt à envoyer.

---

## Runs longs : le lanceur `launchers/launch_eval.sh`

Un benchmark complet (3469 copies × ~30 s) prend **~30 h**. Un run pareil ne doit
jamais dépendre de l'onglet du navigateur : déconnexion, mise en veille, fermeture de
la fenêtre, et le processus est tué.

```bash
launchers/launch_eval.sh            # lancer l'échantillon complet
launchers/launch_eval.sh --status   # copies faites, débit mesuré, heure de fin estimée
launchers/launch_eval.sh --stop     # arrêt propre, sans process orphelin
launchers/launch_eval.sh --help     # toutes les options
```

**Ce qu'il fait à votre place**, et qu'un `nohup` tapé à la main oublie :

- **vérifie avant de partir** — `uv`, config valide, `data.limit: null` (sinon il
  refuse : un run partiel se ferait passer pour complet), aucun run concurrent sur le
  même fichier de sortie (process **et** verrou `flock`), secrets, modèle joignable,
  CSV des labels lisible sur S3 ;
- **détache le run** (`setsid` + `nohup`) : aucun signal du terminal ne l'atteint ;
- **un log horodaté par lancement**, jamais écrasé — `logs/<run>_<horodatage>.log`,
  avec `logs/<run>.latest.log` qui pointe vers le dernier ;
- **contrôle le démarrage** en affichant `N copies au total, M à traiter (K déjà
  faites)` — `0 déjà faites` alors qu'un checkpoint existe signalerait un run reparti
  de zéro ;
- **relance les copies en échec** (3 passages par défaut, `--passes`), et s'arrête dès
  que le reliquat ne diminue plus ;
- **exporte vers S3** en fin de run, sous le nom exact du run, en refusant d'écraser un
  fichier S3 **plus gros** que le fichier local.

Pour une autre approche ou un autre modèle :

```bash
launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml
launchers/launch_eval.sh --model-name gemma4-26b-moe
```

> **Répéter ces options sur `--status` et `--stop`.** Le nom d'un run est
> `<name>_<modèle(s)>` : sans elles, `--status` interrogerait un autre run et `--stop`
> en arrêterait un autre.

`--status` donne un débit **mesuré** sur la passe en cours, plus fiable qu'une
estimation a priori :

```text
Run dictee_end2end_qwen3-6-35b-moe
✔ en cours :
      10489  01:33  uv run scripts/run_benchmark.py --config …
  copies      412/3469 (11%)
  débit       464.5 copies/h (depuis 0h 53min)
  fin estimée 2026-09-06 09:12 (dans 6h 34min)
```

### Depuis un assistant de code

Le lanceur est le point d'entrée unique ; les assistants ne font que l'appeler, sans
dupliquer la moindre logique.

| Assistant | Fichier | Comment |
|---|---|---|
| Claude Code | [`.claude/skills/launch/`](.claude/skills/launch/SKILL.md) | `/launch`, ou « lance l'évaluation complète » |
| openCode | [`.opencode/command/launch.md`](.opencode/command/launch.md) | `/launch`, `/launch status`, `/launch stop` |
| Un autre (Cursor, Codex…) | — | lui faire lire `launchers/README.md`, ou lancer le `.sh` soi-même |

Pas encore d'assistant installé ? `launchers/install_assistant.sh` s'en charge et sait
le brancher sur **llm.lab** plutôt que sur une API payante — llm.lab sert les deux
protocoles attendus (compatible OpenAI pour openCode, API Anthropic Messages pour
Claude Code). Le script sonde l'endpoint avant d'écrire et n'écrit jamais de secret sur
disque. Détails : [`launchers/README.md`](launchers/README.md).

### Piloter un run à la main

Le lanceur couvre le cas normal, débogage compris (`--limit N`, `--foreground`). Pour
un run vraiment interactif, `screen` reste disponible sur Onyxia — `tmux`, lui, n'y est
pas installable :

```bash
screen -S dictee
uv run scripts/run_benchmark.py --config configs/scoring/dictee_end2end.yaml
# Ctrl+A puis D pour détacher ; screen -r dictee pour rattacher
```

> **Ne jamais arrêter un run par son PID.** `uv run …` crée DEUX process — le wrapper
> `uv` et le vrai `python3`. Tuer le wrapper laisse l'enfant orphelin continuer
> d'écrire dans le JSONL ; c'est ainsi que deux runs se sont retrouvés sur le même
> fichier. `--stop` fait le nécessaire ; à la main,
> `pkill -f "run_benchmark.py --config <la config>"`.

> **Un seul run par fichier de sortie.** Un second run visant le même fichier s'arrête
> sur le verrou `<sortie>.lock` : deux runs qui appendent le même JSONL dupliquent les
> copies et faussent les métriques.

---

## Configurer une expérience

Un YAML dans `configs/` = une expérience reproductible. Les configs sont rangées par
famille (`scoring/`, `htr/`, `finetune/`) et documentées dans
[`configs/README.md`](configs/README.md) ; le modèle exhaustivement commenté à copier
est [`configs/scoring/dictee_REFERENCE.yaml`](configs/scoring/dictee_REFERENCE.yaml).

| Champ | Rôle |
|-------|------|
| `approach` | `end_to_end` ou `two_stage` |
| `model.name` | modèle servi par llm.lab (ex. `gemma4-26b-moe`) |
| `concurrency` | copies évaluées en parallèle (défaut 8) |
| `data.images_path` / `data.labels_path` | imagettes et CSV des codes experts (local ou `s3://…`) |
| `data.grid_path` | grille de codage (`configs/grille_dictee_2015.json`) |
| `data.limit` | nombre de copies ; `null` = tout le corpus |
| `grid.scheme` | `simplifiee` (1/9/0) ou `complete` (1/3/4/5/9/0) |
| `prompt.method` | `C` end-to-end (image → code) |
| `prompt.read_final_state` | règle des ratures : lire l'état final corrigé |

---

## Structure du dépôt

```
evaluation_dictee/
├── CLAUDE.md                  ← contexte projet pour humains et IA
├── configs/                   ← une expérience = un YAML (voir configs/README.md)
│   ├── grille_dictee_2015.json    ← grille de codage (mot attendu + fautes connues)
│   └── scoring|htr|finetune/      ← configs de référence, par famille
├── src/evaluation_dictee/
│   ├── config.py              ← configs validées (Pydantic) + secrets
│   ├── data/                  ← chargement images (S3, TIFF 1 bit), grille, labels
│   ├── models/                ← interface Scorer, scorers end-to-end et two-stage
│   ├── pipeline/              ← prompts, évaluation + checkpointing, ré-alignement (Needleman-Wunsch)
│   ├── evaluation/            ← métriques, statistiques, calibration, diagnostics, rapports HTML
│   ├── transcription/         ← pipeline HTR indépendant (Scoledit) : loader, CER/WER, diffs
│   └── utils/                 ← logging, suivi Langfuse, export S3
├── scripts/                   ← points d'entrée Python (un run = un process)
│   ├── run_benchmark.py       ← scoring dictée (les deux approches)
│   ├── run_htr_benchmark.py   ← évaluation HTR Scoledit
│   ├── export_predictions.py  ← export d'un run terminé vers S3
│   └── finetune_htr_scoledit.py   ← fine-tuning QLoRA (GPU H100)
├── launchers/                 ← scripts shell (bash seul ; voir launchers/README.md)
│   ├── launch_eval.sh         ← run complet : détache, surveille, relance, exporte
│   └── install_assistant.sh   ← installe Claude Code / openCode, branchés sur llm.lab
├── .claude/skills/launch/     ← skill Claude Code   ┐ n'appellent que launch_eval.sh,
├── .opencode/command/         ← commande openCode   ┘ aucune logique dupliquée
├── notebooks/                 ← 03 analyse, 04 diagnostic, 05 transcription HTR
├── website/                   ← site Quarto (archi, résultats, métriques, fine-tuning)
├── tests/                     ← suite pytest
└── docs/                      ← décisions, grille de codage, schéma du pipeline
```

---

## Développement

```bash
uv run ruff format src tests scripts        # formatage
uv run ruff check src tests scripts         # lint
uv run pytest                               # toute la suite
uv run pytest tests/test_alignment.py -v    # un fichier
uv run pytest -k "chain_of_thought"         # par motif
```

**Ne jamais committer** les données (`data/`), les checkpoints, les logs (`logs/`) ni
les `.env` : tout est dans le `.gitignore`.

---

## Aide-mémoire

```bash
# ─────────── Installation & configuration (une seule fois) ───────────
uv sync                                          # environnement Python (dev inclus)
cp .env.example .env && nano .env                # hors Onyxia seulement (sinon : Vault)
launchers/install_assistant.sh --check           # (option) assistant de code : tester l'endpoint
launchers/install_assistant.sh --endpoint "$LLM_BASE_URL"   # installer + brancher sur llm.lab

# Langfuse, une seule fois. --env-file .env : Langfuse lit ses clés dans os.environ,
# que .env n'alimente pas seul.
uv run --env-file .env add-langfuse-prompt       # pousse les prompts d'évaluation
uv run --env-file .env add-langfuse-models       # coût théorique /1M tokens (GPU amorti,
                                                 # ajustable dans utils/add_langfuse_models.py)

# ─────────── Vérifier que tout marche ───────────
# S3 :
uv run python -c "from evaluation_dictee.data.loaders import load_labels; \
    print(len(load_labels('s3://projet-production-ecrits-depp/resultat_dictee_2015.csv')), 'copies')"
# Modèle :
uv run python -c "from openai import OpenAI; from evaluation_dictee.config import Secrets; \
    s=Secrets(); c=OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key); \
    print(c.chat.completions.create(model='gemma4-26b-moe', \
    messages=[{'role':'user','content':'Dis bonjour'}], max_tokens=10).choices[0].message.content)"
uv run pytest -q

# ─────────── Runs ───────────
launchers/launch_eval.sh                         # run COMPLET (~3469 copies) — la voie normale
launchers/launch_eval.sh --status                # avancement + heure de fin estimée
launchers/launch_eval.sh --stop                  # arrêt propre
launchers/launch_eval.sh --limit 5               # test rapide de bout en bout, sans export S3
# Autre config / autre modèle : répéter les MÊMES options sur les trois commandes.
launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml
launchers/launch_eval.sh --model-name gemma4-26b-moe

uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml   # un passage, au premier plan
uv run scripts/run_htr_benchmark.py --config configs/htr/htr_REFERENCE.yaml      # transcription seule
uv run scripts/finetune_htr_scoledit.py --config configs/finetune/finetune_REFERENCE.yaml  # GPU H100

# ─────────── Suivre un run en cours ───────────
tail -f logs/<run>.latest.log                    # log du dernier lancement
cat data/processed/<run>_failed_copies.txt       # copies en échec du dernier passage

# ─────────── Export des prédictions vers S3 ───────────
# Le pipeline écrit en local (append + fsync, pour la reprise) ; l'export permet de
# rejouer notebooks et site sans relancer le pipeline. launch_eval.sh le fait seul.
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

## ⚠️ Données sensibles

Les copies sont des **écritures d'élèves mineurs**. Elles **ne quittent jamais le
SSP Cloud** et **ne sont jamais commitées**. Le dossier `/data/` est ignoré par Git.
