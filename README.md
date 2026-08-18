# Évaluation automatique de la dictée des élèves

Évaluer automatiquement la dictée manuscrite d'élèves (CEDRE, école primaire) à
l'aide de modèles multimodaux open weight, et **comparer rigoureusement** le codage
automatique à celui d'un correcteur expert. Collaboration **DEPP × SSP Lab (INSEE)**.

---

## Ce que fait le projet, en bref

1. **Charge** les imagettes de dictée (TIFF 1 bit, depuis S3) et les codes de
   l'annotateur expert (gold standard).
2. **Demande à un modèle multimodal** (gemma4 ou gwen3.6 sur llm.lab) de coder chaque mot de
   la dictée — correct / erreur / absent — directement à partir de l'image et du
   texte de référence, sans étape d'OCR séparée.
3. **Compare** les codes du modèle à ceux de l'expert.
4. **Mesure** la fiabilité (kappa, rappel des fautes, sur-correction) et la
   **calibration de la confiance**, pour décider quels items renvoyer à un humain.

La tâche cible est la **grille simplifiée** : `1` correct / `9` erreur / `0` absent
(voir [docs/decisions.md](docs/decisions.md), décision D2).

### ✅ Deux approches d'évaluation comparables

Le projet implémente **deux architectures** derrière la même interface `Scorer` :

- **`end_to_end` (approche 2)** : un VLM lit l'image ET code en une seule passe.
  Approche par défaut. Config : `configs/scoring/dictee_REFERENCE.yaml`.
- **`two_stage` (approche 1)** : étape 1 = transcription HTR (lecture de l'image en
  texte brut, fautes comprises) ; étape 2 = codage du texte transcrit (sans image,
  éventuellement par un modèle texte plus léger via `model_stage2`). Isole lecture
  et jugement. Config : `configs/scoring/dictee_REFERENCE.yaml`. L'approche se choisit
  via le champ `approach` du YAML ou par les paramètres `--model_name` et 
  `--model_stage2-name`.

### 📄 Évaluation dédiée de la transcription (HTR) sur Scoledit

Indépendamment du codage, on peut mesurer la **fidélité de lecture** d'un modèle sur
l'écriture manuscrite d'enfants via le corpus **Scoledit** (transcriptions de
référence humaines, fautes préservées). Cela permet de comparer les modèles sur la 
seule lecture et de distinguer les erreurs de lecture de celles de jugement. 
Config : `configs/htr/htr_REFERENCE.yaml`, script : `scripts/run_htr_benchmark.py`, 
analyse : `notebooks/05_analyse_transcription_htr.ipynb`.

### 🔨 Fine-tuning HTR (phase ultérieure)

Pour améliorer la fidélité de lecture sur l'écriture d'enfants, on peut
**spécialiser** un VLM par fine-tuning **QLoRA** (LoRA en 4 bits) sur le corpus
Scoledit multi-niveaux (CP→CM2), transcriptions humaines fautes préservées. La
sortie est un adaptateur léger (~50-200 Mo) qui se charge par-dessus le modèle de
base. Nécessite un **GPU H100**. Suivi via **MLflow** (et non Langfuse). Config :
`configs/finetune/finetune_REFERENCE.yaml`, script : `scripts/finetune_htr_scoledit.py`.

### 📖 Documentation pédagogique (site Quarto)

Un **site Quarto** (dossier [`website/`](website/)) présente le projet pour un
public statisticien novice en IA : architecture, résultats des deux approches,
explication détaillée des métriques d'évaluation, et fine-tuning.

```bash
quarto preview website        # aperçu local avec rechargement à chaud
quarto render website         # génère le site statique dans website/_site/
```
> Contexte complet et décisions : voir **[CLAUDE.md](CLAUDE.md)** et les
> **[décisions](docs/decisions.md)** dans [`docs/`](docs/).

---

## 🔧 Démarrage rapide (SSP Cloud / VSCode)

### 1. Installer

```bash
git clone <url-du-depot> evaluation_dictee
cd evaluation_dictee
uv sync
```

> Le paquet s'appelle `evaluation_dictee`. `uv sync` installe toutes les dépendances
> (y compris le groupe `dev`) et configure le mode éditable automatiquement, ce qui
> rend les imports `from evaluation_dictee...` disponibles.

### 2. Configurer les accès (Vault Onyxia + repli `.env`)

Toute la configuration sensible passe par des **variables d'environnement** ; le
code les lit via `Secrets` (Pydantic, `src/evaluation_dictee/config.py`). **Aucun
secret n'est jamais écrit dans le code ni dans les YAML.** Deux façons de fournir
ces variables selon le contexte :

#### Sur le SSP Cloud (recommandé) : le Vault Onyxia

Onyxia intègre un **Vault** (HashiCorp) personnel : un coffre-fort où stocker ses
secrets une fois, puis les **injecter automatiquement comme variables d'environnement**
dans chaque service qu'on lance. On ne recopie ainsi jamais de token en clair.

1. **Stocker les secrets** dans le Vault : `Mon compte` → `Vault` (SecretVault).
   Créer un secret pour le projet (p. ex. `evaluation_dictee`) avec les clés :

   | Clé | Valeur |
   |-----|--------|
   | `LLM_BASE_URL` | `https://llm.lab.sspcloud.fr/api/v1` |
   | `LLM_API_KEY` | ton token llm.lab |
   | `LANGFUSE_BASE_URL` | `https://langfuse.lab.sspcloud.fr` |
   | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | depuis l'UI Langfuse |
   | `MLFLOW_TRACKING_URI` | `https://mlflow.lab.sspcloud.fr` (fine-tuning) |

2. **Injecter dans le service** : au lancement d'un service (VSCode, Jupyter…),
   section `Vault`, référencer ce secret pour qu'Onyxia expose ces clés comme
   variables d'environnement dans le conteneur. Elles sont alors disponibles sans
   fichier `.env`.

3. **Alternative en ligne de commande** — les services Onyxia ont déjà `VAULT_ADDR`
   et `VAULT_TOKEN` positionnés, donc le CLI `vault` fonctionne directement :

   ```bash
   vault kv get <chemin>/evaluation_dictee          # lire les secrets
   # (les identifiants S3 AWS_* sont déjà injectés par Onyxia, rien à faire)
   ```

> Les identifiants **S3** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
> `AWS_SESSION_TOKEN`, `AWS_S3_ENDPOINT`) sont **automatiquement injectés** par
> Onyxia dans chaque service : rien à configurer pour accéder au stockage.

#### En local (hors Onyxia) : le fichier `.env`

Hors SSP Cloud, ou pour un test rapide, `Secrets` retombe sur un fichier `.env`
local (jamais commité, voir `.gitignore`) :

```bash
cp .env.example .env
# éditer .env : LLM_API_KEY, LANGFUSE_*, et si besoin AWS_* / MLFLOW_*
```

`.env.example` documente toutes les variables attendues. Le code ne fait **aucune
différence** entre une variable injectée par le Vault et une variable lue depuis
`.env` : dans les deux cas, elle arrive par l'environnement.

Vérifier l'accès aux données et au modèle :

```bash
# S3 accessible ?
uv run uv run python -c "from evaluation_dictee.data.loaders import load_labels; \
print(len(load_labels('s3://projet-production-ecrits-depp/resultat_dictee_2015.csv')), 'copies')"

# modèle accessible ?
uv run uv run python -c "from openai import OpenAI; from evaluation_dictee.config import Secrets; \
s=Secrets(); c=OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key); \
print(c.chat.completions.create(model='gemma4-26b-moe', \
messages=[{'role':'user','content':'Dis bonjour'}], max_tokens=10).choices[0].message.content)"
```

### 3. Lancer un benchmark

#### **Pour lancer une évalutaion des copies** :
```bash
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml [--model-name gwen3-6-35b-moe] [--model-stage2-name qwen3-6-35b-moe]
```

Cela produit `data/processed/dictee_REFERENCE_<modele>_predictions.jsonl` — le nom
du fichier est **toujours** suffixé par le modèle de l'étape 1 (plus celui de
l'étape 2 s'il diffère), que le modèle vienne du YAML ou de `--model-name`, pour
que deux modèles n'écrasent jamais le même checkpoint. Chaque ligne porte en
outre les champs `model` et `model_stage2`.

Le fichier contient une ligne par item × copie, et tout est journalisé dans
Langfuse : une **session**
par run, une **trace** par copie (entrée/sortie + score d'accord), les appels LLM
en générations imbriquées, et les métriques agrégées du run en Scores et metadata.

**Pour un run complet (long)**, utiliser une session détachable — voir la section
« Runs longs (screen / nohup) » plus bas dans ce README.

**Reprise après crash.** Le benchmark écrit sur disque après CHAQUE copie et
reprend automatiquement où il s'était arrêté : si un run est interrompu
(déconnexion, erreur API, kernel tué), il suffit de **relancer la même commande**
et il saute les copies déjà traitées. Voir « Runs longs » pour les détails.

**Vitesse : évaluation en parallèle.** Les copies sont évaluées concurremment
(le endpoint vLLM batche les requêtes), réglé par `concurrency` dans le YAML
(défaut **8**). C'est le principal levier de temps mural : passer de 1 à N copies
en parallèle divise d'autant la durée tant que le serveur suit. Monter (16, 32…)
si l'endpoint tient, redescendre en cas de timeouts / erreurs 429. Seul le scoring
est parallélisé : l'écriture du JSONL et la reprise restent inchangées.

#### **Pour l'évaluation de la transcription (HTR)** sur Scoledit :

```bash
uv run scripts/run_htr_benchmark.py --config configs/htr/htr_REFERENCE.yaml
```

Cela produit `data/processed/htr_REFERENCE_htr_predictions.jsonl` et affiche
le CER/WER moyens. Analyse dans `notebooks/05_analyse_transcription_htr.ipynb`.
La transcription est elle aussi **parallélisée** (champ `concurrency` du YAML HTR,
défaut 8) ; l'ordre des échantillons en sortie est préservé.

#### **Exporter les prédictions vers S3.** 

Le pipeline écrit en local (append + fsync par copie, pour la reprise sur crash). 
Une fois un run terminé, on pousse le JSONL vers le répertoire `predictions/` du bucket 
S3, afin de **relancer les notebooks et le site Quarto sans réexécuter le pipeline**. 
Rien n'est commité dans Git.

```bash
# scoring — end_to_end OU two_stage (même format, c'est le `name` qui distingue) :
uv run scripts/export_predictions.py --run-name dictee_two_stage_gemma4-26b-moe # type d'approche + nom du modèle

# transcription HTR seule :
uv run scripts/export_predictions.py --config configs/htr/htr_REFERENCE.yaml --htr

# équivalent via la CLI installée :
eval-ecrit export configs/scoring/dictee_REFERENCE.yaml
```

Destination : `$S3_PREDICTIONS_PREFIX/<name>_<modele>_predictions.jsonl` (défaut
`s3://projet-production-ecrits-depp/predictions`, surchargeable par `--dest-prefix`).

#### **Pour le fine-tuning** d'un modèle de transcription (nécessite un GPU H100) :
```bash
uv run scripts/finetune_htr_scoledit.py --config configs/finetune/finetune_REFERENCE.yaml
```
Voir la documentation détaillée dans le script pour les prérequis d'installation
(`unsloth`, `trl`, `bitsandbytes`).

### 4. Analyser les résultats

Trois notebooks, à ouvrir dans **`notebooks/`** et à exécuter cellule par cellule.
Installer d'abord les dépendances notebooks (JupyterLab, matplotlib) puis lancer
JupyterLab via uv :

```bash
uv sync --extra notebooks            # une seule fois
uv run jupyter lab                   # ouvre l'interface
```


| Notebook | Ce qu'il fait | Prérequis |
|----------|---------------|-----------|
| `03_analyse_resultats.ipynb` | métriques globales, prévalence par item avec IC bootstrap, distributions, corrélation modèle vs expert, seuils critiques. Export HTML sélectif pour la DEPP. | un run de benchmark terminé |
| `04_diagnostic.ipynb` | table des copies triées par désaccord, HTML des N pires copies, HTML d'une copie précise par ID (scan + transcription + comparaison expert/modèle) | un run de benchmark terminé |
| `05_analyse_transcription_htr.ipynb` | CER/WER, distribution du CER, HTML des N pires transcriptions et N aléatoires | un run HTR terminé |

Dans chaque notebook, il suffit de changer la variable `RUN_NAME` en tête pour
analyser un autre run — aucun besoin de relancer le benchmark.

**Générer un rapport HTML pour la DEPP** (à partir du notebook 03) : exécuter la
section « 9. Export HTML pour l'équipe DEPP », choisir les sections à inclure,
et le fichier `data/processed/rapport_depp_<RUN>.html` est autonome (assets
inlinés) prêt à envoyer par mail.

---

## Runs longs (screen / nohup)

Un benchmark complet sur 3469 copies × ~30 s prend ~30 h. **Ne jamais lancer un
tel run dans le terminal du navigateur sans protection** : au moindre plantage
réseau, mise en veille, fermeture d'onglet, le processus est tué. Le
checkpointing sauvera les prédictions déjà écrites, mais pas la copie en cours.

> **Note Onyxia** : `tmux` n'est pas disponible dans les services vscode-python
> du SSP Cloud (`sudo apt-get install tmux` échoue avec « No installation
> candidate »). Utiliser `screen` (Option A) ou `nohup` (Option B).

### Avant tout : vérifier qu'aucun run ne tourne, et créer le dossier logs

Deux runs sur le même fichier de sortie dupliquent les copies et faussent les
métriques. Le verrou `<sortie>.lock` fait désormais échouer le second dès le
démarrage, mais autant le constater avant de lancer — quelle que soit l'option
choisie ci-dessous :

```bash
ps -ef | grep run_benchmark | grep -v grep    # doit ne rien renvoyer
screen -ls                                    # ni session détachée oubliée

# À faire une seule fois (nohup échoue si le dossier n'existe pas) :
mkdir -p logs
```

### Option A — screen (recommandé sur Onyxia, généralement disponible)

```bash
# Vérifier la disponibilité :
which screen && echo "OK" || echo "absent"

# Créer une session détachable et lancer le run :
screen -S dictee
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml

# Détacher :         Ctrl+A  puis  D    (le job continue en arrière-plan)
# Rattacher :        screen -r dictee
# Lister sessions :  screen -ls
# Tuer une session : screen -X -S dictee quit
```

### Option B — nohup (toujours disponible, sans interface interactive)

Lancer — **un fichier de log distinct par run**, sinon les sorties de deux runs
s'entrelacent dans le même fichier et deviennent illisibles après coup :

```bash
mkdir -p logs

nohup uv run scripts/run_benchmark.py --config configs/scoring/dictee_end2end.yaml \
      > logs/dictee_end2end.log 2>&1 &

nohup uv run scripts/run_benchmark.py --config configs/scoring/dictee_two_stage.yaml \
      > logs/dictee_two_stage.log 2>&1 &
```

Suivre et contrôler :

```bash
# Suivre le log en direct :
tail -f logs/dictee_end2end.log

# Vérifier que le run tourne (affiche le wrapper `uv run` ET le python) :
ps -ef | grep run_benchmark | grep -v grep

# Arrêter proprement (le checkpointing conserve tout sauf la copie en cours) :
pkill -f "run_benchmark.py --config configs/scoring/dictee_end2end.yaml"
```

> **Ne pas piloter le run par un fichier `.pid`.** `nohup uv run … &` crée DEUX
> process : le wrapper `uv run` et le vrai `python3 scripts/run_benchmark.py`.
> `echo $!` ne capture que le wrapper ; un `kill` sur ce seul PID laisse
> l'enfant Python vivant en orphelin, qui continue d'écrire. C'est ainsi que
> plusieurs runs concurrents se sont retrouvés sur le même JSONL. `pkill -f`
> sur le motif de la config cible les deux d'un coup.

**Vérifier que la reprise a bien pris.** Les premières lignes du log doivent
annoncer le nombre de copies déjà faites :

```bash
head -20 logs/dictee_end2end.log | grep -i "reprise\|copies au total"
```

Si un run annonce `0 déjà faites` alors qu'un checkpoint existe, l'arrêter : le
nom du fichier de sortie ne correspond pas au checkpoint (modèle différent, ou
`name` modifié dans le YAML), et le run repartirait de zéro.

### Surveillance de l'avancement

Pendant un run long, dans une **autre** fenêtre ou onglet, ces commandes donnent
un signal de vie plus fiable que la barre de progression :

```bash
# Retrouver le fichier du run en cours (le nom porte le modèle) :
ls -lt data/processed/*_predictions.jsonl | head

# Compter les copies déjà traitées dans le JSONL (une copie = ~83 lignes) :
wc -l data/processed/dictee_REFERENCE_qwen3-6-35b-moe_predictions.jsonl

# Suivre le compteur en direct (mise à jour toutes les 5 s) :
watch -n 5 "wc -l data/processed/dictee_REFERENCE_qwen3-6-35b-moe_predictions.jsonl"

# Lister les copies en échec (à retenter au prochain lancement) :
cat data/processed/dictee_REFERENCE_qwen3-6-35b-moe_failed_copies.txt
```

> **Un seul run par fichier de sortie.** Un second run visant le même fichier
> s'arrête aussitôt sur le verrou `<sortie>.lock`. Avant de relancer, vérifier
> qu'aucun run ne tourne déjà : `ps -ef | grep run_benchmark`.

### Reprise après crash — mode d'emploi

Le benchmark écrit sur disque après CHAQUE copie évaluée (avec `flush + fsync`).
Conséquences pratiques :

- **Crash ou déconnexion** : relance exactement la même commande. Les copies
  déjà présentes dans `<run>_<modele>_predictions.jsonl` sont automatiquement
  sautées, et le run reprend à la copie suivante. Relancer avec une commande
  *différente* (par ex. en ajoutant `--model-name` alors que le premier
  lancement lisait le modèle du YAML) viserait, avant correction, un autre
  fichier et repartirait de zéro : le nommage est désormais identique dans les
  deux cas.
- **Erreurs API sur des copies isolées** : elles sont loggées dans
  `<run>_<modele>_failed_copies.txt`, la copie fautive est sautée mais le run continue.
  Au prochain lancement, ces copies seront retentées.
- **Repartir de zéro** : supprimer `<run>_<modele>_predictions.jsonl` (ou changer
  `config.name` dans le YAML).

---

## Configurer une expérience

Un fichier YAML dans `configs/` = une expérience reproductible. Les configs sont
rangées par famille (`scoring/`, `htr/`, `finetune/`) et documentées dans
[`configs/README.md`](configs/README.md). Modèle exhaustivement commenté à copier :
[`configs/scoring/dictee_REFERENCE.yaml`](configs/scoring/dictee_REFERENCE.yaml).

| Champ | Rôle |
|-------|------|
| `model.name` | nom du modèle servi par llm.lab (ex. `gemma4-26b-moe`) |
| `data.images_path` | dossier des imagettes (local ou `s3://...`) |
| `data.labels_path` | CSV des codes de l'annotateur (local ou `s3://...`) |
| `data.grid_path` | grille de codage JSON (`configs/grille_dictee_2015.json`) |
| `data.limit` | nombre de copies (mettre `null` pour tout le corpus) |
| `grid.scheme` | `simplifiee` (1/9/0) ou `complete` (1/3/4/5/9/0) |
| `prompt.method` | `C` end-to-end (image → code) |
| `prompt.read_final_state` | règle des ratures : lire l'état final corrigé |

---

## Structure du dépôt

```
evaluation_dictee/
├── CLAUDE.md                  ← contexte du projet pour humains et IA
├── README.md                  ← ce fichier
├── pyproject.toml             ← dépendances + config ruff/mypy/pytest
├── configs/                      ← configs de référence, une par famille (voir configs/README.md)
│   ├── README.md                 ← guide de paramétrage (toutes les familles)
│   ├── grille_dictee_2015.json   ← grille de codage (mot attendu + fautes connues)
│   ├── scoring/dictee_REFERENCE.yaml      ← codage dictée (run_benchmark.py)
│   ├── htr/htr_REFERENCE.yaml             ← évaluation HTR Scoledit (run_htr_benchmark.py)
│   └── finetune/finetune_REFERENCE.yaml   ← fine-tuning HTR QLoRA (finetune_htr_scoledit.py)
├── src/evaluation_dictee/
│   ├── config.py              ← configs validées (Pydantic) + secrets (.env)
│   ├── data/                  ← chargement images (S3, TIFF 1 bit) + grille + labels
│   ├── models/
│   │   ├── base.py            ← interface Scorer + dataclasses de prédiction
│   │   ├── vlm.py             ← scorer end-to-end (approche 1 étape)
│   │   ├── two_stage.py       ← scorer 2 étapes (HTR puis codage texte)
│   │   └── factory.py         ← construit le bon scorer selon la config
│   ├── pipeline/
│   │   ├── prompts.py         ← construction des prompts (dictée + transcription)
│   │   ├── benchmark.py       ← boucle d'évaluation + checkpointing incrémental
│   │   └── alignment.py       ← ré-alignement anti-décalage (Needleman-Wunsch)
│   ├── evaluation/            ← metrics, statistics (bootstrap/Wilson),
│   │   │                        calibration (ECE), report (par item/copie)
│   │   ├── report.py, statistics.py, calibration.py, metrics.py
│   │   ├── diagnostics.py     ← analyse fine des désaccords
│   │   ├── visual_diff.py     ← HTML de comparaison expert / modèle
│   │   └── html_report.py     ← export HTML sélectif (rapport DEPP)
│   ├── transcription/         ← pipeline HTR indépendant (Scoledit)
│   │   ├── scoledit.py        ← loader TEI → texte brut (fautes préservées)
│   │   ├── htr_metrics.py     ← CER, WER (bruts et normalisés)
│   │   ├── htr_benchmark.py   ← run HTR + agrégation métriques
│   │   └── visual_diff.py     ← HTML des pires / N aléatoires
│   └── utils/                 ← logging, suivi Langfuse (traces, prompts, scores)
├── scripts/
│   ├── run_benchmark.py       ← point d'entrée CLI (approches 1 et 2 étapes)
│   ├── run_htr_benchmark.py   ← point d'entrée CLI pour l'évaluation HTR
│   └── finetune_htr_scoledit.py  ← fine-tuning HTR (nécessite un GPU H100)
├── notebooks/
│   ├── 03_analyse_resultats.ipynb   ← analyse statistique + export DEPP
│   ├── 04_diagnostic.ipynb          ← inspection copie par copie
│   └── 05_analyse_transcription_htr.ipynb   ← analyse HTR
├── website/                   ← site Quarto (archi, résultats, métriques, fine-tuning)
├── tests/                     ← 155 tests unitaires (pytest)
└── docs/                      ← décisions, grille de codage, schéma du pipeline
```

---

## Bonnes pratiques développement

Avant chaque commit :

```bash
uv run ruff format src tests scripts        # formatage automatique
uv run ruff check src tests scripts         # lint (attrape les erreurs courantes)
uv run pytest                               # lance toute la suite de tests
uv run pytest tests/test_alignment.py -v    # tester UN fichier précis
uv run pytest -k "chain_of_thought"         # tests dont le nom matche un motif
```

**Ne jamais committer** les données (`data/`), les checkpoints (`checkpoints/`),
les logs (`logs/`) ni les fichiers `.env` : ils sont dans le `.gitignore`.

## Toutes les commandes en un coup d'œil

Section de référence rapide. Chaque commande est détaillée plus haut dans le
README, avec ses prérequis et son contexte d'usage.

```bash
# ─────────── Installation & configuration (une seule fois) ───────────
uv sync                                          # environnement Python (groupe dev inclus d'office)
# Secrets : sur Onyxia, via le Vault (Mon compte > Vault, injecté comme variables
# d'env). En local, repli sur un .env :
cp .env.example .env && nano .env                # renseigner LLM_API_KEY, LANGFUSE_*, S3

# ─────────── Configuration Langfuse (une seule fois) ───────────
# --env-file .env : Langfuse lit ses clés dans os.environ, que .env n'alimente pas seul.
uv run --env-file .env add-langfuse-prompt       # pousse les prompts d'évaluation
uv run --env-file .env add-langfuse-models       # enregistre le coût théorique/1M tokens
                                                 # (prix GPU amorti, ajustables dans
                                                 #  utils/add_langfuse_models.py)

# ─────────── Vérifier que tout marche ───────────
uv run python -c "from evaluation_dictee.data.loaders import load_labels; \
    print(len(load_labels('s3://projet-production-ecrits-depp/resultat_dictee_2015.csv')), 'copies')"
uv run pytest -q                                 # lancer les tests

# ─────────── Benchmark scoring dictée ───────────
# Config de référence prête à l'emploi (approche end_to_end). Pour comparer une
# autre approche/variante (two_stage, chain-of-thought, autre modèle), copier la
# référence et ajuster (voir configs/README.md).
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml

# ─────────── Évaluation de la transcription HTR (Scoledit) ───────────
uv run scripts/run_htr_benchmark.py --config configs/htr/htr_REFERENCE.yaml

# ─────────── Fine-tuning HTR (GPU H100 requis) ───────────
uv run scripts/finetune_htr_scoledit.py --config configs/finetune/finetune_REFERENCE.yaml

# ─────────── Runs longs (session détachable) ───────────
ps -ef | grep run_benchmark | grep -v grep       # AVANT tout : rien ne doit tourner
mkdir -p logs                                    # toujours créer d'abord

# Option A : screen (recommandé sur Onyxia, généralement disponible)
which screen && screen -S dictee                 # puis Ctrl+A D pour détacher
                                                 # screen -r dictee pour rattacher

# Option B : nohup (toujours dispo, sans interface interactive)
# Un log DISTINCT par run, sinon les sorties s'entrelacent.
nohup uv run scripts/run_benchmark.py --config configs/scoring/dictee_end2end.yaml \
      > logs/dictee_end2end.log 2>&1 &
nohup uv run scripts/run_benchmark.py --config configs/scoring/dictee_two_stage.yaml \
      > logs/dictee_two_stage.log 2>&1 &

# ─────────── Surveillance d'un run en cours ───────────
tail -f logs/dictee_end2end.log
head -20 logs/dictee_end2end.log | grep -i "reprise"   # la reprise a-t-elle pris ?
ls -lt data/processed/*_predictions.jsonl | head       # retrouver le fichier du run
watch -n 5 "wc -l data/processed/dictee_end2end_qwen3-6-35b-moe_predictions.jsonl"

# ─────────── Arrêter un run ───────────
# PAS `kill $(cat *.pid)` : `nohup uv run …` crée un wrapper + un python, et `$!`
# ne capture que le wrapper — l'enfant survivrait et continuerait d'écrire.
pkill -f "run_benchmark.py --config configs/scoring/dictee_end2end.yaml"

# ─────────── Analyse des résultats ───────────
uv sync --extra notebooks                          # une seule fois (JupyterLab + matplotlib)
uv run jupyter lab notebooks/03_analyse_resultats.ipynb   # analyse statistique + export DEPP
uv run jupyter lab notebooks/04_diagnostic.ipynb          # inspection copie par copie
uv run jupyter lab notebooks/05_analyse_transcription_htr.ipynb   # analyse HTR

# ─────────── Site de documentation (Quarto) ───────────
quarto preview website                             # aperçu local (rechargement à chaud)
quarto render website                              # génère website/_site/

# ─────────── Qualité de code (avant tout commit) ───────────
uv run ruff format src tests scripts
uv run ruff check src tests scripts
uv run pytest
```

---

## ⚠️ Données sensibles

Les copies sont des **écritures d'élèves mineurs**. Elles **ne quittent jamais le
SSP Cloud** et **ne sont jamais commitées**. Le dossier `/data/` est ignoré par Git
(voir [`.gitignore`](.gitignore)) ; seul le `.env.example` (sans secret) est versionné.