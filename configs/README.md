# `configs/` — configurations d'expériences

Toute expérience du projet se pilote par un fichier **YAML** (aucun paramètre
codé en dur ailleurs). Un fichier = un run reproductible, validé au chargement
par un schéma [Pydantic](../src/evaluation_dictee/config.py).

Ce dossier contient **trois familles** de configs, chacune consommée par un
script différent, plus la grille de codage (une donnée, pas une config).

Chaque famille fournit **une config de référence** (`*_REFERENCE.yaml`) : un
fichier valide, prêt à lancer, et exhaustivement commenté. C'est le point de
départ — on le copie pour créer sa propre expérience (voir plus bas).

```
configs/
├── README.md                          ← ce guide
├── grille_dictee_2015.json            ← grille de codage (mot attendu + fautes/item)
├── scoring/dictee_REFERENCE.yaml      ← codage de la dictée   → run_benchmark.py
├── htr/htr_REFERENCE.yaml             ← transcription seule    → run_htr_benchmark.py
└── finetune/finetune_REFERENCE.yaml   ← fine-tuning LoRA (GPU) → finetune_htr_scoledit.py
```

---

## Démarrage rapide

```bash
# 1. Scoring dictée (codage image → codes). Commence par un petit run.
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml

# 2. Éval de transcription (CER/WER) sur Scoledit.
uv run scripts/run_htr_benchmark.py --config configs/htr/htr_REFERENCE.yaml

# 3. Fine-tuning HTR (à lancer sur un service GPU).
uv run scripts/finetune_htr_scoledit.py --config configs/finetune/finetune_REFERENCE.yaml
```

> **Astuce paramétrage** : dans presque tous les fichiers, **`limit`** borne le
> nombre de copies (mettre `5`–`20` pour un test, `null` pour tout le corpus) et
> **`model.name`** est le seul champ à changer pour tester un autre modèle.

---

## Famille 1 — Scoring dictée (`scoring/`)

C'est la famille principale. Schéma : `ExperimentConfig`. Le fichier
**`dictee_REFERENCE.yaml`** documente **chaque paramètre** avec ses valeurs
possibles et par défaut : commence par le lire, puis copie-le pour ta config.

### Les champs, en un coup d'œil

| Chemin | Rôle | Valeurs / défaut |
|---|---|---|
| `name` | nom unique du run (session Langfuse + préfixe des sorties) | libre, unique |
| `seed` | reproductibilité | `42` |
| `approach` | 1 passe ou 2 étapes | `end_to_end` \| `two_stage` |
| `model.name` | modèle servi par vLLM (nom exact llm.lab) | ex. `gemma4-26b-moe` |
| `model.kind` | modalité | `vlm` \| `llm` \| `htr` |
| `model.temperature` | échantillonnage (0 = déterministe) | `0.0` |
| `model.max_tokens` | plafond de génération (marge anti-troncature) | `2048`, **mettre `8192`** |
| `model.max_retries` | essais si réponse non parsable | `2` |
| `model.disable_thinking` | coupe le `<think>` des modèles Qwen3/R1 | `true` |
| `model.structured_output` | force un JSON conforme au schéma (anti « non transcrite ») | `true` |
| `data.corpus` | corpus | `dictee` \| `production_ecrite` |
| `data.images_path` | imagettes (local ou `s3://…`) | requis |
| `data.labels_path` | codes experts = vérité terrain (local ou `s3://…`) | requis |
| `data.grid_path` | grille JSON | `configs/grille_dictee_2015.json` |
| `data.limit` | nb de copies (`null` = tout) | `null` |
| `grid.scheme` | modalités des codes | `simplifiee` (1/erreur/0) \| `complete` |
| `prompt.method` | méthode A/B/C/D | `C` |
| `prompt.n_few_shot` | exemples dans le prompt | `0` |
| `prompt.enforce_faithful` | consigne anti-sur-correction | `true` |
| `prompt.read_final_state` | ratures : lire l'état final | `true` |
| `prompt.chain_of_thought` | verbalise la comparaison avant le code | `false` |

### Décliner en variantes (à partir de la référence)

`dictee_REFERENCE.yaml` couvre l'approche end-to-end zéro-shot. Les autres
variantes se créent en copiant la référence et en changeant peu de champs :

| Variante | Champs à changer |
|---|---|
| Autre modèle | `model.name` (+ `disable_thinking: true` si modèle *thinking* : Qwen3, R1…) |
| Chain-of-thought | `prompt.chain_of_thought: true` |
| Deux étapes (HTR + codage) | `approach: two_stage` + ajouter le bloc `model_stage2` |
| Run de fumée (traçage) | `data.limit: 5` et un `name` dédié |

> **Comparaison équitable end-to-end vs two-stage** : garder le *même* modèle des
> deux côtés isole l'effet de l'architecture (1 vs 2 étapes) de celui du modèle.

---

## Famille 2 — Transcription HTR (`htr/`)

Mesure la fidélité de lecture (CER/WER) sur le corpus Scoledit, indépendamment
du codage. Modèle commenté à copier : **`htr_REFERENCE.yaml`**. Consommée par
`run_htr_benchmark.py`, qui lit le YAML comme un simple dictionnaire — **les clés
attendues sont donc `scans_dir` / `annotations_dir`** (et non `*_path`) :

```yaml
name: htr_REFERENCE
model:
  name: gemma4-26b-moe        # ← seul champ à changer pour tester un autre VLM
  kind: vlm
  temperature: 0.0
  max_tokens: 2048
  max_retries: 2
data:
  scans_dir:       s3://…/scoledit/scans/CE1/
  annotations_dir: s3://…/scoledit/annotation/CE1/
  limit: 10                   # null = tout le corpus
read_final_state: true        # ratures : lire l'état final
```

---

## Famille 3 — Fine-tuning LoRA (`finetune/`)

Fine-tuning HTR (QLoRA) sur Scoledit, **à lancer depuis un service GPU** du SSP
Cloud (H100 recommandé). Schéma propre (dataclass dans le script), distinct des
deux autres : blocs `base_model`, `data` (`scans_root`/`annotations_root`,
niveaux, ratios de split), `lora` (rang, couches ciblées) et `training`
(epochs, batch, LR, scheduler…). Suivi via **MLflow** (et non Langfuse). Modèle
commenté à copier : **`finetune_REFERENCE.yaml`** (tous les champs + leurs
défauts).

---

## Créer sa propre config (recette)

1. **Copier** le modèle adapté à ta famille :
   ```bash
   cp configs/scoring/dictee_REFERENCE.yaml configs/scoring/dictee_MONMODELE_zeroshot.yaml
   ```
2. **Renommer** le champ `name` (unique — sinon tu écrases le checkpoint d'un
   autre run) et régler `model.name` sur le nom exact affiché sur
   [llm.lab.sspcloud.fr](https://llm.lab.sspcloud.fr).
3. **Vérifier** `data.images_path` / `labels_path`, et mettre `data.limit` à une
   petite valeur pour un premier essai.
4. **Lancer** et itérer :
   ```bash
   uv run scripts/run_benchmark.py --config configs/scoring/dictee_MONMODELE_zeroshot.yaml
   ```
5. Passer `limit: null` une fois la chaîne validée (voir le README racine pour
   les runs longs : `screen`/`nohup`, checkpointing, reprise après crash).

---

## Bon à savoir

- **Secrets** : aucun identifiant dans les YAML. Les accès S3 et le modèle
  passent par des variables d'environnement (`.env`, jamais commité — voir
  `.env.example`).
- **Données sensibles** : écritures d'élèves mineurs. Les microdonnées ne
  quittent jamais le SSP Cloud ; aucune image dans Git (`data/` est ignoré).
- **`max_tokens`** : une copie fait 83 items. En JSON (et *a fortiori* en CoT),
  prévoir `8192` pour éviter une troncature qui casse le parsing.
- **Modèles « thinking »** (Qwen3, DeepSeek-R1, QwQ) : toujours
  `disable_thinking: true`, sinon le bloc `<think>` casse le JSON.
