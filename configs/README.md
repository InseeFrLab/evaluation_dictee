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

> **Comparer deux modèles sans dupliquer les YAML** : `run_benchmark.py` (et la
> commande `eval-ecrit benchmark`) acceptent `--model-name` (étape 1 / unique
> étape en end_to_end) et `--model-stage2-name` (étape 2, two_stage
> uniquement) pour surcharger `model.name` / `model_stage2.name` sans éditer le
> fichier. Les noms passés doivent correspondre EXACTEMENT à ceux servis sur
> [llm.lab.sspcloud.fr](https://llm.lab.sspcloud.fr).
>
> Le fichier de sortie est **toujours** suffixé par le(s) modèle(s) utilisé(s) —
> `data/processed/<name>_<modele>_predictions.jsonl` — que le modèle vienne du
> YAML ou d'une surcharge CLI, pour ne jamais écraser le checkpoint d'un autre
> modèle. Le suffixe n'est ajouté qu'une fois (cf. `run_output_name`), et le
> modèle de l'étape 2 n'y figure que s'il diffère de celui de l'étape 1. Le nom
> du modèle est en outre inscrit dans **chaque ligne** du JSONL (champs `model`
> et `model_stage2`), donc l'information survit à une fusion ou à un renommage.
>
> Exemple : comparer `gemma4-26b-moe` à `qwen3.6-35b-moe` sur les deux
> approches (4 fichiers de prédictions) :
> ```bash
> # qwen3-6-35b-moe (modèle inscrit dans les YAML, rien à surcharger)
> uv run scripts/run_benchmark.py --config configs/scoring/dictee_end2end.yaml
> uv run scripts/run_benchmark.py --config configs/scoring/dictee_two_stage.yaml
>
> # gemma4-26b-moe (surcharge du modèle)
> uv run scripts/run_benchmark.py --config configs/scoring/dictee_end2end.yaml \
>       --model-name gemma4-26b-moe
> uv run scripts/run_benchmark.py --config configs/scoring/dictee_two_stage.yaml \
>       --model-name gemma4-26b-moe --model-stage2-name gemma4-26b-moe
> ```
> → `dictee_end2end_qwen3-6-35b-moe_predictions.jsonl`,
> `dictee_two_stage_qwen3-6-35b-moe_predictions.jsonl`,
> `dictee_end2end_gemma4-26b-moe_predictions.jsonl`,
> `dictee_two_stage_gemma4-26b-moe_predictions.jsonl`.
>
> **Un seul run à la fois par fichier de sortie** : un second run visant le même
> fichier échoue immédiatement sur le verrou `<sortie>.lock`. Deux runs qui
> appendent le même JSONL dupliquent les copies et faussent les métriques.

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
| `model.disable_thinking` | `false` = le modèle raisonne avant de répondre | `true` |
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
| Autre modèle | `model.name` |
| Raisonnement natif (thinking) | `disable_thinking: false` + `max_tokens: 16384`, **modèles Qwen3 seulement** |
| Chain-of-thought (champ « comparaison ») | `prompt.chain_of_thought: true` |
| Deux étapes (HTR + codage) | `approach: two_stage` + ajouter le bloc `model_stage2` |
| Run de fumée (traçage) | `data.limit: 5` et un `name` dédié |

> **Comparaison équitable end-to-end vs two-stage** : garder le *même* modèle des
> deux côtés isole l'effet de l'architecture (1 vs 2 étapes) de celui du modèle.

### Options de raisonnement : testées, NON retenues

Cinq variantes de prompt ont été mesurées sur la dictée 2015, à modèle et corpus
constants, contre le run de référence du **même** modèle. Elles sont toutes à `false`
par défaut. Le tableau donne l'écart de kappa mesuré ; ★ signale un intervalle de
confiance à 95 % (bootstrap par grappes) excluant zéro.

| Option | `qwen3-6-35b-moe` | `qwen3-8-27b` | `gemma4-26b-moe` | Copies |
|---|---|---|---|---|
| `chain_of_thought` | −0,031 ★ | +0,015 | −0,037 ★ | 500 |
| `count_items` | −0,015 | +0,030 ★ | −0,051 ★ | 500 |
| `count_items` + `enforce_count` + `check_neighbours` | +0,006 | +0,033 ★ | −0,075 ★ | 500 |
| `disable_thinking: false` (raisonnement natif) | inexploitable | non testé | inexploitable | 20 |
| **`show_error_examples`** | **+0,016** | **+0,079 ★** | **+0,041 ★** | **100** |

**Ce qu'il faut en retenir.** Aucune des quatre premières n'améliore le codage de façon
générale : le meilleur modèle reste `qwen3-6-35b-moe` **sans aucune option**. Le
raisonnement natif est inexploitable en end-to-end (45 % de copies perdues par
troncature, ~61 h projetées contre ~5 h). Le vote majoritaire entre les trois modèles
ne dépasse pas non plus le meilleur modèle isolé.

**Pourquoi elles échouent.** L'analyse des copies les plus mal codées a montré que
**90 % des fautes manquées sont des items que le modèle transcrit à l'identique du mot
attendu** : il ne voit pas la différence. Le goulot est perceptif, pas logique — or ces
options agissent toutes sur le raisonnement. Recoder mécaniquement à partir de la
transcription du modèle donne d'ailleurs un kappa PIRE que son propre codage (0,554
contre 0,603) : son jugement est bon, c'est sa lecture qui bloque.

`show_error_examples` est la seule option qui vise la lecture, et la seule qui
progresse sur les trois modèles — à confirmer sur 500 copies.

> **Ne pas activer deux options de raisonnement dans un même run** : l'écart mesuré ne
> serait plus imputable à l'une ou à l'autre.

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
- **Deux mécanismes de raisonnement, à ne pas confondre** :
  - `model.disable_thinking: false` active le **raisonnement natif** du modèle. La
    sortie JSON n'est PAS modifiée (le raisonnement arrive dans un champ
    `reasoning_content` séparé, écrit dans `<run>_<modele>_reasoning.jsonl`, une
    ligne par copie). C'est du niveau **copie**, pas item : un bloc pour les 83
    items. Réservé aux modèles Qwen3 — mesuré le 11/09/2026, `gemma4-26b-moe` en
    thinking consomme 16 384 tokens de raisonnement sans jamais rendre de JSON, et
    `qwen3-vl` n'a pas ce mode. Il n'existe **aucun réglage d'effort** :
    `reasoning_effort` et `thinking_budget` sont acceptés par l'API mais inertes.
  - `prompt.chain_of_thought: true` ajoute un champ **« comparaison » par item**
    dans le JSON : le schéma de sortie change, et le raisonnement devient
    attribuable item par item. Coût mesuré : +28 à +46 % de tokens.
- **Ne pas activer les deux ensemble** dans un même run : un écart de performance ne
  serait plus attribuable à l'un ou à l'autre.
- **`max_tokens` et raisonnement** : le raisonnement est facturé sur le même budget
  que la réponse. 8192 suffit sans raisonnement, il faut **16384** avec. Une
  génération tronquée code toute la copie en `?` — le pipeline émet désormais un
  avertissement « Génération TRONQUÉE » au lieu d'échouer en silence.
