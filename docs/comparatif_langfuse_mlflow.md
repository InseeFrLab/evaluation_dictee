# Langfuse vs MLflow pour tracer les appels LLM d'un projet OCR

Test réalisé le 28/10/2026 sur `evaluation_dictee` (branche `comparatif-mlflow`). Les deux outils
coexistent dans le code : `TRACKING_BACKEND=langfuse|mlflow` (défaut `langfuse`) et reçoivent les
mêmes informations (versionnage de prompts, traces, scores, sessions, `user_id`, tokens, coût).

**Versions** : Langfuse serveur **3.132.0** (instance partagée `langfuse.lab.sspcloud.fr`, vérifié
via `/api/public/health`) ; SDK Python Langfuse **4.18.0** sur la branche (4.7.1 verrouillé sur
`main`). La doc indique un serveur ≥ 3.63.0 pour le SDK v4 : les versions sont donc compatibles
sur le papier. MLflow serveur **3.15.1** (instance perso Onyxia), client **3.17.2**.
Modèle des runs comparés : `gemma4-26b-moe` (`qwen3-8-27b` testé sur 5 copies seulement).

**Niveau de preuve** : chaque ligne des tableaux est étiquetée *mesuré* (je l'ai exécuté),
*documentation* (issu de la doc officielle, non testé ici) ou *non vérifié*. Je n'ai pas de
navigateur : **aucun jugement sur l'interface n'est basé sur une utilisation réelle**.

## Grille de comparaison (run de 50 copies aléatoires)

Échantillon : 50 copies tirées au hasard (graine 2026, `data.sample_seed`), identiques pour les
deux backends, dictée 2015, méthode C, `gemma4-26b-moe`, 8 copies en parallèle. 1 copie vierge
écartée des métriques dans les deux runs. Accord brut 84,8 % (Langfuse) contre 84,7 % (MLflow),
kappa 0,554 contre 0,549 : écart dû au modèle, pas à l'outil.

| Critère | Langfuse | MLflow | Avantage | Preuve |
|---------|----------|--------|----------|--------|
| **Difficulté d'implémentation** | Backend de 75 lignes de code. Session, user et tags se propagent seuls (`propagate_attributes`), wrapper OpenAI, lien prompt explicite. Le coût demande un script serveur à part (39 lignes). | Backend de 147 lignes : adaptateur de span (`CopyTrace`), `ContextVar` pour session et user, processeur de span pour le coût, flush explicite. 5 frictions (voir plus bas). | Langfuse | mesuré |
| **Gestion de l'interface** | Non évaluée. | Non évaluée. | — | non vérifié |
| **Lecture d'une trace** | Champs complets lus le 28/10 à 17:41 (nom, spans, version de prompt, scores, coût natif, tokens, image en média). **Illisible aujourd'hui pour le run de 50 copies** : le serveur annonce 37 traces sur 51 mais renvoie des listes vides. | `get_trace` en 0,6 s : spans, version de prompt, 6 scores, coût, tokens, latence. 50 traces sur 50, de façon reproductible. | MLflow (fiabilité observée) | mesuré |
| **Lecture par session** | Entité session native (`/sessions/{id}`). Non relisible pour ce run (voir ci-dessus). | Filtre par métadonnée : 50 traces en 1,4 s. Pas d'entité session : totaux (coût, tokens, accord moyen) calculés côté client, 0,01 s avec pandas. | À égalité sur le papier | mesuré pour MLflow, non vérifié pour Langfuse |
| **Lecture par user_id** | Filtre `userId` natif. Non relisible pour ce run. | Filtre par métadonnée : 73 traces du user en 1,6 s, agrégats côté client. | À égalité sur le papier | idem |
| **Métriques d'évaluation** | Scores sur une trace de synthèse `run-summary`, mélangés aux scores par copie dans l'API. | Vrai run : `search_runs` en 0,4 s, 18 métriques et paramètres, comparaison de runs native. | MLflow | mesuré |
| **Tableaux de bord d'accueil** | Home personnalisable (widgets coût, latence, scores, usage, tableaux croisés, export JSON). La doc date la page d'accueil-dashboard de juillet 2026 : à vérifier sur un serveur 3.132. | Onglet « Overview » (latence, volume, tokens, coût, scores) à partir de MLflow 3.10 ; notre serveur est en 3.15.1. Peu ou pas de personnalisation documentée. | Langfuse (si la doc s'applique à notre version) | documentation |
| **Versionnage de prompts** | Labels `production`, `staging`, `latest` (automatique), commit message, diff dans l'UI. Création, promotion et rollback testés. Tags au niveau du prompt seulement (liste), pas par version. Labels protégés selon l'offre. Suppression par API refusée (405). | Alias `production`, `staging` + tags clé/valeur **par version et par prompt**, modifiables après coup (ex. `status: validated`, `in-review`, `archived`), commit message, onglet Compare. Création, promotion, rollback, recherche par tag testés. | MLflow pour les tags, Langfuse pour la rapidité | mesuré (UI : documentation) |
| **Temps de logging** | 2,2 s sur 175 s de run (1,3 %) | 5,6 s sur 187 s (3,0 %) | Langfuse, écart faible | mesuré |
| **Fiabilité de l'enregistrement et de la lecture** | Envoi OK (statut 201), mais lectures incohérentes sur cette instance : 200 puis 404 pour la même trace, listes vides avec un total non nul. | Aucun incident observé. | MLflow (sur ces instances) | mesuré |

### Temps de logging en détail (50 copies, secondes cumulées)

| Poste | Langfuse | MLflow |
|-------|----------|--------|
| Ouverture et fermeture du run | 0,00 | 2,35 |
| Ouverture des traces (50) | 0,85 | 1,46 |
| Mise à jour et scores des traces | 0,04 | 0,08 |
| Fermeture des traces | 0,01 | 0,02 |
| Métriques de fin de run | 0,00 | 0,25 |
| Flush final | 1,34 | 1,48 |
| **Total** | **2,24** | **5,63** |

Le temps passé ici est négligeable dans les deux cas. Le micro-benchmark précédent (flush de
15 s pour 100 traces avec images, contre 3,6 s) ne se reproduit pas sur un vrai run : les pièces
jointes partent pendant l'évaluation, il ne reste que 1,5 s à vider à la fin. Le coût de MLflow vient
surtout de l'ouverture et de la fermeture du run (2,35 s).

### Versionnage de prompts : temps des opérations (s)

| Opération | Langfuse | MLflow |
|-----------|----------|--------|
| Créer une version | 0,05 à 0,10 | 0,8 à 1,3 |
| Promouvoir / rollback | 0,05 | 0,84 |
| Charger par label ou alias | 0,02 | 1,08 |
| Poser un tag de version | — (non supporté) | 0,29 |

Serveur MLflow distant : chaque appel coûte environ une seconde. Sans conséquence au chargement
dans un run (les prompts sont mis en cache), notable lors d'un script de publication.

## Ce qui a été testé

| Test | Langfuse (main) | MLflow (branche) |
|------|-----------------|------------------|
| Run réel, 5 copies de la dictée 2015, `gemma4-26b-moe`, méthode C | 36,6 s, accord 87,0 % | 42,2 s, accord 86,7 % |
| Micro-benchmark : 100 traces avec image de 60 Ko, 8 threads | 0,37 s chemin chaud, 3,6 s flush inclus | 0,61 s chemin chaud, 15,3 s flush inclus |
| Serveur de tracking injoignable | messages de retry, pas de crash observé (micro-benchmark seul, pas un run complet) | run complet testé : il continue, avec des tracebacks bruyants |
| Requêtes, cycle de vie des prompts | voir ci-dessous | voir ci-dessous |

Les écarts de durée et d'accord des runs réels sont du bruit (latence du modèle, non-déterminisme
du décodage), pas un effet de l'outil. Seul le micro-benchmark isole le surcoût de traçage.

## Comparatif par cas d'usage

| Cas d'usage | Langfuse | MLflow 3.17 |
|-------------|----------|-------------|
| **Traces** | Trace par copie, génération imbriquée via `langfuse.openai`. | Trace par copie, span LLM via `mlflow.openai.autolog()`. Équivalent. |
| **Images (OCR)** | Base64 sorti de la trace et stocké en objet média (`@@@langfuseMedia:…@@@`). | Base64 sorti en pièce jointe (`mlflow-attachment://…`). Trace de 25 Ko pour une image de 59 Ko. Équivalent, mais l'envoi des pièces jointes ralentit le flush (voir plus bas). |
| **Sessions** | `session_id` propagé par `propagate_attributes` à toutes les traces. | `session_id` de trace (`mlflow.trace.session`), à réappliquer à chaque trace : pas de propagation automatique, d'où une `ContextVar` dans `tracking.py`. Filtrable par session. |
| **user_id** | Idem, propagé. | `user` de trace, à réappliquer aussi. Filtrable. |
| **Scores** | Score sur la trace (`score_trace`). Les 80 971 scores existants de l'instance prouvent que ça marche en production. | Assessment « feedback » (`log_feedback`), source typée (humain, code, LLM juge). Filtrable (`feedback.raw_agreement > 0.8`). |
| **Métriques agrégées du run** | Scores sur une trace de synthèse `run-summary:…` : détournement. | Run MLflow natif : params, métriques, tags. Comparaison de runs en tableau et graphiques, c'est le point fort de MLflow pour un benchmark. |
| **Versionnage de prompts** | Labels `production` et `latest` automatiques, `create_prompt`. | Registre : `register_prompt` + alias `production`. Création de version, promotion et rollback testés. Pas d'API de diff (UI seulement). |
| **Lien prompt ↔ trace** | Explicite (`langfuse_prompt=`), version visible sur la génération. | Automatique : `load_prompt` dans la trace pose `mlflow.linkedPrompts`. Moins de code. Filtre `prompt = 'Dictation/2'`. |
| **Coût** | Prix par token configurable par modèle (`add_langfuse_models.py`) : 0,001 € par génération affiché. | Pas de table de prix personnalisée (seul le catalogue public est connu, coût `None` pour `llm.lab`). Contournement : un processeur de span (`tracking._ajouter_cout`) pose `mlflow.llm.cost` avec les mêmes prix (`utils/model_prices.py`). Résultat identique à Langfuse (0,00101 € par copie), agrégé sur la trace. Une dizaine de lignes de code à maintenir. |
| **Tags** | Liste de tags (`run:…`, `thinking:off`…), filtre par appartenance. | Tags clé/valeur uniquement : la liste est jointe en une chaîne et les paramètres deviennent des tags filtrables (`tags.method = 'C'`). |
| **Traçabilité du code** | Non. | Commit git, branche et dépôt capturés automatiquement. |

## Métriques communes aux deux versions

Langfuse et MLflow ne comptent pas les tokens de la même façon, et aucun des deux ne donne de
total par run. Pour que les deux versions reçoivent **les mêmes métriques**, le pipeline les mesure
lui-même (`utils/usage.py`, prix dans `utils/model_prices.py`) et les envoie à l'identique.

| Niveau | Métriques (mêmes noms des deux côtés) | Langfuse | MLflow |
|--------|---------------------------------------|----------|--------|
| Par copie | `raw_agreement`, `duration_s`, `input_tokens`, `output_tokens`, `cost_eur`, `n_attempts` | scores de la trace | feedbacks de la trace |
| Par run | `raw_agreement`, `cohen_kappa`, `kappa_mots`, `kappa_ponctuation`, `taux_sous_detection`, `biais_taux_faute_pts`, `n_items`, `n_blank`, `n_illegible`, `n_items_ecartes`, `n_non_transcribed`, `median_seconds_per_copy`, `n_copies_llm`, `total_input_tokens`, `total_output_tokens`, `total_cost_eur`, `mean_cost_eur_per_copy`, `mean_output_tokens_per_copy` | scores de la trace `run-summary:…` | métriques du run MLflow |
| Trace (natif) | tokens, coût, latence, version du prompt, session, user | oui | oui |

Écarts qui subsistent, dus aux outils :
- MLflow a les métriques de run dans un vrai run (comparable en tableau) ; Langfuse les range sur
  une trace de synthèse.
- Langfuse garde les tags en liste ; MLflow les met en clé/valeur.
- Coût natif : Langfuse le calcule côté serveur (prix à enregistrer avec `add-langfuse-models`) ;
  MLflow via le processeur de span. Les deux valent 0,00101 € pour 3541 tokens d'entrée et 2747
  de sortie avec `gemma4-26b-moe`.

### Premiers chiffres (5 copies, dictée 2015, méthode C, run MLflow)

| Modèle | Accord brut | Kappa | Coût / copie | Tokens entrée / sortie (5 copies) | Médiane / copie |
|--------|-------------|-------|--------------|-----------------------------------|------------------|
| `gemma4-26b-moe` | 87,5 % | 0,572 | 0,0010 € | 17 705 / 13 780 | 27,8 s |
| `qwen3-8-27b` | 89,6 % | 0,580 | 0,0026 € | 32 900 / 14 247 | 45,8 s |

Échantillon de 5 copies : aucune différence de qualité n'est établie. Le prix de `qwen3-8-27b`
repose sur un débit **estimé** (1 200 tokens/s, modèle dense), à mesurer avant d'en tirer une
conclusion de coût.

## Effort de migration

7 fichiers modifiés, +203/−117 lignes hors `uv.lock`. `pipeline/benchmark.py` n'a pas changé :
`tracking.py` expose un adaptateur (`CopyTrace`) avec les mêmes méthodes `update` et
`score_trace`. 242 tests passent. Frictions rencontrées :

- MLflow refuse les espaces dans un nom de prompt : « Text coding » devient `Text-coding`.
- Pas de propagation d'attributs : session et user doivent être réappliqués à chaque trace.
- Les tags ne sont pas des listes.
- Pas de prix personnalisés : le coût se calcule dans le code (`utils/model_prices.py`, partagé avec Langfuse).
- Sans `try/except` autour de `set_experiment`, un MLflow injoignable fait planter le run.
  Langfuse n'a pas planté dans le micro-benchmark (retries silencieux).

## Points faibles observés

- **MLflow, flush en rafale** : 15,3 s contre 3,6 s pour 100 traces envoyées d'un coup (micro-benchmark).
  Sur un vrai run de 50 copies, le flush ne prend que 1,5 s (les pièces jointes partent au fil de
  l'eau). À appeler quand même avant de quitter (`flush()` dans `run_benchmark.py`).
- **MLflow, opérations de prompt lentes** : environ 1 s par appel au registre distant (contre
  0,02 à 0,1 s pour Langfuse). Sans effet au chargement en run (cache), gênant pour des scripts.
- **MLflow, bruit si le serveur tombe** : le run termine, mais les logs se remplissent de tracebacks.
- **Langfuse, ingestion et lecture incomplètes côté serveur** (testé avec le code de `main`, SDK
  4.7.1, puis avec la branche, SDK 4.18.0 : même symptôme). L'envoi réussit (le SDK reçoit
  `status 201`, aucune erreur dans les logs), mais :
  - sur 5 copies, `main` est complet (6 traces sur 6) ; la branche a perdu 6 traces sur 6 dans un
    cas (session `cmp2_qwen3-8-27b`) ;
  - sur 50 copies, **`main` perd 4 copies consécutives (0036 à 0039)** : 47 traces lues sur 51, et
    18 copies sur 46 sans leurs scores ; la branche en perd 14 (37 sur 51) ;
  - les lectures basculent d'un appel à l'autre entre « liste pleine, total à 0 » et « liste vide,
    total non nul », et `GET /traces/<id>` répond tantôt 200, tantôt 404.
  Le SDK et ma branche ne sont donc pas en cause. Les runs de 100 copies du 23/10 sont lisibles :
  quelque chose a changé côté serveur (version 3.132.0), probablement sa file d'ingestion ou la base
  de lecture. `auth_check()` échoue aussi (champ `organization` absent de `/api/public/projects`).
  À signaler aux administrateurs de l'instance.
- **Instance MLflow testée = perso**, donc non partagée. Langfuse est l'instance d'équipe (on y voit
  les traces d'autres utilisateurs).

## Non testé

**Interface utilisateur** (je n'ai testé que les API : la ligne « gestion de l'interface » de la grille
est à noter par vous), tableaux de bord réels (documentation seulement), relecture Langfuse du run de
50 copies (impossible, voir plus haut), annotation manuelle des traces, jeux de données
et expériences Langfuse, `mlflow.genai.evaluate` et LLM-juge, prix personnalisés MLflow, passage à
l'échelle (3 469 copies), droits d'accès sur une instance partagée.

## Conclusion pour les projets OCR

**Recommandation (révisée après le run de 50 copies) : Langfuse pour tracer l'inférence, MLflow pour
les métriques de run et l'entraînement, à condition que l'instance Langfuse soit réparée.**

Ce que le benchmark établit :
- **Langfuse est plus simple à intégrer** (75 contre 147 lignes de backend, propagation automatique
  de session et user, wrapper OpenAI) et **un peu plus rapide** (2,2 s contre 5,6 s de logging sur
  un run de 3 minutes : l'écart est négligeable).
- **MLflow est meilleur pour les métriques d'évaluation** (vrai run, comparaison native, lecture en
  0,4 s) et **plus souple sur les tags de prompt** (tags clé/valeur par version, modifiables).
- **Versionnage de prompts** : les deux savent promouvoir, revenir en arrière, et commenter chaque
  version. Langfuse est plus rapide (0,05 s contre 0,8 s), MLflow permet d'y accrocher un statut.
- **Fiabilité** : sur ces deux instances, MLflow s'est comportée sans incident, alors que Langfuse
  accepte les traces mais ne les relit pas de façon fiable. Ce point **pèse lourd** : un outil de
  traçage qu'on ne peut pas relire n'a pas de valeur. Il vient de l'instance, pas du produit, mais
  c'est la situation actuelle.

Ce que le benchmark n'établit pas : la qualité de l'interface et des tableaux de bord (non utilisés),
et la tenue de Langfuse hors de cette instance.

**Arbitrage selon la suite** :
1. **Instance Langfuse réparée (versions serveur/SDK alignées, ingestion vérifiée)** : garder
   Langfuse pour les traces et les prompts, MLflow pour les métriques de run et le fine-tuning.
   C'est la configuration la plus confortable : peu de code, coût natif, dashboards.
2. **Instance Langfuse non réparée à court terme** : MLflow seul est viable (traces, sessions, user,
   scores, prompts avec tags, coût, métriques) au prix d'environ 150 lignes à maintenir, de noms de
   prompts sans espace, d'un `flush()` systématique et d'opérations de prompt plus lentes.
3. **Projet sans appel LLM via API** (HTR/OCR classique) : MLflow seul, c'est déjà l'outil du
   fine-tuning.

**Suite possible** : tester les deux interfaces avec un jeu de critères noté à la main (lecture d'une
trace, filtre par session et user, tableau de bord d'accueil, changement de label ou d'alias d'un
prompt), puis compléter la ligne « gestion de l'interface » de la grille.
