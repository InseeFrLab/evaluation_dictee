# Journal des décisions méthodologiques

Ce fichier trace les décisions structurantes du projet et leur justification.
Format inspiré des ADR (Architecture Decision Records).

---

## D1 — Commencer par la dictée

**Décision** : la phase 1 porte sur la dictée, pas sur la production écrite libre.

**Raison** : la dictée a un texte de référence connu, ce qui rend l'évaluation
mesurable (CER/WER exacts, codage item par item). La production libre n'a pas de
référence et dépend davantage de la qualité HTR.

---

## D2 — Grille simplifiée 1/erreur/0 en cible principale

**Décision** : la cible principale du codage regroupe les codes 3/4/5 en un seul
code « erreur » (1 = correct, 9 = erreur, 0 = absent). La grille complète à 6 codes
reste un objectif secondaire.

**Raison** : sur le test pilote (3 copies, 249 items), la simplification fait passer
le kappa de 0,785 à 0,879 et fait disparaître 8 des 17 désaccords — exactement ceux
portant sur la frontière lexical/grammatical, là où les correcteurs humains eux-mêmes
divergent le plus.

**À valider avec la DEPP** : la distinction lexical/grammatical reste-t-elle
indispensable à certaines exploitations (suivi fin des compétences) ? Si oui, la
traiter comme un sur-codage optionnel.

---

## D3 — Règle des ratures : lire l'état final

**Décision** : quand l'élève rature et réécrit, on lit la version finale (corrigée
par l'élève).

**Raison** : tranché par la DEPP. Élimine 2 désaccords du test pilote. Inscrit dans
tous les prompts (`pipeline/prompts.py`).

---

## D4 — Score de confiance par item obligatoire

**Décision** : chaque code prédit doit s'accompagner d'un score de confiance, pour
permettre un renvoi ciblé des items incertains vers un codeur humain.

**Raison** : besoin DEPP confirmé. Le livrable décisionnel central est la courbe
« taux de renvoi humain vs taux d'erreur résiduel » (`evaluation/calibration.py`).

---

## D5 — Modèle principal Qwen2.5-VL, tester le 7B avant le 72B

**Décision** : Qwen2.5-VL est le modèle principal. On évalue d'abord le 7B (1 GPU
H100) et on ne passe au 72B (4 GPU, TP=4) que si nécessaire.

**Raison** : calibrer le coût GPU. Si le 7B atteint déjà κ > 0,75 en un seul appel,
le 72B n'est pas requis pour la phase 1.

---

## D6 — Fine-tuning : scoring oui, HTR conditionné

**Décision** : les ~3 000 copies annotées suffisent pour fine-tuner le scoring
(cible 1/erreur/0). Le fine-tuning HTR est conditionné à la production préalable
d'un ground truth de transcription mot à mot (les annotations actuelles sont des
codes, pas des transcriptions).

---

## Questions ouvertes (à trancher avec la DEPP)

- Localisation des corrections multi-annotateurs (un seul code disponible pour l'instant).
- Sémantique du code 9 sur les items mots (hors grille écrite).
- Existence de scans en niveaux de gris (au lieu du TIFF 1 bit).
- Deadline réelle des copies CEDRE 2026 sur S3.
- Seuil de « performance satisfaisante » (QWK) différencié par critère.
- Répartition précise des tâches avec TEKLIA.
- Validation RGPD du niveau de sécurité SSP Cloud pour des données de mineurs.

---

## D7 — Raisonnement des modèles : ce qui est possible, et à quel prix

**Décision** : le raisonnement est évalué comme un **bras d'expérience séparé**
(`configs/scoring/dictee_end2end_thinking.yaml`), sous sa forme **native**
(`model.disable_thinking: false`), sur les modèles Qwen3 uniquement. Le schéma de
sortie reste identique au bras sans raisonnement, pour que l'écart mesuré soit
imputable au seul raisonnement.

**Deux mécanismes à ne pas confondre** :

| | Raisonnement natif | Champ « comparaison » (`prompt.chain_of_thought`) |
|---|---|---|
| Activation | paramètre du modèle | consigne de prompt + schéma JSON étendu |
| Sortie structurée | **inchangée** (champ `reasoning_content` séparé) | **modifiée** (un champ de plus par item) |
| Granularité | la **copie** (un bloc pour les 83 items) | l'**item** |
| Coût mesuré (83 items) | ~9 700 tokens vs ~4 400 | +28 à +46 % de tokens |

**Mesures du 11/09/2026 sur llm.lab**, qui motivent la décision :

- L'endpoint isole le raisonnement dans `reasoning_content` : le JSON reste conforme
  au schéma même avec `structured_output`. La mise en garde historique (« le bloc
  `<think>` casse le JSON ») ne vaut plus.
- **Aucun réglage d'effort n'existe** : `reasoning_effort` (`low`/`high`) et
  `thinking_budget` sont acceptés par l'API mais **sans aucun effet** — volumes de
  raisonnement identiques à l'octet près. C'est binaire, activé ou non.
- **Aucun raisonnement par item n'est possible** à la granularité d'appel actuelle :
  une requête = une copie = 83 items. Un raisonnement attribuable à un item doit donc
  vivre dans la sortie structurée (c'est le rôle du champ « comparaison »), ou exiger
  83 appels par copie (~288 000 appels sur l'échantillon — hors de portée).
- Support par modèle : `qwen3-6-35b-moe` et `qwen3-8-27b` (activé par défaut, à couper
  explicitement), `gemma4-26b-moe` (désactivé par défaut, et **inutilisable** : le
  raisonnement s'emballe, consomme 16 384 tokens et ne rend aucun JSON),
  `qwen3-vl` (pas de raisonnement natif).

**Conséquences pour le pipeline** :

- Le raisonnement est stocké dans `<run>_<modele>_reasoning.jsonl`, **une ligne par
  copie**. Il ne peut pas aller dans le JSONL de prédictions, qui est de niveau item :
  83 lignes par copie × ~13 000 caractères donneraient plusieurs Go.
- `max_tokens` doit passer à **16384** : le raisonnement est facturé sur le même budget
  que la réponse. Une génération tronquée code toute la copie en `?`.
- Le pipeline journalise désormais les générations tronquées (`finish_reason=length`)
  et le temps de traitement par copie. Sans ces deux garde-fous, le run CoT de juillet
  2026 a consommé **43 h pour coder 802 copies en « non transcrite »**, sans qu'aucun
  message ne signale la cause.

**Reste ouvert** : la forme « champ comparaison » n'est pas abandonnée, mais son
implémentation actuelle a trois défauts recensés (ordre des clés non garanti par le
décodage contraint, champ `reason` demandé au prompt mais absent du schéma, perte de
la comparaison lors d'un ré-alignement). À corriger avant tout run qui l'utiliserait.

---

## D8 — Copies vierges et illisibles : écartées des métriques, jamais perdues

**Décision** : une copie **vierge** (l'élève n'a rien écrit) ou **illisible** (l'expert
n'a pas pu lire, code `i`) est écartée des métriques de performance, conservée dans le
JSONL avec un motif d'exclusion, et listée dans un fichier dédié pour vérification
humaine.

**Raison** : dans les deux cas, l'expert n'a rendu **aucun jugement auquel comparer le
modèle**. Les compter revient à imputer au modèle un défaut d'annotation ou de
numérisation. Le cas d'école est `dictee_2015_0204`, codée `i` sur ses 83 items :
elle ressortait « pire copie du corpus, 0 % d'accord » alors qu'il n'y a aucun
désaccord de jugement à constater.

**Mécanique** :

- Un item est inévaluable si son code expert est `i` (illisible) ou vide
  (`reference.est_evaluable`). Une copie est vierge si sa densité d'encre passe sous
  `data.blank_ink_threshold` (2,5 % par défaut, seul le pré-imprimé marquant la page).
- Une copie n'est écartée **en entier** que si elle est **majoritairement**
  inexploitable. En dessous de ce seuil, seuls les items concernés sortent des
  métriques : quelques mots illisibles ne justifient pas de perdre les 70 autres.
- Chaque ligne du JSONL porte un champ `exclusion` (`vierge`, `illisible`, ou absent).
  Toute relecture du fichier — métriques, notebooks, site — applique donc le même
  filtre, sans le réinventer.
- Les copies écartées sont listées dans
  `data/processed/<run>_<modele>_copies_ecartees.csv` (copie, motif, items concernés,
  items au total). **Cette liste doit être relue à l'œil** : une copie peut être
  déclarée vierge pour une mauvaise raison — numérisation trop pâle, seuil d'encre mal
  réglé — auquel cas c'est le seuil qu'il faut corriger, pas la copie qu'il faut
  oublier.

**Effet mesuré** (500 copies, `comptage_strict` / qwen3-6-35b-moe) : 6 copies vierges
et 13 items illisibles écartés, soit 511 décisions sur 41 500. Le kappa passe de
**0,601 à 0,582**. Il BAISSE, et c'est normal : les copies vierges étaient codées
correctement à 100 % sans aucun appel modèle, et gonflaient donc artificiellement les
métriques. Ce qu'elles mesuraient, c'est la qualité de l'heuristique d'encre, pas
celle du modèle.

**Conséquence** : les métriques publiées avant cette correction sont légèrement
optimistes, d'environ 0,02 de kappa sur cet échantillon.

---

## D9 — Un bras n'alimente le site que sur le corpus complet

**Décision** : les bras d'expérience (testés sur un échantillon partiel) sont exportés
sur S3 dans `predictions/experimentations/`, un sous-dossier que le site ne liste
jamais. Les runs de référence (corpus complet) restent à la racine de `predictions/`,
seul dossier que le site lit.

**Raison** : le 15/09/2026, l'export des cinq bras testés dans cette phase (CoT,
comptage, comptage+, exemples, thinking) avait été fait avec la même convention de
nommage que les runs de référence, dans le même dossier plat. `modeles_exportes()`
(`website/_analyse.py`) déduit la liste des modèles affichés du contenu de
`predictions/` : elle avait donc fait apparaître **13 faux modèles** (`cot_qwen3-6-
35b-moe`, `comptage_strict_gemma4-26b-moe`…) dans les menus déroulants, avant qu'un
rendu du site n'ait lieu.

**Corrections successives, et pourquoi la seconde l'a emporté** :
1. Un filtre par liste de noms connus (`BRAS_EXPERIMENTAUX`) dans `website/
   _analyse.py`. Fonctionnel, mais à maintenir à la main : un sixième bras oublié dans
   la liste reproduit le bug à l'identique, et les fichiers restent mélangés dans
   `predictions/` — trompeur pour quiconque parcourt le bucket à la main (DEPP,
   TEKLIA), pas seulement pour le rendu du site.
2. **Un sous-dossier S3** (`predictions/experimentations/`). `_noms_exportes()` fait un
   `fs.ls()` non récursif sur `predictions/` : un fichier posé dans le sous-dossier est
   invisible du site *par construction*, sans aucune liste à maintenir. Cette option a
   été retenue ; la première a été retirée du code.

**Le critère de routage n'est PAS « lancé via `launchers/launch_eval.sh` »**, mais
« couvre le corpus complet ». `scripts/export_predictions.py` compare l'effectif
RÉELLEMENT présent dans le fichier de prédictions au nombre de copies du corpus (CSV
des labels), avec une marge de 10 % pour les copies perdues (échecs API). Ce choix
délibéré évite un piège : `data.limit` du YAML vaut `null` même pour un run lancé avec
`--limit 500` en ligne de commande (c'est ainsi que tous les bras de ce projet ont été
lancés) — s'appuyer sur `data.limit` aurait classé ces runs comme des runs complets, et
reproduit le bug initial. Le launcher reste le moyen recommandé de mener un run complet
(fiabilité, checkpointing), mais ce n'est pas ce que le routage vérifie.

**Conséquence pour `scripts/rapport_bras.py`** : `_localiser()` cherche désormais un
run local, puis à la racine de `predictions/`, puis dans son sous-dossier — sans
savoir a priori laquelle des deux catégories il cherche, ce n'est pas son rôle de le
deviner.
