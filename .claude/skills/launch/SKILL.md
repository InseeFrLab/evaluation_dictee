---
name: launch
description: Lance, surveille ou arrête une évaluation COMPLÈTE de la dictée (~3 500 copies, ~30 h) sur le SSP Cloud. À utiliser dès que l'utilisateur demande de lancer un run, un benchmark, une évaluation sur tout l'échantillon, ou de savoir où en est un run en cours. Le run survit à la fermeture de la session.
---

# Lancer une évaluation complète de la dictée

Toute la logique est dans **`launchers/launch_eval.sh`** — un script bash autonome, sans
dépendance à un assistant. Ton rôle ici est de l'appeler avec les bons arguments et
d'expliquer ce qu'il répond. **N'improvise pas de `nohup` à la main** : le script
règle des pièges que la ligne de commande manuelle oublie systématiquement (verrou,
orphelins, log écrasé, export sous un mauvais nom).

## Les trois commandes

```bash
launchers/launch_eval.sh                      # lancer (config par défaut : dictee_end2end)
launchers/launch_eval.sh --status             # où en est le run : copies, débit, ETA
launchers/launch_eval.sh --stop               # arrêter proprement
```

Si l'utilisateur vise une autre approche ou un autre modèle, ajouter les mêmes options
aux **trois** commandes, sinon `--status`/`--stop` viseraient un autre run :

```bash
launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml
launchers/launch_eval.sh --model-name gemma4-26b-moe
```

`launchers/launch_eval.sh --help` liste tout le reste (`--limit`, `--passes`,
`--no-export`, `--dest-prefix`, `--skip-checks`, `--foreground`).

## Ce que fait le script, pour pouvoir l'expliquer

1. **Vérifications avant 30 h de calcul** : `uv` présent, config valide, `data.limit`
   bien à `null` (sinon il refuse : un run partiel se ferait passer pour complet),
   aucun run concurrent sur le même fichier de sortie (process + verrou `flock`),
   secrets lisibles, modèle joignable, CSV des labels lisible sur S3.
2. **Lancement détaché** (`setsid` + `nohup`) : le run survit à la fermeture de
   l'onglet, à une déconnexion réseau, à une mise en veille.
3. **Vérification du démarrage** : il attend la ligne `N copies au total, M à traiter
   (K déjà faites)` et l'affiche. C'est le contrôle qui compte — `0 déjà faites` alors
   qu'un checkpoint existe signalerait que le run vise le mauvais fichier.
4. **Relance des échecs** : jusqu'à `--passes` passages (3 par défaut). Le pipeline
   reprend seul les copies listées dans `failed_copies.txt`. La chaîne s'arrête dès
   qu'il n'y a plus d'échec, ou dès que le reliquat cesse de diminuer.
5. **Export S3 final** sous le nom exact du run, avec un garde-fou : il refuse
   d'écraser un fichier S3 **plus gros** que le fichier local (protection contre
   l'écrasement d'un run complet par un run partiel).

## À savoir pour répondre à l'utilisateur

- **Durée** : ~3 469 copies. Compter des heures ; `--status` donne un débit mesuré et
  une heure de fin estimée, bien plus fiable qu'une estimation a priori.
- **Reprise** : après un crash ou un `--stop`, relancer **exactement la même commande**.
  Les copies déjà écrites sont sautées, rien n'est perdu sauf la copie en cours.
- **Nommage** : tout est suffixé par le(s) modèle(s) — `<name>_<modèle>_predictions.jsonl`.
  Deux modèles n'écrasent jamais le même fichier, et peuvent tourner en parallèle.
- **Logs** : `logs/<run>_<horodatage>.log`, un par lancement, jamais écrasé ;
  `logs/<run>.latest.log` pointe sur le dernier.
- **Jamais** `kill` sur un PID : `uv run …` crée un wrapper **et** un `python3`, et
  tuer le seul wrapper laisse l'enfant écrire en orphelin. `--stop` s'en charge
  correctement (chaîne d'abord, puis `pkill -f` sur le motif exact du run).
- **Données sensibles** : ne jamais afficher ni copier d'images ou de transcriptions
  d'élèves hors du SSP Cloud, et rien de tout cela dans Git.

## Après le run

Les prédictions sont sur S3 (export automatique) et en local dans
`data/processed/`. L'analyse se fait dans `notebooks/03_analyse_resultats.ipynb`
(métriques) et `notebooks/04_diagnostic.ipynb` (copies les plus en désaccord).
