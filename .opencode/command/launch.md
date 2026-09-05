---
description: Lance, surveille ou arrête une évaluation complète de la dictée (~3 500 copies)
agent: build
---

# Évaluation complète de la dictée

Arguments reçus : `$ARGUMENTS`

Toute la logique est dans `launchers/launch_eval.sh`, un script bash autonome à la
racine du dépôt. Ne réécris pas de `nohup` à la main : le script règle des pièges que
la ligne de commande manuelle oublie (verrou sur le fichier de sortie, process
orphelins, log écrasé, export S3 sous un mauvais nom).

## Ce que tu dois faire

1. Lis `$ARGUMENTS` pour déterminer l'intention :
   - vide, « lance », « démarre », « run complet » → **lancer** ;
   - « où en est », « avancement », « status », « statut » → **surveiller** ;
   - « arrête », « stop », « tue » → **arrêter**.
2. Exécute la commande correspondante depuis la racine du dépôt :

```bash
launchers/launch_eval.sh            # lancer (défaut : configs/scoring/dictee_end2end.yaml)
launchers/launch_eval.sh --status   # avancement : copies faites, débit, heure de fin estimée
launchers/launch_eval.sh --stop     # arrêt propre (chaîne de relance + benchmark)
```

3. Si `$ARGUMENTS` mentionne une autre approche (`two_stage`) ou un autre modèle
   (`gemma4-26b-moe`…), ajoute les options — et **répète-les sur `--status` et
   `--stop`**, sinon ces commandes viseraient un autre run :

```bash
launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml
launchers/launch_eval.sh --model-name gemma4-26b-moe
```

`launchers/launch_eval.sh --help` liste les autres options (`--limit`, `--passes`,
`--no-export`, `--dest-prefix`, `--skip-checks`, `--foreground`).

4. Restitue la sortie du script à l'utilisateur : elle contient déjà le résultat des
   vérifications, le chemin du log et les commandes de suivi. N'invente rien.

## Points à connaître pour répondre

- Le run est détaché (`setsid` + `nohup`) : il **survit** à la fermeture de l'onglet,
  à une déconnexion ou à une mise en veille. ~3 469 copies, plusieurs heures.
- Le script refuse de partir si un run tourne déjà sur le même fichier de sortie, ou
  si la config contient un `data.limit` (un run partiel se ferait passer pour complet).
- Après une interruption, relancer **exactement la même commande** : les copies déjà
  écrites sont sautées (checkpoint dans `data/processed/<run>_predictions.jsonl`).
- Les copies en échec sont relancées automatiquement (3 passes par défaut), puis le
  fichier est exporté vers S3 sous le nom exact du run.
- N'utilise **jamais** `kill` sur un PID : `uv run` crée un wrapper *et* un `python3` ;
  tuer le wrapper laisse l'enfant écrire en orphelin. `--stop` fait ça correctement.
- Données d'élèves mineurs : aucune image ni transcription ne sort du SSP Cloud, et
  rien de tout cela ne va dans Git.
