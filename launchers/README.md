# `launchers/` — scripts shell du projet

Ce dossier rassemble les **scripts shell** : ceux qui lancent les traitements longs,
et ceux qui préparent le poste de travail. Les points d'entrée Python restent dans
[`scripts/`](../scripts/) — un script de `scripts/` fait *un* passage et s'arrête ;
un lanceur de `launchers/` l'enveloppe pour qu'il survive à la session, se surveille
et se rattrape tout seul.

Tout ici tourne avec `bash` seul : ni `uv`, ni notebook, ni assistant IA requis pour
démarrer.

## Lancer un traitement long

| Lanceur | Ce qu'il lance | Durée typique |
|---|---|---|
| [`launch_eval.sh`](launch_eval.sh) | `scripts/run_benchmark.py` sur l'échantillon complet (~3469 copies), puis l'export S3 | ~30 h |

## Préparer le poste de travail

| Script | Ce qu'il fait |
|---|---|
| [`install_assistant.sh`](install_assistant.sh) | Installe Claude Code et/ou openCode et les branche, si on le souhaite, sur un serveur LLM interne (llm.lab) plutôt que sur une API payante |

```bash
launchers/install_assistant.sh --check                       # teste l'endpoint, n'écrit rien
launchers/install_assistant.sh --endpoint "$LLM_BASE_URL"    # installe et configure
launchers/install_assistant.sh opencode --print              # montre la config sans l'écrire
```

Le script **sonde** l'endpoint avant d'écrire quoi que ce soit, parce que les deux
outils ne parlent pas le même protocole :

- **openCode** parle les API **compatibles OpenAI** : llm.lab fonctionne directement.
  Le script écrit un provider dans `opencode.json` et y déclare les modèles
  effectivement servis, découverts via `<endpoint>/models`.
- **Claude Code** parle l'**API Anthropic Messages**. `ANTHROPIC_BASE_URL` doit donc
  désigner une base dont `<base>/v1/messages` répond — Claude Code ajoute ce suffixe
  lui-même. Le script teste la *forme* de la réponse (une passerelle Open WebUI
  renvoie « 400 Model not found » sur n'importe quelle route : le code HTTP ne prouve
  rien) et refuse d'écrire une configuration qui échouerait en silence.

> llm.lab (SSP Cloud) sert **les deux** protocoles : `…/api/v1/chat/completions` pour
> openCode et `…/api/v1/messages` pour Claude Code — soit `ANTHROPIC_BASE_URL` =
> `https://llm.lab.sspcloud.fr/api`. Les deux assistants fonctionnent donc sans
> abonnement.

**Aucun secret n'est écrit sur disque** : les configurations référencent le *nom*
d'une variable d'environnement (`{env:LLM_API_KEY}` côté openCode, `${LLM_API_KEY}`
dans le profil shell), alimentée par le Vault Onyxia. Une config openCode existante
est sauvegardée avant modification, et le bloc ajouté au profil shell est délimité,
donc remplacé — jamais dupliqué — à chaque relance.

## Pourquoi un lanceur plutôt qu'un `nohup` tapé à la main

Sur le SSP Cloud, un run de 30 h lancé depuis le terminal du navigateur meurt à la
première déconnexion. La parade tient en une ligne de `nohup`… et en une demi-douzaine
de précautions qu'on oublie systématiquement à 18 h un vendredi :

- créer `logs/` **avant** (sinon `nohup` échoue sans rien dire) ;
- un fichier de log **distinct par lancement**, sinon deux runs entrelacent leurs
  sorties et le log devient illisible après coup ;
- vérifier qu'**aucun run ne tourne déjà** sur le même fichier de sortie : deux runs
  qui appendent le même JSONL dupliquent les copies et faussent les métriques ;
- vérifier que la **reprise a bien pris** (`N à traiter (K déjà faites)`) — un run qui
  repart de zéro sur un checkpoint existant, c'est 30 h à refaire ;
- ne **jamais** arrêter le run par son PID : `uv run …` crée un wrapper *et* un
  `python3`, et tuer le wrapper laisse l'enfant orphelin continuer d'écrire.

Le lanceur fait tout cela, à chaque fois, dans le bon ordre.

## Usage

```bash
launchers/launch_eval.sh            # lancer
launchers/launch_eval.sh --status   # avancement : copies, débit mesuré, heure de fin
launchers/launch_eval.sh --stop     # arrêt propre
launchers/launch_eval.sh --help     # toutes les options
```

Les options qui identifient le run (`--config`, `--model-name`, `--model-stage2-name`)
doivent être **répétées sur les trois commandes** : le nom d'un run est
`<name>_<modèle(s)>`, et sans elles `--status` interrogerait un autre run.

Documentation complète : section « Runs longs » du [README](../README.md).

## Conventions pour ajouter un script

Les six règles ci-dessous valent pour tout le dossier ; les points 2 et 4 concernent
surtout les lanceurs de runs longs.

1. **Bash seul.** Un script d'ici doit tourner sans `uv`, sans notebook et sans
   assistant. Il peut évidemment *appeler* `uv run` ; il ne doit pas en dépendre
   pour démarrer.
2. **Trois modes** : lancement (par défaut), `--status`, `--stop`. Un run qu'on ne
   sait pas surveiller ni arrêter proprement est un run qu'on relancera de travers.
3. **Vérifier avant d'agir.** Tout ce qui peut échouer vite (config, secrets, accès
   réseau, run concurrent, protocole d'un endpoint) doit échouer *avant* le `setsid`
   ou l'écriture d'un fichier, dans le terminal de l'utilisateur, pas dans un log que
   personne ne lira. Sonder plutôt que supposer : un code HTTP ne prouve pas qu'une
   API parle le protocole attendu.
4. **Un nom de run = une source unique de vérité.** Dériver les chemins (checkpoint,
   log, export) d'une seule fonction, jamais les recomposer à la main : c'est ainsi
   qu'on écrase le fichier d'un autre modèle.
5. **Jamais de secret sur disque.** Écrire le *nom* d'une variable d'environnement,
   pas sa valeur. Sauvegarder tout fichier de config modifié, et délimiter tout bloc
   ajouté à un fichier partagé pour pouvoir le remplacer sans le dupliquer.
6. **Le script est le contrat.** Les skills et commandes d'assistants
   (`.claude/skills/`, `.opencode/command/`) ne font que l'appeler : aucune logique
   ne doit vivre en double dans un fichier Markdown.
