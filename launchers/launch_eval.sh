#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  launch_eval.sh — lance une évaluation COMPLÈTE de la dictée (~3 500 copies)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Point d'entrée UNIQUE et autonome : ce script ne dépend d'aucun assistant IA.
#  Les skills/commandes (.claude/skills/launch, .opencode/command/launch.md) ne
#  font que l'appeler.
#
#  Ce qu'il garantit, et qu'un `nohup` tapé à la main oublie souvent :
#    1. le run survit à la fermeture du navigateur / de la session (setsid+nohup) ;
#    2. un seul run par fichier de sortie (test du verrou flock AVANT de lancer) ;
#    3. un fichier de log DISTINCT et horodaté par lancement (jamais écrasé) ;
#    4. vérifications préalables (uv, secrets, modèle, S3) avant 30 h de calcul ;
#    5. relance automatique des copies en échec (le pipeline les reprend seul) ;
#    6. export S3 final sous le nom EXACT du run (jamais celui d'un autre run) ;
#    7. `--status` (avancement + ETA) et `--stop` (arrêt propre, sans orphelin).
#
#  Usage :
#     launchers/launch_eval.sh                       # end2end, échantillon complet
#     launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml
#     launchers/launch_eval.sh --status
#     launchers/launch_eval.sh --stop
#
#  Voir `launchers/launch_eval.sh --help` pour toutes les options.
# ═══════════════════════════════════════════════════════════════════════════════

set -Eeuo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Valeurs par défaut ────────────────────────────────────────────────────────
CONFIG="configs/scoring/dictee_end2end.yaml"
MODEL_NAME=""
MODEL_STAGE2_NAME=""
LIMIT_CLI=""
PASSES=3
DO_EXPORT=1
EXPORT_EXPLICIT=0
FORCE_EXPORT=0
DEST_PREFIX=""
SKIP_CHECKS=0
FOREGROUND=0
MODE="launch"          # launch | status | stop | chain
RUN_NAME=""            # résolu par probe_config(), ou passé au worker
STAMP=""
LOG=""

# Renseignés par probe_config()
APPROACH=""; MODEL=""; MODEL2=""; LIMIT=""; CONCURRENCY=""; LABELS=""; IMAGES=""
JSONL=""; FAILED=""; LOCK=""

# ── Affichage ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  B=$'\033[1m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; N=$'\033[0m'
else
  B=""; R=""; G=""; Y=""; C=""; N=""
fi
ok()   { printf '%s✔%s %s\n'  "$G" "$N" "$*"; }
warn() { printf '%s!%s %s\n'  "$Y" "$N" "$*"; }
info() { printf '%s·%s %s\n'  "$C" "$N" "$*"; }
step() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
die()  { printf '%s✖%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
launch_eval.sh — évaluation complète de la dictée, résistante à la perte de session.

MODES
  (aucun)                  Lance le run en arrière-plan (setsid + nohup) et rend la main.
  --status                 Avancement du run : copies faites, débit, ETA, échecs.
  --stop                   Arrête proprement le run (chaîne + benchmark, sans orphelin).
  --help                   Cette aide.

OPTIONS DE LANCEMENT
  --config PATH            YAML du run.   [défaut : configs/scoring/dictee_end2end.yaml]
  --model-name NAME        Surcharge model.name (étape 1 / unique étape).
  --model-stage2-name NAME Surcharge model_stage2.name (two_stage uniquement).
  --limit N                Run PARTIEL sur N copies (test). Désactive l'export S3.
  --passes N               Nombre de passes max, relance des échecs incluse. [défaut : 3]
  --no-export              Ne pas exporter vers S3 à la fin.
  --export                 Forcer l'export même avec --limit.
  --dest-prefix S3URI      Destination de l'export.   [défaut : $S3_PREDICTIONS_PREFIX]
  --force-export           Exporter même si le fichier local est plus petit que le
                           fichier déjà présent sur S3 (garde-fou anti-écrasement).
  --skip-checks            Sauter les appels de vérification (modèle, S3). Déconseillé.
  --foreground             Exécuter la chaîne dans le terminal (débogage, CI).

EXEMPLES
  launchers/launch_eval.sh
  launchers/launch_eval.sh --config configs/scoring/dictee_two_stage.yaml
  launchers/launch_eval.sh --model-name gemma4-26b-moe
  launchers/launch_eval.sh --limit 5 --skip-checks        # fumée, 5 copies
  launchers/launch_eval.sh --status
  launchers/launch_eval.sh --stop

Les sorties d'un run sont TOUTES nommées d'après <name>_<modèle(s)> :
  data/processed/<run>_predictions.jsonl     prédictions (checkpoint de reprise)
  data/processed/<run>_failed_copies.txt     copies en échec du dernier passage
  logs/<run>_<horodatage>.log                log du lancement (jamais écrasé)
  logs/<run>.latest.log                      lien vers le log du dernier lancement
EOF
}

# ── Analyse des arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)            CONFIG="$2"; shift 2 ;;
    --model-name)        MODEL_NAME="$2"; shift 2 ;;
    --model-stage2-name) MODEL_STAGE2_NAME="$2"; shift 2 ;;
    --limit)             LIMIT_CLI="$2"; shift 2 ;;
    --passes)            PASSES="$2"; shift 2 ;;
    --no-export)         DO_EXPORT=0; EXPORT_EXPLICIT=1; shift ;;
    --export)            DO_EXPORT=1; EXPORT_EXPLICIT=1; shift ;;
    --dest-prefix)       DEST_PREFIX="$2"; shift 2 ;;
    --force-export)      FORCE_EXPORT=1; shift ;;
    --skip-checks)       SKIP_CHECKS=1; shift ;;
    --foreground)        FOREGROUND=1; shift ;;
    --status)            MODE="status"; shift ;;
    --stop)              MODE="stop"; shift ;;
    --chain-worker)      MODE="chain"; shift ;;          # interne
    --run-name)          RUN_NAME="$2"; shift 2 ;;       # interne
    --log)               LOG="$2"; shift 2 ;;            # interne
    --stamp)             STAMP="$2"; shift 2 ;;          # interne
    --help|-h)           usage; exit 0 ;;
    *) die "Option inconnue : $1  (voir --help)" ;;
  esac
done

# ── Résolution du nom du run — SOURCE UNIQUE DE VÉRITÉ ────────────────────────
# Le nom des fichiers de sortie est `run_output_name(config)`, c.-à-d. le champ
# `name` du YAML suffixé par le(s) modèle(s) EFFECTIVEMENT utilisés (surcharges
# --model-name comprises). C'est la seule façon de viser le bon checkpoint et de
# n'exporter que le fichier de CE run — jamais celui d'un autre modèle.
probe_config() {
  [[ -f $CONFIG ]] || die "Config introuvable : $CONFIG"
  local out
  if ! out=$(uv run python - "$CONFIG" "$MODEL_NAME" "$MODEL_STAGE2_NAME" <<'PY' 2>&1
import sys
from evaluation_dictee.config import load_config, override_model_names, run_output_name

cfg = load_config(sys.argv[1])
cfg = override_model_names(cfg, sys.argv[2] or None, sys.argv[3] or None)
print("RUN_NAME=" + run_output_name(cfg))
print("APPROACH=" + str(cfg.approach))
print("MODEL=" + cfg.model.name)
print("MODEL2=" + (cfg.model_stage2.name if cfg.model_stage2 else ""))
print("LIMIT=" + ("" if cfg.data.limit is None else str(cfg.data.limit)))
print("CONCURRENCY=" + str(cfg.concurrency))
print("LABELS=" + cfg.data.labels_path)
print("IMAGES=" + cfg.data.images_path)
PY
  ); then
    printf '%s\n' "$out" >&2
    die "Config illisible ou invalide : $CONFIG"
  fi
  local line key value
  while IFS= read -r line; do
    case "$line" in
      RUN_NAME=*|APPROACH=*|MODEL=*|MODEL2=*|LIMIT=*|CONCURRENCY=*|LABELS=*|IMAGES=*)
        key="${line%%=*}"; value="${line#*=}"
        printf -v "$key" '%s' "$value" ;;
    esac
  done <<<"$out"
  [[ -n $RUN_NAME ]] || die "Impossible de résoudre le nom du run depuis $CONFIG"
  set_paths
}

set_paths() {
  JSONL="data/processed/${RUN_NAME}_predictions.jsonl"
  FAILED="data/processed/${RUN_NAME}_failed_copies.txt"
  LOCK="${JSONL}.lock"
}

# ── Petits utilitaires ────────────────────────────────────────────────────────

# Nombre de copies DISTINCTES présentes dans le JSONL (une copie = ~83 lignes).
count_done() {
  local f="${1:-$JSONL}" n
  if [[ -s $f ]]; then
    n=$(grep -o '"copy_id"[[:space:]]*:[[:space:]]*"[^"]*"' "$f" | sort -u | wc -l) || n=0
    echo "${n// /}"
  else
    echo 0
  fi
}

# Le fichier de prédictions est-il verrouillé par un run en cours ?
# Retourne 0 (vrai) si un run le tient, 1 sinon. `flock -n` teste sans bloquer ;
# le verrou est relâché par le noyau à la mort du process, donc pas de faux positif
# après un crash. `9<>` ouvre sans tronquer (le verrou porte sur l'inode).
is_locked() {
  command -v flock >/dev/null 2>&1 || return 1
  [[ -e $LOCK ]] || return 1
  ( exec 9<>"$LOCK"; flock -n 9 ) && return 1 || return 0
}

# Motif identifiant le benchmark de CE run — et de lui seul. La config ne suffit
# pas : deux modèles sur la même config sont des runs distincts (fichiers de sortie
# distincts) qui ont le droit de tourner en même temps. La surcharge --model-name
# fait donc partie du motif, dans l'ordre exact où la chaîne passe les arguments.
bench_pattern() {
  local pat="run_benchmark.py --config ${CONFIG}"
  if [[ -n $MODEL_NAME ]];        then pat+=" --model-name ${MODEL_NAME}"; fi
  if [[ -n $MODEL_STAGE2_NAME ]]; then pat+=" --model-stage2-name ${MODEL_STAGE2_NAME}"; fi
  printf '%s' "$pat"
}

# Un run (chaîne ou benchmark) tourne-t-il déjà pour CE run ?
running_pids() {
  { pgrep -f -- "launch_eval.sh --chain-worker --run-name ${RUN_NAME}" || true
    pgrep -f -- "$(bench_pattern)" || true
  } | sort -u
}

human_duration() {
  local s=$1
  if (( s < 0 )); then echo "?"; return; fi
  printf '%dh %02dmin' $(( s / 3600 )) $(( (s % 3600) / 60 ))
}

latest_log() {
  ls -1t "logs/${RUN_NAME}_"*.log 2>/dev/null | head -1 || true
}

# Rappel des surcharges, pour que les commandes suggérées visent bien CE run.
run_flags() {
  local f="--config ${CONFIG}"
  if [[ -n $MODEL_NAME ]];        then f+=" --model-name ${MODEL_NAME}"; fi
  if [[ -n $MODEL_STAGE2_NAME ]]; then f+=" --model-stage2-name ${MODEL_STAGE2_NAME}"; fi
  printf '%s' "$f"
}

# ── Vérifications préalables ──────────────────────────────────────────────────
preflight() {
  step "1. Vérifications préalables"

  command -v uv >/dev/null 2>&1 || die "\`uv\` introuvable. Voir README (uv sync)."
  [[ -f scripts/run_benchmark.py ]] || die "Lancé hors du dépôt ? scripts/run_benchmark.py absent."
  ok "dépôt : $PROJECT_ROOT"
  ok "config : $CONFIG  (approche $APPROACH, concurrency $CONCURRENCY)"
  if [[ -n $MODEL2 ]]; then
    ok "modèles : $MODEL (étape 1) + $MODEL2 (étape 2)"
  else
    ok "modèle : $MODEL"
  fi
  ok "run : $RUN_NAME"

  mkdir -p logs data/processed
  ok "dossiers logs/ et data/processed/ prêts"

  # Échantillon complet attendu : un `limit` oublié dans le YAML produirait un run
  # partiel qui se ferait passer pour complet (et écraserait l'export S3 du run complet).
  if [[ -n $LIMIT_CLI ]]; then
    warn "run PARTIEL demandé : $LIMIT_CLI copies (--limit). Même checkpoint que le run complet."
    if (( EXPORT_EXPLICIT == 0 )); then
      DO_EXPORT=0
      warn "export S3 désactivé (un JSONL partiel ne doit pas écraser l'export du run complet)."
    fi
  elif [[ -n $LIMIT ]]; then
    die "$CONFIG contient data.limit: $LIMIT — ce run ne traiterait que $LIMIT copies.
    Pour l'échantillon complet, mettre \`limit: null\` dans le YAML.
    Pour un test délibéré, relancer avec --limit $LIMIT."
  else
    ok "data.limit: null → échantillon complet"
  fi

  # Un seul run par fichier de sortie (le pipeline pose un flock ; on le constate ici).
  local pids; pids="$(running_pids)"
  if [[ -n $pids ]]; then
    printf '%s\n' "$pids" | while read -r p; do
      [[ -n $p ]] && ps -o pid=,etime=,args= -p "$p" 2>/dev/null || true
    done
    die "Un run est DÉJÀ en cours pour $RUN_NAME (PID ci-dessus).
    Deux runs sur le même JSONL dupliquent les copies et faussent les métriques.
    Avancement : launchers/launch_eval.sh --status $(run_flags)
    Arrêt      : launchers/launch_eval.sh --stop   $(run_flags)"
  fi
  if is_locked; then
    die "Le verrou $LOCK est tenu par un autre process (run lancé autrement ?).
    Vérifier : ps -ef | grep run_benchmark"
  fi
  ok "aucun run concurrent sur $(basename "$JSONL")"

  # État de la reprise : combien de copies sont déjà au chaud ?
  local done_n; done_n="$(count_done)"
  if (( done_n > 0 )); then
    ok "reprise : $done_n copies déjà présentes dans $JSONL (elles seront sautées)"
  else
    info "aucun checkpoint : le run part de zéro"
  fi

  if (( SKIP_CHECKS == 1 )); then
    warn "vérifications secrets/modèle/S3 sautées (--skip-checks)"
    return 0
  fi

  # 30 h de calcul méritent 20 s de vérification : secrets, endpoint modèle, S3.
  info "test des accès (secrets, modèle, S3)…"
  local out
  if ! out=$(uv run python - "$MODEL" "$LABELS" <<'PY' 2>&1
import sys
from evaluation_dictee.config import Secrets

model, labels_path = sys.argv[1], sys.argv[2]
s = Secrets()
if not s.llm_base_url:
    sys.exit("LLM_BASE_URL vide : configurer le Vault Onyxia ou .env (cf. README).")
print(f"endpoint : {s.llm_base_url}")

from openai import OpenAI

client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=5
)
print(f"modèle {model} : joignable")

from evaluation_dictee.data.loaders import load_labels

print(f"labels : {len(load_labels(labels_path))} copies dans {labels_path}")
PY
  ); then
    printf '%s\n' "$out" >&2
    die "Vérifications échouées — corriger avant de lancer 30 h de calcul.
    Secrets : Vault Onyxia (Mon compte > Vault) ou .env, cf. README §2.
    Pour passer outre : --skip-checks"
  fi
  while IFS= read -r line; do
    if [[ -n $line ]]; then ok "$line"; fi
  done <<<"$out"
}

# ── Lancement ─────────────────────────────────────────────────────────────────
launch() {
  probe_config
  preflight

  STAMP="$(date +%Y%m%d-%H%M%S)"
  LOG="logs/${RUN_NAME}_${STAMP}.log"
  : >"$LOG"
  ln -sfn "$(basename "$LOG")" "logs/${RUN_NAME}.latest.log"

  local -a worker=( "$SELF" --chain-worker --run-name "$RUN_NAME"
                    --config "$CONFIG" --passes "$PASSES" --log "$LOG" --stamp "$STAMP" )
  if [[ -n $MODEL_NAME ]];        then worker+=( --model-name "$MODEL_NAME" ); fi
  if [[ -n $MODEL_STAGE2_NAME ]]; then worker+=( --model-stage2-name "$MODEL_STAGE2_NAME" ); fi
  if [[ -n $LIMIT_CLI ]];         then worker+=( --limit "$LIMIT_CLI" ); fi
  if [[ -n $DEST_PREFIX ]];       then worker+=( --dest-prefix "$DEST_PREFIX" ); fi
  if (( DO_EXPORT == 0 ));        then worker+=( --no-export ); fi
  if (( FORCE_EXPORT == 1 ));     then worker+=( --force-export ); fi

  local EXPORT_LABEL
  if (( DO_EXPORT == 1 )); then
    EXPORT_LABEL="oui, à la fin du run (nom : ${RUN_NAME}_predictions.jsonl)"
  elif [[ -n $LIMIT_CLI ]]; then
    EXPORT_LABEL="non — run partiel (--limit). Forcer avec --export."
  else
    EXPORT_LABEL="non (--no-export)"
  fi

  step "2. Lancement"
  if (( FOREGROUND == 1 )); then
    warn "--foreground : le run meurt avec ce terminal. Ctrl+C l'interrompt."
    bash "${worker[@]}" 2>&1 | tee -a "$LOG"
    return
  fi

  # setsid : nouvelle session, donc AUCUN signal du terminal (SIGHUP à la fermeture
  # de l'onglet, SIGINT d'un Ctrl+C) n'atteint le run. nohup en renfort si setsid
  # est absent. </dev/null : le worker ne doit jamais attendre une saisie.
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup bash "${worker[@]}" >>"$LOG" 2>&1 </dev/null &
  else
    nohup bash "${worker[@]}" >>"$LOG" 2>&1 </dev/null &
    warn "setsid absent : repli sur nohup seul (protégé du SIGHUP, pas du groupe de process)."
  fi
  disown 2>/dev/null || true

  # Laisser au run le temps d'annoncer sa reprise, et le vérifier (README : un run
  # qui annonce « 0 déjà faites » alors qu'un checkpoint existe vise le mauvais fichier).
  info "démarrage en cours…"
  local i
  for i in $(seq 1 60); do
    if grep -q "copies au total" "$LOG" 2>/dev/null; then break; fi
    if ! running_pids | grep -q . && (( i > 5 )); then break; fi
    sleep 2
  done

  step "3. État du démarrage"
  if grep -q "copies au total" "$LOG" 2>/dev/null; then
    # Cette seule ligne porte les trois nombres qui comptent : total, à traiter,
    # déjà faites. « 0 déjà faites » alors qu'un checkpoint existe = mauvais fichier.
    ok "$(grep -h "copies au total" "$LOG" | tail -1 | tr -s ' ')"
  elif [[ -n "$(running_pids)" ]]; then
    warn "pas encore de ligne « copies au total » (chargement S3 en cours). Le run tourne."
  else
    tail -20 "$LOG" >&2
    die "Le run s'est arrêté immédiatement — voir le log ci-dessus : $LOG"
  fi

  cat <<EOF

${B}Run lancé — il survit à la fermeture de cette session.${N}
  run          ${RUN_NAME}
  log          ${LOG}   (alias : logs/${RUN_NAME}.latest.log)
  prédictions  ${JSONL}
  passes       ${PASSES} max (relance automatique des copies en échec)
  export S3    ${EXPORT_LABEL}

${B}Suivre / contrôler${N}
  launchers/launch_eval.sh --status $(run_flags)
  launchers/launch_eval.sh --stop   $(run_flags)
  tail -f ${LOG}

Reprise : si le run est interrompu, relancer exactement la même commande —
les copies déjà écrites sont sautées.
EOF
}

# ── Chaîne exécutée en arrière-plan : passes + relance des échecs + export ─────
chain() {
  set_paths
  local -a base=( uv run scripts/run_benchmark.py --config "$CONFIG" )
  if [[ -n $MODEL_NAME ]];        then base+=( --model-name "$MODEL_NAME" ); fi
  if [[ -n $MODEL_STAGE2_NAME ]]; then base+=( --model-stage2-name "$MODEL_STAGE2_NAME" ); fi
  if [[ -n $LIMIT_CLI ]];         then base+=( --limit "$LIMIT_CLI" ); fi

  echo "#RUN_META run_name=${RUN_NAME} config=${CONFIG} stamp=${STAMP} jsonl=${JSONL}"
  echo "#CHAIN_START epoch=$(date +%s) date=$(date -Is)"
  echo "commande : ${base[*]}"

  local pass=1 rc=0 last_rc=0 done_before done_after todo prev_todo=-1 mark
  while (( pass <= PASSES )); do
    done_before="$(count_done)"
    echo
    echo "#PASS_START pass=${pass} epoch=$(date +%s) done=${done_before}"
    echo "════════ PASSE ${pass}/${PASSES} — $(date -Is) — ${done_before} copies déjà faites ════════"
    mark=$(wc -l <"$LOG")

    rc=0
    "${base[@]}" || rc=$?
    last_rc=$rc
    done_after="$(count_done)"
    echo "════════ PASSE ${pass} terminée (code ${rc}) — ${done_after} copies au total ════════"

    # Les copies en échec du passage sont archivées : le pipeline réécrit
    # failed_copies.txt en mode « w » et ne le vide jamais, donc sans archive
    # horodatée on ne saurait plus quelle passe a produit quel échec.
    if [[ -s $FAILED ]]; then
      cp -f "$FAILED" "logs/${RUN_NAME}_${STAMP}_failed_pass${pass}.txt"
      echo "échecs de la passe ${pass} archivés : logs/${RUN_NAME}_${STAMP}_failed_pass${pass}.txt"
    fi

    # « N à traiter » annoncé au DÉBUT de cette passe : c'est le reliquat que la
    # passe précédente n'a pas su traiter. 0 ⇒ plus rien à faire.
    todo=$(tail -n +"$((mark + 1))" "$LOG" | grep -oE '[0-9]+ à traiter' | tail -1 | grep -oE '^[0-9]+' || true)
    todo="${todo:-inconnu}"

    if (( rc == 0 )) && ! tail -n +"$((mark + 1))" "$LOG" | grep -q "copie(s) en échec"; then
      echo "#CHAIN_DONE reason=complete pass=${pass}"
      break
    fi
    if (( rc != 0 )) && [[ $done_after == "$done_before" ]]; then
      echo "#CHAIN_DONE reason=echec_sans_progres pass=${pass} rc=${rc}"
      echo "La passe a échoué sans traiter la moindre copie : inutile de réessayer."
      break
    fi
    if [[ $todo != "inconnu" && $todo == "$prev_todo" ]]; then
      echo "#CHAIN_DONE reason=echecs_persistants pass=${pass} restant=${todo}"
      echo "${todo} copies échouent de façon reproductible : à diagnostiquer à la main."
      break
    fi
    prev_todo="$todo"
    if (( pass == PASSES )); then
      echo "#CHAIN_DONE reason=passes_epuisees pass=${pass}"
      break
    fi
    echo "Copies restantes en échec → relance (passe $((pass + 1))/${PASSES}) dans 60 s…"
    sleep 60
    pass=$((pass + 1))
  done

  # ── Bilan ───────────────────────────────────────────────────────────────────
  echo
  echo "════════ BILAN — $(date -Is) ════════"
  echo "run           : ${RUN_NAME}"
  echo "copies faites : $(count_done)"
  echo "prédictions   : ${JSONL}"
  if [[ -s $FAILED ]]; then
    echo "échecs restants (dernier passage) : $(wc -l <"$FAILED") — voir ${FAILED}"
  else
    echo "échecs restants : aucun"
  fi
  grep -h "Accord brut\|Kappa de Cohen\|copies non transcrites" "$LOG" | tail -3 || true

  # ── Export S3 ───────────────────────────────────────────────────────────────
  if (( DO_EXPORT == 0 )); then
    echo "#EXPORT skipped=--no-export"
    echo "Export S3 non demandé. Pour l'exporter plus tard :"
    echo "  uv run scripts/export_predictions.py --run-name ${RUN_NAME}"
    return 0
  fi
  if (( last_rc != 0 )); then
    echo "#EXPORT skipped=run_incomplet rc=${last_rc}"
    echo "Le run s'est terminé en erreur : pas d'export automatique (un JSONL"
    echo "incomplet écraserait un export S3 plus complet). Après diagnostic :"
    echo "  uv run scripts/export_predictions.py --run-name ${RUN_NAME}"
    return 0
  fi

  echo
  echo "════════ EXPORT S3 ════════"
  # Garde-fou : la destination est TOUJOURS <run_name>_predictions.jsonl, donc
  # jamais le fichier d'un autre run/modèle. Reste le risque d'écraser SON PROPRE
  # export par une version plus courte (run partiel, JSONL tronqué) : on refuse.
  local guard grc=0
  guard=$(uv run python - "$JSONL" "${DEST_PREFIX}" <<'PY' 2>&1
import os
import sys

import fsspec

from evaluation_dictee.config import Secrets

local = sys.argv[1]
prefix = (sys.argv[2] or Secrets().s3_predictions_prefix).rstrip("/")
dest = f"{prefix}/{os.path.basename(local)}"
local_size = os.path.getsize(local)
print(f"DEST={dest}")
print(f"LOCAL_SIZE={local_size}")
fs, path = fsspec.core.url_to_fs(dest)
if fs.exists(path):
    remote_size = fs.info(path)["size"]
    print(f"REMOTE_SIZE={remote_size}")
    if local_size < remote_size:
        sys.exit(3)
else:
    print("REMOTE_SIZE=absent")
PY
  ) || grc=$?
  echo "$guard"
  if (( grc == 3 )); then
    echo "#EXPORT skipped=plus_petit_que_S3"
    echo "Le fichier local est PLUS PETIT que celui déjà sur S3 : export refusé"
    echo "pour ne pas écraser un run plus complet. Forcer avec --force-export."
    (( FORCE_EXPORT == 1 )) || return 0
    echo "--force-export : on exporte quand même."
  elif (( grc != 0 )); then
    echo "#EXPORT skipped=verification_impossible rc=${grc}"
    echo "Accès S3 non vérifiable : export manuel après diagnostic —"
    echo "  uv run scripts/export_predictions.py --run-name ${RUN_NAME}"
    return 0
  fi

  local -a export_cmd=( uv run scripts/export_predictions.py --run-name "$RUN_NAME" )
  if [[ -n $DEST_PREFIX ]]; then export_cmd+=( --dest-prefix "$DEST_PREFIX" ); fi
  echo "commande : ${export_cmd[*]}"
  if "${export_cmd[@]}"; then
    echo "#EXPORT ok"
  else
    echo "#EXPORT failed"
  fi
  echo "════════ FIN — $(date -Is) ════════"
}

# ── Avancement ────────────────────────────────────────────────────────────────
status() {
  probe_config
  local log; log="$(latest_log)"
  local done_n total pids
  done_n="$(count_done)"
  pids="$(running_pids)"

  step "Run ${RUN_NAME}"
  if [[ -n $pids ]]; then
    ok "en cours :"
    printf '%s\n' "$pids" | while read -r p; do
      [[ -n $p ]] && ps -o pid=,etime=,args= -p "$p" 2>/dev/null | sed 's/^/    /' || true
    done
  else
    warn "aucun process en cours pour ce run."
  fi

  if [[ -z $log ]]; then
    warn "aucun log dans logs/ pour ce run (jamais lancé via launch_eval.sh ?)"
  else
    info "log : $log"
  fi

  total=""
  if [[ -n $log ]]; then
    total=$(grep -hoE '[0-9]+ copies au total' "$log" | tail -1 | grep -oE '^[0-9]+' || true)
  fi
  if [[ -n $total && $total -gt 0 ]]; then
    printf '  copies      %s/%s (%d%%)\n' "$done_n" "$total" $(( done_n * 100 / total ))
  else
    printf '  copies      %s (total inconnu tant que le chargement S3 n’a pas fini)\n' "$done_n"
  fi

  # Débit et ETA depuis le marqueur de la passe en cours (posé par la chaîne).
  if [[ -n $log ]]; then
    local pstart pdone now elapsed made rate eta
    pstart=$(grep -h '^#PASS_START' "$log" | tail -1 || true)
    if [[ -n $pstart ]]; then
      # shellcheck disable=SC2001
      pdone=$(sed 's/.*done=\([0-9]*\).*/\1/' <<<"$pstart")
      # shellcheck disable=SC2001
      pstart=$(sed 's/.*epoch=\([0-9]*\).*/\1/' <<<"$pstart")
      now=$(date +%s); elapsed=$(( now - pstart )); made=$(( done_n - pdone ))
      if (( elapsed > 60 && made > 0 )); then
        rate=$(awk -v m="$made" -v e="$elapsed" 'BEGIN{printf "%.1f", m*3600/e}')
        printf '  débit       %s copies/h (depuis %s)\n' "$rate" "$(human_duration "$elapsed")"
        if [[ -n $total && $total -gt 0 ]]; then
          eta=$(awk -v r="$total" -v d="$done_n" -v m="$made" -v e="$elapsed" \
                    'BEGIN{printf "%d", (r-d)*e/m}')
          printf '  fin estimée %s (dans %s)\n' "$(date -d "+${eta} seconds" '+%F %H:%M' 2>/dev/null || echo '?')" "$(human_duration "$eta")"
        fi
      else
        info "débit non mesurable pour l’instant (passe trop récente)"
      fi
    fi
  fi

  if [[ -f $JSONL ]]; then
    printf '  dernière écriture  %s\n' "$(date -r "$JSONL" '+%F %H:%M:%S')"
  fi
  if [[ -s $FAILED ]]; then
    warn "$(wc -l <"$FAILED") copie(s) en échec au dernier passage — $FAILED"
  fi
  if [[ -n $log ]] && grep -q '^#CHAIN_DONE' "$log"; then
    info "$(grep -h '^#CHAIN_DONE' "$log" | tail -1)"
    # `|| true` : la chaîne peut avoir fini le run sans avoir encore écrit sa ligne
    # d'export ; avec `pipefail`, un grep bredouille tuerait --status.
    { grep -h '^#EXPORT' "$log" || true; } | tail -1 | while IFS= read -r l; do
      if [[ -n $l ]]; then info "$l"; fi
    done
  fi

  if [[ -n $log ]]; then
    step "Dernières lignes du log"
    tail -5 "$log" | sed 's/^/    /'
  fi
}

# ── Arrêt propre ──────────────────────────────────────────────────────────────
stop() {
  probe_config
  step "Arrêt du run ${RUN_NAME}"

  # ORDRE IMPORTANT : la chaîne d'abord, sinon elle relance une passe dès que le
  # benchmark meurt. Puis `pkill -f` sur le motif de la config, qui tue d'un coup
  # le wrapper `uv run` ET le `python3 scripts/run_benchmark.py` — un kill par PID
  # laisserait l'enfant orphelin continuer d'écrire dans le JSONL.
  if pkill -f -- "launch_eval.sh --chain-worker --run-name ${RUN_NAME}"; then
    ok "chaîne de relance arrêtée"
  else
    info "aucune chaîne de relance en cours"
  fi
  if pkill -f -- "$(bench_pattern)"; then
    ok "benchmark arrêté (wrapper uv + python)"
  else
    info "aucun benchmark en cours pour $CONFIG"
  fi

  sleep 3
  local leftover; leftover="$(running_pids)"
  if [[ -n $leftover ]]; then
    warn "process encore vivants, insistance (SIGKILL) :"
    printf '%s\n' "$leftover" | while read -r p; do
      [[ -n $p ]] && ps -o pid=,args= -p "$p" 2>/dev/null | sed 's/^/    /' || true
    done
    pkill -9 -f -- "$(bench_pattern)" || true
  fi

  ok "arrêt terminé — $(count_done) copies conservées dans $JSONL"
  info "reprise : relancer exactement la même commande de lancement."
}

case "$MODE" in
  launch) launch ;;
  status) status ;;
  stop)   stop ;;
  chain)  chain ;;
esac
