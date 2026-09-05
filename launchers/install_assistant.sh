#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  install_assistant.sh — installe et configure Claude Code et/ou openCode
# ═══════════════════════════════════════════════════════════════════════════════
#
#  But : qu'un⋅e nouveau⋅elle arrivant⋅e sur le SSP Cloud puisse lancer le projet
#  avec l'assistant de son choix, y compris branché sur un serveur LLM interne
#  (llm.lab) plutôt que sur une API payante.
#
#  Deux outils, deux façons de brancher un endpoint — la différence n'est pas un
#  détail, et ce script ne la masque pas :
#
#    • openCode   parle nativement les API **compatibles OpenAI**. Un endpoint
#                 vLLM / Open WebUI comme llm.lab fonctionne directement : le
#                 script écrit un provider dans opencode.json et découvre les
#                 modèles servis.
#    • Claude Code parle l'**API Anthropic Messages**. `ANTHROPIC_BASE_URL` route
#                 vers un proxy ou une passerelle, mais celle-ci DOIT parler ce
#                 protocole. Pointer Claude Code directement sur un endpoint
#                 OpenAI-compatible ne marche pas — il faut une passerelle de
#                 traduction (LiteLLM, claude-code-router…). Le script teste
#                 l'endpoint et refuse d'écrire une config qui échouerait en
#                 silence.
#
#  Aucun secret n'est écrit sur disque : les configs référencent une VARIABLE
#  d'environnement (`{env:LLM_API_KEY}` pour openCode, `${LLM_API_KEY}` pour le
#  profil shell), alimentée par le Vault Onyxia.
#
#  Usage :
#     launchers/install_assistant.sh                       # les deux, sans endpoint
#     launchers/install_assistant.sh opencode --endpoint https://llm.lab.sspcloud.fr/api/v1
#     launchers/install_assistant.sh --check               # ne touche à rien, teste
#
#  Voir `launchers/install_assistant.sh --help`.
# ═══════════════════════════════════════════════════════════════════════════════

set -Eeuo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"

# ── Valeurs par défaut ────────────────────────────────────────────────────────
TARGET="both"                 # claude | opencode | both
ENDPOINT="${LLM_BASE_URL:-}"  # repris de l'environnement Onyxia s'il y est
KEY_VAR="LLM_API_KEY"
PROVIDER_ID="llmlab"
MODELS=""                     # liste explicite ; sinon découverte via /models
DEFAULT_MODEL=""
CLAUDE_MODEL=""            # modèle servi par l'endpoint, pour Claude Code
CLAUDE_CONTEXT=""          # fenêtre de contexte réelle de ce modèle, en tokens
OPENCODE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
SHELL_PROFILE="$HOME/.bashrc"
WRITE_PROFILE=1
DO_INSTALL=1
DO_CONFIG=1
DRY_RUN=0
FORCE=0
NPM_PKG="@ai-sdk/openai-compatible"

# ── Affichage ─────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  B=$'\033[1m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; N=$'\033[0m'
else
  B=""; R=""; G=""; Y=""; C=""; N=""
fi
ok()   { printf '%s✔%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "$*"; }
info() { printf '%s·%s %s\n' "$C" "$N" "$*"; }
step() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
die()  { printf '%s✖%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
install_assistant.sh — installe et configure Claude Code et/ou openCode.

CIBLE (premier argument, optionnel)
  claude | opencode | both        [défaut : both]

ENDPOINT
  --endpoint URL         Serveur LLM à utiliser.  [défaut : $LLM_BASE_URL]
                         openCode : endpoint compatible OpenAI (ex. llm.lab).
                         Claude Code : passerelle parlant l'API Anthropic Messages.
  --api-key-var VAR      Variable d'env contenant la clé.  [défaut : LLM_API_KEY]
                         Seul le NOM est écrit dans les configs, jamais la valeur.
  --models a,b,c         Modèles à déclarer.  [défaut : découverts via <endpoint>/models]
  --default-model NAME   Modèle par défaut openCode.  [défaut : le premier de la liste]
  --claude-model NAME    Modèle que Claude Code demandera à l'endpoint.
                         [défaut : --default-model, sinon le premier découvert]
  --claude-context N     Fenêtre de contexte réelle du modèle, en tokens. Sans elle,
                         Claude Code suppose 200 000 et le dit à chaque démarrage.
  --provider-id ID       Identifiant du provider openCode.  [défaut : llmlab]
  --npm PKG              Paquet AI SDK du provider openCode.
                         [défaut : @ai-sdk/openai-compatible ; utiliser
                          @ai-sdk/openai pour un endpoint de type /v1/responses]

MODES
  --check                Ne rien installer ni écrire : tester l'endpoint et sortir.
  --print                Afficher les configs qui SERAIENT écrites, sans rien écrire.
  --no-install           Ne pas installer les binaires (déjà présents).
  --no-config            Installer seulement, ne rien configurer.
  --force                Réinstaller / écraser même si déjà présent ou incompatible.

CHEMINS
  --opencode-dir DIR     Dossier de config openCode.  [défaut : ~/.config/opencode]
  --shell-profile FILE   Profil shell à compléter.    [défaut : ~/.bashrc]
  --no-profile           Ne pas toucher au profil shell (afficher les exports).

EXEMPLES
  launchers/install_assistant.sh
  launchers/install_assistant.sh opencode --endpoint https://llm.lab.sspcloud.fr/api/v1
  launchers/install_assistant.sh --check --endpoint https://llm.lab.sspcloud.fr/api/v1
  launchers/install_assistant.sh claude --endpoint https://ma-passerelle-litellm/anthropic

NOTE — pourquoi les deux outils ne se configurent pas pareil
  openCode parle les API compatibles OpenAI : llm.lab fonctionne directement.
  Claude Code parle l'API Anthropic Messages : un endpoint OpenAI-compatible ne
  suffit PAS, il faut une passerelle de traduction devant (LiteLLM…). Le script
  teste l'endpoint et le dit clairement plutôt que d'écrire une config muette.
EOF
}

# ── Analyse des arguments ─────────────────────────────────────────────────────
case "${1:-}" in
  claude|opencode|both) TARGET="$1"; shift ;;
esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)       ENDPOINT="$2"; shift 2 ;;
    --api-key-var)    KEY_VAR="$2"; shift 2 ;;
    --models)         MODELS="$2"; shift 2 ;;
    --default-model)  DEFAULT_MODEL="$2"; shift 2 ;;
    --claude-model)   CLAUDE_MODEL="$2"; shift 2 ;;
    --claude-context) CLAUDE_CONTEXT="$2"; shift 2 ;;
    --provider-id)    PROVIDER_ID="$2"; shift 2 ;;
    --npm)            NPM_PKG="$2"; shift 2 ;;
    --opencode-dir)   OPENCODE_DIR="$2"; shift 2 ;;
    --shell-profile)  SHELL_PROFILE="$2"; shift 2 ;;
    --no-profile)     WRITE_PROFILE=0; shift ;;
    --check)          DO_INSTALL=0; DO_CONFIG=0; shift ;;
    --print)          DRY_RUN=1; shift ;;
    --no-install)     DO_INSTALL=0; shift ;;
    --no-config)      DO_CONFIG=0; shift ;;
    --force)          FORCE=1; shift ;;
    --help|-h)        usage; exit 0 ;;
    *) die "Option inconnue : $1  (voir --help)" ;;
  esac
done

wants() { [[ $TARGET == both || $TARGET == "$1" ]]; }

# ── Sonde de l'endpoint ───────────────────────────────────────────────────────
# Deux protocoles, deux tests. On ne devine pas : on interroge.

DISCOVERED_MODELS=""
OPENAI_OK=0
ANTHROPIC_OK=0
ANTHROPIC_BASE=""
ANTHROPIC_SHAPE=""

# GET <base>/models — l'inventaire standard d'une API compatible OpenAI.
probe_openai() {
  local base="${ENDPOINT%/}" body key
  key="${!KEY_VAR:-}"
  body=$(curl -sS -m 20 -H "Authorization: Bearer ${key}" "${base}/models" 2>/dev/null) || return 1
  DISCOVERED_MODELS=$(printf '%s' "$body" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(1)
data = payload.get("data") if isinstance(payload, dict) else None
if not isinstance(data, list):
    sys.exit(1)
ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
if not ids:
    sys.exit(1)
print(",".join(ids))
') || return 1
  return 0
}

# Claude Code construit lui-même l'URL « <ANTHROPIC_BASE_URL>/v1/messages ». Ce
# qu'on cherche n'est donc pas une route, c'est la BASE qui, suffixée de
# /v1/messages, tombe juste — écrire l'endpoint tel quel donnerait .../v1/v1/messages.
# On teste donc les bases plausibles : l'endpoint, et l'endpoint privé de son /v1.
#
# Et le code HTTP ne prouve rien : une passerelle Open WebUI répond
# « 400 Model not found » sur n'importe quelle route. Le seul discriminant fiable
# est la FORME de la réponse — l'API Anthropic renvoie soit un message
# ({"type":"message","content":[...]}), soit une erreur enveloppée
# ({"type":"error","error":{"type":...}}). Un {"detail": "..."} est autre chose.
probe_anthropic() {
  local endpoint="${ENDPOINT%/}" key base body verdict model
  local -a bases=("$endpoint")
  if [[ $endpoint == */v1 ]]; then bases+=("${endpoint%/v1}"); fi
  key="${!KEY_VAR:-}"
  model="${DISCOVERED_MODELS%%,*}"
  model="${model:-claude-opus-5}"

  for base in "${bases[@]}"; do
    body=$(curl -sS -m 30 -X POST "${base}/v1/messages" \
      -H 'content-type: application/json' \
      -H 'anthropic-version: 2023-06-01' \
      -H "x-api-key: ${key}" \
      -H "authorization: Bearer ${key}" \
      -d "{\"model\":\"${model}\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
      2>/dev/null) || continue
    verdict=$(printf '%s' "$body" | python3 -c '
import json, sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(1)
if not isinstance(payload, dict):
    sys.exit(1)
# Réponse complète : le protocole est parlé ET la requête a abouti.
if payload.get("type") == "message" and isinstance(payload.get("content"), list):
    print("repond")
    sys.exit(0)
# Erreur enveloppée à la mode Anthropic : le protocole est parlé, la requête non.
error = payload.get("error")
if payload.get("type") == "error" and isinstance(error, dict) and error.get("type"):
    print("protocole")
    sys.exit(0)
sys.exit(1)
') || continue
    ANTHROPIC_BASE="$base"
    ANTHROPIC_SHAPE="$verdict"
    return 0
  done
  return 1
}

probe_endpoint() {
  step "Endpoint"
  if [[ -z $ENDPOINT ]]; then
    info "aucun endpoint fourni (--endpoint absent, \$LLM_BASE_URL vide)."
    info "les assistants garderont leur configuration par défaut (API en ligne)."
    return 0
  fi
  info "endpoint : $ENDPOINT"
  if [[ -n ${!KEY_VAR:-} ]]; then
    ok "clé lue dans \$$KEY_VAR (non affichée, jamais écrite sur disque)"
  else
    warn "\$$KEY_VAR est vide — les tests partiront sans jeton ; sur Onyxia, injecter le secret par le Vault."
  fi

  if probe_openai; then
    OPENAI_OK=1
    local n; n=$(awk -F, '{print NF}' <<<"$DISCOVERED_MODELS")
    ok "API compatible OpenAI : $n modèle(s) servis — openCode peut s'y brancher directement"
    info "modèles : $(cut -c1-160 <<<"$DISCOVERED_MODELS")"
  else
    warn "pas de réponse exploitable sur ${ENDPOINT%/}/models"
    warn "si l'endpoint est une API OpenAI, vérifier le suffixe (…/v1) et la clé."
  fi

  if probe_anthropic; then
    ANTHROPIC_OK=1
    if [[ $ANTHROPIC_SHAPE == repond ]]; then
      ok "API Anthropic Messages : ${ANTHROPIC_BASE}/v1/messages répond — Claude Code peut s'y brancher"
    else
      ok "API Anthropic Messages : ${ANTHROPIC_BASE}/v1/messages parle le protocole (requête refusée : modèle ou clé)"
      warn "vérifier le modèle et la clé avant de compter dessus."
    fi
    if [[ $ANTHROPIC_BASE != "${ENDPOINT%/}" ]]; then
      info "ANTHROPIC_BASE_URL sera « $ANTHROPIC_BASE » : Claude Code ajoute /v1/messages lui-même."
    fi
  else
    info "pas d'API Anthropic Messages ici — attendu pour un serveur OpenAI/vLLM."
    info "Claude Code ne pourra donc PAS s'y brancher sans passerelle de traduction."
  fi
}

# ── Installation des binaires ─────────────────────────────────────────────────
# On télécharge l'installeur officiel dans un fichier temporaire avant de
# l'exécuter, plutôt que `curl | bash` : la taille est affichée et le script
# reste inspectable en cas de doute.
install_from() {
  local name="$1" url="$2" tmp
  tmp=$(mktemp "/tmp/install-${name}.XXXXXX.sh")
  info "téléchargement de l'installeur officiel : $url"
  if ! curl -fsSL -m 120 "$url" -o "$tmp"; then
    rm -f "$tmp"
    warn "téléchargement impossible (réseau ? proxy ?)"
    return 1
  fi
  ok "installeur reçu ($(wc -c <"$tmp") octets) : $tmp"
  if (( DRY_RUN == 1 )); then
    info "--print : installeur NON exécuté"
    rm -f "$tmp"; return 0
  fi
  bash "$tmp" || { rm -f "$tmp"; return 1; }
  rm -f "$tmp"
  return 0
}

install_tool() {
  local name="$1" bin="$2" url="$3" npm_name="$4" current=""
  if command -v "$bin" >/dev/null 2>&1; then
    current=$("$bin" --version 2>/dev/null | head -1 || true)
    if (( FORCE == 0 )); then
      ok "$name déjà installé ($(command -v "$bin")${current:+ — $current}) — rien à faire"
      return 0
    fi
    info "$name déjà installé ($current) mais --force demandé : réinstallation"
  fi
  if install_from "$name" "$url"; then
    ok "$name installé"
  elif command -v npm >/dev/null 2>&1; then
    warn "repli sur npm : npm install -g $npm_name"
    (( DRY_RUN == 1 )) || npm install -g "$npm_name"
  else
    warn "$name non installé : ni l'installeur officiel ni npm ne sont utilisables."
    warn "installer manuellement — $url"
    return 1
  fi
}

# ── Configuration openCode ────────────────────────────────────────────────────
write_opencode_config() {
  local models="$1" default_model="$2" target="$OPENCODE_DIR/opencode.json"

  # Un opencode.jsonc existant est la config réelle : c'est lui qu'on complète.
  if [[ -f "$OPENCODE_DIR/opencode.jsonc" && ! -f "$target" ]]; then
    target="$OPENCODE_DIR/opencode.jsonc"
  fi

  local rendered
  rendered=$(python3 - "$target" "$PROVIDER_ID" "$ENDPOINT" "$KEY_VAR" "$models" "$default_model" "$NPM_PKG" 2>&1 <<'PY'
import json
import os
import re
import sys

target, provider_id, base_url, key_var, models_csv, default_model, npm_pkg = sys.argv[1:8]

config = {}
if os.path.isfile(target):
    raw = open(target, encoding="utf-8").read()
    # Un .jsonc peut porter des commentaires : on les retire pour relire la config
    # existante, et on la réécrit en JSON strict (que les deux extensions acceptent).
    stripped = re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE)
    try:
        config = json.loads(stripped) if stripped.strip() else {}
    except json.JSONDecodeError as exc:
        sys.exit(f"CONFIG_ILLISIBLE:{exc}")

config.setdefault("$schema", "https://opencode.ai/config.json")
providers = config.setdefault("provider", {})
providers[provider_id] = {
    "name": provider_id,
    "npm": npm_pkg,
    "options": {
        "baseURL": base_url.rstrip("/"),
        # Le NOM de la variable, jamais sa valeur : aucun secret sur disque.
        "apiKey": "{env:%s}" % key_var,
    },
    "models": {m: {} for m in models_csv.split(",") if m},
}
if default_model:
    config["model"] = f"{provider_id}/{default_model}"

print(json.dumps(config, indent=2, ensure_ascii=False))
PY
  ) || {
    case "$rendered" in
      CONFIG_ILLISIBLE:*)
        warn "config openCode existante illisible : ${rendered#CONFIG_ILLISIBLE:}"
        warn "elle n'est PAS écrasée. La corriger, ou la déplacer, puis relancer."
        return 1 ;;
      *) warn "génération de la config openCode impossible : $rendered"; return 1 ;;
    esac
  }

  if (( DRY_RUN == 1 )); then
    info "--print — config openCode qui serait écrite dans $target :"
    sed 's/^/    /' <<<"$rendered"
    return 0
  fi

  mkdir -p "$OPENCODE_DIR"
  if [[ -f $target ]]; then
    local backup="${target}.$(date +%Y%m%d-%H%M%S).bak"
    cp -p "$target" "$backup"
    ok "sauvegarde de la config existante : $backup"
  fi
  printf '%s\n' "$rendered" >"$target"
  ok "config openCode écrite : $target"
  info "provider « $PROVIDER_ID » → $ENDPOINT (clé lue dans \$$KEY_VAR au démarrage)"
  if [[ -n $default_model ]]; then info "modèle par défaut : $PROVIDER_ID/$default_model"; fi
  return 0
}

configure_opencode() {
  step "Configuration openCode"
  if [[ -z $ENDPOINT ]]; then
    info "pas d'endpoint : openCode garde sa configuration par défaut."
    return 0
  fi
  if (( OPENAI_OK == 0 && FORCE == 0 )); then
    warn "l'endpoint n'a pas répondu comme une API compatible OpenAI : config NON écrite."
    warn "vérifier l'URL (souvent le suffixe /v1) et la clé, puis relancer."
    warn "pour écrire quand même : --force."
    return 0
  fi

  local models="$MODELS"
  if [[ -z $models ]]; then
    models="$DISCOVERED_MODELS"
  fi
  if [[ -z $models ]]; then
    warn "aucun modèle connu (ni --models, ni découverte) : config NON écrite."
    return 0
  fi
  local default_model="$DEFAULT_MODEL"
  if [[ -z $default_model ]]; then
    default_model="${models%%,*}"
  fi
  write_opencode_config "$models" "$default_model" || return 0
}

# ── Configuration Claude Code ─────────────────────────────────────────────────
# Sur un endpoint maison, les identifiants de modèle sont ceux du serveur, pas ceux
# d'Anthropic : sans ces variables, Claude Code réclamerait « claude-opus-5 » et se
# ferait répondre « Model not found ». Les trois alias couvrent aussi les tâches de
# fond (résumés, titres), qui visent sinon un modèle Haiku inexistant ici.
claude_exports() {
  local base="${ANTHROPIC_BASE:-${ENDPOINT%/}}"
  cat <<EOF
export ANTHROPIC_BASE_URL="${base}"
export ANTHROPIC_AUTH_TOKEN="\${${KEY_VAR}:-}"
EOF
  if [[ -n $CLAUDE_MODEL ]]; then
    cat <<EOF
export ANTHROPIC_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${CLAUDE_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${CLAUDE_MODEL}"
EOF
  fi
  if [[ -n $CLAUDE_CONTEXT ]]; then
    printf 'export CLAUDE_CODE_MAX_CONTEXT_TOKENS="%s"\n' "$CLAUDE_CONTEXT"
  fi
}

configure_claude() {
  step "Configuration Claude Code"
  if [[ -z $ENDPOINT ]]; then
    info "pas d'endpoint : Claude Code garde l'API Anthropic par défaut"
    info "(authentification interactive au premier lancement : \`claude\`)."
    return 0
  fi

  if [[ -z $CLAUDE_MODEL ]]; then
    CLAUDE_MODEL="${DEFAULT_MODEL:-${DISCOVERED_MODELS%%,*}}"
  fi

  if (( ANTHROPIC_OK == 0 )); then
    warn "cet endpoint ne parle pas l'API Anthropic Messages."
    cat <<EOF
    Claude Code ne sait PAS parler une API OpenAI : ANTHROPIC_BASE_URL doit
    pointer vers une passerelle qui expose /v1/messages. Trois issues :
      1. laisser Claude Code sur l'API Anthropic (rien à faire) et réserver
         l'endpoint interne à openCode, qui le gère nativement ;
      2. monter une passerelle de traduction OpenAI → Anthropic (LiteLLM,
         claude-code-router) devant ${ENDPOINT%/}, puis relancer ce script
         avec --endpoint <url-de-la-passerelle> ;
      3. --force pour écrire quand même les variables (déconseillé : les appels
         échoueront en 404 sans que la cause soit évidente).
EOF
    (( FORCE == 1 )) || return 0
    warn "--force : variables écrites malgré tout."
  fi

  if (( DRY_RUN == 1 )); then
    info "--print — variables qui seraient ajoutées à $SHELL_PROFILE :"
    claude_exports | sed 's/^/    /'
    return 0
  fi
  if (( WRITE_PROFILE == 0 )); then
    info "--no-profile : ajouter ces lignes vous-même à votre shell —"
    claude_exports | sed 's/^/    /'
    return 0
  fi

  # Bloc délimité et remplacé à l'identique : relancer le script ne duplique rien.
  python3 - "$SHELL_PROFILE" "$(claude_exports)" <<'PY'
import os
import sys

profile, block = sys.argv[1], sys.argv[2]
start = "# >>> evaluation_dictee — endpoint Claude Code >>>"
end = "# <<< evaluation_dictee — endpoint Claude Code <<<"
new = f"{start}\n{block}\n{end}\n"

text = open(profile, encoding="utf-8").read() if os.path.isfile(profile) else ""
if start in text and end in text:
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    text = head + new + tail.lstrip("\n")
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + new
open(profile, "w", encoding="utf-8").write(text)
PY
  ok "variables ajoutées à $SHELL_PROFILE (bloc délimité, remplacé à chaque relance)"
  info "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE:-${ENDPOINT%/}}"
  if [[ -n $CLAUDE_MODEL ]]; then
    info "modèle demandé à l'endpoint : $CLAUDE_MODEL (alias opus/sonnet/haiku compris)"
  else
    warn "aucun modèle connu : Claude Code réclamera un modèle Anthropic que l'endpoint ne sert pas. Passer --claude-model."
  fi
  info "ANTHROPIC_AUTH_TOKEN lu dans \$$KEY_VAR au démarrage du shell — aucun jeton en clair"
  warn "ouvrir un nouveau terminal, ou : source $SHELL_PROFILE"
  cat <<EOF

    Deux avertissements attendus au premier lancement de \`claude\` :
      • « connectors are disabled … auth source is set » — normal : ces variables
        prennent le pas sur une connexion claude.ai. Pour revenir à un compte
        Anthropic, commenter le bloc dans $SHELL_PROFILE (ou \`unset ANTHROPIC_BASE_URL
        ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL\`).
      • « <modèle> isn't described by this version's model catalog » — normal pour
        un modèle maison : Claude Code suppose alors 200 000 tokens de contexte.
        Passer --claude-context <N> pour lui donner la vraie valeur.
EOF
}

# ── PATH ──────────────────────────────────────────────────────────────────────
ensure_path() {
  local dir="$1"
  [[ -d $dir ]] || return 0
  case ":$PATH:" in *":$dir:"*) return 0 ;; esac
  if (( WRITE_PROFILE == 0 || DRY_RUN == 1 )); then
    warn "$dir n'est pas dans le PATH — ajouter : export PATH=\"$dir:\$PATH\""
    return 0
  fi
  if ! grep -qF "$dir" "$SHELL_PROFILE" 2>/dev/null; then
    printf '\nexport PATH="%s:$PATH"\n' "$dir" >>"$SHELL_PROFILE"
    ok "$dir ajouté au PATH dans $SHELL_PROFILE"
  fi
}

# ── Bilan ─────────────────────────────────────────────────────────────────────
summary() {
  step "Bilan"
  if wants claude; then
    if command -v claude >/dev/null 2>&1; then
      ok "claude : $(command -v claude) — $(claude --version 2>/dev/null | head -1)"
    else
      warn "claude : non installé"
    fi
  fi
  if wants opencode; then
    if command -v opencode >/dev/null 2>&1; then
      ok "opencode : $(command -v opencode) — $(opencode --version 2>/dev/null | head -1)"
    else
      warn "opencode : non installé"
    fi
  fi
  cat <<EOF

${B}Et ensuite${N}
  cd ${PROJECT_ROOT}
  claude          # puis : « lance l'évaluation complète »  (skill .claude/skills/launch)
  opencode        # puis : /launch                          (.opencode/command/launch.md)

Le lancement lui-même ne dépend d'aucun assistant :
  launchers/launch_eval.sh --help
EOF
}

# ── Déroulé ───────────────────────────────────────────────────────────────────
command -v curl >/dev/null 2>&1 || die "curl est requis."
command -v python3 >/dev/null 2>&1 || die "python3 est requis (lecture/écriture des configs JSON)."

probe_endpoint

if (( DO_INSTALL == 1 )); then
  step "Installation"
  if wants claude; then
    install_tool "Claude Code" claude "https://claude.ai/install.sh" "@anthropic-ai/claude-code" || true
  fi
  if wants opencode; then
    install_tool "openCode" opencode "https://opencode.ai/install" "opencode-ai" || true
  fi
  ensure_path "$HOME/.local/bin"
  ensure_path "$HOME/.opencode/bin"
fi

if (( DO_CONFIG == 1 )); then
  if wants opencode; then configure_opencode; fi
  if wants claude;   then configure_claude;   fi
fi

if (( DO_INSTALL == 0 && DO_CONFIG == 0 )); then
  step "Rien n'a été installé ni écrit (mode --check, ou --no-install + --no-config)."
else
  summary
fi
