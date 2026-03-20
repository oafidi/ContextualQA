#!/bin/bash

# ============================================================
#  watchdog_qa.sh
#  Surveille datasets/processed/qa_types_3051_4575.json
#  toutes les minutes et relance qa_types_detector.py si besoin
# ============================================================

TARGET_FILE="datasets/processed/qa_types_7626_7863.json"
SCRIPT_NAME="qa_types_detector.py"
LINE_THRESHOLD=10600
CHECK_INTERVAL=60  # secondes

# ---------- couleurs ----------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

get_line_count() {
    if [[ ! -f "$TARGET_FILE" ]]; then
        echo -1
        return
    fi
    wc -l < "$TARGET_FILE"
}

kill_and_restart() {
    log "${YELLOW}  Aucun changement détecté et lignes < ${LINE_THRESHOLD}. Redémarrage...${NC}"

    # Tuer tous les processus python qui font tourner le script
    PIDS=$(pgrep -f "$SCRIPT_NAME")
    if [[ -n "$PIDS" ]]; then
        log "${RED} Kill PID(s) : $PIDS${NC}"
        kill $PIDS
        sleep 2
        # Force kill si toujours actif
        PIDS=$(pgrep -f "$SCRIPT_NAME")
        if [[ -n "$PIDS" ]]; then
            log "${RED} Force kill (SIGKILL) PID(s) : $PIDS${NC}"
            kill -9 $PIDS
        fi
    else
        log "${YELLOW}  Aucun processus '$SCRIPT_NAME' trouvé à tuer.${NC}"
    fi

    # Relancer en arrière-plan
    log "${GREEN} Lancement de : python $SCRIPT_NAME${NC}"
    nohup python "$SCRIPT_NAME" >> logs/qa_detector.log 2>&1 &
    log "${GREEN} Processus relancé avec PID $!${NC}"
}

# ---------- initialisation ----------
mkdir -p logs

if [[ ! -f "$TARGET_FILE" ]]; then
    log "${RED} Fichier introuvable : $TARGET_FILE${NC}"
    log "   Le watchdog continuera à surveiller jusqu'à ce qu'il apparaisse."
fi

PREV_COUNT=$(get_line_count)
log "${CYAN} Watchdog démarré. Lignes initiales : ${PREV_COUNT} | Seuil : ${LINE_THRESHOLD}${NC}"
log "${CYAN} Fichier surveillé : ${TARGET_FILE}${NC}"
log "${CYAN} Intervalle de vérification : ${CHECK_INTERVAL}s${NC}"

# ---------- boucle principale ----------
while true; do
    sleep "$CHECK_INTERVAL"

    CURRENT_COUNT=$(get_line_count)

    if [[ "$CURRENT_COUNT" -eq -1 ]]; then
        log "${RED} Fichier toujours introuvable. En attente...${NC}"
        PREV_COUNT=-1
        continue
    fi

    if [[ "$CURRENT_COUNT" -ne "$PREV_COUNT" ]]; then
        # --- Le fichier a changé ---
        DIFF=$(( CURRENT_COUNT - PREV_COUNT ))
        if [[ $DIFF -gt 0 ]]; then
            DIFF_STR="+${DIFF}"
        else
            DIFF_STR="${DIFF}"
        fi
        log "${GREEN} Changement détecté : ${PREV_COUNT} → ${CURRENT_COUNT} lignes (${DIFF_STR})${NC}"
        PREV_COUNT="$CURRENT_COUNT"

    else
        # --- Pas de changement ---
        log "${YELLOW} Pas de changement : ${CURRENT_COUNT} lignes.${NC}"

        if [[ "$CURRENT_COUNT" -gt "$LINE_THRESHOLD" ]]; then
            log "${CYAN} Seuil dépassé (${CURRENT_COUNT} > ${LINE_THRESHOLD}). Fin du watchdog.${NC}"
            exit 0
        else
            log "${RED} Sous le seuil (${CURRENT_COUNT} ≤ ${LINE_THRESHOLD}).${NC}"
            kill_and_restart
        fi
    fi
done