#!/bin/bash

TARGET_FILE="datasets/processed/qa_types_7626_7863.json"
SCRIPT_NAME="question_types_detector.py"
LINE_THRESHOLD=10600
CHECK_INTERVAL=60  # seconds

# ---------- colors ----------
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
    log "${YELLOW}  No change detected and lines < ${LINE_THRESHOLD}. Restarting...${NC}"

    # Kill all python processes running the script
    PIDS=$(pgrep -f "$SCRIPT_NAME")
    if [[ -n "$PIDS" ]]; then
        log "${RED} Killing PID(s): $PIDS${NC}"
        kill $PIDS
        sleep 2
        # Force kill if still running
        PIDS=$(pgrep -f "$SCRIPT_NAME")
        if [[ -n "$PIDS" ]]; then
            log "${RED} Force kill (SIGKILL) PID(s): $PIDS${NC}"
            kill -9 $PIDS
        fi
    else
        log "${YELLOW}  No '$SCRIPT_NAME' process found to kill.${NC}"
    fi

    # Restart in background
    log "${GREEN} Starting: python $SCRIPT_NAME${NC}"
    nohup python "$SCRIPT_NAME" >> logs/qa_detector.log 2>&1 &
    log "${GREEN} Process restarted with PID $!${NC}"
}

# ---------- initialization ----------
mkdir -p logs

if [[ ! -f "$TARGET_FILE" ]]; then
    log "${RED} File not found: $TARGET_FILE${NC}"
    log "   The watchdog will keep monitoring until it appears."
fi

PREV_COUNT=$(get_line_count)
log "${CYAN} Watchdog started. Initial lines: ${PREV_COUNT} | Threshold: ${LINE_THRESHOLD}${NC}"
log "${CYAN} Monitored file: ${TARGET_FILE}${NC}"
log "${CYAN} Check interval: ${CHECK_INTERVAL}s${NC}"

# ---------- main loop ----------
while true; do
    sleep "$CHECK_INTERVAL"

    CURRENT_COUNT=$(get_line_count)

    if [[ "$CURRENT_COUNT" -eq -1 ]]; then
        log "${RED} File still not found. Waiting...${NC}"
        PREV_COUNT=-1
        continue
    fi

    if [[ "$CURRENT_COUNT" -ne "$PREV_COUNT" ]]; then
        # --- File has changed ---
        DIFF=$(( CURRENT_COUNT - PREV_COUNT ))
        if [[ $DIFF -gt 0 ]]; then
            DIFF_STR="+${DIFF}"
        else
            DIFF_STR="${DIFF}"
        fi
        log "${GREEN} Change detected: ${PREV_COUNT} → ${CURRENT_COUNT} lines (${DIFF_STR})${NC}"
        PREV_COUNT="$CURRENT_COUNT"

    else
        # --- No change ---
        log "${YELLOW} No change: ${CURRENT_COUNT} lines.${NC}"

        if [[ "$CURRENT_COUNT" -gt "$LINE_THRESHOLD" ]]; then
            log "${CYAN} Threshold reached (${CURRENT_COUNT} > ${LINE_THRESHOLD}). Stopping watchdog.${NC}"
            exit 0
        else
            log "${RED} Below threshold (${CURRENT_COUNT} ≤ ${LINE_THRESHOLD}).${NC}"
            kill_and_restart
        fi
    fi
done