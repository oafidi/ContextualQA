#!/bin/bash

TARGET_FILE="/mnt/c/Users/omar1/Documents/Projects/genAi/gemini_scraper/datasets/questions_1526_3050.json"

# Stop the watchdog once the file reaches this many lines (scraping is done).
LINE_THRESHOLD=9150

# How often to check the file, in seconds.
CHECK_INTERVAL=180

# ---------- colors ----------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'   # No Color — resets color after each log line

# ---------- log helper ----------
log() {
    # Prints a timestamped message to the terminal.
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ---------- line counter ----------
get_line_count() {
    # Returns -1 if the file does not exist yet (scraper hasn't created it).
    # Otherwise returns the number of lines via wc -l.
    if [[ ! -f "$TARGET_FILE" ]]; then
        echo -1
        return
    fi
    wc -l < "$TARGET_FILE"
}

if [[ ! -f "$TARGET_FILE" ]]; then
    log "${RED}File not found: $TARGET_FILE${NC}"
    log "   The watchdog will keep monitoring until it appears."
fi

PREV_COUNT=$(get_line_count)
log "${CYAN}Watchdog started. Initial lines: ${PREV_COUNT} | Threshold: ${LINE_THRESHOLD}${NC}"
log "${CYAN}Monitored file : ${TARGET_FILE}${NC}"
log "${CYAN}Check interval : ${CHECK_INTERVAL}s${NC}"

# How many alert emails have been sent so far.
SEND_COUNT=0

# Maximum number of alert emails before giving up and exiting.
SEND_MAX_TIMES=3

# =============================================================================
# MAIN LOOP
# =============================================================================
while true; do
    sleep "$CHECK_INTERVAL"

    CURRENT_COUNT=$(get_line_count)

    # --- File does not exist yet ---
    if [[ "$CURRENT_COUNT" -eq -1 ]]; then
        log "${RED}File still not found. Waiting...${NC}"
        # Do NOT update PREV_COUNT here — keep it at -1 so the
        # comparison still works once the file appears.
        continue
    fi

    # --- File has grown → scraper is making progress ---
    if [[ "$CURRENT_COUNT" -gt "$PREV_COUNT" ]]; then
        DIFF=$(( CURRENT_COUNT - PREV_COUNT ))
        log "${GREEN}Change detected: ${PREV_COUNT} → ${CURRENT_COUNT} lines (+${DIFF})${NC}"

        # Update baseline for the next check.
        PREV_COUNT="$CURRENT_COUNT"

        # Reset the alert counter — progress is happening again.
        SEND_COUNT=0

    # --- File has NOT grown → possible problem ---
    else
        log "${YELLOW}No change: ${CURRENT_COUNT} lines.${NC}"

        # Check if we already reached the target (scraping finished normally).
        if [[ "$CURRENT_COUNT" -ge "$LINE_THRESHOLD" ]]; then
            log "${CYAN}Threshold reached (${CURRENT_COUNT} >= ${LINE_THRESHOLD}). Scraping complete. Stopping watchdog.${NC}"
            exit 0

        # Still below threshold and no progress → send alert email.
        elif [[ "$SEND_COUNT" -lt "$SEND_MAX_TIMES" ]]; then
            SEND_COUNT=$(( SEND_COUNT + 1 ))
            log "${RED}No progress detected. Sending alert email (${SEND_COUNT}/${SEND_MAX_TIMES})...${NC}"
            python watchdogs/email_sender.py

        # Reached max alerts with no recovery → give up.
        else
            log "${RED}Max alerts reached (${SEND_MAX_TIMES}). Scraper appears stuck. Exiting watchdog.${NC}"
            exit 1

        fi

    fi

done