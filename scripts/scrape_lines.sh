#!/bin/zsh
# Morning scraper — fetch today's PrizePicks lines + model predictions.
# Scheduled to run at 11:00 AM PT via launchd (most lineups confirmed by then).

PROJECT="/Users/vinayaks/Desktop/projects/MLB_Predictions"
PYTHON="$PROJECT/.venv/bin/python"
LOG="$PROJECT/logs/scrape_lines.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
cd "$PROJECT" && "$PYTHON" -m src.prizepicks.scrape_lines >> "$LOG" 2>&1
echo "exit $?" >> "$LOG"
