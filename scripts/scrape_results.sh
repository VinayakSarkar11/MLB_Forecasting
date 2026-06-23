#!/bin/zsh
# Evening scraper — fetch MLB box scores after all games complete.
# Scheduled to run at 11:00 PM PT via launchd.

PROJECT="/Users/vinayaks/Desktop/projects/MLB_Predictions"
PYTHON="$PROJECT/.venv/bin/python"
LOG="$PROJECT/logs/scrape_results.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
TODAY=$(date '+%Y-%m-%d')
cd "$PROJECT" && "$PYTHON" -m src.prizepicks.scrape_results --date "$TODAY" >> "$LOG" 2>&1
echo "exit $?" >> "$LOG"
