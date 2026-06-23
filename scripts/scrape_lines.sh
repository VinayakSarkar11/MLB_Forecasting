#!/bin/zsh
# Morning scraper — fetch today's PrizePicks lines + model predictions.
# Scheduled to run at 11:00 AM PT via launchd.
# Uploads DB to S3 after scraping so GitHub Actions can read it for results.

PROJECT="/Users/vinayaks/Desktop/projects/MLB_Predictions"
PYTHON="$PROJECT/.venv/bin/python"
LOG="$PROJECT/logs/scrape_lines.log"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
cd "$PROJECT" && "$PYTHON" -m src.prizepicks.scrape_lines >> "$LOG" 2>&1

# Sync DB to S3 so the results scraper on GitHub Actions can pick it up
"$PYTHON" - >> "$LOG" 2>&1 <<'PYEOF'
import boto3
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)
db = Path('data/prizepicks_history.db')
if db.exists():
    s3.upload_file(str(db), 'statcast-surge-raw-data-vs', 'prizepicks/prizepicks_history.db')
    print('DB uploaded to S3.')
else:
    print('No DB file found — skipping S3 upload.')
PYEOF

echo "exit $?" >> "$LOG"
