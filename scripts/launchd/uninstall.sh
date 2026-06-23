#!/bin/zsh
# Remove the launchd jobs.

PLIST_DIR="$HOME/Library/LaunchAgents"

for plist in com.mlbpred.scrape_lines.plist com.mlbpred.scrape_results.plist; do
    dst="$PLIST_DIR/$plist"
    if [ -f "$dst" ]; then
        launchctl unload "$dst" 2>/dev/null || true
        rm "$dst"
        echo "Removed: $plist"
    fi
done
