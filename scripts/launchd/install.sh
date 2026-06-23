#!/bin/zsh
# Install (or reinstall) the launchd jobs for the PrizePicks scrapers.
# Run once: bash scripts/launchd/install.sh

set -e

PLIST_DIR="$HOME/Library/LaunchAgents"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for plist in com.mlbpred.scrape_lines.plist com.mlbpred.scrape_results.plist; do
    src="$SCRIPT_DIR/$plist"
    dst="$PLIST_DIR/$plist"

    # Unload first if already installed
    if launchctl list | grep -q "${plist%.plist}"; then
        launchctl unload "$dst" 2>/dev/null || true
    fi

    cp "$src" "$dst"
    launchctl load "$dst"
    echo "Loaded: $plist"
done

echo ""
echo "Jobs scheduled:"
echo "  11:00 AM PT — scrape_lines   (PrizePicks lines + model predictions)"
echo "  11:00 PM PT — scrape_results (MLB box scores for today)"
echo ""
echo "Logs:"
echo "  logs/scrape_lines.log"
echo "  logs/scrape_results.log"
echo ""
echo "To check status:  launchctl list | grep mlbpred"
echo "To uninstall:     bash scripts/launchd/uninstall.sh"
