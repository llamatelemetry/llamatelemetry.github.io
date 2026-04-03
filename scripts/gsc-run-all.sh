#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '1. Listing Search Console sites\n'
"$SCRIPT_DIR/gsc-list-sites.sh"

printf '\n2. Submitting sitemap\n'
"$SCRIPT_DIR/gsc-submit-sitemap.sh"

printf '\n3. Listing submitted sitemaps\n'
"$SCRIPT_DIR/gsc-list-sitemaps.sh"

printf '\n4. Inspecting default URLs\n'
"$SCRIPT_DIR/gsc-inspect-defaults.sh"
