#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gsc-common.sh
source "$SCRIPT_DIR/gsc-common.sh"

require_cmd curl

API_URL="https://www.googleapis.com/webmasters/v3/sites/$(site_url_encoded)/sitemaps/$(sitemap_url_encoded)"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'PUT %s\n' "$API_URL"
  exit 0
fi

curl -sS -X PUT \
  -H "$(auth_header)" \
  "$API_URL"
printf '\n'
