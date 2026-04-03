#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gsc-common.sh
source "$SCRIPT_DIR/gsc-common.sh"

require_cmd curl
require_cmd jq

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s <url> [languageCode]\n' "$(basename "$0")" >&2
  exit 1
fi

INSPECTION_URL="$1"
LANG_CODE="${2:-en-US}"

PAYLOAD="$(jq -cn \
  --arg inspectionUrl "$INSPECTION_URL" \
  --arg siteUrl "$SITE_URL" \
  --arg languageCode "$LANG_CODE" \
  '{inspectionUrl: $inspectionUrl, siteUrl: $siteUrl, languageCode: $languageCode}')"

curl -sS -X POST \
  -H "$(auth_header)" \
  -H "Content-Type: application/json" \
  "https://searchconsole.googleapis.com/v1/urlInspection/index:inspect" \
  -d "$PAYLOAD" | jq
