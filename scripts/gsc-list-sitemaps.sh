#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gsc-common.sh
source "$SCRIPT_DIR/gsc-common.sh"

require_cmd curl
require_cmd jq

curl -s \
  -H "$(auth_header)" \
  "https://www.googleapis.com/webmasters/v3/sites/$(site_url_encoded)/sitemaps" | jq
