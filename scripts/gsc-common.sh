#!/usr/bin/env bash
set -euo pipefail

SITE_URL="${SITE_URL:-https://llamatelemetry.github.io/}"
SITEMAP_URL="${SITEMAP_URL:-https://llamatelemetry.github.io/sitemap.xml}"
DEFAULT_URLS=(
  "https://llamatelemetry.github.io/"
  "https://llamatelemetry.github.io/get-started/"
  "https://llamatelemetry.github.io/reference/"
)

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

require_cmd() {
  if ! have_cmd "$1"; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

urlencode() {
  python3 - "$1" <<'PY'
import sys, urllib.parse
print(urllib.parse.quote(sys.argv[1], safe=''))
PY
}

get_access_token() {
  if [[ -n "${ACCESS_TOKEN:-}" ]]; then
    printf '%s\n' "$ACCESS_TOKEN"
    return 0
  fi

  require_cmd gcloud
  gcloud auth application-default print-access-token
}

auth_header() {
  printf 'Authorization: Bearer %s' "$(get_access_token)"
}

site_url_encoded() {
  urlencode "$SITE_URL"
}

sitemap_url_encoded() {
  urlencode "$SITEMAP_URL"
}
