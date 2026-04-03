#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gsc-common.sh
source "$SCRIPT_DIR/gsc-common.sh"

for url in "${DEFAULT_URLS[@]}"; do
  printf '\n=== %s ===\n' "$url"
  "$SCRIPT_DIR/gsc-inspect-url.sh" "$url" \
    | jq '.inspectionResult.indexStatusResult | {
        verdict,
        coverageState,
        indexingState,
        robotsTxtState,
        pageFetchState,
        googleCanonical,
        userCanonical,
        lastCrawlTime,
        referringUrls,
        sitemap
      }'
done
