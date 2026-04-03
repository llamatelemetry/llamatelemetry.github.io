#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gsc-common.sh
source "$SCRIPT_DIR/gsc-common.sh"

require_cmd gcloud

printf 'Starting Google Application Default Credentials login for Search Console API.\n'
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/webmasters

printf '\nAccess token preview:\n'
TOKEN="$(get_access_token)"
printf '%s...\n' "${TOKEN:0:24}"
