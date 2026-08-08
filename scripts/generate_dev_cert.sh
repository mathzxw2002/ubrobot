#!/usr/bin/env bash
# Regenerate the local self-signed TLS dev certificate for the Operator Console.
#
# The console serves TLS from assets/key.pem + assets/cert.pem (see
# src/chat_ui/app.py). These are per-machine dev credentials and are NOT
# tracked in git (see .gitignore). Run this once on any fresh clone that
# starts the console with UBROBOT_CHAT_TLS enabled (the default is "on").
#
#   ./scripts/generate_dev_cert.sh          # 365-day self-signed localhost cert
#   OPENSSL_CN="192.168.18.233" ./scripts/generate_dev_cert.sh   # custom CN
#
# Safety: this regenerates a NEW key/cert pair and overwrites the existing
# files. Browsers will show a fresh self-signed warning until you accept the
# new certificate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${REPO_ROOT}/assets"
KEY_FILE="${ASSETS_DIR}/key.pem"
CERT_FILE="${ASSETS_DIR}/cert.pem"
CN="${OPENSSL_CN:-localhost}"

mkdir -p "${ASSETS_DIR}"

if ! command -v openssl >/dev/null 2>&1; then
    echo "error: openssl is required to generate the dev certificate" >&2
    exit 1
fi

openssl req \
    -x509 \
    -newkey rsa:4096 \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -days 365 \
    -nodes \
    -subj "/CN=${CN}/OU=ubrobot-dev/O=UBRobot"

echo "Generated dev certificate for CN=${CN}:"
echo "  key : ${KEY_FILE}"
echo "  cert: ${CERT_FILE}"
echo
echo "Note: never commit these files; they are local development credentials."
