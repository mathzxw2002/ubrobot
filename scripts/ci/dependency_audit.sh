#!/usr/bin/env bash
# Dependency vulnerability scan for the pinned operator-console runtime.
#
# Runs pip-audit against requirements-operator-console.txt. Known CVEs in the
# pinned gradio 5.50.0 + pillow 11.3.0 are allowlisted HERE, deliberately:
#
#   - gradio==5.50.0 is pinned by requirements-operator-console.txt (and
#     requirements.txt). Its 2026 advisories (PYSEC-2026-63..66, 211,
#     2178-2179) are fixed only in gradio 6.x, a major upgrade that also
#     unlocks pillow>=12 (gradio 5.50 requires pillow<12). Upgrading gradio
#     also drags fastapi/starlette and needs a chat_ui compatibility pass
#     (removed APIs like gr.Chatbot(type=...)). Tracked as a dedicated task
#     (docs/plans/2026-08-08-production-hardening.md, P2.5).
#   - pillow==11.3.0 is the highest version gradio 5.50 permits; its 2026
#     advisories (PYSEC-2026-165, 2249-2257, 2874, 3451/3453-3454,
#     3493-3496) need pillow>=12.1.1, which likewise requires gradio 6.x.
#
#   - starlette is pinned to 0.49.1 (fixes PYSEC-2026-1942). Its remaining
#     advisories (PYSEC-2026-161, 248-249, 2280-2281) need starlette 1.x,
#     which requires gradio 6.x (same upgrade).
#
# These ignores are REMOVED when the gradio 6 upgrade lands. Any OTHER
# vulnerability fails the build.
set -euo pipefail

python -m pip_audit \
  -r requirements-operator-console.txt \
  --progress-spinner off \
  --ignore-vuln PYSEC-2026-63 \
  --ignore-vuln PYSEC-2026-64 \
  --ignore-vuln PYSEC-2026-65 \
  --ignore-vuln PYSEC-2026-66 \
  --ignore-vuln PYSEC-2026-211 \
  --ignore-vuln PYSEC-2026-2178 \
  --ignore-vuln PYSEC-2026-2179 \
  --ignore-vuln PYSEC-2026-161 \
  --ignore-vuln PYSEC-2026-248 \
  --ignore-vuln PYSEC-2026-249 \
  --ignore-vuln PYSEC-2026-2280 \
  --ignore-vuln PYSEC-2026-2281 \
  --ignore-vuln PYSEC-2026-165 \
  --ignore-vuln PYSEC-2026-2249 \
  --ignore-vuln PYSEC-2026-2250 \
  --ignore-vuln PYSEC-2026-2251 \
  --ignore-vuln PYSEC-2026-2252 \
  --ignore-vuln PYSEC-2026-2253 \
  --ignore-vuln PYSEC-2026-2254 \
  --ignore-vuln PYSEC-2026-2255 \
  --ignore-vuln PYSEC-2026-2256 \
  --ignore-vuln PYSEC-2026-2257 \
  --ignore-vuln PYSEC-2026-2874 \
  --ignore-vuln PYSEC-2026-3451 \
  --ignore-vuln PYSEC-2026-3453 \
  --ignore-vuln PYSEC-2026-3454 \
  --ignore-vuln PYSEC-2026-3493 \
  --ignore-vuln PYSEC-2026-3494 \
  --ignore-vuln PYSEC-2026-3495 \
  --ignore-vuln PYSEC-2026-3496
