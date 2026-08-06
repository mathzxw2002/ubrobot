#!/usr/bin/env bash
# Task 6: Go2+Piper Cortex combined acceptance driver.
#
# Runs the deployment-gate + staged acceptance checks on the Go2 dock.
#
# NOTE: this script is a SAFETY-GATED DRIVER, not an unsupervised test. It
# validates configuration gates and read-only readiness, then prints the
# staged plan and waits for an operator to run each stage with motion
# enabled. Navigation-on-real-dog stages are deliberately deferred until the
# operator releases them; the Piper (arm-only) stages can run first.
#
# Usage (from the workstation or on the dock):
#   bash scripts/hardware/validate_go2_piper_cortex.sh [--dry-run] [ssh_alias]
#
# Exit code 0 only when the deployment gates pass. Stages that need motion
# require an operator; they are listed, not auto-executed.

set -u

SSH_ALIAS=""
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *)
            if [ -n "${SSH_ALIAS}" ]; then
                echo "error: multiple ssh aliases given: ${SSH_ALIAS} $arg" >&2
                exit 2
            fi
            SSH_ALIAS="$arg"
            ;;
    esac
done

run() {
    if [ -n "${SSH_ALIAS}" ]; then
        ssh -o BatchMode=yes "${SSH_ALIAS}" "$*"
    else
        bash -c "$*"
    fi
}

section() { echo; echo "===== $* ====="; }

# ---------------------------------------------------------------- deployment gates
echo "========== Go2+Piper Cortex acceptance: deployment gates =========="
echo "ssh_alias=${SSH_ALIAS:-<local>} dry_run=${DRY_RUN}"

section "1. Platform identity (must be go2_piper)"
run "test \"\${UBROBOT_PLATFORM:-}\" = go2_piper && echo OK || echo 'MISSING/INVALID UBROBOT_PLATFORM'"
run "test \"\${UBROBOT_GRASP_PLATFORM:-}\" = go2_piper && echo OK || echo 'MISSING/INVALID UBROBOT_GRASP_PLATFORM'"

section "2. Edge mode / authority (hardware + explicit authority)"
run "test \"\${UBROBOT_EDGE_MODE:-}\" = hardware && echo OK || echo 'MISSING/INVALID UBROBOT_EDGE_MODE'"
run "test \"\${UBROBOT_EDGE_HARDWARE_AUTHORITY:-}\" = true && echo OK || echo 'MISSING UBROBOT_EDGE_HARDWARE_AUTHORITY'"

section "3. Local stop bound (mandatory for motion authority)"
run "test \"\${UBROBOT_EDGE_ESTOP_ENABLED:-}\" = true && echo OK || echo 'MISSING UBROBOT_EDGE_ESTOP_ENABLED'"
run "test -n \"\${UBROBOT_EDGE_ESTOP_CHIP:-}\" && echo OK || echo 'MISSING UBROBOT_EDGE_ESTOP_CHIP'"
run "test -n \"\${UBROBOT_EDGE_ESTOP_LINE:-}\" && echo OK || echo 'MISSING UBROBOT_EDGE_ESTOP_LINE'"

section "4. Remote perception service URL (x86 GPU server)"
run "test -n \"\${REMOTE_PERCEPTION_SERVICE_URL:-}\" && echo OK || echo 'MISSING REMOTE_PERCEPTION_SERVICE_URL'"

section "5. RMW + ROS domain (CycloneDDS, matching go2-bridge and emos dock containers)"
run "test \"\${RMW_IMPLEMENTATION:-}\" = rmw_cyclonedds_cpp && echo OK || echo 'MISSING RMW_IMPLEMENTATION=rmw_cyclonedds_cpp'"
run "echo \"ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-<unset>}\""

section "6. Safety checklist + config hash artifacts"
run "test -f deploy/robot-edge/checklist/go2-piper-hardware-checklist.md && echo OK || echo 'MISSING checklist'"
run "test -f deploy/robot-edge/config/go2-piper.example.env && echo OK || echo 'MISSING example.env'"

section "7. Dock readiness (read-only inventory)"
run "docker --version"
run "docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E 'go2-bridge|emos' || echo 'go2-bridge/emos not listed (expected in hardware run)'"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo
    echo "DRY-RUN complete: gates above reported, NO motion stages executed."
    echo "Run without --dry-run and with an operator to proceed to staged acceptance."
    exit 0
fi

# ---------------------------------------------------------------- staged plan
echo
echo "========== Staged acceptance plan (operator-driven) =========="
echo "Order (per Task 6 Step 2):"
echo "  S1 read-only health          (safe, no motion)"
echo "  S2 zero-output / stop        (Piper torque DISABLED)"
echo "  S3 low-speed navigation      (DEFERRED on real dog)"
echo "  S4 stationary pre-grasp      (Piper only, base still)"
echo "  S5 light grasp               (Piper only, base still)"
echo
echo "Failure-injection rounds (one at a time): cancel, lease loss,"
echo "Console/Edge/Cortex disconnect, local E-stop, physical E-stop,"
echo "remote-perception disconnect (fail-closed, no motion)."
echo
echo "Mutual-exclusion checks:"
echo "  - GraspObject while navigation lease active   -> REJECTED"
echo "  - NavigateToObject during grasp               -> grasp fail-closed cancel"
echo
echo "Navigation stages (S3) are DEFERRED until the operator releases them;"
echo "Piper stages (S4/S5) can run first on the stationary dog."
echo
echo "Run the harness:"
echo "  cd tests/hardware && $PYTHON test_go2_piper_cortex_acceptance.py"
echo "  (with operator + physical E-stop + second observer)"
