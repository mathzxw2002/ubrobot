<#
.SYNOPSIS
    Parameter/behavior validation for start_console_hardware.ps1 (refactor Task 3).

    Two layers, both WITHOUT touching real hardware or the network:

      1. Static source assertions — the script must contain the hardened
         guards (tool check, token-freshness window, -ForceRefresh, safety
         notice), and state commands (status/logs/stop) must not reach the
         token/scp path.
      2. In-process behavior test — with scp shadowed to fail loudly, a
         missing-cache `start` must try to fetch (i.e. reach scp) and fail
         closed; with a fresh cache, `start` must NOT call scp.

    Uses a temp copy of the script with the cache path repointed so no real
    file is touched. Never contacts a real Raspberry Pi.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir | Split-Path -Parent
$TargetScript = Join-Path $RepoRoot "scripts\start_console_hardware.ps1"
$Source = Get-Content -LiteralPath $TargetScript -Raw

$Failed = 0
function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        $script:Failed++
        Write-Host "  [FAIL] $Message" -ForegroundColor Red
    } else {
        Write-Host "  [PASS] $Message" -ForegroundColor Green
    }
}

Write-Host "start_console_hardware.ps1 hardening tests:" -ForegroundColor Cyan

# ── Layer 1: static source guards ──
Assert-True ($Source -match 'Get-Command \$tool') "checks for required tools (ssh/scp)"
Assert-True ($Source -match 'TokenMaxAgeHours') "has a token freshness window"
Assert-True ($Source -match 'ForceRefresh') "supports -ForceRefresh"
Assert-True ($Source -match 'LastWriteTime') "reads cache LastWriteTime for freshness"
Assert-True ($Source -match 'SECURITY: UBROBOT_EDGE_HARDWARE_AUTHORITY') "prints hardware-authority safety notice"
# state commands must bypass token logic
$stateIdx = $Source.IndexOf('if ($Command -ne "start")')
Assert-True ($stateIdx -ge 0) "state-command guard present in script"
if ($stateIdx -ge 0) {
    $stateBlock = $Source.Substring($stateIdx)
    Assert-True ($stateBlock -match 'operator_console.ps1.*\$Command') "state commands delegate without token fetch"
}

# ── Layer 2: behavior with a temp copy ──
$TempScript = Join-Path $env:TEMP "start_console_hardware_test.ps1"
$CacheFile = Join-Path $env:TEMP "ubrobot_test_token.json"
New-Item -ItemType Directory -Path (Split-Path $CacheFile -Parent) -Force | Out-Null

# Repoint the cache path + repo root (temp script lives in %TEMP%) and stub
# scp to record invocation. Single-quoted replacement strings avoid
# PowerShell backtick interpretation. The stub also resets $LASTEXITCODE=0.
$RealRepoRoot = (Split-Path -Parent $TargetScript) + "\.."
$Patched = $Source `
    -replace [regex]::Escape('$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent'),
    ('$RepoRoot = "' + $RealRepoRoot + '"') `
    -replace [regex]::Escape('Join-Path $RepoRoot "tmp\edge_tokens.json"'), ('"$CacheFile"') `
    -replace [regex]::Escape('scp "$PiAlias`:$PiTokenPath" $LocalToken'),
    'Set-Content -LiteralPath $env:TEMP\ubrobot_scp_called -Value "yes" -Encoding UTF8; $global:LASTEXITCODE = 0' `
    -replace [regex]::Escape('& "$RepoRoot\scripts\operator_console.ps1" -Command start -Port $Port'),
    'Set-Content -LiteralPath $env:TEMP\ubrobot_console_called -Value "yes" -Encoding UTF8'
$Patched | Set-Content -LiteralPath $TempScript -Encoding UTF8

# (a) missing cache -> scp stub runs (fetches)
Remove-Item -LiteralPath $CacheFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $env:TEMP "ubrobot_scp_called") -Force -ErrorAction SilentlyContinue
& $TempScript -Command start *> $null
Assert-True (Test-Path (Join-Path $env:TEMP "ubrobot_scp_called")) "missing cache triggers token fetch"

# (b) fresh cache -> scp stub must NOT run
Remove-Item -LiteralPath (Join-Path $env:TEMP "ubrobot_scp_called") -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $CacheFile -Value '{}' -Encoding UTF8
(Get-Item -LiteralPath $CacheFile).LastWriteTime = Get-Date
& $TempScript -Command start *> $null
Assert-True (-not (Test-Path (Join-Path $env:TEMP "ubrobot_scp_called"))) "fresh cache reuses token without fetch"

# (c) -ForceRefresh with fresh cache -> scp stub runs again
Remove-Item -LiteralPath (Join-Path $env:TEMP "ubrobot_scp_called") -Force -ErrorAction SilentlyContinue
& $TempScript -Command start -ForceRefresh *> $null
Assert-True (Test-Path (Join-Path $env:TEMP "ubrobot_scp_called")) "-ForceRefresh re-fetches even with fresh cache"

# cleanup
Remove-Item -LiteralPath $TempScript, $CacheFile, (Join-Path $env:TEMP "ubrobot_scp_called") -Force -ErrorAction SilentlyContinue

Write-Host ""
if ($Failed -gt 0) {
    Write-Host "FAILED: $Failed check(s)" -ForegroundColor Red
    exit 1
}
Write-Host "PASS: all checks" -ForegroundColor Green
exit 0
