<#
.SYNOPSIS
    One-command launcher: Operator Console → Robot Edge (hardware mode).
    Pulls the edge token from Raspberry Pi, sets every env var, and starts
    the console. Works with the existing operator_console.ps1 lifecycle
    (start / status / logs / stop).

.EXAMPLE
    .\scripts\start_console_hardware.ps1           # start
    .\scripts\start_console_hardware.ps1 status    # check health + port
    .\scripts\start_console_hardware.ps1 logs      # tail logs
    .\scripts\start_console_hardware.ps1 stop      # graceful shutdown
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "status", "logs", "stop")]
    [string]$Command = "start",

    [ValidateRange(1, 65535)]
    [int]$Port = 7863,

    [string]$EdgeHost = "192.168.18.233",

    [ValidateRange(1, 65535)]
    [int]$EdgePort = 8780,

    [string]$PiAlias = "rasp_pi",
    [string]$PiTokenPath = "/home/china/ubrobot-builds/20260805-9207018/deploy/robot-edge/config/tokens.json",

    # Re-fetch the edge token even when a fresh cache exists (< 24h default).
    [switch]$ForceRefresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$LocalToken = Join-Path $RepoRoot "tmp\edge_tokens.json"
# Token cache freshness window. A cache younger than this is reused; older
# ones are re-fetched so a rotated edge token never silently goes stale.
$TokenMaxAgeHours = 24

# ── state commands (status / logs / stop) don't need token setup ──
if ($Command -ne "start") {
    & "$RepoRoot\scripts\operator_console.ps1" -Command $Command -Port $Port
    exit $LASTEXITCODE
}

# ── validate tooling before touching the network ──
foreach ($tool in @("ssh", "scp")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required tool '$tool' not found on PATH. Install OpenSSH client first."
    }
}

# ── ensure token file is available locally (fresh enough) ──
$TokenExists = Test-Path -LiteralPath $LocalToken
$TokenFresh = $false
if ($TokenExists) {
    $Age = (Get-Date) - (Get-Item -LiteralPath $LocalToken).LastWriteTime
    $TokenFresh = $Age.TotalHours -le $TokenMaxAgeHours
}
if (-not $TokenExists -or -not $TokenFresh -or $ForceRefresh) {
    if ($TokenExists -and -not $TokenFresh) {
        Write-Host "Token cache is older than $TokenMaxAgeHours hours; re-fetching from $PiAlias." -ForegroundColor Yellow
    }
    if ($ForceRefresh -and $TokenExists) {
        Write-Host "Force refresh requested; re-fetching token from $PiAlias." -ForegroundColor Yellow
    }
    Write-Host "Fetching edge token from $PiAlias ..." -ForegroundColor Cyan
    $null = New-Item -ItemType Directory -Path (Split-Path $LocalToken -Parent) -Force
    scp "$PiAlias`:$PiTokenPath" $LocalToken
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to scp token file from Pi. Check SSH connectivity."
    }
    Write-Host "Token cached at $LocalToken" -ForegroundColor Green
} else {
    Write-Host "Using fresh token cache at $LocalToken (LastWriteTime $((Get-Item -LiteralPath $LocalToken).LastWriteTime.ToString('s')))." -ForegroundColor DarkGray
}

# ── env vars for robot-edge hardware mode ──
$env:UBROBOT_CHAT_BACKEND  = "robot-edge"
$env:UBROBOT_EDGE_URL      = "http://${EdgeHost}:${EdgePort}"
$env:UBROBOT_EDGE_TOKEN_FILE = $LocalToken
$env:UBROBOT_CHAT_MEDIA    = "off"
$env:UBROBOT_EDGE_HARDWARE_AUTHORITY = "true"
$env:UBROBOT_EDGE_ESTOP_EXEMPTED = "true"
# TLS off: Edge link is already over unencrypted local-network HTTP;
# self-signed cert would break PowerShell health checks. Use plain HTTP
# for the local browser→console link during development.
$env:UBROBOT_CHAT_TLS      = "off"

# Safety notice: hardware authority + E-stop exemption is a deliberate,
# owner-approved combination (ADR-0002: power cable is the final cutoff).
Write-Host "" -ForegroundColor Yellow
Write-Host "SECURITY: UBROBOT_EDGE_HARDWARE_AUTHORITY=true with UBROBOT_EDGE_ESTOP_EXEMPTED=true." -ForegroundColor Yellow
Write-Host "The physical E-stop is NOT bound; the operator's power cable is the final cutoff." -ForegroundColor Yellow
Write-Host "Ensure no motion can start unattended and a human is at the power switch." -ForegroundColor Yellow
Write-Host "" -ForegroundColor Yellow

Write-Host "Backend : robot-edge @ $env:UBROBOT_EDGE_URL" -ForegroundColor Cyan
Write-Host "Token   : $LocalToken" -ForegroundColor Cyan
Write-Host "Browser : http://localhost:$Port" -ForegroundColor Green

& "$RepoRoot\scripts\operator_console.ps1" -Command start -Port $Port
