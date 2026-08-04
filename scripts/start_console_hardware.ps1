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
    [string]$PiTokenPath = "/home/china/ubrobot-builds/m7-20260803/deploy/robot-edge/config/tokens.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$LocalToken = Join-Path $RepoRoot "tmp\edge_tokens.json"

# ── state commands (status / logs / stop) don't need token setup ──
if ($Command -ne "start") {
    & "$RepoRoot\scripts\operator_console.ps1" -Command $Command -Port $Port
    exit $LASTEXITCODE
}

# ── ensure token file is available locally ──
if (-not (Test-Path -LiteralPath $LocalToken)) {
    Write-Host "Fetching edge token from $PiAlias ..." -ForegroundColor Cyan
    $null = New-Item -ItemType Directory -Path (Split-Path $LocalToken -Parent) -Force
    scp "$PiAlias`:$PiTokenPath" $LocalToken
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to scp token file from Pi. Check SSH connectivity."
    }
    Write-Host "Token cached at $LocalToken" -ForegroundColor Green
}

# ── env vars for robot-edge hardware mode ──
$env:UBROBOT_CHAT_BACKEND  = "robot-edge"
$env:UBROBOT_EDGE_URL      = "http://${EdgeHost}:${EdgePort}"
$env:UBROBOT_EDGE_TOKEN_FILE = $LocalToken
$env:UBROBOT_CHAT_MEDIA    = "off"
$env:UBROBOT_CHAT_TLS      = "on"

Write-Host "Backend : robot-edge @ $env:UBROBOT_EDGE_URL" -ForegroundColor Cyan
Write-Host "Token   : $LocalToken" -ForegroundColor Cyan
Write-Host "Browser : https://localhost:$Port" -ForegroundColor Green

& "$RepoRoot\scripts\operator_console.ps1" -Command start -Port $Port
