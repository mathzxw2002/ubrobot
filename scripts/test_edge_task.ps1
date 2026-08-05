<#
.SYNOPSIS
    End-to-end self-test: Operator Console / Robot Edge / Cortex planner chain.

.DESCRIPTION
    1. Checks the local Operator Console health (http://127.0.0.1:7863).
    2. Checks Robot Edge health, capabilities, and camera frame on the Pi.
    3. Optionally submits a task (default: navigate to the chair) directly to
       Robot Edge /v1/commands and prints the live event timeline until the
       command reaches a terminal state (succeeded / failed / cancelled).

    This bypasses the Gradio UI, so it isolates the Pi-side stack
    (relay -> ARK planner -> Cortex -> NavigateToObject -> Kompass -> guard).
    If this script succeeds but the UI fails, the problem is console-side.

.EXAMPLE
    .\scripts\test_edge_task.ps1 -HealthOnly
        Only run health checks (console, edge, capabilities, camera frame).

.EXAMPLE
    .\scripts\test_edge_task.ps1
        Submit the default task "导航到前面的椅子" and stream its timeline.

.EXAMPLE
    .\scripts\test_edge_task.ps1 -Task "你好"
        Submit a custom task text.
#>

[CmdletBinding()]
param(
    [string]$Task = "导航到前面的椅子",

    [string]$EdgeHost = "192.168.18.233",

    [ValidateRange(1, 65535)]
    [int]$EdgePort = 8780,

    [ValidateRange(1, 65535)]
    [int]$ConsolePort = 7863,

    [string]$TokenFile = "",

    [ValidateRange(10, 900)]
    [int]$TimeoutSec = 180,

    [switch]$HealthOnly
)

$ErrorActionPreference = "Stop"
if (-not $TokenFile) {
    $TokenFile = Join-Path (Split-Path -Parent $PSScriptRoot) "tmp\edge_tokens.json"
}
$EdgeBase = "http://${EdgeHost}:${EdgePort}"
$TerminalStates = @("succeeded", "failed", "cancelled")

function Write-Step([string]$Text) {
    Write-Host "`n== $Text ==" -ForegroundColor Cyan
}

# ── load operator token (needs task.submit scope for submission) ──
if (-not (Test-Path -LiteralPath $TokenFile)) {
    throw "Token file not found: $TokenFile`nRun scripts\start_console_hardware.ps1 once to fetch it from the Pi."
}
$Tokens = Get-Content -LiteralPath $TokenFile -Raw | ConvertFrom-Json
$SubmitToken = $null
$ObserveToken = $null
foreach ($Prop in $Tokens.PSObject.Properties) {
    $Scopes = @($Prop.Value)
    if ($null -eq $ObserveToken -and $Scopes -contains "observe") { $ObserveToken = $Prop.Name }
    if ($Scopes -contains "task.submit" -and $Scopes -contains "observe") { $SubmitToken = $Prop.Name; break }
}
if ($null -eq $ObserveToken) { throw "No token with 'observe' scope found in $TokenFile" }
$ObserveHeaders = @{ "Authorization" = "Bearer $ObserveToken" }

$Failed = $false

# ── 1. local console health ──
Write-Step "Operator Console (local, port $ConsolePort)"
try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$ConsolePort/api/health/ready" -TimeoutSec 3
    Write-Host ("OK   status={0} backend={1} mode={2}" -f $Health.status, $Health.backend, $Health.execution_mode) -ForegroundColor Green
} catch {
    Write-Host "FAIL console not ready: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "     start it with: .\scripts\start_console_hardware.ps1" -ForegroundColor Yellow
    $Failed = $true
}

# ── 2. edge health + capabilities + camera ──
Write-Step "Robot Edge ($EdgeBase)"
try {
    $Ready = Invoke-RestMethod -Uri "$EdgeBase/v1/health/ready" -Headers $ObserveHeaders -TimeoutSec 5
    Write-Host ("OK   ready: {0}" -f ($Ready | ConvertTo-Json -Compress -Depth 3)) -ForegroundColor Green
} catch {
    Write-Host "FAIL edge health: $($_.Exception.Message)" -ForegroundColor Red
    $Failed = $true
}
try {
    $Caps = Invoke-RestMethod -Uri "$EdgeBase/v1/capabilities" -Headers $ObserveHeaders -TimeoutSec 5
    $Names = $Caps.capabilities.PSObject.Properties.Name -join ", "
    Write-Host "OK   capabilities: $Names" -ForegroundColor Green
} catch {
    Write-Host "WARN capabilities: $($_.Exception.Message)" -ForegroundColor Yellow
}
try {
    $null = Invoke-WebRequest -Uri "$EdgeBase/v1/camera/frame" -Headers $ObserveHeaders -TimeoutSec 15
    Write-Host "OK   camera frame: JPEG available" -ForegroundColor Green
} catch {
    $Code = $_.Exception.Response.StatusCode.value__
    if ($Code -eq 404) {
        Write-Host "WARN camera frame: 404 (no frame yet - camera offline or still starting)" -ForegroundColor Yellow
    } else {
        Write-Host "WARN camera frame: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

if ($HealthOnly) {
    Write-Host "`nHealth checks done." -ForegroundColor Cyan
    if ($Failed) { exit 1 } else { exit 0 }
}

# ── 3. submit task and stream timeline ──
if ($null -eq $SubmitToken) { throw "No token with 'task.submit' scope found in $TokenFile" }
$SubmitHeaders = @{ "Authorization" = "Bearer $SubmitToken" }

Write-Step "Submit task: $Task"
$CorrelationId = [guid]::NewGuid().ToString()
$Body = @{
    text           = $Task
    correlation_id = $CorrelationId
    operator_id    = "manual-test-script"
    nonce          = [guid]::NewGuid().ToString()
    timestamp      = ([DateTimeOffset]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ss.fffffff+00:00")
} | ConvertTo-Json -Compress

try {
    # Encode body as UTF-8 bytes so Chinese/unicode task text survives the
    # HTTP hop (PowerShell 5.1 Invoke-RestMethod mangles non-ASCII strings).
    $BodyBytes = [System.Text.Encoding]::UTF8.GetBytes($Body)
    $Accepted = Invoke-RestMethod -Method Post -Uri "$EdgeBase/v1/commands" `
        -Headers $SubmitHeaders -ContentType "application/json; charset=utf-8" -Body $BodyBytes -TimeoutSec 10
} catch {
    $Detail = $_.ErrorDetails.Message
    throw "Command rejected: $($_.Exception.Message) $Detail"
}
$CommandId = $Accepted.command_id
Write-Host "accepted: command_id=$CommandId" -ForegroundColor Green

Write-Step "Event timeline"
$After = 0
$Sw = [System.Diagnostics.Stopwatch]::StartNew()
$Terminal = $null
while ($Sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    try {
        $Resp = Invoke-RestMethod -Uri "$EdgeBase/v1/events?after=$After" -Headers $ObserveHeaders -TimeoutSec 10
    } catch {
        Start-Sleep -Milliseconds 500
        continue
    }
    foreach ($Event in @($Resp.events)) {
        $After = [Math]::Max($After, [int]$Event.sequence)
        if ($Event.command_id -ne $CommandId) { continue }
        $Stamp = "{0,7:n1}s" -f $Sw.Elapsed.TotalSeconds
        $Msg = $Event.message
        if ($Msg.Length -gt 160) { $Msg = $Msg.Substring(0, 160) + "..." }
        Write-Host ("{0}  [{1}] {2}" -f $Stamp, $Event.state, $Msg)
        if ($TerminalStates -contains $Event.state) { $Terminal = $Event.state }
    }
    if ($Terminal) { break }
    Start-Sleep -Milliseconds 300
}

Write-Host ""
if ($Terminal -eq "succeeded") {
    Write-Host "RESULT: succeeded in $("{0:n1}" -f $Sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
    exit 0
} elseif ($Terminal) {
    Write-Host "RESULT: $Terminal after $("{0:n1}" -f $Sw.Elapsed.TotalSeconds)s" -ForegroundColor Red
    exit 1
} else {
    Write-Host "RESULT: TIMEOUT after ${TimeoutSec}s - command may still be running." -ForegroundColor Red
    Write-Host "Cancel it from the console UI, or check Pi logs: ssh rasp_pi docker logs emos-cortex-recipe" -ForegroundColor Yellow
    exit 2
}
