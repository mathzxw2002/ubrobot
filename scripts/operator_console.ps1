[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "status", "logs", "stop")]
    [string]$Command = "status",

    [ValidateRange(1, 65535)]
    [int]$Port = 7863,

    [string]$Python = "python",

    [switch]$Follow
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $ScriptDirectory
$LogDirectory = Join-Path $RepositoryRoot "logs"
$PidFile = Join-Path $LogDirectory "operator-console-$Port.pid"
$TokenFile = Join-Path $LogDirectory "operator-console-$Port.token"
$StdoutLog = Join-Path $LogDirectory "operator-console-$Port.stdout.log"
$StderrLog = Join-Path $LogDirectory "operator-console-$Port.stderr.log"
$ApplicationPath = Join-Path $RepositoryRoot "src\chat_ui\app.py"

function Get-Listener {
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Get-ManagedProcess {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $null
    }
    $ProcessId = [int](Get-Content -LiteralPath $PidFile -Raw)
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $Process) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $TokenFile -Force -ErrorAction SilentlyContinue
        return $null
    }
    $ProcessDetails = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $ProcessDetails -or $ProcessDetails.CommandLine -notmatch "chat_ui[\\/]app\.py") {
        throw "PID file points to an unrelated process ($ProcessId); refusing to manage it."
    }
    return $Process
}

function Get-ServiceUrl {
    $Scheme = if ($env:UBROBOT_CHAT_TLS -and $env:UBROBOT_CHAT_TLS.ToLowerInvariant() -ne "off") {
        "https"
    } else {
        "http"
    }
    return "${Scheme}://127.0.0.1:$Port"
}

function Show-Status {
    $Managed = Get-ManagedProcess
    $Listener = Get-Listener
    if ($null -eq $Listener) {
        Write-Output "Operator Console is stopped; port $Port is free."
        return
    }
    $ManagedText = if ($null -ne $Managed -and $Managed.Id -eq $Listener.OwningProcess) {
        "managed"
    } else {
        "unmanaged"
    }
    Write-Output "Operator Console listener: PID $($Listener.OwningProcess), port $Port ($ManagedText)."
    try {
        $Health = Invoke-RestMethod -Uri "$(Get-ServiceUrl)/api/health/ready" -TimeoutSec 2
        Write-Output "Health: $($Health.status); backend=$($Health.backend); voice=$($Health.voice_provider); mode=$($Health.execution_mode)."
    } catch {
        Write-Output "Health endpoint unavailable: $($_.Exception.Message)"
    }
}

switch ($Command) {
    "start" {
        New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
        $Existing = Get-ManagedProcess
        $Listener = Get-Listener
        if ($null -ne $Existing -or $null -ne $Listener) {
            $Owner = if ($null -ne $Listener) { $Listener.OwningProcess } else { $Existing.Id }
            throw "Cannot start Operator Console: port $Port is already used by PID $Owner. Run status or stop first."
        }

        if (-not $env:UBROBOT_CHAT_BACKEND) { $env:UBROBOT_CHAT_BACKEND = "cortex-mock" }
        if (-not $env:UBROBOT_CHAT_MEDIA) { $env:UBROBOT_CHAT_MEDIA = "off" }
        if (-not $env:UBROBOT_VOICE_PROVIDER) { $env:UBROBOT_VOICE_PROVIDER = "off" }
        if (-not $env:UBROBOT_CHAT_TLS) { $env:UBROBOT_CHAT_TLS = "off" }
        $env:UBROBOT_CHAT_PORT = "$Port"
        $env:PYTHONPATH = "$(Join-Path $RepositoryRoot 'src');$(Join-Path $RepositoryRoot 'src\chat_ui')"
        $ShutdownToken = [guid]::NewGuid().ToString("N")
        $env:UBROBOT_SHUTDOWN_TOKEN = $ShutdownToken

        $Process = Start-Process -FilePath $Python `
            -ArgumentList @("-u", "`"$ApplicationPath`"") `
            -WorkingDirectory $RepositoryRoot `
            -RedirectStandardOutput $StdoutLog `
            -RedirectStandardError $StderrLog `
            -WindowStyle Hidden `
            -PassThru
        Set-Content -LiteralPath $PidFile -Value $Process.Id -NoNewline
        Set-Content -LiteralPath $TokenFile -Value $ShutdownToken -NoNewline

        $Deadline = [DateTime]::UtcNow.AddSeconds(30)
        do {
            Start-Sleep -Milliseconds 250
            if ($Process.HasExited) {
                $Tail = if (Test-Path -LiteralPath $StderrLog) {
                    (Get-Content -LiteralPath $StderrLog -Tail 20) -join [Environment]::NewLine
                } else { "No stderr log was created." }
                Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $TokenFile -Force -ErrorAction SilentlyContinue
                throw "Operator Console exited during startup.`n$Tail"
            }
            try {
                $Ready = Invoke-RestMethod -Uri "$(Get-ServiceUrl)/api/health/ready" -TimeoutSec 1
                if ($Ready.status -eq "ready") {
                    Write-Output "Operator Console started: $(Get-ServiceUrl) (PID $($Process.Id))."
                    Write-Output "Logs: $StdoutLog and $StderrLog"
                    exit 0
                }
            } catch {
                # Startup is still in progress.
            }
        } while ([DateTime]::UtcNow -lt $Deadline)
        throw "Operator Console did not become ready within 30 seconds. Check $StderrLog."
    }

    "status" {
        Show-Status
    }

    "logs" {
        $Files = @($StdoutLog, $StderrLog) | Where-Object { Test-Path -LiteralPath $_ }
        if ($Files.Count -eq 0) {
            throw "No logs found for port $Port."
        }
        Get-Content -LiteralPath $Files -Tail 100 -Wait:$Follow
    }

    "stop" {
        $Process = Get-ManagedProcess
        if ($null -eq $Process) {
            $Listener = Get-Listener
            if ($null -ne $Listener) {
                throw "Port $Port is owned by unmanaged PID $($Listener.OwningProcess); refusing to stop it."
            }
            Write-Output "Operator Console is already stopped."
            exit 0
        }
        if (-not (Test-Path -LiteralPath $TokenFile)) {
            throw "Shutdown token is missing; refusing to terminate PID $($Process.Id) automatically."
        }
        $Token = Get-Content -LiteralPath $TokenFile -Raw
        try {
            Invoke-RestMethod `
                -Method Post `
                -Uri "$(Get-ServiceUrl)/api/admin/shutdown" `
                -Headers @{ "X-UBRobot-Shutdown-Token" = $Token } `
                -TimeoutSec 3 | Out-Null
        } catch {
            throw "Graceful shutdown request failed; process was left running: $($_.Exception.Message)"
        }

        $Deadline = [DateTime]::UtcNow.AddSeconds(15)
        do {
            Start-Sleep -Milliseconds 200
            $Process.Refresh()
        } while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $Deadline)
        if (-not $Process.HasExited) {
            throw "PID $($Process.Id) did not stop within 15 seconds; process was left running."
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $TokenFile -Force -ErrorAction SilentlyContinue
        Write-Output "Operator Console stopped cleanly; port $Port is free."
    }
}
