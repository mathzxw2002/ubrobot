[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$ReportDirectory = "logs\validation"
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $ScriptDirectory
$ResolvedReportDirectory = Join-Path $RepositoryRoot $ReportDirectory
New-Item -ItemType Directory -Path $ResolvedReportDirectory -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ReportPath = Join-Path $ResolvedReportDirectory "operator-console-software-$Timestamp.md"
$StartedAt = Get-Date
$env:UBROBOT_CHAT_BACKEND = "cortex-mock"
$env:UBROBOT_CHAT_MEDIA = "off"
$env:UBROBOT_VOICE_PROVIDER = "mock"
$env:UBROBOT_CHAT_TLS = "off"
$env:UBROBOT_CHAT_LOG_LEVEL = "CRITICAL"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::DeprecationWarning,ignore::ResourceWarning"

function Invoke-ValidationCommand {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Output "Running $Name..."
    $CaptureId = [guid]::NewGuid().ToString("N")
    $StdoutCapture = Join-Path $ResolvedReportDirectory ".$CaptureId.stdout.tmp"
    $StderrCapture = Join-Path $ResolvedReportDirectory ".$CaptureId.stderr.tmp"
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepositoryRoot `
        -RedirectStandardOutput $StdoutCapture `
        -RedirectStandardError $StderrCapture `
        -Wait `
        -PassThru `
        -NoNewWindow
    $Stdout = if (Test-Path -LiteralPath $StdoutCapture) {
        Get-Content -LiteralPath $StdoutCapture -Raw -Encoding UTF8
    } else { "" }
    $Stderr = if (Test-Path -LiteralPath $StderrCapture) {
        Get-Content -LiteralPath $StderrCapture -Raw -Encoding UTF8
    } else { "" }
    Remove-Item -LiteralPath $StdoutCapture -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StderrCapture -Force -ErrorAction SilentlyContinue
    $Output = ($Stdout + $Stderr).Trim()
    $ExitCode = $Process.ExitCode
    Write-Output $Output
    return [pscustomobject]@{
        Name = $Name
        Command = "$Python $($Arguments -join ' ')"
        ExitCode = $ExitCode
        Output = $Output.Trim()
    }
}

$Software = Invoke-ValidationCommand `
    -Name "software-only unit/integration suite" `
    -Arguments @("-m", "unittest", "discover", "-s", "tests/cortex_navigation", "-p", "test_*.py", "-q")
$EndToEnd = Invoke-ValidationCommand `
    -Name "process-level mock acceptance suite" `
    -Arguments @("-m", "unittest", "tests.e2e.test_operator_console_mock", "-v")
$PythonVersion = (& $Python --version 2>&1 | Out-String).Trim()
$Passed = $Software.ExitCode -eq 0 -and $EndToEnd.ExitCode -eq 0
$ResultText = if ($Passed) { "PASS" } else { "FAIL" }
$FinishedAt = Get-Date

$Report = @"
# Operator Console M1-M4 Software Validation

- Result: **$ResultText**
- Started: $($StartedAt.ToString("o"))
- Finished: $($FinishedAt.ToString("o"))
- Python: $PythonVersion
- Backend: cortex-mock
- Voice provider: mock
- Media: off
- Hardware authority: **false**
- Hardware/ROS/cloud tests: **not executed**

## Assertions

- UI/API -> InteractionRuntime -> TaskRuntime -> Cortex Mock success path.
- Navigation planning/running/feedback/succeeded timeline retention.
- Concurrent status query without a second Cortex dispatch.
- Queue retention, normal cancel, spoken/UI emergency stop, and queue supersede.
- Critical safety event metadata and no-hardware-authority banner.
- Operator event replay/reconnect and Mock voice WebSocket reconnect.
- Capability Registry and explicit execution/authority descriptors.
- JSON-safe telemetry DTOs with unavailable/stale/disconnected semantics.
- Fixture-only Cortex/telemetry adapters with no ROS or hardware SDK imports.

## Commands and sanitized output

### $($Software.Name)

~~~text
$($Software.Command)
$($Software.Output)
~~~

### $($EndToEnd.Name)

~~~text
$($EndToEnd.Command)
$($EndToEnd.Output)
~~~

This report intentionally records only fixed execution modes and sanitized test
output. It does not read or include cloud credentials.
"@

Set-Content -LiteralPath $ReportPath -Value $Report -Encoding UTF8
Write-Output "Validation report: $ReportPath"
if (-not $Passed) {
    exit 1
}
