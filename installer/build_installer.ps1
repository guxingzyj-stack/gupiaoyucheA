$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$IssFile = Join-Path $ScriptDir 'guanghuan_stock.iss'

$Candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
)

$Iscc = $null
foreach ($Candidate in $Candidates) {
    if ($Candidate -and (Test-Path $Candidate)) {
        $Iscc = $Candidate
        break
    }
}

if (-not $Iscc) {
    $Command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($Command) {
        $Iscc = $Command.Source
    }
}

if (-not $Iscc) {
    Write-Host 'ISCC.exe was not found.' -ForegroundColor Yellow
    Write-Host 'Please install Inno Setup 6, then run this script again.'
    Write-Host 'Installer script:' $IssFile
    exit 1
}

if (-not (Test-Path $IssFile)) {
    throw "Installer script not found: $IssFile"
}

Push-Location $ScriptDir
try {
    & $Iscc $IssFile
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compile failed. Exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host 'Installer build completed. Output directory:' $ScriptDir -ForegroundColor Green
