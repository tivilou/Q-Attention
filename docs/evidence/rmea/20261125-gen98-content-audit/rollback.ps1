$ErrorActionPreference = 'Stop'
$taskDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$original = Join-Path $taskDir 'selector.original.py'
$modified = Join-Path $taskDir 'selector.py'
$expected = (Get-Content -Raw -LiteralPath (Join-Path $taskDir 'rollback-original-sha256.txt')).Trim()
if ((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Preserved original hash mismatch'
}
Copy-Item -LiteralPath $original -Destination $modified -Force
if ((Get-FileHash -LiteralPath $modified -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Rollback verification failed'
}
Write-Output 'RESTORED_ORIGINAL_SELECTOR'
