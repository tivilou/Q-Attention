$ErrorActionPreference='Stop'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$baseline=Join-Path $root 'screen.original.py'
$sha=[System.Security.Cryptography.SHA256]::Create()
$digest=[BitConverter]::ToString($sha.ComputeHash([IO.File]::ReadAllBytes($baseline))).Replace('-','').ToLower()
$sha.Dispose()
if ($digest -ne '50ce858f78938151b87ac18711efff388fb0456a06c21df90108a7edea6933ab') {throw 'Original identity differs'}
Copy-Item -LiteralPath $baseline -Destination (Join-Path $root 'screen.py') -Force
Write-Output 'ROLLBACK_VERIFIED_ORIGINAL_SCREEN'
