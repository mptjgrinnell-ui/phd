$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

git config core.hooksPath .githooks

Write-Host "Configured core.hooksPath=.githooks"
Write-Host "post-commit hook will snapshot into backup/backup.db after each commit."

