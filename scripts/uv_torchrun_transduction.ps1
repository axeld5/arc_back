#!/usr/bin/env pwsh
param(
  [Parameter(Mandatory=$true, Position=0)]
  [ValidateSet("data-gen","sft","rl")]
  [string]$Task,

  [Parameter(Position=1)]
  [int]$NProc = 1,

  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$ForwardArgs
)

function Show-Usage {
  @"
Usage:
  .\scripts\uv_torchrun_transduction.ps1 -Task <data-gen|sft|rl> [-NProc 1] [-- <args>]

Examples:
  .\scripts\uv_torchrun_transduction.ps1 -Task data-gen -- --data_dir training_data --output transduction/train_dataset.json
  .\scripts\uv_torchrun_transduction.ps1 -Task sft -NProc 2
  .\scripts\uv_torchrun_transduction.ps1 -Task rl -NProc 2
"@
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "'uv' is not installed or not on PATH. See https://docs.astral.sh/uv/"
  exit 1
}

# Ensure we run from repo root so uv sees pyproject.toml
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

switch ($Task) {
  'data-gen' { $Module = 'transduction.data_gen' }
  'sft'      { $Module = 'transduction.training.sft' }
  'rl'       { $Module = 'transduction.training.rl' }
  default    { Show-Usage; exit 2 }
}

Write-Host "Running: uv run torchrun --standalone --nproc_per_node=$NProc -m $Module $ForwardArgs"
if ($null -ne $ForwardArgs -and $ForwardArgs.Length -gt 0) {
  uv run torchrun --standalone --nproc_per_node=$NProc -m $Module @ForwardArgs
} else {
  uv run torchrun --standalone --nproc_per_node=$NProc -m $Module
}

