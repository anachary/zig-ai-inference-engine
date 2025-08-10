param(
  [ValidateSet('png','svg')]
  [string]$Format = 'png'
)

$ErrorActionPreference = 'Stop'

function Has-Command($name) {
  try { return (Get-Command $name -ErrorAction SilentlyContinue) -ne $null } catch { return $false }
}

function Render-Mermaid($inputPath, $outputPath) {
  if (Has-Command 'mmdc') {
    Write-Host "[mmdc] $inputPath -> $outputPath"
    & mmdc -i $inputPath -o $outputPath
  } elseif (Has-Command 'npx') {
    Write-Host "[npx mermaid-cli] $inputPath -> $outputPath"
    & npx @mermaid-js/mermaid-cli -i $inputPath -o $outputPath
  } else {
    throw "Neither 'mmdc' nor 'npx' is available. Install Node.js and mermaid-cli (mmdc) or use npx."
  }
  if ($LASTEXITCODE -ne 0) { throw "Render failed (exit code $LASTEXITCODE)" }
}

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Input = Join-Path $RepoRoot 'docs/diagrams/ollama_gguf_llama_flow.mmd'
$OutDir = Join-Path $RepoRoot 'docs/images'
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$OutExt = if ($Format -eq 'svg') { '.svg' } else { '.png' }
$Output = Join-Path $OutDir ('ollama_gguf_llama_flow' + $OutExt)

Render-Mermaid -inputPath $Input -outputPath $Output

Write-Host "Rendered: $Output"

