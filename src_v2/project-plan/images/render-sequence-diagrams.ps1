param(
  [ValidateSet('svg','png','both')]
  [string]$Format = 'svg',
  [switch]$NoUpdateMarkdown
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
    throw "Neither 'mmdc' nor 'npx' is available. Please install Node.js and either mermaid-cli (mmdc) or allow using npx."
  }
  if ($LASTEXITCODE -ne 0) { throw "Render failed for $inputPath -> $outputPath (exit code $LASTEXITCODE)" }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcDir = Join-Path $ScriptDir 'src'
$MdPath = Join-Path (Split-Path $ScriptDir -Parent) 'sequence-diagrams.md'

$Pairs = @(
  @{ In = 'cli-user-flow.mmd'; OutBase = 'cli-user-flow' },
  @{ In = 'parsing-flow.mmd'; OutBase = 'parsing-flow' },
  @{ In = 'runtime-execution-llama.mmd'; OutBase = 'runtime-execution-llama' }
)

foreach ($p in $Pairs) {
  $inFile = Join-Path $SrcDir $p.In
  if (-not (Test-Path $inFile)) {
    Write-Warning "Missing Mermaid source: $inFile"
    continue
  }

  if ($Format -in @('svg','both')) {
    $outSvg = Join-Path $ScriptDir ($p.OutBase + '.svg')
    Render-Mermaid -inputPath $inFile -outputPath $outSvg
  }
  if ($Format -in @('png','both')) {
    $outPng = Join-Path $ScriptDir ($p.OutBase + '.png')
    Render-Mermaid -inputPath $inFile -outputPath $outPng
  }
}

if (-not $NoUpdateMarkdown) {
  if (Test-Path $MdPath) {
    $ext = switch ($Format) {
      'png' { '.png' }
      default { '.svg' }
    }
    $content = Get-Content -Path $MdPath -Raw
    $content = $content -replace 'images/cli-user-flow\.(svg|png)', ('images/cli-user-flow' + $ext)
    $content = $content -replace 'images/parsing-flow\.(svg|png)', ('images/parsing-flow' + $ext)
    $content = $content -replace 'images/runtime-execution-llama\.(svg|png)', ('images/runtime-execution-llama' + $ext)
    Set-Content -Path $MdPath -Value $content -NoNewline
    Write-Host "Updated $MdPath to use extension: $ext"
  } else {
    Write-Warning "Markdown file not found: $MdPath"
  }
}

Write-Host "Sequence diagrams rendered successfully."

