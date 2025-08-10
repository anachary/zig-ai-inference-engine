param(
  [ValidateSet('svg','png','both')]
  [string]$Format = 'png',
  [string]$Theme = 'dark',
  [string]$BackgroundColor = '#0b1220',
  [switch]$UseDefaultConfig,
  [switch]$NoUpdateMarkdown
)

$ErrorActionPreference = 'Stop'

function Has-Command($name) {
  try { return (Get-Command $name -ErrorAction SilentlyContinue) -ne $null } catch { return $false }
}

function Render-Mermaid($inputPath, $outputPath, $Theme, $BackgroundColor, $ConfigPath) {
  $args = @('-i', $inputPath, '-o', $outputPath, '-t', $Theme, '-b', $BackgroundColor)
  if ($ConfigPath) { $args += @('-C', $ConfigPath) }

  if (Has-Command 'mmdc') {
    Write-Host "[mmdc] $inputPath -> $outputPath (theme=$Theme bg=$BackgroundColor config=$ConfigPath)"
    & mmdc @args
  } elseif (Has-Command 'npx') {
    Write-Host "[npx mermaid-cli] $inputPath -> $outputPath (theme=$Theme bg=$BackgroundColor config=$ConfigPath)"
    & npx @mermaid-js/mermaid-cli @args
  } else {
    throw "Neither 'mmdc' nor 'npx' is available. Please install Node.js and either mermaid-cli (mmdc) or allow using npx."
  }
  if ($LASTEXITCODE -ne 0) { throw "Render failed for $inputPath -> $outputPath (exit code $LASTEXITCODE)" }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcDir = Join-Path $ScriptDir 'src'
$MdPath = Join-Path (Split-Path $ScriptDir -Parent) 'sequence-diagrams.md'
$DefaultConfigPath = Join-Path $ScriptDir 'mermaid-config.json'
$ConfigPathToUse = $null
if ($UseDefaultConfig -and (Test-Path $DefaultConfigPath)) { $ConfigPathToUse = $DefaultConfigPath }

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
    Render-Mermaid -inputPath $inFile -outputPath $outSvg -Theme $Theme -BackgroundColor $BackgroundColor -ConfigPath $ConfigPathToUse
  }
  if ($Format -in @('png','both')) {
    $outPng = Join-Path $ScriptDir ($p.OutBase + '.png')
    Render-Mermaid -inputPath $inFile -outputPath $outPng -Theme $Theme -BackgroundColor $BackgroundColor -ConfigPath $ConfigPathToUse
  }
}

if (-not $NoUpdateMarkdown) {
  if (Test-Path $MdPath) {
    $ext = switch ($Format) {
      'svg' { '.svg' }
      'both' { '.png' } # prefer PNG for broad compat
      default { '.png' }
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

Write-Host "Sequence diagrams rendered successfully (format=$Format theme=$Theme bg=$BackgroundColor)."

