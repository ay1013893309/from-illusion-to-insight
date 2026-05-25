param(
    [ValidateSet("heuristic", "llm")]
    [string]$Backend = "heuristic",

    [string]$Model = "deepseek-chat",

    [int]$TopK = 5,

    [string]$TargetProject = "lucene",

    [string]$ArtifactDir = "artifacts",

    [string]$DataDir = "data/file_level_pairs"
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    if ($PSScriptRoot) {
        return (Resolve-Path $PSScriptRoot).Path
    }
    $scriptPath = $MyInvocation.MyCommand.Path
    if ($scriptPath) {
        return (Resolve-Path (Split-Path -Parent $scriptPath)).Path
    }
    return (Get-Location).Path
}

function Resolve-Python {
    param(
        [string]$ProjectRoot
    )

    $candidate = Join-Path (Split-Path -Parent $ProjectRoot) ".venv\Scripts\python.exe"
    if (Test-Path $candidate) {
        return (Resolve-Path $candidate).Path
    }

    $fallback = Get-Command python -ErrorAction SilentlyContinue
    if ($fallback) {
        return $fallback.Source
    }

    throw "No Python interpreter found. Expected .venv\\Scripts\\python.exe or python in PATH."
}

$projectRoot = Resolve-ProjectRoot
$pythonExe = Resolve-Python -ProjectRoot $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

$dataRoot = Join-Path $projectRoot $DataDir
$artifactRoot = Join-Path $projectRoot $ArtifactDir
$manifestPath = Join-Path $dataRoot "manifest.csv"

if (-not (Test-Path $manifestPath)) {
    throw "Missing manifest file at $manifestPath. Run prepare-bulk first."
}

$targetDir = Join-Path $dataRoot $TargetProject
if (-not (Test-Path $targetDir)) {
    throw "Target project directory not found: $targetDir"
}

$allJsonl = Get-ChildItem $dataRoot -Recurse -Filter *.jsonl | Where-Object { -not $_.PSIsContainer }
$targetFiles = Get-ChildItem $targetDir -Filter *.jsonl | Where-Object { -not $_.PSIsContainer }
$sourceFiles = $allJsonl | Where-Object { $_.DirectoryName -ne $targetDir }

if ($targetFiles.Count -eq 0) {
    throw "No target jsonl files found under $targetDir"
}

if ($sourceFiles.Count -eq 0) {
    throw "No source jsonl files found outside target project $TargetProject"
}

New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$artifactPath = Join-Path $artifactRoot ("cpdp_{0}_source.pkl" -f $TargetProject)
$outputPath = Join-Path $artifactRoot ("{0}_cpdp_predictions_{1}.csv" -f $TargetProject, $Backend)

Write-Host "Target project: $TargetProject"
Write-Host "Backend: $Backend"
Write-Host "Model: $Model"
Write-Host "Top-K retrieval: $TopK"
Write-Host "Source files: $($sourceFiles.Count)"
Write-Host "Target files: $($targetFiles.Count)"
Write-Host "Artifact: $artifactPath"
Write-Host "Output: $outputPath"

& $pythonExe -B -m cpdp_change_aware.cli fit `
    --sources $sourceFiles.FullName `
    --artifact-path $artifactPath

& $pythonExe -B -m cpdp_change_aware.cli predict `
    --artifact-path $artifactPath `
    --target $targetFiles.FullName `
    --output $outputPath `
    --backend $Backend `
    --model $Model `
    --top-k $TopK

Write-Host ""
Write-Host "Completed cross-project prediction for $TargetProject."
Write-Host "Predictions saved to: $outputPath"
