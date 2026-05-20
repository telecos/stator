param(
    [string]$EdgeRepo = "Q:\Edge\src",
    [string]$EdgeBuildDir = "out\stator_release_candidate",
    [string]$OutputDir = "target\release-gate",
    [switch]$AllowDirty,
    [switch]$SkipMandatoryChecks,
    [switch]$SkipEdgeValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-GateStep {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Host "==> $Name"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Get-CargoPackage {
    param(
        [object]$Metadata,
        [string]$Name
    )

    $package = $Metadata.packages | Where-Object { $_.name -eq $Name } | Select-Object -First 1
    if ($null -eq $package) {
        throw "Cargo package '$Name' not found"
    }
    return $package
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
$commit = (& git rev-parse HEAD).Trim()
$status = (& git status --porcelain=v1)
$isClean = [string]::IsNullOrWhiteSpace(($status -join "`n"))

if ($branch -ne "main") {
    throw "Stator release candidates must be validated from branch 'main'; current branch is '$branch'"
}
if (-not $isClean -and -not $AllowDirty) {
    throw "Stator working tree must be clean. Re-run only for local script testing with -AllowDirty."
}

$outputPath = Join-Path $repoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$metadata = cargo metadata --format-version 1 --no-deps | ConvertFrom-Json
$publishOrder = @("stator_jse", "stator_jse_ffi", "st8")
$packages = @()
foreach ($name in $publishOrder) {
    $package = Get-CargoPackage -Metadata $metadata -Name $name
    $packages += [ordered]@{
        name = $package.name
        version = $package.version
        manifest_path = $package.manifest_path
    }
}
$privatePackage = Get-CargoPackage -Metadata $metadata -Name "stator_jse_test262"

$abiSource = Get-Content "crates\stator_ffi\src\lib.rs" -Raw
$abi = [ordered]@{}
foreach ($component in @("MAJOR", "MINOR", "PATCH")) {
    $pattern = "STATOR_FFI_ABI_VERSION_$component\s*:\s*u32\s*=\s*(\d+)"
    if ($abiSource -notmatch $pattern) {
        throw "Unable to read STATOR_FFI_ABI_VERSION_$component from crates\stator_ffi\src\lib.rs"
    }
    $abi[$component.ToLowerInvariant()] = [int]$Matches[1]
}
$abi["packed"] = (($abi["major"] -shl 16) -bor ($abi["minor"] -shl 8) -bor $abi["patch"])

if (-not $SkipMandatoryChecks) {
    Invoke-GateStep "cargo fmt --all" "cargo" @("fmt", "--all")
    Invoke-GateStep "cargo clippy --workspace -- -D warnings" "cargo" @("clippy", "--workspace", "--", "-D", "warnings")
    Invoke-GateStep "cargo build --workspace" "cargo" @("build", "--workspace")
    Invoke-GateStep "cargo test --workspace" "cargo" @("test", "--workspace")
    Invoke-GateStep "cargo build --workspace --release" "cargo" @("build", "--workspace", "--release")
}

Invoke-GateStep "generated header/ABI build" "cargo" @("build", "-p", "stator_jse_ffi")
Invoke-GateStep "generated header is committed" "git" @("diff", "--exit-code", "--", "crates\stator_ffi\include\stator.h")
Invoke-GateStep "FFI ABI contract tests" "cargo" @("test", "-p", "stator_jse_ffi", "--test", "abi_contract")

$packageDryRuns = @()
foreach ($name in $publishOrder) {
    $dirtyArg = @()
    if ($AllowDirty) {
        $dirtyArg = @("--allow-dirty")
    }
    Invoke-GateStep "cargo package --no-verify -p $name" "cargo" (@("package", "--no-verify", "-p", $name) + $dirtyArg)
    Invoke-GateStep "cargo publish --dry-run --no-verify -p $name" "cargo" (@("publish", "--dry-run", "--no-verify", "-p", $name) + $dirtyArg)
    $packageDryRuns += [ordered]@{
        package = $name
        package_command = "cargo package --no-verify -p $name"
        publish_dry_run_command = "cargo publish --dry-run --no-verify -p $name"
    }
}

$edgeCommands = @(
    "gn gen $EdgeBuildDir",
    "autoninja -C $EdgeBuildDir components\edge_stator_jse:edge_stator_jse",
    "autoninja -C $EdgeBuildDir components\edge_stator_jse:edge_stator_jse_conformance components\edge_stator_jse:edge_stator_jse_perfproof",
    "$EdgeBuildDir\edge_stator_jse_conformance.exe --output-json=$OutputDir\edge_stator_jse_conformance.json",
    "$EdgeBuildDir\edge_stator_jse_perfproof.exe --output-json=$OutputDir\edge_stator_jse_perfproof.json"
)

if (-not $SkipEdgeValidation) {
    if (-not (Test-Path $EdgeRepo)) {
        throw "Edge repo not found: $EdgeRepo"
    }
    Push-Location $EdgeRepo
    try {
        Invoke-GateStep "Edge candidate GN generation" "gn" @("gen", $EdgeBuildDir)
        Invoke-GateStep "Edge candidate component build" "autoninja" @("-C", $EdgeBuildDir, "components\edge_stator_jse:edge_stator_jse")
        Invoke-GateStep "Edge candidate smoke build" "autoninja" @("-C", $EdgeBuildDir, "components\edge_stator_jse:edge_stator_jse_conformance", "components\edge_stator_jse:edge_stator_jse_perfproof")
        Invoke-GateStep "Edge direct conformance smoke" (Join-Path $EdgeBuildDir "edge_stator_jse_conformance.exe") @("--output-json=$(Join-Path $outputPath 'edge_stator_jse_conformance.json')")
        Invoke-GateStep "Edge perfproof smoke" (Join-Path $EdgeBuildDir "edge_stator_jse_perfproof.exe") @("--output-json=$(Join-Path $outputPath 'edge_stator_jse_perfproof.json')")
    }
    finally {
        Pop-Location
    }
}

$report = [ordered]@{
    schema = "stator-edge-prepublish-gate.v1"
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    stator = [ordered]@{
        repo = $repoRoot.Path
        branch = $branch
        commit = $commit
        clean = $isClean
        mandatory_checks_required = -not $SkipMandatoryChecks
    }
    crates = $packages
    private_crates = @([ordered]@{
        name = $privatePackage.name
        version = $privatePackage.version
        publish = $false
    })
    ffi_abi = $abi
    package_dry_runs = $packageDryRuns
    generated_header = [ordered]@{
        path = "crates\stator_ffi\include\stator.h"
        check = "cargo build -p stator_jse_ffi; git diff --exit-code -- crates\stator_ffi\include\stator.h; cargo test -p stator_jse_ffi --test abi_contract"
    }
    edge = [ordered]@{
        repo = $EdgeRepo
        build_dir = $EdgeBuildDir
        skipped = [bool]$SkipEdgeValidation
        required_commands = $edgeCommands
    }
}

$reportPath = Join-Path $outputPath "edge-prepublish-gate-metadata.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $reportPath
Write-Host "Wrote release candidate metadata: $reportPath"
