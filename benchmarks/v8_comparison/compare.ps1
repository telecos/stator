# Stator vs V8 Performance Comparison
#
# This script runs identical JS micro-benchmarks on V8 (via Node.js)
# and Stator (via Criterion), then produces a comparison table.
#
# Usage:
#   cd <repo-root>
#   pwsh benchmarks/v8_comparison/compare.ps1
#
# Prerequisites:
#   - Node.js in PATH
#   - Rust toolchain (cargo bench must work)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $RepoRoot

Write-Host "`n=== Stator vs V8 Performance Comparison ===" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host ""

# ── Step 1: Run V8 benchmarks ───────────────────────────────────────────
Write-Host "▸ Running V8 (Node.js) benchmarks..." -ForegroundColor Yellow
$v8Output = node benchmarks/v8_comparison/benchmarks.js 2>&1
$v8JsonLine = ($v8Output | Where-Object { $_ -match "^V8_BENCHMARK_RESULTS_JSON=" }) -replace "^V8_BENCHMARK_RESULTS_JSON=", ""
$v8Results = $v8JsonLine | ConvertFrom-Json

Write-Host "  ✓ V8 benchmarks complete ($($v8Results.Count) tests)" -ForegroundColor Green
$v8Output | Where-Object { $_ -notmatch "^V8_BENCHMARK_RESULTS_JSON=" } | ForEach-Object { Write-Host "  $_" }

# ── Step 2: Run Stator benchmarks ───────────────────────────────────────
Write-Host "`n▸ Running Stator (Criterion) benchmarks..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..."

$statorOutput = cargo bench --package stator_core --bench engine_benchmarks -- --output-format=bencher 2>&1
$statorLines = $statorOutput | Where-Object { $_ -match "^test .+ bench:" }

# Parse Criterion bencher output: "test <name> ... bench:      1234 ns/iter (+/- 56)"
$statorResults = @{}
foreach ($line in $statorLines) {
    if ($line -match "^test\s+(\S+)\s+\.\.\.\s+bench:\s+([\d,]+)\s+ns/iter") {
        $name = $Matches[1]
        $ns = [long]($Matches[2] -replace ",", "")
        $statorResults[$name] = $ns
    }
}

Write-Host "  ✓ Stator benchmarks complete ($($statorResults.Count) parsed)" -ForegroundColor Green

# ── Step 3: Build comparison table ───────────────────────────────────────
Write-Host "`n`n" -NoNewline
Write-Host "╔══════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    STATOR vs V8 PERFORMANCE COMPARISON                          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$header = "{0,-28} {1,14} {2,14} {3,10}" -f "Benchmark", "V8 (µs)", "Stator (µs)", "Ratio"
Write-Host $header -ForegroundColor White
Write-Host ("-" * 70)

$wins = 0
$losses = 0

foreach ($v8 in $v8Results) {
    $name = $v8.name
    $v8Us = [math]::Round($v8.median_ns / 1000.0, 1)

    $statorNs = $statorResults[$name]
    if ($null -eq $statorNs) {
        $row = "{0,-28} {1,14} {2,14} {3,10}" -f $name, $v8Us, "N/A", "-"
        Write-Host $row -ForegroundColor DarkGray
        continue
    }

    $statorUs = [math]::Round($statorNs / 1000.0, 1)
    $ratio = if ($v8Us -gt 0) { [math]::Round($statorUs / $v8Us, 2) } else { 0 }

    $color = if ($ratio -le 1.0) { "Green" } elseif ($ratio -le 3.0) { "Yellow" } else { "Red" }
    $label = if ($ratio -le 1.0) { "✓ ${ratio}x" } else { "${ratio}x" }

    if ($ratio -le 1.0) { $wins++ } else { $losses++ }

    $row = "{0,-28} {1,14} {2,14} {3,10}" -f $name, $v8Us, $statorUs, $label
    Write-Host $row -ForegroundColor $color
}

Write-Host ("-" * 70)
Write-Host ""
Write-Host "Summary: $wins benchmarks faster or equal, $losses benchmarks slower" -ForegroundColor Cyan
Write-Host ""
Write-Host "Notes:" -ForegroundColor DarkGray
Write-Host "  - V8 includes JIT compilation; Stator is currently interpreter-only" -ForegroundColor DarkGray
Write-Host "  - Ratio = Stator/V8 (lower is better; <1.0 means Stator is faster)" -ForegroundColor DarkGray
Write-Host "  - V8 uses median of multiple iterations; Stator uses Criterion median" -ForegroundColor DarkGray
Write-Host ""

Pop-Location
