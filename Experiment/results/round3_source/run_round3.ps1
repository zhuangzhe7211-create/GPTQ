$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.hf_cache'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
if (-not (Test-Path 'data/round3_holdout/manifest.json')) {
    python prepare_round3_holdout.py
    if ($LASTEXITCODE -ne 0) { throw 'Holdout preparation failed' }
}
New-Item -ItemType Directory -Path results/round3_logs -Force | Out-Null
foreach ($pilotLayer in 6,12,18) {
    foreach ($pilotSeed in 0,1,2) {
        # Existing results deliberately cause the Python runner to refuse overwrite.
        if (Test-Path "results/round3_l${pilotLayer}_s$pilotSeed/metrics.json") { throw 'Existing run: use new output paths before rerunning' }
        python -u run_pilot.py --iterate --model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --device cuda --dtype float32 --calibration "data/wikitext2_20260926_s$pilotSeed/calibration.jsonl" --validation data/wikitext2_20260926_s0/validation.jsonl --target "model.layers.$pilotLayer.self_attn.o_proj" --samples 64 --eval-samples 64 --seq-len 256 --bits 4 --groups 4 --rho 0.5 --seed $pilotSeed --out "results/round3_l${pilotLayer}_s$pilotSeed" > "results/round3_logs/l${pilotLayer}_s$pilotSeed.log" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Scan failed: layer $pilotLayer seed $pilotSeed" }
        Write-Output "Completed layer $pilotLayer seed $pilotSeed"
    }
}
python -u evaluate_iteration.py
if ($LASTEXITCODE -ne 0) { throw 'Held-out evaluation failed' }
