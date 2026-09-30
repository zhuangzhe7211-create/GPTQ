$env:HF_HOME='C:\GPTQ\Experiment\.hf_cache'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
foreach ($seed in 0..2) {
    if (-not (Test-Path "results/round4_prefix_s$seed/manifest.json")) {
        python build_round4_prefix.py --seed $seed *> "results/round4_prefix_s$seed.log"
        if ($LASTEXITCODE -ne 0) { throw "prefix failed: $seed" }
    }
}
foreach ($prefix in 'rtn','gptq') {
    foreach ($module in 'attn','mlp') {
        $target = if ($module -eq 'attn') { 'model.layers.12.self_attn.o_proj' } else { 'model.layers.12.mlp.up_proj' }
        foreach ($seed in 0..2) {
            $folder = "results/round4_${prefix}_${module}_s$seed"
            if (Test-Path "$folder/metrics.json") {
                $metrics = Get-Content "$folder/metrics.json" -Raw | ConvertFrom-Json
                if ($metrics._summary) { Write-Output "ALREADY COMPLETE $folder"; continue }
                throw "Incomplete result exists: $folder. Preserve and investigate it before restarting."
            }
            $extra = @()
            if ($prefix -eq 'gptq') { $extra = @('--prefix-checkpoint',"results/round4_prefix_s$seed") }
            python run_pilot.py --model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --device cuda --dtype float32 --target $target --samples 64 --eval-samples 64 --seq-len 256 --bits 4 --seed $seed --calibration "data/wikitext2_20260926_s$seed/calibration.jsonl" --validation data/wikitext2_20260926_s0/validation.jsonl --round4 --out $folder @extra *> "$folder.log"
            if ($LASTEXITCODE -ne 0) { throw "scan failed: $folder" }
            Write-Output "DONE $folder"
        }
    }
}
python evaluate_round4.py *> results/round4_holdout.log
if ($LASTEXITCODE -ne 0) { throw 'holdout evaluation failed' }
