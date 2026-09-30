$env:HF_HOME='C:\GPTQ\Experiment\.hf_cache'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
foreach ($layer in 12,18) {
    foreach ($seed in 0..2) {
        $folder="results/round6_l${layer}_s$seed"
        if (Test-Path "$folder/metrics.json") {
            $metrics=Get-Content "$folder/metrics.json" -Raw | ConvertFrom-Json
            if ($metrics._summary.complete) { Write-Output "ALREADY COMPLETE $folder"; continue }
            throw "Incomplete result exists: $folder"
        }
        python run_round6.py --layer $layer --seed $seed *> "$folder.log"
        if ($LASTEXITCODE -ne 0) { throw "Scan failed: $folder" }
        Write-Output "DONE $folder"
    }
}
python evaluate_round6.py *> results/round6_holdout.log
if ($LASTEXITCODE -ne 0) { throw 'External evaluation failed' }
