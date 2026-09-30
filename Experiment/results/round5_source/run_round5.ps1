$env:HF_HOME='C:\GPTQ\Experiment\.hf_cache'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
foreach ($prefix in 'rtn','gptq') {
    foreach ($seed in 0..2) {
        $folder="results/round5_${prefix}_s$seed"
        if (Test-Path "$folder/metrics.json") {
            $metrics=Get-Content "$folder/metrics.json" -Raw | ConvertFrom-Json
            if ($metrics._summary.complete) { Write-Output "ALREADY COMPLETE $folder"; continue }
            throw "Incomplete result exists: $folder"
        }
        python run_round5.py --prefix $prefix --seed $seed *> "$folder.log"
        if ($LASTEXITCODE -ne 0) { throw "Scan failed: $folder" }
        Write-Output "DONE $folder"
    }
}
python evaluate_round5.py *> results/round5_holdout.log
if ($LASTEXITCODE -ne 0) { throw 'Holdout failed' }
