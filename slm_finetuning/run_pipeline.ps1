# run_pipeline.ps1

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor DarkGray
    . ".\venv\Scripts\Activate.ps1"
}

Write-Host "Starting SLM Fine-Tuning Pipeline..." -ForegroundColor Cyan

# 1. Train microsoft/Phi-3.5-mini-instruct
Write-Host "`n--- Training Phi-3.5-mini-instruct ---" -ForegroundColor Green
python train.py --model_id "microsoft/Phi-3.5-mini-instruct" --epochs 4
if ($LASTEXITCODE -ne 0) { 
    Write-Error "Training Phi-3.5 failed"
    exit $LASTEXITCODE 
}

# 2. Train Qwen/Qwen2.5-7B-Instruct
Write-Host "`n--- Training Qwen2.5-7B-Instruct ---" -ForegroundColor Green
python train.py --model_id "Qwen/Qwen2.5-7B-Instruct" --epochs 3
if ($LASTEXITCODE -ne 0) { 
    Write-Error "Training Qwen2.5 failed"
    exit $LASTEXITCODE 
}

# 3. Evaluate Phi-3.5-mini-instruct
Write-Host "`n--- Evaluating Phi-3.5-mini-instruct ---" -ForegroundColor Yellow
python evaluate.py --model_id "microsoft/Phi-3.5-mini-instruct" --adapter_dir "./lora_adapter_Phi-3.5-mini-instruct" --test_file "test_set_Phi-3.5-mini-instruct.json"

# 4. Evaluate Qwen/Qwen2.5-7B-Instruct
Write-Host "`n--- Evaluating Qwen2.5-7B-Instruct ---" -ForegroundColor Yellow
python evaluate.py --model_id "Qwen/Qwen2.5-7B-Instruct" --adapter_dir "./lora_adapter_Qwen2.5-7B-Instruct" --test_file "test_set_Qwen2.5-7B-Instruct.json"

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Pipeline completed! Review the accuracy percentages above." -ForegroundColor White
Write-Host "To patch your JSON files with predictions and certainty scores, pick the winning model and run one of these commands:" -ForegroundColor White
Write-Host "For Qwen2.5:" -ForegroundColor DarkGray
Write-Host "  python predict.py --model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir ./lora_adapter_Qwen2.5-7B-Instruct" -ForegroundColor DarkGray
Write-Host "For Phi-3.5:" -ForegroundColor DarkGray
Write-Host "  python predict.py --model_id microsoft/Phi-3.5-mini-instruct --adapter_dir ./lora_adapter_Phi-3.5-mini-instruct" -ForegroundColor DarkGray
