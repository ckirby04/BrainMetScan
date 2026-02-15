# Stacking v3b: Two experiments
#   Experiment 1: 2 models (no TTA) with v3 structural fixes
#   Experiment 2: All 4 models (no TTA) with v3 structural fixes

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

Write-Host "============================================================"
Write-Host "EXPERIMENT 1: 2 models, no TTA, v3 structural fixes"
Write-Host "  Reusing stacking_cache_v2 predictions (no regeneration)"
Write-Host "============================================================"

python scripts/train_stacking.py `
    --models improved_24patch,improved_36patch `
    --regen-overlap 0.5 `
    --min-component 20 `
    --cache-dir stacking_cache_v2 `
    --fg-ratio 0.7 `
    --patch-size 64 `
    --stacking-patch 32 `
    --stacking-overlap 0.5

# Rename experiment 1 results
Write-Host "Saving experiment 1 results..."
Copy-Item model/stacking_v3_results.json model/stacking_v3b_2model_results.json
Copy-Item model/stacking_v3_results_per_case.json model/stacking_v3b_2model_per_case.json
Copy-Item model/stacking_v3_classifier.pth model/stacking_v3b_2model_classifier.pth

Write-Host ""
Write-Host "============================================================"
Write-Host "EXPERIMENT 2: All 4 models, no TTA, v3 structural fixes"
Write-Host "  Cache: stacking_cache_4model (will generate if needed)"
Write-Host "============================================================"

python scripts/train_stacking.py `
    --models exp1_8patch,exp3_12patch_maxfn,improved_24patch,improved_36patch `
    --regen-overlap 0.5 `
    --min-component 20 `
    --cache-dir stacking_cache_4model `
    --fg-ratio 0.7 `
    --patch-size 64 `
    --stacking-patch 32 `
    --stacking-overlap 0.5

# Rename experiment 2 results
Write-Host "Saving experiment 2 results..."
Copy-Item model/stacking_v3_results.json model/stacking_v3b_4model_results.json
Copy-Item model/stacking_v3_results_per_case.json model/stacking_v3b_4model_per_case.json
Copy-Item model/stacking_v3_classifier.pth model/stacking_v3b_4model_classifier.pth

Write-Host ""
Write-Host "============================================================"
Write-Host "DONE - Compare results:"
Write-Host "  model/stacking_v3b_2model_results.json"
Write-Host "  model/stacking_v3b_4model_results.json"
Write-Host "============================================================"
