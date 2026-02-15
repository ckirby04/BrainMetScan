#!/bin/bash
# Stacking v3b: Two experiments
#   Experiment 1: 2 models (no TTA) with v3 structural fixes
#   Experiment 2: All 4 models (no TTA) with v3 structural fixes
#
# Both reuse existing prediction caches where possible.
# Results are renamed between experiments to avoid overwriting.

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "EXPERIMENT 1: 2 models, no TTA, v3 structural fixes"
echo "  Reusing stacking_cache_v2 predictions (no regeneration)"
echo "============================================================"

python scripts/train_stacking.py \
    --models improved_24patch,improved_36patch \
    --regen-overlap 0.5 \
    --min-component 20 \
    --cache-dir stacking_cache_v2 \
    --fg-ratio 0.7 \
    --patch-size 64 \
    --stacking-patch 32 \
    --stacking-overlap 0.5

# Rename experiment 1 results before experiment 2 overwrites them
echo "Saving experiment 1 results..."
cp model/stacking_v3_results.json model/stacking_v3b_2model_results.json
cp model/stacking_v3_results_per_case.json model/stacking_v3b_2model_per_case.json
cp model/stacking_v3_classifier.pth model/stacking_v3b_2model_classifier.pth

echo ""
echo "============================================================"
echo "EXPERIMENT 2: All 4 models, no TTA, v3 structural fixes"
echo "  Cache: stacking_cache_4model (will generate if needed)"
echo "============================================================"

python scripts/train_stacking.py \
    --models exp1_8patch,exp3_12patch_maxfn,improved_24patch,improved_36patch \
    --regen-overlap 0.5 \
    --min-component 20 \
    --cache-dir stacking_cache_4model \
    --fg-ratio 0.7 \
    --patch-size 64 \
    --stacking-patch 32 \
    --stacking-overlap 0.5

# Rename experiment 2 results
echo "Saving experiment 2 results..."
cp model/stacking_v3_results.json model/stacking_v3b_4model_results.json
cp model/stacking_v3_results_per_case.json model/stacking_v3b_4model_per_case.json
cp model/stacking_v3_classifier.pth model/stacking_v3b_4model_classifier.pth

echo ""
echo "============================================================"
echo "DONE - Compare results:"
echo "  model/stacking_v3b_2model_results.json"
echo "  model/stacking_v3b_4model_results.json"
echo "============================================================"
