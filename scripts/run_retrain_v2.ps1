# Train all 4 v2 ensemble models at 256^3
# Safe to Ctrl+C and re-run - fully resumable

python scripts/retrain_256.py --model exp1_8patch
python scripts/retrain_256.py --model exp3_12patch_maxfn
python scripts/retrain_256.py --model improved_24patch
python scripts/retrain_256.py --model improved_36patch
