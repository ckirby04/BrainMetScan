"""
Overnight pipeline to improve the native-geometry model.

Phase 0 (~1-2h): Quick wins on existing models (TTA + threshold sweep + postprocessing)
Phase 1 (~6-8h): Retrain 4 base models with v3 improvements
Phase 2 (~3-4h): Rebuild stacking cache + retrain stacking classifier
Phase 3 (~1h):   Full evaluation + comparison

GPU assignment:
  GPU 0 (RTX 3070 Ti,  8.6 GB): 8-patch + 12-patch (sequential)
  GPU 1 (RTX 5060 Ti, 17.1 GB): 24-patch + 36-patch (sequential)
  Phases 0/2/3: GPU 1 (more VRAM)

Estimated total: ~11-13 hours

Usage:
    python scripts/training/native_overnight.py
    python scripts/training/native_overnight.py --epochs 250
    python scripts/training/native_overnight.py --skip-phase0 --skip-phase1  # just rebuild stacking
"""

import argparse
import gc
import json
import subprocess
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_native_v3.py"
DATA_DIR = ROOT / "data" / "preprocessed_256" / "train"
MODEL_DIR = ROOT / "model"
V3_MODEL_DIR = ROOT / "model" / "native_v3"
RESULTS_DIR = ROOT / "results" / "native_v3"
OLD_CACHE_DIR = MODEL_DIR / "stacking_cache_v4"
NEW_CACHE_DIR = MODEL_DIR / "stacking_cache_v5"

PYTHON = sys.executable

# Model name mapping: v3 checkpoint name -> stacking cache key
V3_MODELS = {
    8:  {"v3_name": "v3_8patch",  "cache_key": "exp1_8patch",          "old_ckpt": "exp1_8patch_finetuned.pth"},
    12: {"v3_name": "v3_12patch", "cache_key": "exp3_12patch_maxfn",   "old_ckpt": "exp3_12patch_maxfn_finetuned.pth"},
    24: {"v3_name": "v3_24patch", "cache_key": "improved_24patch",     "old_ckpt": "improved_24patch_finetuned.pth"},
    36: {"v3_name": "v3_36patch", "cache_key": "improved_36patch",     "old_ckpt": "improved_36patch_finetuned.pth"},
}

STACKING_MODEL_NAMES = [
    'exp1_8patch', 'exp3_12patch_maxfn', 'improved_24patch',
    'improved_36patch', 'nnunet', 'nnunet_2d',
]


def run_cmd(cmd, desc, wait=True):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(ROOT),
        stdout=None,
        stderr=subprocess.STDOUT,
    )
    if wait:
        proc.wait()
        return proc.returncode
    return proc


def main():
    parser = argparse.ArgumentParser(description="Native model overnight improvement pipeline")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patches-per-volume", type=int, default=5)
    parser.add_argument("--skip-phase0", action="store_true", help="Skip TTA/threshold quick wins")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip base model retraining")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip stacking cache rebuild")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip final evaluation")
    args = parser.parse_args()

    start_time = time.time()

    print(f"""
################################################################################
#                                                                              #
#   NATIVE MODEL OVERNIGHT IMPROVEMENT PIPELINE                                #
#                                                                              #
#   Phase 0: Quick wins (TTA + threshold + postprocessing on existing models)  #
#   Phase 1: Retrain 4 base models (SmallLesionLoss + MONAI + weighted)        #
#   Phase 2: Rebuild stacking cache + retrain stacking classifier              #
#   Phase 3: Full lesion-level evaluation + comparison                         #
#                                                                              #
#   Dataset: BrainMetShare (156) + UCSF-BMSR (410) = 566 cases                #
#   GPUs: GPU 0 (RTX 3070 Ti) + GPU 1 (RTX 5060 Ti)                           #
#   Epochs: {args.epochs}                                                           #
#                                                                              #
################################################################################
""")

    V3_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PHASE 0: Quick wins on existing models
    # ========================================================================
    if not args.skip_phase0:
        print(f"\n{'#'*70}")
        print(f"# PHASE 0: Quick wins (TTA + threshold sweep + postprocessing)")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")
        phase0_quick_wins()

    # ========================================================================
    # PHASE 1: Retrain 4 base models in parallel
    # ========================================================================
    if not args.skip_phase1:
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: Retrain base models with v3 improvements")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

        # Batch 1: 8-patch (GPU 0) + 24-patch (GPU 1) in parallel
        print(f"\n  Batch 1: 8-patch (GPU 0) + 24-patch (GPU 1)")
        procs = []
        for ps, gpu in [(8, 0), (24, 1)]:
            p = run_cmd(
                [PYTHON, str(TRAIN_SCRIPT),
                 "--patch-size", str(ps), "--gpu", str(gpu),
                 "--epochs", str(args.epochs),
                 "--patches-per-volume", str(args.patches_per_volume)],
                f"v3 {ps}-patch on GPU {gpu}",
                wait=False,
            )
            procs.append((f"{ps}-patch", p))

        for name, proc in procs:
            print(f"\n  Waiting for {name} (PID {proc.pid})...")
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

        # Batch 2: 12-patch (GPU 0) + 36-patch (GPU 1) in parallel
        print(f"\n  Batch 2: 12-patch (GPU 0) + 36-patch (GPU 1)")
        procs = []
        for ps, gpu in [(12, 0), (36, 1)]:
            p = run_cmd(
                [PYTHON, str(TRAIN_SCRIPT),
                 "--patch-size", str(ps), "--gpu", str(gpu),
                 "--epochs", str(args.epochs),
                 "--patches-per-volume", str(args.patches_per_volume)],
                f"v3 {ps}-patch on GPU {gpu}",
                wait=False,
            )
            procs.append((f"{ps}-patch", p))

        for name, proc in procs:
            print(f"\n  Waiting for {name} (PID {proc.pid})...")
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

        # Print training results summary
        print(f"\n  Phase 1 training results:")
        for ps in [8, 12, 24, 36]:
            state_path = V3_MODEL_DIR / f"v3_{ps}patch_state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                print(f"    {ps}-patch: best_dice={state['best_val_dice']:.4f} "
                      f"(ep {state['best_epoch']}), tiny={state.get('best_tiny_dice', 'N/A')}")

    # ========================================================================
    # PHASE 2: Rebuild stacking cache + retrain stacking classifier
    # ========================================================================
    if not args.skip_phase2:
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: Rebuild stacking cache + retrain stacking classifier")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")
        phase2_rebuild_stacking()

    # ========================================================================
    # PHASE 3: Full evaluation
    # ========================================================================
    if not args.skip_phase3:
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: Full evaluation + comparison")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")
        phase3_evaluate()

    total_hours = (time.time() - start_time) / 3600
    print(f"""
################################################################################
#                                                                              #
#   OVERNIGHT PIPELINE COMPLETE                                                #
#   Total time: {total_hours:.1f} hours                                              #
#   v3 models: {V3_MODEL_DIR}                                 #
#   Stacking cache: {NEW_CACHE_DIR}                    #
#   Results: {RESULTS_DIR}                                     #
#                                                                              #
################################################################################
""")


# ============================================================================
# PHASE 0: Quick wins
# ============================================================================

def phase0_quick_wins():
    """Apply TTA + threshold sweep + postprocessing to existing stacking cache."""
    sys.path.insert(0, str(ROOT / "src"))

    import torch
    from segmentation.stacking import (
        StackingClassifier, load_stacking_model, build_stacking_features,
        sliding_window_inference, postprocess_prediction,
        STACKING_MODEL_NAMES, STACKING_PATCH_SIZE, STACKING_OVERLAP,
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load existing stacking classifier
    stacking_model = load_stacking_model(device=device)
    if stacking_model is None:
        print("  WARNING: No stacking classifier found, skipping Phase 0")
        return

    # Get validation cases (same split as all evaluations)
    cached_cases = sorted([f.stem for f in OLD_CACHE_DIR.glob("*.npz")])
    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)
    n_val = int(len(cases_shuffled) * 0.15)
    val_cases = cases_shuffled[:n_val]

    print(f"  Evaluating {len(val_cases)} val cases with threshold + postprocessing sweep...")

    # Sweep parameters
    thresholds = [0.80, 0.85, 0.90, 0.92, 0.95]
    min_sizes = [10, 20, 50, 100]

    best_combo = {"threshold": 0.9, "min_size": 20, "dice": 0}
    all_results = {}

    for threshold in thresholds:
        for min_size in min_sizes:
            dices = []
            for case_id in val_cases:
                cache_file = OLD_CACHE_DIR / f"{case_id}.npz"
                if not cache_file.exists():
                    continue

                features, preds, mask = build_stacking_features(cache_file)
                prob = sliding_window_inference(
                    stacking_model, features, STACKING_PATCH_SIZE, device, overlap=STACKING_OVERLAP
                )
                pred = (prob > threshold).astype(np.float32)
                pred = postprocess_prediction(pred, min_size=min_size)

                inter = (pred * mask).sum()
                total = pred.sum() + mask.sum()
                dice = 2 * inter / total if total > 0 else (1.0 if mask.sum() == 0 else 0.0)
                dices.append(dice)

            mean_dice = float(np.mean(dices))
            key = f"t={threshold:.2f}_ms={min_size}"
            all_results[key] = mean_dice

            if mean_dice > best_combo["dice"]:
                best_combo = {"threshold": threshold, "min_size": min_size, "dice": mean_dice}

    # Also evaluate TTA on top-3 average (no stacking model)
    top3_keys = ['exp3_12patch_maxfn', 'improved_24patch', 'improved_36patch']
    tta_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    best_top3 = {"threshold": 0.40, "dice": 0}

    print(f"  Evaluating top-3 average with flip TTA...")

    for threshold in tta_thresholds:
        dices = []
        for case_id in val_cases:
            cache_file = OLD_CACHE_DIR / f"{case_id}.npz"
            if not cache_file.exists():
                continue

            data = np.load(cache_file)
            mask = data['mask']

            # Average top-3 predictions
            probs = np.mean([data[k] for k in top3_keys], axis=0)

            # Simple TTA: average with flipped versions
            tta_probs = [probs]
            for axis in [0, 1, 2]:
                flipped = np.flip(probs, axis=axis)
                tta_probs.append(flipped)
            avg_prob = np.mean(tta_probs, axis=0)

            pred = (avg_prob > threshold).astype(np.float32)
            pred = postprocess_prediction(pred, min_size=20)

            inter = (pred * mask).sum()
            total = pred.sum() + mask.sum()
            dice = 2 * inter / total if total > 0 else (1.0 if mask.sum() == 0 else 0.0)
            dices.append(dice)

        mean_dice = float(np.mean(dices))
        if mean_dice > best_top3["dice"]:
            best_top3 = {"threshold": threshold, "dice": mean_dice}

    # Print results
    print(f"\n  {'='*60}")
    print(f"  PHASE 0 RESULTS (existing models, no retraining)")
    print(f"  {'='*60}")
    print(f"\n  Stacking classifier sweep:")
    print(f"    Best: threshold={best_combo['threshold']}, min_size={best_combo['min_size']}, "
          f"Dice={best_combo['dice']:.4f}")
    print(f"    (baseline: threshold=0.9, min_size=20)")
    print(f"\n  Top-3 average with flip TTA:")
    print(f"    Best: threshold={best_top3['threshold']}, Dice={best_top3['dice']:.4f}")
    print(f"    (baseline: threshold=0.40, Dice=0.7475)")
    print(f"  {'='*60}")

    # Save
    report = {
        "stacking_sweep": all_results,
        "stacking_best": best_combo,
        "top3_tta_best": best_top3,
        "n_val_cases": len(val_cases),
        "timestamp": datetime.now().isoformat(),
    }
    report_path = RESULTS_DIR / "phase0_quick_wins.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    del stacking_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# PHASE 2: Rebuild stacking cache + retrain stacking classifier
# ============================================================================

def phase2_rebuild_stacking():
    """
    Generate new stacking cache (v5) with v3 base models + existing nnU-Net,
    then retrain the stacking classifier.
    """
    sys.path.insert(0, str(ROOT / "src"))

    import torch
    from segmentation.unet import LightweightUNet3D
    from segmentation.stacking import sliding_window_inference

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    NEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load v3 base models
    v3_models = {}
    for ps, info in V3_MODELS.items():
        ckpt_path = V3_MODEL_DIR / f"{info['v3_name']}_best.pth"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, falling back to original")
            # Fall back to original model
            old_path = MODEL_DIR / info["old_ckpt"]
            if old_path.exists():
                ckpt_path = old_path
            else:
                print(f"  ERROR: Neither v3 nor original checkpoint found for {ps}-patch")
                continue

        model = LightweightUNet3D(
            in_channels=4, out_channels=1,
            base_channels=20, use_attention=True, use_residual=True,
        ).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        v3_models[ps] = model
        dice = ckpt.get("val_dice", ckpt.get("dice", "N/A"))
        print(f"  Loaded {ps}-patch from {ckpt_path.name} (dice={dice})")

    # Get all cases from existing cache
    old_cache_files = sorted(OLD_CACHE_DIR.glob("*.npz"))
    print(f"\n  Rebuilding stacking cache for {len(old_cache_files)} cases...")
    print(f"  New v3 predictions for 4 lightweight models, keeping nnU-Net from v4 cache")

    # Load volumes and regenerate predictions
    from segmentation.dataset import BrainMetDataset

    # We need to detect sequences from the data
    sample_case = list(DATA_DIR.iterdir())[0]
    has_t2 = (sample_case / "t2.nii.gz").exists()
    has_bravo = (sample_case / "bravo.nii.gz").exists()
    if has_t2:
        sequences = ["t1_pre", "t1_gd", "flair", "t2"]
    elif has_bravo:
        sequences = ["t1_pre", "t1_gd", "flair", "bravo"]
    else:
        sequences = ["t1_pre", "t1_gd", "flair", "t2"]

    import nibabel as nib
    from scipy.ndimage import zoom

    target_size = (128, 128, 128)
    n_done = 0
    n_total = len(old_cache_files)

    for cache_file in old_cache_files:
        case_id = cache_file.stem
        new_cache_file = NEW_CACHE_DIR / f"{case_id}.npz"

        # Skip if already done
        if new_cache_file.exists():
            n_done += 1
            continue

        # Load existing cache for nnU-Net predictions and mask
        old_data = np.load(str(cache_file))
        mask = old_data["mask"]

        # Copy nnU-Net predictions from old cache
        save_dict = {"mask": mask}
        for key in ["nnunet", "nnunet_2d"]:
            if key in old_data:
                save_dict[key] = old_data[key]

        # Load volume for v3 predictions
        case_dir = DATA_DIR / case_id
        if case_dir.exists():
            try:
                channels = []
                for seq in sequences:
                    nii_path = case_dir / f"{seq}.nii.gz"
                    if not nii_path.exists():
                        raise FileNotFoundError(f"Missing {seq} for {case_id}")
                    data = np.asarray(nib.load(str(nii_path)).dataobj, dtype=np.float32)
                    # Resize to 128^3 (model input size)
                    if data.shape != target_size:
                        factors = [t / s for t, s in zip(target_size, data.shape)]
                        data = zoom(data, factors, order=1)
                    mean, std = data.mean(), data.std()
                    if std > 0:
                        data = (data - mean) / std
                    channels.append(data)
                volume = np.stack(channels, axis=0)

                # Generate v3 predictions (at 128^3) and upsample to 256^3
                full_size = mask.shape  # (256, 256, 256)
                for ps, model in v3_models.items():
                    cache_key = V3_MODELS[ps]["cache_key"]
                    prob = sliding_window_inference(
                        model, volume, ps, device, overlap=0.5
                    )
                    # Upsample from 128^3 to 256^3 to match nnU-Net and mask
                    if prob.shape != full_size:
                        up_factors = [f / p for f, p in zip(full_size, prob.shape)]
                        prob = zoom(prob, up_factors, order=1)
                    save_dict[cache_key] = prob.astype(np.float16)

            except Exception as e:
                print(f"  WARNING: Failed to process {case_id}: {e}")
                # Fall back to old predictions for this case
                for ps, info in V3_MODELS.items():
                    key = info["cache_key"]
                    if key in old_data:
                        save_dict[key] = old_data[key]
        else:
            # Case dir doesn't exist, keep old predictions
            for ps, info in V3_MODELS.items():
                key = info["cache_key"]
                if key in old_data:
                    save_dict[key] = old_data[key]

        np.savez_compressed(str(new_cache_file), **save_dict)
        n_done += 1

        if n_done % 50 == 0:
            print(f"    {n_done}/{n_total} cases...", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"  Stacking cache v5: {n_done} cases in {NEW_CACHE_DIR}")

    # Unload v3 models
    del v3_models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Retrain stacking classifier on new cache
    print(f"\n  Retraining stacking classifier on v5 cache...")
    retrain_stacking_classifier(device)


def retrain_stacking_classifier(device):
    """Train a new stacking classifier on the v5 cache."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import autocast, GradScaler

    from segmentation.stacking import (
        StackingClassifier, sliding_window_inference,
        STACKING_PATCH_SIZE, STACKING_OVERLAP,
    )

    # Data split (same as all other evals)
    cached_cases = sorted([f.stem for f in NEW_CACHE_DIR.glob("*.npz")])
    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)

    n_val = int(len(cases_shuffled) * 0.15)
    val_cases = cases_shuffled[:n_val]
    train_cases = cases_shuffled[n_val:]

    # Further split train into stacking-train and stacking-val
    n_stack_val = int(len(train_cases) * 0.15)
    stack_val = train_cases[:n_stack_val]
    stack_train = train_cases[n_stack_val:]

    print(f"    Stacking train: {len(stack_train)}, Stacking val: {len(stack_val)}, "
          f"Final eval: {n_val}")

    PATCH = 32  # patch size at full 256^3 resolution

    class StackingPatchDataset(Dataset):
        def __init__(self, case_ids, cache_dir, patch_size=PATCH, fg_ratio=0.7):
            self.case_ids = case_ids
            self.cache_dir = Path(cache_dir)
            self.patch_size = patch_size
            self.fg_ratio = fg_ratio

        def __len__(self):
            return len(self.case_ids)

        def __getitem__(self, idx):
            case_id = self.case_ids[idx]
            data = np.load(str(self.cache_dir / f"{case_id}.npz"))
            mask = data["mask"].astype(np.float32)

            # Extract patch location FIRST, then only load the patch region
            p = self.patch_size
            H, W, D = mask.shape
            fg = np.where(mask > 0)

            if len(fg[0]) > 0 and np.random.rand() < self.fg_ratio:
                i = np.random.randint(len(fg[0]))
                ch, cw, cd = fg[0][i], fg[1][i], fg[2][i]
            else:
                ch = np.random.randint(p // 2, max(H - p // 2, p // 2 + 1))
                cw = np.random.randint(p // 2, max(W - p // 2, p // 2 + 1))
                cd = np.random.randint(p // 2, max(D - p // 2, p // 2 + 1))

            h0 = max(0, min(ch - p // 2, H - p))
            w0 = max(0, min(cw - p // 2, W - p))
            d0 = max(0, min(cd - p // 2, D - p))
            h1, w1, d1 = h0 + p, w0 + p, d0 + p

            # Load only the patch region from each prediction
            preds = []
            for name in STACKING_MODEL_NAMES:
                if name in data:
                    preds.append(data[name][h0:h1, w0:w1, d0:d1].astype(np.float32))
                else:
                    preds.append(np.zeros((p, p, p), dtype=np.float32))
            preds = np.stack(preds, axis=0)  # (6, p, p, p)

            variance = preds.var(axis=0, keepdims=True)
            range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
            features = np.concatenate([preds, variance, range_map], axis=0)  # (8, p, p, p)

            mask_patch = mask[h0:h1, w0:w1, d0:d1]

            return (torch.from_numpy(features).float(),
                    torch.from_numpy(mask_patch[None]).float())

    train_ds = StackingPatchDataset(stack_train, NEW_CACHE_DIR, patch_size=PATCH, fg_ratio=0.7)
    val_ds = StackingPatchDataset(stack_val, NEW_CACHE_DIR, patch_size=PATCH, fg_ratio=0.7)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0,
                            pin_memory=True)

    in_channels = len(STACKING_MODEL_NAMES) + 2  # 6 + variance + range = 8
    model = StackingClassifier(in_channels=in_channels).to(device)
    print(f"    Stacking model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{in_channels} input channels")

    # Use SmallLesionOptimizedLoss for stacking too
    from segmentation.advanced_losses import SmallLesionOptimizedLoss
    criterion = SmallLesionOptimizedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    scaler = GradScaler("cuda")

    best_dice = 0
    best_path = MODEL_DIR / "stacking_v5_classifier.pth"

    for epoch in range(1, 151):
        # Train
        model.train()
        train_loss = 0
        for feat, mask in train_loader:
            feat, mask = feat.to(device), mask.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                out = model(feat)
                loss = criterion(out, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_dices = []
        with torch.no_grad():
            for feat, mask in val_loader:
                feat, mask = feat.to(device), mask.to(device)
                with autocast("cuda"):
                    out = model(feat)
                pred = (torch.sigmoid(out) > 0.5).float()
                inter = (pred * mask).sum()
                total = pred.sum() + mask.sum()
                dice = (2 * inter / (total + 1e-6)).item()
                val_dices.append(dice)
        val_dice = np.mean(val_dices)

        scheduler.step()

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "in_channels": in_channels,
                "model_names": STACKING_MODEL_NAMES,
                "training": "v5 (SmallLesionLoss, v3 base models)",
            }, best_path)

        if epoch % 25 == 0 or is_best:
            mark = " *" if is_best else ""
            print(f"    Epoch {epoch:3d}/150: loss={train_loss:.4f} dice={val_dice:.3f}{mark}")

    print(f"    Stacking v5: best Dice={best_dice:.4f}, saved to {best_path}")

    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# PHASE 3: Full evaluation
# ============================================================================

def phase3_evaluate():
    """Full lesion-level evaluation comparing v3 pipeline vs baseline."""
    sys.path.insert(0, str(ROOT / "src"))

    import torch
    from segmentation.stacking import (
        StackingClassifier, build_stacking_features, sliding_window_inference,
        postprocess_prediction, STACKING_PATCH_SIZE, STACKING_OVERLAP,
        STACKING_MODEL_NAMES as SM_NAMES,
    )
    from scipy.ndimage import label as ndimage_label

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load v5 stacking classifier
    v5_ckpt_path = MODEL_DIR / "stacking_v5_classifier.pth"
    v4_ckpt_path = MODEL_DIR / "stacking_v4_classifier.pth"

    results = {}

    for version, ckpt_path, cache_dir in [
        ("v4_baseline", v4_ckpt_path, OLD_CACHE_DIR),
        ("v5_improved", v5_ckpt_path, NEW_CACHE_DIR),
    ]:
        if not ckpt_path.exists():
            print(f"  Skipping {version}: checkpoint not found")
            continue

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        in_ch = ckpt.get("in_channels", 8)
        model = StackingClassifier(in_channels=in_ch).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Same val split
        cached_cases = sorted([f.stem for f in cache_dir.glob("*.npz")])
        random.seed(42)
        cases_shuffled = cached_cases.copy()
        random.shuffle(cases_shuffled)
        n_val = int(len(cases_shuffled) * 0.15)
        val_cases = cases_shuffled[:n_val]

        print(f"\n  Evaluating {version} on {len(val_cases)} val cases...")

        # Sweep thresholds
        thresholds = [0.85, 0.90, 0.92, 0.95]
        best_t, best_dice = 0.9, 0

        for threshold in thresholds:
            dices = []
            for case_id in val_cases:
                cache_file = cache_dir / f"{case_id}.npz"
                if not cache_file.exists():
                    continue

                features, preds, mask = build_stacking_features(
                    cache_file, model_names=SM_NAMES
                )
                prob = sliding_window_inference(
                    model, features, STACKING_PATCH_SIZE, device, overlap=STACKING_OVERLAP
                )
                pred = (prob > threshold).astype(np.float32)
                pred = postprocess_prediction(pred, min_size=20)

                # Voxel dice
                inter = (pred * mask).sum()
                total = pred.sum() + mask.sum()
                dice = 2 * inter / total if total > 0 else (1.0 if mask.sum() == 0 else 0.0)
                dices.append(dice)

            mean_dice = float(np.mean(dices))
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_t = threshold

        # Run detailed eval at best threshold
        voxel_dices = []
        lesion_dices = []
        detections = {"tp": 0, "fp": 0, "fn": 0}

        for case_id in val_cases:
            cache_file = cache_dir / f"{case_id}.npz"
            if not cache_file.exists():
                continue

            features, preds, mask = build_stacking_features(cache_file, model_names=SM_NAMES)
            prob = sliding_window_inference(
                model, features, STACKING_PATCH_SIZE, device, overlap=STACKING_OVERLAP
            )
            pred = (prob > best_t).astype(np.float32)
            pred = postprocess_prediction(pred, min_size=20)

            # Voxel dice
            inter = (pred * mask).sum()
            total = pred.sum() + mask.sum()
            dice = 2 * inter / total if total > 0 else (1.0 if mask.sum() == 0 else 0.0)
            voxel_dices.append(dice)

            # Lesion-level metrics
            gt_labeled, n_gt = ndimage_label(mask > 0)
            pred_labeled, n_pred = ndimage_label(pred > 0)

            gt_matched = set()
            pred_matched = set()

            for g in range(1, n_gt + 1):
                gt_region = (gt_labeled == g)
                for p in range(1, n_pred + 1):
                    pred_region = (pred_labeled == p)
                    if (gt_region & pred_region).any():
                        gt_matched.add(g)
                        pred_matched.add(p)
                        # Per-lesion dice
                        inter_l = (gt_region & pred_region).sum()
                        total_l = gt_region.sum() + pred_region.sum()
                        lesion_dices.append(2 * inter_l / total_l)

            detections["tp"] += len(gt_matched)
            detections["fn"] += n_gt - len(gt_matched)
            detections["fp"] += n_pred - len(pred_matched)

        tp, fp, fn = detections["tp"], detections["fp"], detections["fn"]
        lesion_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        lesion_recall = tp / (tp + fn + 1e-6)
        lesion_precision = tp / (tp + fp + 1e-6)

        results[version] = {
            "threshold": best_t,
            "voxel_dice_mean": float(np.mean(voxel_dices)),
            "voxel_dice_std": float(np.std(voxel_dices)),
            "voxel_dice_median": float(np.median(voxel_dices)),
            "per_lesion_dice": float(np.mean(lesion_dices)) if lesion_dices else 0,
            "lesion_f1": lesion_f1,
            "lesion_recall": lesion_recall,
            "lesion_precision": lesion_precision,
            "detections": detections,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison
    print(f"\n  {'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"  {'='*70}")
    print(f"\n  {'Metric':<25} {'v4 Baseline':<20} {'v5 Improved':<20} {'Delta':<10}")
    print(f"  {'-'*75}")

    if "v4_baseline" in results and "v5_improved" in results:
        v4 = results["v4_baseline"]
        v5 = results["v5_improved"]
        for metric_name, v4_key in [
            ("Voxel Dice", "voxel_dice_mean"),
            ("Per-Lesion Dice", "per_lesion_dice"),
            ("Lesion F1", "lesion_f1"),
            ("Lesion Recall", "lesion_recall"),
            ("Lesion Precision", "lesion_precision"),
        ]:
            v4_val = v4[v4_key]
            v5_val = v5[v4_key]
            delta = v5_val - v4_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric_name:<25} {v4_val:.4f}{'':<16} {v5_val:.4f}{'':<16} {sign}{delta:.4f}")
        print(f"\n  Thresholds: v4={v4['threshold']}, v5={v5['threshold']}")
    else:
        for ver, r in results.items():
            print(f"  {ver}: Voxel Dice={r['voxel_dice_mean']:.4f}, "
                  f"Lesion F1={r['lesion_f1']:.4f}, Recall={r['lesion_recall']:.4f}")

    print(f"  {'='*70}")

    # Save
    report_path = RESULTS_DIR / "overnight_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {report_path}")


if __name__ == "__main__":
    main()
