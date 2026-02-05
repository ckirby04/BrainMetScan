"""
Complete training and demonstration pipeline
- Trains segmentation model for 100 epochs
- Builds RAG database
- Runs example case through full pipeline
- Saves outputs
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.segmentation.dataset import BrainMetDataset, get_train_val_split
from src.segmentation.unet import LightweightUNet3D, CombinedLoss, count_parameters


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)

    return dice.item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(loader, desc="Validation")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)

    return avg_loss, avg_dice


def train_segmentation_model(data_dir, output_dir, num_epochs=100, device='cuda'):
    """Train segmentation model"""
    print("\n" + "="*60)
    print("STEP 1: Training Segmentation Model")
    print("="*60 + "\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get train/val split (returns case directory paths)
    train_cases, val_cases = get_train_val_split(data_dir, val_ratio=0.2, seed=42)

    # Create datasets from the split cases
    print("Loading datasets...")
    # For train/val split, we need to create separate datasets or use indices
    # Let's create the full dataset and get indices
    full_dataset = BrainMetDataset(
        data_dir=data_dir,
        sequences=['t1_gd'],
        patch_size=(64, 64, 64)
    )

    # Map case paths to indices
    case_to_idx = {case.name: idx for idx, case in enumerate(full_dataset.cases)}
    train_idx = [case_to_idx[case.name] for case in train_cases if case.name in case_to_idx]
    val_idx = [case_to_idx[case.name] for case in val_cases if case.name in case_to_idx]

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create model
    print("\nInitializing model...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = LightweightUNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {device}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_val_dice = 0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, output_dir / 'best_model.pth')
            print(f"Saved best model (Dice: {val_dice:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
    }, output_dir / 'final_model.pth')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Segmentation training complete!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}\n")

    return output_dir / 'best_model.pth'


def build_rag_database(data_dir, metadata_path, output_dir, device='cuda'):
    """Build RAG database"""
    print("\n" + "="*60)
    print("STEP 2: Building RAG Database")
    print("="*60 + "\n")

    # Run build_database.py script
    cmd = [
        'python', 'src/rag/build_database.py',
        '--data_dir', str(data_dir),
        '--metadata_path', str(metadata_path),
        '--output_dir', str(output_dir),
        '--device', device
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building RAG database: {result.stderr}")
        raise RuntimeError("RAG database building failed")

    print(result.stdout)

    db_path = Path(output_dir) / 'chromadb'

    print(f"\n{'='*60}")
    print(f"RAG database built successfully!")
    print(f"Database location: {db_path}")
    print(f"{'='*60}\n")

    return db_path


def run_example_pipeline(case_dir, model_path, db_path, output_dir):
    """Run example case through full pipeline"""
    print("\n" + "="*60)
    print("STEP 3: Running Example Pipeline")
    print("="*60 + "\n")

    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = case_dir.name
    print(f"Processing case: {case_id}")

    # Run segmentation inference
    print("\nRunning segmentation...")
    seg_output_dir = output_dir / 'segmentation'
    seg_output_dir.mkdir(exist_ok=True)

    cmd = [
        'python', 'src/segmentation/inference.py',
        '--case_dir', str(case_dir),
        '--model_path', str(model_path),
        '--output_dir', str(seg_output_dir)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Segmentation error: {result.stderr}")
    else:
        print(f"Segmentation complete: {result.stdout}")

    # Run RAG query
    print("\nRunning RAG query...")
    rag_output_dir = output_dir / 'rag'
    rag_output_dir.mkdir(exist_ok=True)

    cmd = [
        'python', 'src/rag/query.py',
        '--case_dir', str(case_dir),
        '--db_path', str(db_path),
        '--output_dir', str(rag_output_dir),
        '--k_cases', '5',
        '--k_facts', '3'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"RAG query error: {result.stderr}")
    else:
        print(f"RAG query complete: {result.stdout}")

    # Create summary
    summary = {
        'case_id': case_id,
        'case_dir': str(case_dir),
        'segmentation_output': str(seg_output_dir / f'{case_id}_prediction.nii.gz'),
        'rag_report': str(rag_output_dir / f'{case_id}_report.txt'),
        'rag_results': str(rag_output_dir / f'{case_id}_results.json')
    }

    summary_path = output_dir / 'pipeline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Example pipeline complete!")
    print(f"Case: {case_id}")
    print(f"Outputs saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Complete training and demo pipeline')

    parser.add_argument('--train_dir', type=str, default='train',
                        help='Training data directory')
    parser.add_argument('--metadata_path', type=str, default='metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--seg_output_dir', type=str, default='outputs/segmentation',
                        help='Segmentation output directory')
    parser.add_argument('--rag_output_dir', type=str, default='outputs/rag',
                        help='RAG output directory')
    parser.add_argument('--example_case', type=str, default='train/Mets_005',
                        help='Example case directory')
    parser.add_argument('--demo_output_dir', type=str, default='outputs/demo',
                        help='Demo output directory')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--skip_segmentation_training', action='store_true',
                        help='Skip segmentation training')
    parser.add_argument('--skip_rag_building', action='store_true',
                        help='Skip RAG database building')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("BRAIN METASTASIS COMPLETE PIPELINE")
    print("="*60)

    # Step 1: Train segmentation model
    if not args.skip_segmentation_training:
        model_path = train_segmentation_model(
            data_dir=args.train_dir,
            output_dir=args.seg_output_dir,
            num_epochs=args.num_epochs,
            device=args.device
        )
    else:
        model_path = Path(args.seg_output_dir) / 'best_model.pth'
        print(f"Skipping segmentation training, using existing model: {model_path}")

    # Step 2: Build RAG database
    if not args.skip_rag_building:
        db_path = build_rag_database(
            data_dir=args.train_dir,
            metadata_path=args.metadata_path,
            output_dir=args.rag_output_dir,
            device=args.device
        )
    else:
        db_path = Path(args.rag_output_dir) / 'chromadb'
        print(f"Skipping RAG building, using existing database: {db_path}")

    # Step 3: Run example pipeline
    summary = run_example_pipeline(
        case_dir=args.example_case,
        model_path=model_path,
        db_path=db_path,
        output_dir=args.demo_output_dir
    )

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE!")
    print("="*60)
    print("\nPipeline Summary:")
    print(f"  Segmentation model: {model_path}")
    print(f"  RAG database: {db_path}")
    print(f"  Example outputs: {args.demo_output_dir}")
    print("\nExample case outputs:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
