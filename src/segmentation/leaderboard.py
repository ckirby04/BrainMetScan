"""
Leaderboard for tracking model performance across different configurations.
Automatically updates during training and provides ensemble performance estimates.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import os


class Leaderboard:
    """Track and compare model performance across configurations."""

    def __init__(self, leaderboard_path: Optional[str] = None):
        if leaderboard_path is None:
            # Default to project model directory
            project_dir = Path(__file__).parent.parent.parent
            self.path = project_dir / "model" / "leaderboard.json"
        else:
            self.path = Path(leaderboard_path)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        """Load leaderboard from disk."""
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {
            "models": {},
            "ensemble_estimate": {},
            "last_updated": None
        }

    def _save(self):
        """Save leaderboard to disk."""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def update(
        self,
        model_name: str,
        patch_size: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        tiny_dice: Optional[float] = None,
        small_dice: Optional[float] = None,
        medium_dice: Optional[float] = None,
        large_dice: Optional[float] = None,
        sensitivity: Optional[float] = None,
        specificity: Optional[float] = None,
        model_path: Optional[str] = None,
        is_best: bool = False
    ):
        """Update leaderboard with new metrics."""

        key = f"{model_name}_{patch_size}patch"

        current = self.data["models"].get(key, {
            "model_name": model_name,
            "patch_size": patch_size,
            "best_val_dice": 0,
            "best_tiny_dice": 0,
            "best_epoch": 0,
            "history": []
        })

        # Record this epoch
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "timestamp": datetime.now().isoformat()
        }

        if tiny_dice is not None:
            entry["tiny_dice"] = tiny_dice
        if small_dice is not None:
            entry["small_dice"] = small_dice
        if medium_dice is not None:
            entry["medium_dice"] = medium_dice
        if large_dice is not None:
            entry["large_dice"] = large_dice
        if sensitivity is not None:
            entry["sensitivity"] = sensitivity
        if specificity is not None:
            entry["specificity"] = specificity

        # Update best scores
        if val_dice > current.get("best_val_dice", 0):
            current["best_val_dice"] = val_dice
            current["best_epoch"] = epoch
            current["best_val_loss"] = val_loss
            if model_path:
                current["best_model_path"] = model_path

        if tiny_dice is not None and tiny_dice > current.get("best_tiny_dice", 0):
            current["best_tiny_dice"] = tiny_dice
            current["best_tiny_epoch"] = epoch

        if small_dice is not None and small_dice > current.get("best_small_dice", 0):
            current["best_small_dice"] = small_dice

        if medium_dice is not None and medium_dice > current.get("best_medium_dice", 0):
            current["best_medium_dice"] = medium_dice

        if large_dice is not None and large_dice > current.get("best_large_dice", 0):
            current["best_large_dice"] = large_dice

        # Keep last 10 history entries to avoid huge files
        current["history"].append(entry)
        current["history"] = current["history"][-10:]
        current["latest"] = entry

        self.data["models"][key] = current

        # Update ensemble estimates
        self._update_ensemble_estimate()

        self._save()

    def _update_ensemble_estimate(self):
        """Estimate ensemble performance from best models at each scale."""
        models = self.data["models"]

        if not models:
            return

        # Group by patch size
        by_patch = {}
        for key, model in models.items():
            ps = model["patch_size"]
            if ps not in by_patch or model["best_val_dice"] > by_patch[ps]["best_val_dice"]:
                by_patch[ps] = model

        # Ensemble estimate: weighted combination based on lesion size specialization
        # Smaller patches -> better for tiny lesions
        # Larger patches -> better for large lesions + context

        ensemble = {
            "models_used": list(by_patch.keys()),
            "strategy": "size-weighted routing",
            "notes": "Smaller patches handle tiny lesions, larger patches handle context"
        }

        # Estimate overall dice as weighted average
        if by_patch:
            dices = [m["best_val_dice"] for m in by_patch.values()]
            ensemble["estimated_overall_dice"] = max(dices) * 1.05  # Ensemble typically +5%

            # Tiny lesions: use smallest patch model
            smallest_patch = min(by_patch.keys())
            ensemble["estimated_tiny_dice"] = by_patch[smallest_patch].get("best_tiny_dice", 0)

            # Large lesions: use largest patch model
            largest_patch = max(by_patch.keys())
            ensemble["estimated_large_dice"] = by_patch[largest_patch].get("best_large_dice", 0)

        self.data["ensemble_estimate"] = ensemble

    def get_summary(self) -> str:
        """Get formatted leaderboard summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL LEADERBOARD")
        lines.append("=" * 70)

        if not self.data["models"]:
            lines.append("No models trained yet.")
            return "\n".join(lines)

        # Sort by patch size
        sorted_models = sorted(
            self.data["models"].items(),
            key=lambda x: x[1]["patch_size"]
        )

        lines.append(f"{'Model':<25} {'Patch':>6} {'Best Dice':>10} {'Tiny Dice':>10} {'Epoch':>6}")
        lines.append("-" * 70)

        for key, model in sorted_models:
            name = model["model_name"][:20]
            patch = f"{model['patch_size']}³"
            dice = f"{model['best_val_dice']*100:.1f}%"
            tiny = f"{model.get('best_tiny_dice', 0)*100:.1f}%"
            epoch = model.get("best_epoch", 0)

            lines.append(f"{name:<25} {patch:>6} {dice:>10} {tiny:>10} {epoch:>6}")

        # Ensemble estimate
        ens = self.data.get("ensemble_estimate", {})
        if ens:
            lines.append("")
            lines.append("-" * 70)
            lines.append("ENSEMBLE ESTIMATE (theoretical)")
            lines.append("-" * 70)
            lines.append(f"Models: {ens.get('models_used', [])}")
            lines.append(f"Strategy: {ens.get('strategy', 'N/A')}")
            if "estimated_overall_dice" in ens:
                lines.append(f"Estimated Overall Dice: {ens['estimated_overall_dice']*100:.1f}%")
            if "estimated_tiny_dice" in ens:
                lines.append(f"Estimated Tiny Dice: {ens['estimated_tiny_dice']*100:.1f}%")

        lines.append("")
        lines.append(f"Last updated: {self.data.get('last_updated', 'Never')}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def print_summary(self):
        """Print leaderboard to console."""
        print(self.get_summary())

    def get_best_model(self, metric: str = "val_dice") -> Optional[dict]:
        """Get the best model by a specific metric."""
        if not self.data["models"]:
            return None

        best = None
        best_score = -1

        for key, model in self.data["models"].items():
            if metric == "val_dice":
                score = model.get("best_val_dice", 0)
            elif metric == "tiny_dice":
                score = model.get("best_tiny_dice", 0)
            elif metric == "small_dice":
                score = model.get("best_small_dice", 0)
            elif metric == "large_dice":
                score = model.get("best_large_dice", 0)
            else:
                score = model.get(f"best_{metric}", 0)

            if score > best_score:
                best_score = score
                best = model

        return best


# Global leaderboard instance
_leaderboard = None

def get_leaderboard() -> Leaderboard:
    """Get global leaderboard instance."""
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = Leaderboard()
    return _leaderboard


def update_leaderboard(**kwargs):
    """Convenience function to update global leaderboard."""
    get_leaderboard().update(**kwargs)


def print_leaderboard():
    """Print global leaderboard."""
    get_leaderboard().print_summary()


if __name__ == "__main__":
    # Demo/test
    lb = Leaderboard()

    # Simulate some training results
    lb.update(
        model_name="tiny_lesion",
        patch_size=16,
        epoch=50,
        train_loss=0.15,
        val_loss=0.18,
        val_dice=0.65,
        tiny_dice=0.55,
        small_dice=0.70,
        medium_dice=0.75,
        large_dice=0.80
    )

    lb.update(
        model_name="tiny_lesion",
        patch_size=48,
        epoch=75,
        train_loss=0.12,
        val_loss=0.15,
        val_dice=0.72,
        tiny_dice=0.45,
        small_dice=0.75,
        medium_dice=0.80,
        large_dice=0.85
    )

    lb.print_summary()
