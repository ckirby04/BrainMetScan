"""
Brain Metastasis Segmentation Demo - Professional Edition
Interactive web interface with modern UI/UX for clinical demonstration

Consolidated into 1.23 project. Uses legacy ensemble for demo visualization
and the production SmartEnsemble API for remote inference.
"""

import sys
import os

# Add project root to path for package imports
_project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _project_root)

import gradio as gr
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io
from PIL import Image
from skimage import measure
from datetime import datetime
import json

from src.segmentation.unet import LightweightUNet3D
from src.segmentation.stacking import (
    StackingClassifier, load_stacking_model, run_stacking_inference,
    build_stacking_features, STACKING_MODEL_NAMES, STACKING_THRESHOLD,
    STACKING_IN_CHANNELS, DISPLAY_NAMES, sliding_window_inference,
)
from demo.legacy_ensemble import EnsembleSegmentationModel, create_ensemble_model


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path(__file__).parent.parent / "model" / "best_model.pth"
MODEL_DIR = Path(__file__).parent.parent / "model"
# Use preprocessed_256 data from 1.21 (already resized to 256³, has t2 sequences)
# This is the exact data the ensemble models were trained on
PREPROCESSED_DIR = Path(__file__).parent.parent.parent / "1.21" / "data" / "preprocessed_256" / "train"
SUPERSET_DIR = Path(__file__).parent.parent.parent / "Superset" / "full" / "train"
DATA_DIR = PREPROCESSED_DIR if PREPROCESSED_DIR.exists() else (SUPERSET_DIR if SUPERSET_DIR.exists() else Path(__file__).parent.parent / "data" / "train")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stacking cache directory (base model predictions at 128^3)
STACKING_CACHE_DIR = Path(__file__).parent.parent / "model" / "stacking_cache_v4"

# Precomputed inference cache directory
DEMO_CACHE_DIR = Path(__file__).parent.parent / "outputs" / "demo_cache"

# Global caches
_model = None
_ensemble_model = None
_model_info = {}
_case_cache = {}
_prediction_cache = {}
_probability_cache = {}  # Cache probability maps for threshold adjustment


def _load_precomputed_cache():
    """Load precomputed stacking/ensemble results from disk into memory."""
    if not DEMO_CACHE_DIR.exists():
        return 0
    loaded = 0
    for npz_path in sorted(DEMO_CACHE_DIR.glob("*_ensemble.npz")):
        case_name = npz_path.stem.replace("_ensemble", "")
        cache_key = f"{case_name}_ensemble"
        if cache_key in _probability_cache:
            continue
        try:
            data = np.load(str(npz_path))
            result = {
                'fused': data['fused'].astype(np.float32),
                'agreement': data['agreement'].astype(np.float32),
                'individual': {},
            }
            for key in data.files:
                if key.startswith('individual_'):
                    model_name = key[len('individual_'):]
                    result['individual'][model_name] = data[key].astype(np.float32)
            _probability_cache[cache_key] = result
            _prediction_cache[case_name] = result['fused']
            loaded += 1
        except Exception:
            pass
    return loaded


# Load precomputed results at import time
_precomputed_count = _load_precomputed_cache()
if _precomputed_count > 0:
    print(f"Loaded {_precomputed_count} precomputed inference results from {DEMO_CACHE_DIR}")


# ============================================================================
# CUSTOM CSS THEME
# ============================================================================

CUSTOM_CSS = """
/* ============================================================================
   PREMIUM DESIGN SYSTEM - Inspired by Apple & OpenAI
   ============================================================================ */

/* Import Premium Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');

/* CSS Variables - Titanium, Gloss Black & Rapid Blue Theme */
:root {
    /* Chevrolet Rapid Blue */
    --rapid-blue: #0C4DA2;
    --rapid-blue-light: #1E6FD9;
    --rapid-blue-dark: #083A7A;

    /* Gloss Black */
    --gloss-black: #0a0a0a;
    --gloss-black-light: #1a1a1a;
    --gloss-black-medium: #2d2d2d;

    /* Titanium */
    --titanium: #878681;
    --titanium-light: #a8a8a3;
    --titanium-dark: #5c5c58;
    --titanium-pale: #e8e8e6;

    /* Semantic Colors */
    --primary: var(--rapid-blue);
    --primary-hover: var(--rapid-blue-light);
    --text-primary: var(--gloss-black);
    --text-secondary: var(--titanium);
    --text-tertiary: var(--titanium-light);
    --bg-primary: #f4f4f3;
    --bg-secondary: #ffffff;
    --bg-tertiary: var(--titanium-pale);
    --bg-dark: var(--gloss-black);
    --border-color: rgba(135, 134, 129, 0.2);
    --shadow-sm: 0 2px 8px rgba(10, 10, 10, 0.06);
    --shadow-md: 0 4px 20px rgba(10, 10, 10, 0.1);
    --shadow-lg: 0 12px 40px rgba(10, 10, 10, 0.15);
    --radius-sm: 12px;
    --radius-md: 16px;
    --radius-lg: 24px;
    --transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
}

/* Global Styles */
.gradio-container {
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    min-height: 100vh;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Premium Header - Gloss Black */
.header-container {
    background: linear-gradient(180deg, var(--gloss-black) 0%, var(--gloss-black-light) 100%);
    padding: 56px 40px;
    border-radius: 0;
    margin: -16px -16px 32px -16px;
    border-bottom: 3px solid var(--rapid-blue);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(12, 77, 162, 0.1) 0%, transparent 50%);
    pointer-events: none;
}

.header-title {
    color: #ffffff !important;
    font-size: 3.2em !important;
    font-weight: 600 !important;
    margin: 0 !important;
    letter-spacing: -0.03em;
    line-height: 1.1;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.header-subtitle {
    color: var(--titanium-light) !important;
    font-size: 1.25em !important;
    margin-top: 12px !important;
    font-weight: 400;
    letter-spacing: -0.01em;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--rapid-blue);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 500;
    margin-top: 20px;
    box-shadow: 0 4px 12px rgba(12, 77, 162, 0.4);
}

/* Card System */
.card {
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
    padding: 24px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
    transition: var(--transition);
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* Section Labels */
.section-label {
    font-size: 0.7em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-tertiary);
    margin-bottom: 12px;
}

/* Premium Button Styling - Rapid Blue & Titanium */
button.primary, .primary-btn, button[variant="primary"] {
    background: linear-gradient(180deg, var(--rapid-blue-light) 0%, var(--rapid-blue) 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    padding: 14px 28px !important;
    border-radius: 980px !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 12px rgba(12, 77, 162, 0.3) !important;
    letter-spacing: -0.01em;
}

button.primary:hover, .primary-btn:hover, button[variant="primary"]:hover {
    background: linear-gradient(180deg, var(--rapid-blue) 0%, var(--rapid-blue-dark) 100%) !important;
    transform: scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(12, 77, 162, 0.4) !important;
}

button.secondary, .secondary-btn, button[variant="secondary"] {
    background: var(--gloss-black) !important;
    border: 1px solid var(--titanium-dark) !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 14px 28px !important;
    border-radius: 980px !important;
    transition: var(--transition) !important;
}

button.secondary:hover, .secondary-btn:hover, button[variant="secondary"]:hover {
    background: var(--gloss-black-light) !important;
    border-color: var(--titanium) !important;
}

/* Sidebar Styling */
.sidebar-section {
    margin-bottom: 32px;
}

.sidebar-title {
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-tertiary);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
}

/* Form Controls */
input, select, textarea {
    font-family: inherit !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-color) !important;
    padding: 14px 16px !important;
    font-size: 1em !important;
    transition: var(--transition) !important;
    background: var(--bg-secondary) !important;
}

input:focus, select:focus, textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.1) !important;
    outline: none !important;
}

/* Premium Slider - Titanium & Rapid Blue */
input[type="range"] {
    height: 6px !important;
    border-radius: 3px !important;
    background: linear-gradient(90deg, var(--titanium-pale) 0%, var(--titanium-light) 100%) !important;
    border: none !important;
}

input[type="range"]::-webkit-slider-thumb {
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    background: linear-gradient(180deg, var(--rapid-blue-light) 0%, var(--rapid-blue) 100%) !important;
    border: 2px solid white !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(12, 77, 162, 0.4) !important;
    transition: var(--transition) !important;
}

input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.15) !important;
    box-shadow: 0 4px 16px rgba(12, 77, 162, 0.5) !important;
}

/* Tab Navigation - Rapid Blue Style */
.tab-nav, .tabs, .tabitem {
    border-bottom: 2px solid var(--titanium-pale) !important;
    background: transparent !important;
    padding: 0 !important;
    gap: 0 !important;
}

.tab-nav button, .tabs button, button.tab-nav, div[role="tablist"] button {
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 0.95em !important;
    padding: 16px 24px !important;
    color: var(--titanium) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    transition: var(--transition) !important;
    margin-bottom: -2px !important;
}

.tab-nav button:hover, .tabs button:hover, div[role="tablist"] button:hover {
    color: var(--gloss-black) !important;
}

.tab-nav button.selected, .tabs button.selected, div[role="tablist"] button[aria-selected="true"],
button.selected, .tab-nav .selected, .tabs .selected {
    color: var(--rapid-blue) !important;
    background: transparent !important;
    border-bottom: 3px solid var(--rapid-blue) !important;
    font-weight: 600 !important;
}

/* Override Gradio default orange/red tab indicator */
.gradio-container button.selected,
.gradio-container .tab-nav button.selected,
.gradio-container div[role="tablist"] button[aria-selected="true"] {
    color: #0C4DA2 !important;
    border-color: #0C4DA2 !important;
    border-bottom-color: #0C4DA2 !important;
    background: transparent !important;
}

/* Remove any orange/red borders or indicators */
.gradio-container button:focus,
.gradio-container button:active {
    outline: none !important;
    box-shadow: none !important;
    border-color: #0C4DA2 !important;
}

/* Tab panel borders */
.tabitem, .tab-content, div[role="tabpanel"] {
    border-color: var(--titanium-pale) !important;
}

/* Image Display - Clean Presentation */
.image-container, .image-preview {
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    background: #000;
}

/* Legend - Titanium Style */
.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    background: linear-gradient(180deg, #ffffff 0%, var(--titanium-pale) 100%);
    border-radius: 980px;
    font-size: 0.85em;
    font-weight: 500;
    color: var(--gloss-black);
    border: 1px solid var(--titanium-light);
}

.legend-color {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Stats Display - Titanium Style */
.stat-card {
    background: linear-gradient(180deg, #ffffff 0%, var(--titanium-pale) 100%);
    padding: 24px;
    border-radius: var(--radius-md);
    text-align: center;
    border: 1px solid var(--titanium-light);
    transition: var(--transition);
}

.stat-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
    border-color: var(--rapid-blue);
}

.stat-value {
    font-size: 2.5em;
    font-weight: 600;
    line-height: 1;
    color: var(--gloss-black);
    letter-spacing: -0.02em;
}

.stat-label {
    font-size: 0.85em;
    color: var(--titanium);
    margin-top: 8px;
    font-weight: 500;
}

/* Premium Footer - Gloss Black */
.footer {
    text-align: center;
    padding: 48px 20px;
    color: var(--titanium-light);
    font-size: 0.9em;
    border-top: none;
    margin-top: 48px;
    background: linear-gradient(180deg, var(--gloss-black-light) 0%, var(--gloss-black) 100%);
}

.footer p {
    margin: 4px 0;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 24px;
    margin-top: 16px;
}

.footer-links a {
    color: var(--rapid-blue-light);
    text-decoration: none;
    font-weight: 500;
}

/* Keyboard Hints - Subtle */
.shortcut-hint {
    background: var(--bg-tertiary);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8em;
    color: var(--text-tertiary);
    font-family: 'SF Mono', Monaco, monospace;
}

/* Accordion - Clean */
.accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
}

.accordion-header {
    background: var(--bg-tertiary) !important;
    padding: 16px !important;
    font-weight: 500 !important;
}

/* Markdown Styling */
.prose h3, .markdown h3 {
    font-size: 0.75em !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-tertiary) !important;
    margin: 24px 0 12px 0 !important;
}

/* Quick Stats Card */
.quick-stats {
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    padding: 16px;
    font-size: 0.9em;
    color: var(--text-secondary);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.fade-in {
    animation: fadeIn 0.4s cubic-bezier(0.25, 0.1, 0.25, 1);
}

.loading {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .header-title {
        font-size: 2em !important;
    }

    .header-container {
        padding: 32px 20px;
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
}

::-webkit-scrollbar-thumb {
    background: #c1c1c6;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #a1a1a6;
}

/* ============================================================================
   GRADIO SPECIFIC OVERRIDES - Force Rapid Blue on ALL tab indicators
   ============================================================================ */

/* Target all possible Gradio tab selectors */
.gradio-container .tabs > .tab-nav > button.selected,
.gradio-container .tabitem,
.gradio-container [class*="tab"] button.selected,
.gradio-container [class*="tab"] button[aria-selected="true"],
.gradio-container .svelte-1kcgrqr.selected,
button.svelte-1kcgrqr.selected,
.tab-nav > button.selected,
div.tabs > div > button.selected {
    color: #0C4DA2 !important;
    border-bottom: 3px solid #0C4DA2 !important;
    background: transparent !important;
    border-color: #0C4DA2 !important;
}

/* Override any orange/red accent colors */
.gradio-container *::selection {
    background: rgba(12, 77, 162, 0.3);
}

/* Force blue on focus states */
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container select:focus-visible {
    outline: 2px solid #0C4DA2 !important;
    outline-offset: 2px;
}

/* Override Gradio's default accent color variable */
.gradio-container {
    --color-accent: #0C4DA2 !important;
    --color-accent-soft: rgba(12, 77, 162, 0.15) !important;
    --primary-500: #0C4DA2 !important;
    --primary-600: #083A7A !important;
}

/* Radio buttons and checkboxes */
.gradio-container input[type="radio"]:checked,
.gradio-container input[type="checkbox"]:checked {
    accent-color: #0C4DA2 !important;
}

/* Slider active state */
.gradio-container input[type="range"]::-webkit-slider-thumb {
    background: linear-gradient(180deg, #1E6FD9 0%, #0C4DA2 100%) !important;
}

/* Any remaining orange elements */
.gradio-container [style*="orange"],
.gradio-container [style*="#f"] {
    border-color: #0C4DA2 !important;
}

/* ============================================================================
   EVIDENCE CARDS - RAG Citation Panel
   ============================================================================ */

.evidence-card {
    background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
    border: 1px solid #5c5c58;
    border-radius: 12px;
    margin-bottom: 10px;
    overflow: hidden;
    transition: all 0.2s ease;
}

.evidence-card:hover {
    border-color: #0C4DA2;
    box-shadow: 0 2px 12px rgba(12, 77, 162, 0.2);
}

.evidence-card summary {
    padding: 14px 18px;
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ffffff;
    font-weight: 500;
    font-size: 0.9em;
}

.evidence-card summary::-webkit-details-marker { display: none; }

.evidence-card summary::before {
    content: '\\25B6';
    font-size: 0.7em;
    color: #878681;
    transition: transform 0.2s ease;
}

.evidence-card[open] summary::before {
    transform: rotate(90deg);
}

.evidence-card .evidence-body {
    padding: 0 18px 14px 18px;
    color: #a8a8a3;
    font-size: 0.85em;
    line-height: 1.6;
    border-top: 1px solid rgba(92, 92, 88, 0.3);
}

.evidence-score-bar {
    height: 3px;
    border-radius: 2px;
    margin-top: 6px;
}

.evidence-score-high { background: linear-gradient(90deg, #10b981, #34d399); }
.evidence-score-med { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.evidence-score-low { background: linear-gradient(90deg, #878681, #a8a8a3); }

/* RECIST Response Badges */
.recist-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 8px 20px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.2em;
    letter-spacing: 0.05em;
}

.recist-cr { background: linear-gradient(180deg, #10b981 0%, #059669 100%); color: white; }
.recist-pr { background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%); color: white; }
.recist-sd { background: linear-gradient(180deg, #f59e0b 0%, #d97706 100%); color: white; }
.recist-pd { background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%); color: white; }

/* Corpus stats card */
.corpus-stats-card {
    background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
    border: 1px solid #5c5c58;
    border-radius: 12px;
    padding: 16px;
    font-size: 0.85em;
    color: #a8a8a3;
    line-height: 1.6;
}

.corpus-stats-card strong {
    color: #ffffff;
}
"""


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the single segmentation model (fallback)"""
    global _model, _model_info

    if _model is not None:
        return _model

    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if 'args' in checkpoint:
        model_args = checkpoint['args']
        base_channels = model_args.get('base_channels', 20)
        depth = model_args.get('depth', 3)
        use_residual = model_args.get('use_residual', True)
    else:
        base_channels = 20
        depth = 3
        use_residual = True

    _model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=base_channels,
        depth=depth,
        dropout_p=0.0,
        use_residual=use_residual
    ).to(DEVICE)

    if 'model_state_dict' in checkpoint:
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model_info['epoch'] = checkpoint.get('epoch', 'unknown')
        _model_info['val_dice'] = checkpoint.get('val_dice', 0)
    else:
        _model.load_state_dict(checkpoint)

    _model.eval()
    print(f"Model loaded! (Epoch {_model_info.get('epoch', '?')}, Dice: {_model_info.get('val_dice', 0):.2%})")

    return _model


def load_stacking_model_cached():
    """Load the stacking classifier (6 models) for demo inference."""
    global _ensemble_model, _model_info

    if _ensemble_model is not None:
        return _ensemble_model

    print("=" * 50)
    print("Loading stacking classifier...")
    print("=" * 50)

    # Try stacking classifier first
    stacking = load_stacking_model(MODEL_DIR, DEVICE)
    if stacking is not None:
        _ensemble_model = stacking
        _model_info['ensemble'] = True
        _model_info['stacking'] = True
        _model_info['num_models'] = 6
        _model_info['model_names'] = list(STACKING_MODEL_NAMES)
        _model_info['val_dice'] = 0.7763
        _model_info['epoch'] = 'stacking_v4'

        print(f"Stacking classifier loaded: 6 base models")
        print(f"Models: {STACKING_MODEL_NAMES}")
        print(f"Stacking Dice: 0.7763")
        print(f"Threshold: {STACKING_THRESHOLD}")
        print(f"Using preprocessed data from: {DATA_DIR}")
        print("=" * 50)
        return _ensemble_model

    # Fallback to legacy ensemble
    print("Stacking checkpoint not found, trying legacy ensemble...")
    try:
        _ensemble_model = create_ensemble_model(MODEL_DIR, DEVICE)
        ensemble_info = _ensemble_model.get_ensemble_info()
        avg_dice = np.mean([info.get('val_dice', 0) for info in ensemble_info['models'].values()])
        _model_info['ensemble'] = True
        _model_info['stacking'] = False
        _model_info['num_models'] = ensemble_info['num_models']
        _model_info['model_names'] = ensemble_info['model_names']
        _model_info['val_dice'] = avg_dice
        _model_info['epoch'] = 'ensemble'
        print(f"Legacy ensemble loaded: {ensemble_info['num_models']} models")
        print("=" * 50)
        return _ensemble_model
    except Exception as e:
        print(f"Error loading ensemble: {e}")
        print("Falling back to single model")
        return None


# Keep backward-compatible alias
load_ensemble_model = load_stacking_model_cached


# ============================================================================
# DATA LOADING
# ============================================================================

def get_available_cases():
    """Get list of available cases, preferring top Dice scores from manifest.

    Only shows cases that have stacking cache entries (if stacking cache exists).
    """
    # If a precomputed manifest exists, use its ranked list
    manifest_path = DEMO_CACHE_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            top = manifest.get('top_cases', [])
            # Verify each case still exists on disk
            cases = [
                entry['case_name'] for entry in top
                if (DATA_DIR / entry['case_name']).is_dir()
            ]
            if cases:
                return cases
        except Exception:
            pass

    # Build set of cases with stacking cache entries
    stacking_cases = set()
    if STACKING_CACHE_DIR.exists():
        for npz_path in STACKING_CACHE_DIR.glob("*.npz"):
            stacking_cases.add(npz_path.stem)

    # Fallback: scan directory, filter to those with stacking cache
    cases = []
    if DATA_DIR.exists():
        for case_dir in sorted(DATA_DIR.iterdir()):
            if case_dir.is_dir():
                name = case_dir.name
                # Only include cases with stacking cache (if cache exists)
                if stacking_cases and name not in stacking_cases:
                    continue
                valid_prefixes = ('Mets_', 'BMS_Mets_', 'UCSF_', 'BraTS_', 'Yale_')
                if any(name.startswith(p) for p in valid_prefixes):
                    seg_file = case_dir / 'seg.nii.gz'
                    if seg_file.exists():
                        cases.append(name)
    return cases[:25]


def load_case_data(case_name):
    """Load MRI sequences and ground truth for a case"""
    global _case_cache

    if case_name in _case_cache:
        return _case_cache[case_name]

    case_dir = DATA_DIR / case_name

    sequences = {}
    # Check if t2 exists (Superset data), otherwise use bravo (original data)
    seq_names = ['t1_pre', 't1_gd', 'flair']
    if (case_dir / "t2.nii.gz").exists():
        seq_names.append('t2')
        fourth_seq = 't2'
    else:
        seq_names.append('bravo')
        fourth_seq = 'bravo'

    for seq in seq_names:
        nii = nib.load(str(case_dir / f"{seq}.nii.gz"))
        sequences[seq] = nii.get_fdata().astype(np.float32)

    # Store under standardized name for inference
    sequences['fourth'] = sequences[fourth_seq]

    seg_nii = nib.load(str(case_dir / "seg.nii.gz"))
    ground_truth = seg_nii.get_fdata().astype(np.float32)

    _case_cache[case_name] = (sequences, ground_truth)
    return sequences, ground_truth


# ============================================================================
# INFERENCE
# ============================================================================

def normalize_for_inference(img):
    """Normalize image for model input"""
    mean = np.mean(img)
    std = np.std(img)
    if std > 0:
        return (img - mean) / std
    return img - mean


def run_full_inference(model, sequences):
    """Run inference on the full volume using sliding window (single model)"""
    images = []
    # Use 'fourth' which is set to either t2 or bravo depending on data
    for seq in ['t1_pre', 't1_gd', 'flair', 'fourth']:
        img = normalize_for_inference(sequences[seq])
        images.append(img)

    volume = np.stack(images, axis=0)
    H, W, D = volume.shape[1:]
    patch_size = 96

    output = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    stride = patch_size // 2

    h_starts = list(range(0, max(1, H - patch_size + 1), stride))
    if H > patch_size and (H - patch_size) not in h_starts:
        h_starts.append(H - patch_size)

    w_starts = list(range(0, max(1, W - patch_size + 1), stride))
    if W > patch_size and (W - patch_size) not in w_starts:
        w_starts.append(W - patch_size)

    d_starts = list(range(0, max(1, D - patch_size + 1), stride))
    if D > patch_size and (D - patch_size) not in d_starts:
        d_starts.append(D - patch_size)

    with torch.no_grad():
        for h_start in h_starts:
            for w_start in w_starts:
                for d_start in d_starts:
                    h_end = min(h_start + patch_size, H)
                    w_end = min(w_start + patch_size, W)
                    d_end = min(d_start + patch_size, D)

                    patch = volume[:, h_start:h_end, w_start:w_end, d_start:d_end]

                    pad_h = patch_size - patch.shape[1]
                    pad_w = patch_size - patch.shape[2]
                    pad_d = patch_size - patch.shape[3]

                    if pad_h > 0 or pad_w > 0 or pad_d > 0:
                        patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')

                    input_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(DEVICE)
                    pred = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]

                    valid_h = h_end - h_start
                    valid_w = w_end - w_start
                    valid_d = d_end - d_start

                    output[h_start:h_end, w_start:w_end, d_start:d_end] += pred[:valid_h, :valid_w, :valid_d]
                    count[h_start:h_end, w_start:w_end, d_start:d_end] += 1

    output = output / np.maximum(count, 1)
    return output


def run_stacking_demo_inference(model_or_ensemble, sequences, case_name=None):
    """
    Run inference using stacking classifier or legacy ensemble.

    If stacking: loads base model predictions from stacking cache, runs stacking
    classifier, upsamples from 128^3 to original MRI size.
    If legacy ensemble: delegates to legacy predict_ensemble.

    Returns:
        dict with 'fused', 'individual', and 'agreement' maps
    """
    # Check if this is a stacking classifier
    is_stacking = isinstance(model_or_ensemble, StackingClassifier)

    if is_stacking and case_name:
        # Try stacking cache
        cache_path = STACKING_CACHE_DIR / f"{case_name}.npz"
        if cache_path.exists():
            target_size = sequences['t1_gd'].shape  # e.g. (256, 256, 256)
            result = run_stacking_inference(
                cache_path, model_or_ensemble, DEVICE,
                target_size=target_size,
            )
            return result

    # Fallback: legacy ensemble inference
    if hasattr(model_or_ensemble, 'predict_ensemble'):
        images = []
        for seq in ['t1_pre', 't1_gd', 'flair', 'fourth']:
            img = normalize_for_inference(sequences[seq])
            images.append(img)
        volume = np.stack(images, axis=0)
        result = model_or_ensemble.predict_ensemble(
            volume,
            fusion_method='mean',
            return_individual=True,
            use_matched_patch_sizes=True,
        )
        return result

    # No stacking cache and not a legacy ensemble — return empty
    return {'fused': np.zeros_like(sequences['t1_gd']), 'individual': {}, 'agreement': None}


# Keep backward-compatible alias
def run_ensemble_inference(ensemble_model, sequences):
    """Legacy wrapper — delegates to run_stacking_demo_inference."""
    return run_stacking_demo_inference(ensemble_model, sequences)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_lesion_features(mask, image=None):
    """Extract radiomic features from segmentation mask"""
    features = {}

    labeled_mask = measure.label(mask > 0.5)
    regions = measure.regionprops(labeled_mask, intensity_image=image)

    features['num_lesions'] = len(regions)

    if len(regions) == 0:
        return {
            'num_lesions': 0,
            'total_volume': 0,
            'mean_lesion_volume': 0,
            'max_lesion_volume': 0,
            'min_lesion_volume': 0,
            'lesion_volumes': [],
            'lesion_centroids': [],
            'lesion_slices': []
        }

    volumes = [r.area for r in regions]
    centroids = [r.centroid for r in regions]

    features['total_volume'] = sum(volumes)
    features['mean_lesion_volume'] = np.mean(volumes)
    features['max_lesion_volume'] = max(volumes)
    features['min_lesion_volume'] = min(volumes)
    features['lesion_volumes'] = volumes
    features['lesion_centroids'] = centroids

    lesion_slices = []
    for r in regions:
        coords = r.coords
        z_coords = coords[:, 2]
        lesion_slices.append((z_coords.min(), z_coords.max(), r.centroid[2]))

    features['lesion_slices'] = lesion_slices

    return features


# ============================================================================
# VISUALIZATION - MULTI-VIEW
# ============================================================================

def create_multiview_visualization(sequences, prediction, ground_truth, slice_indices, show_overlay=True, confidence_threshold=0.5, slice_axis="Axial"):
    """Create a single-axis 4-panel visualization (MRI, Confidence, GT, Comparison).

    Args:
        slice_axis: One of "Axial", "Sagittal", "Coronal"
    """
    fig = plt.figure(figsize=(16, 5), facecolor='#1a1a2e')
    gs = GridSpec(1, 4, figure=fig, wspace=0.1)

    mri = sequences['t1_gd']
    H, W, D = mri.shape
    ax_idx, sag_idx, cor_idx = slice_indices

    if slice_axis == "Sagittal":
        view_name = "Sagittal"
        mri_slice = mri[sag_idx, :, :]
        pred_slice = prediction[sag_idx, :, :]
        gt_slice = ground_truth[sag_idx, :, :]
    elif slice_axis == "Coronal":
        view_name = "Coronal"
        mri_slice = mri[:, cor_idx, :]
        pred_slice = prediction[:, cor_idx, :]
        gt_slice = ground_truth[:, cor_idx, :]
    else:
        view_name = "Axial"
        mri_slice = mri[:, :, ax_idx]
        pred_slice = prediction[:, :, ax_idx]
        gt_slice = ground_truth[:, :, ax_idx]

    # Rotate for proper orientation
    mri_slice = np.rot90(mri_slice)
    pred_slice = np.rot90(pred_slice)
    gt_slice = np.rot90(gt_slice)

    # Normalize MRI
    vmin, vmax = np.percentile(mri_slice, [1, 99])
    mri_display = np.clip((mri_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Column 0: Raw MRI
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(mri_display, cmap='gray')
    ax0.set_title(f'{view_name} - MRI', color='white', fontsize=12, fontweight='bold', pad=8)
    ax0.axis('off')
    ax0.set_facecolor('#1a1a2e')

    # Column 1: MRI + Prediction overlay (confidence heatmap)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mri_display, cmap='gray')
    if show_overlay and np.any(pred_slice > 0.1):
        pred_rgba = plt.cm.hot(pred_slice)
        pred_rgba[..., 3] = pred_slice * 0.7
        ax1.imshow(pred_rgba)
        if np.any(pred_slice > confidence_threshold):
            ax1.contour(pred_slice, levels=[confidence_threshold], colors=['cyan'], linewidths=1.5)
    ax1.set_title(f'{view_name} - Confidence', color='#ef4444', fontsize=12, fontweight='bold', pad=8)
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')

    # Column 2: MRI + Ground Truth overlay
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(mri_display, cmap='gray')
    if show_overlay and np.any(gt_slice > 0.5):
        gt_rgba = np.zeros((*gt_slice.shape, 4))
        gt_mask = gt_slice > 0.5
        gt_rgba[gt_mask, 1] = 1.0
        gt_rgba[gt_mask, 3] = 0.6
        ax2.imshow(gt_rgba)
    ax2.set_title(f'{view_name} - Ground Truth', color='#10b981', fontsize=12, fontweight='bold', pad=8)
    ax2.axis('off')
    ax2.set_facecolor('#1a1a2e')

    # Column 3: Comparison
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(mri_display, cmap='gray')
    if show_overlay:
        overlay = np.zeros((*mri_display.shape, 4))
        pred_mask = pred_slice > confidence_threshold
        gt_mask = gt_slice > 0.5
        overlap = pred_mask & gt_mask
        fp = pred_mask & ~gt_mask
        fn = gt_mask & ~pred_mask

        overlay[fp, 0] = 1.0
        overlay[fp, 3] = 0.6
        overlay[fn, 1] = 1.0
        overlay[fn, 3] = 0.6
        overlay[overlap, 0] = 1.0
        overlay[overlap, 1] = 1.0
        overlay[overlap, 3] = 0.7
        ax3.imshow(overlay)
    ax3.set_title(f'{view_name} - Comparison', color='#fbbf24', fontsize=12, fontweight='bold', pad=8)
    ax3.axis('off')
    ax3.set_facecolor('#1a1a2e')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#fbbf24', label='True Positive (Overlap)'),
        mpatches.Patch(facecolor='#ef4444', label='False Positive (Pred Only)'),
        mpatches.Patch(facecolor='#10b981', label='False Negative (GT Only)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, facecolor='#1a1a2e', edgecolor='none',
               labelcolor='white', framealpha=0)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_slice_navigator(ground_truth, prediction, current_slice, confidence_threshold=0.5):
    """Create an enhanced slice distribution visualization"""

    fig, ax = plt.subplots(figsize=(14, 3), facecolor='white')

    depth = ground_truth.shape[2]
    slices = np.arange(depth)

    gt_per_slice = np.sum(ground_truth > 0.5, axis=(0, 1))
    pred_per_slice = np.sum(prediction > confidence_threshold, axis=(0, 1))

    # Normalize
    gt_norm = gt_per_slice / (gt_per_slice.max() + 1e-8)
    pred_norm = pred_per_slice / (pred_per_slice.max() + 1e-8)

    # Fill area plots
    ax.fill_between(slices, 0, gt_norm, alpha=0.4, color='#10b981', label='Ground Truth')
    ax.fill_between(slices, 0, pred_norm, alpha=0.4, color='#ef4444', label='Prediction')

    # Line plots
    ax.plot(slices, gt_norm, color='#059669', linewidth=2)
    ax.plot(slices, pred_norm, color='#dc2626', linewidth=2)

    # Current slice marker
    ax.axvline(x=current_slice, color='#3b82f6', linestyle='-', linewidth=3, alpha=0.8)
    ax.scatter([current_slice], [0.5], s=150, color='#3b82f6', zorder=5, marker='v')

    # Annotate lesion regions
    gt_lesion_slices = np.where(gt_per_slice > 0)[0]
    if len(gt_lesion_slices) > 0:
        ax.axvspan(gt_lesion_slices[0], gt_lesion_slices[-1], alpha=0.1, color='#10b981')
        ax.annotate(f'Lesion Region\n(Slices {gt_lesion_slices[0]+1}-{gt_lesion_slices[-1]+1})',
                    xy=((gt_lesion_slices[0] + gt_lesion_slices[-1])/2, 1.05),
                    ha='center', fontsize=9, color='#059669', fontweight='bold')

    ax.set_xlabel('Slice Number', fontsize=11, fontweight='500')
    ax.set_ylabel('Relative Lesion Content', fontsize=11, fontweight='500')
    ax.set_xlim(-1, depth)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def compute_relaxed_dice(pred_binary, gt_binary, margin=2):
    """Compute relaxed Dice score with a tolerance margin (in voxels).

    A predicted voxel is considered a true positive if it falls within
    `margin` voxels of any ground truth voxel, and vice versa.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(3, 1)  # 6-connectivity

    # Dilate GT and pred by margin
    gt_dilated = binary_dilation(gt_binary > 0.5, structure=struct, iterations=margin)
    pred_dilated = binary_dilation(pred_binary > 0.5, structure=struct, iterations=margin)

    # Relaxed TP: pred voxels within margin of GT + GT voxels within margin of pred
    tp_pred = np.sum((pred_binary > 0.5) & gt_dilated)
    tp_gt = np.sum((gt_binary > 0.5) & pred_dilated)

    total_pred = np.sum(pred_binary > 0.5)
    total_gt = np.sum(gt_binary > 0.5)

    if total_pred + total_gt == 0:
        return 1.0
    return float((tp_pred + tp_gt) / (total_pred + total_gt + 1e-8))


def create_metrics_chart(gt_features, pred_features, dice_score, relaxed_dice=None):
    """Create a visual metrics comparison chart with Dice, Relaxed Dice @2, volumes, and lesion count."""

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor='white')

    # Chart 1: Volume Comparison Bar Chart
    ax1 = axes[0]
    metrics = ['Total Vol.', 'Mean Vol.', 'Max Vol.']
    gt_vals = [gt_features['total_volume'], gt_features['mean_lesion_volume'], gt_features['max_lesion_volume']]
    pred_vals = [pred_features['total_volume'], pred_features['mean_lesion_volume'], pred_features['max_lesion_volume']]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width/2, gt_vals, width, label='Ground Truth', color='#10b981', alpha=0.8)
    ax1.bar(x + width/2, pred_vals, width, label='Prediction', color='#ef4444', alpha=0.8)

    ax1.set_ylabel('Voxels', fontweight='500')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=9)
    ax1.set_title('Volume Metrics', fontweight='bold', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Chart 2: Dice Score Gauge
    ax2 = axes[1]
    ax2.set_aspect('equal')
    theta = np.linspace(np.pi, 0, 100)
    ax2.fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.2, color='gray')

    filled_theta = np.linspace(np.pi, np.pi - (dice_score * np.pi), 100)
    if dice_score >= 0.7:
        color = '#10b981'
    elif dice_score >= 0.5:
        color = '#3b82f6'
    elif dice_score >= 0.3:
        color = '#f59e0b'
    else:
        color = '#ef4444'

    ax2.fill_between(np.cos(filled_theta), 0, np.sin(filled_theta), alpha=0.8, color=color)
    ax2.text(0, 0.3, f'{dice_score:.1%}', ha='center', va='center', fontsize=28, fontweight='bold', color=color)
    ax2.text(0, -0.1, 'Dice Score', ha='center', va='center', fontsize=12, color='#64748b')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.3, 1.2)
    ax2.axis('off')
    ax2.set_title('Segmentation Accuracy', fontweight='bold', fontsize=12)

    # Chart 3: Relaxed Dice @2 Gauge
    ax_rd = axes[2]
    ax_rd.set_aspect('equal')
    ax_rd.fill_between(np.cos(theta), 0, np.sin(theta), alpha=0.2, color='gray')

    rd = relaxed_dice if relaxed_dice is not None else 0.0
    filled_rd = np.linspace(np.pi, np.pi - (rd * np.pi), 100)
    if rd >= 0.8:
        rd_color = '#10b981'
    elif rd >= 0.6:
        rd_color = '#3b82f6'
    elif rd >= 0.4:
        rd_color = '#f59e0b'
    else:
        rd_color = '#ef4444'

    ax_rd.fill_between(np.cos(filled_rd), 0, np.sin(filled_rd), alpha=0.8, color=rd_color)
    ax_rd.text(0, 0.3, f'{rd:.1%}', ha='center', va='center', fontsize=28, fontweight='bold', color=rd_color)
    ax_rd.text(0, -0.1, 'Relaxed Dice @2', ha='center', va='center', fontsize=11, color='#64748b')
    ax_rd.set_xlim(-1.2, 1.2)
    ax_rd.set_ylim(-0.3, 1.2)
    ax_rd.axis('off')
    ax_rd.set_title('Boundary-Tolerant Accuracy', fontweight='bold', fontsize=12)

    # Chart 4: Lesion Count
    ax3 = axes[3]
    categories = ['Ground Truth', 'Prediction']
    counts = [gt_features['num_lesions'], pred_features['num_lesions']]
    bar_colors = ['#10b981', '#ef4444']

    bars = ax3.bar(categories, counts, color=bar_colors, alpha=0.8, width=0.5)

    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(int(count)), ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax3.set_ylabel('Number of Lesions', fontweight='500')
    ax3.set_title('Lesion Detection', fontweight='bold', fontsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(0, max(counts) * 1.3 + 1)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


# ============================================================================
# ENSEMBLE VISUALIZATION
# ============================================================================

def create_ensemble_details_visualization(sequences, ensemble_result, ground_truth, slice_idx, confidence_threshold=0.5):
    """
    Create multi-panel view showing individual model predictions and stacking result.

    Supports up to 6 base models (4 patch + 2 nnU-Net) in a 3x4 grid:
      Row 0: 4 patch model predictions (8-patch, 12-patch, 24-patch, 36-patch)
      Row 1: nnU-Net 3D, nnU-Net 2D, Agreement Map, Confidence Histogram
      Row 2: Stacking Output, Ground Truth, Comparison (FP/FN/TP), Metrics summary
    Falls back to 2x4 grid for <= 3 models.
    """
    mri = sequences['t1_gd']
    mri_slice = np.rot90(mri[:, :, slice_idx])
    gt_slice = np.rot90(ground_truth[:, :, slice_idx])

    # Normalize MRI
    vmin, vmax = np.percentile(mri_slice, [1, 99])
    mri_display = np.clip((mri_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)

    individual_preds = ensemble_result.get('individual', {})
    fused_pred = ensemble_result['fused'][:, :, slice_idx]
    fused_slice = np.rot90(fused_pred)

    agreement = ensemble_result.get('agreement', None)
    if agreement is not None:
        agreement_slice = np.rot90(agreement[:, :, slice_idx])

    model_names = list(individual_preds.keys())
    n_models = len(model_names)

    # 6 colors for up to 6 models
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#a78bfa', '#f59e0b', '#ec4899']

    # Determine layout
    if n_models > 3:
        n_rows = 3
        fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    else:
        n_rows = 2
        fig = plt.figure(figsize=(16, 8), facecolor='#1a1a2e')
    gs = GridSpec(n_rows, 4, figure=fig, hspace=0.25, wspace=0.15)

    def _plot_pred(ax, pred_slice, color_idx, title):
        ax.imshow(mri_display, cmap='gray')
        if np.any(pred_slice > 0.1):
            pred_rgba = plt.cm.hot(pred_slice)
            pred_rgba[..., 3] = pred_slice * 0.7
            ax.imshow(pred_rgba)
            if np.any(pred_slice > confidence_threshold):
                ax.contour(pred_slice, levels=[confidence_threshold],
                           colors=[colors[color_idx % len(colors)]], linewidths=2)
        ax.set_title(title, color=colors[color_idx % len(colors)],
                     fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    # Plot individual models
    for i, name in enumerate(model_names):
        pred = individual_preds[name]
        pred_slice = np.rot90(pred[:, :, slice_idx])
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        display_name = DISPLAY_NAMES.get(name, name)
        _plot_pred(ax, pred_slice, i, display_name)

    # Determine which row/cols remain for agreement, histogram, fused, GT, comparison
    if n_models > 4:
        # Row 1 has models at cols 0,1 (nnunet, nnunet_2d); agreement at col 2, histogram at col 3
        agree_pos = (1, 2)
        hist_pos = (1, 3)
        fused_pos = (2, 0)
        gt_pos = (2, 1)
        comp_pos = (2, 2)
        metrics_pos = (2, 3)
    elif n_models > 3:
        # 4 models fill row 0; row 1 has agreement, fused, GT, comparison
        agree_pos = (1, 0)
        hist_pos = (1, 3)
        fused_pos = (2, 0)
        gt_pos = (2, 1)
        comp_pos = (2, 2)
        metrics_pos = (2, 3)
    else:
        # <= 3 models: row 0 has models + agreement; row 1 has fused, GT, comp, hist
        agree_pos = (0, 3)
        hist_pos = (1, 3)
        fused_pos = (1, 0)
        gt_pos = (1, 1)
        comp_pos = (1, 2)
        metrics_pos = None

    # Agreement map
    ax_agree = fig.add_subplot(gs[agree_pos[0], agree_pos[1]])
    ax_agree.imshow(mri_display, cmap='gray')
    if agreement is not None and np.any(agreement_slice > 0):
        agree_rgba = np.zeros((*agreement_slice.shape, 4))
        max_agree = int(agreement_slice.max())
        # Color scale: more agreement = more red/opaque
        for n_agree in range(1, max_agree + 1):
            mask = agreement_slice >= n_agree
            frac = n_agree / max(max_agree, 1)
            agree_rgba[mask] = [1.0, max(0, 0.8 - 0.6 * frac), max(0, 0.2 - 0.2 * frac),
                                0.2 + 0.5 * frac]
        ax_agree.imshow(agree_rgba)
    ax_agree.set_title('Agreement Map', color='#fbbf24', fontsize=11, fontweight='bold', pad=8)
    ax_agree.axis('off')
    ax_agree.set_facecolor('#1a1a2e')

    # Fused prediction (stacking output)
    ax_fused = fig.add_subplot(gs[fused_pos[0], fused_pos[1]])
    ax_fused.imshow(mri_display, cmap='gray')
    if np.any(fused_slice > 0.1):
        fused_rgba = plt.cm.hot(fused_slice)
        fused_rgba[..., 3] = fused_slice * 0.7
        ax_fused.imshow(fused_rgba)
        if np.any(fused_slice > confidence_threshold):
            ax_fused.contour(fused_slice, levels=[confidence_threshold], colors=['cyan'], linewidths=2)
    ax_fused.set_title('Stacking Classifier', color='#0C4DA2', fontsize=11, fontweight='bold', pad=8)
    ax_fused.axis('off')
    ax_fused.set_facecolor('#1a1a2e')

    # Ground Truth
    ax_gt = fig.add_subplot(gs[gt_pos[0], gt_pos[1]])
    ax_gt.imshow(mri_display, cmap='gray')
    if np.any(gt_slice > 0.5):
        gt_rgba = np.zeros((*gt_slice.shape, 4))
        gt_mask = gt_slice > 0.5
        gt_rgba[gt_mask, 1] = 1.0
        gt_rgba[gt_mask, 3] = 0.6
        ax_gt.imshow(gt_rgba)
    ax_gt.set_title('Ground Truth', color='#10b981', fontsize=11, fontweight='bold', pad=8)
    ax_gt.axis('off')
    ax_gt.set_facecolor('#1a1a2e')

    # Comparison (FP/FN/TP)
    ax_comp = fig.add_subplot(gs[comp_pos[0], comp_pos[1]])
    ax_comp.imshow(mri_display, cmap='gray')
    overlay = np.zeros((*mri_display.shape, 4))
    pred_mask = fused_slice > confidence_threshold
    gt_mask = gt_slice > 0.5
    tp_mask = pred_mask & gt_mask
    fp_mask = pred_mask & ~gt_mask
    fn_mask = gt_mask & ~pred_mask

    overlay[fp_mask, 0] = 1.0
    overlay[fp_mask, 3] = 0.6
    overlay[fn_mask, 1] = 1.0
    overlay[fn_mask, 3] = 0.6
    overlay[tp_mask, 0] = 1.0
    overlay[tp_mask, 1] = 1.0
    overlay[tp_mask, 3] = 0.7
    ax_comp.imshow(overlay)
    ax_comp.set_title('Comparison', color='#fbbf24', fontsize=11, fontweight='bold', pad=8)
    ax_comp.axis('off')
    ax_comp.set_facecolor('#1a1a2e')

    # Confidence histogram
    ax_hist = fig.add_subplot(gs[hist_pos[0], hist_pos[1]])
    ax_hist.set_facecolor('#2d2d2d')
    fused_flat = ensemble_result['fused'].flatten()
    positive_vals = fused_flat[fused_flat > 0.01]
    if len(positive_vals) > 0:
        ax_hist.hist(positive_vals, bins=50, color='#ef4444', alpha=0.7, edgecolor='none')
        ax_hist.axvline(x=confidence_threshold, color='cyan', linestyle='--', linewidth=2,
                        label=f'Threshold: {confidence_threshold}')
        ax_hist.legend(loc='upper right', fontsize=9, facecolor='#2d2d2d', edgecolor='none', labelcolor='white')
    ax_hist.set_xlabel('Confidence', color='white', fontsize=10)
    ax_hist.set_ylabel('Voxels', color='white', fontsize=10)
    ax_hist.set_title('Confidence Distribution', color='white', fontsize=11, fontweight='bold', pad=8)
    ax_hist.tick_params(colors='white')
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['bottom'].set_color('white')
    ax_hist.spines['left'].set_color('white')

    # Metrics summary panel (only for 3-row layout)
    if metrics_pos is not None:
        ax_metrics = fig.add_subplot(gs[metrics_pos[0], metrics_pos[1]])
        ax_metrics.set_facecolor('#1a1a2e')
        ax_metrics.axis('off')

        # Compute slice-level metrics
        gt_flat = gt_slice.flatten() > 0.5
        pred_flat = fused_slice.flatten() > confidence_threshold
        tp_count = int((gt_flat & pred_flat).sum())
        fp_count = int((~gt_flat & pred_flat).sum())
        fn_count = int((gt_flat & ~pred_flat).sum())
        slice_dice = (2 * tp_count) / (2 * tp_count + fp_count + fn_count + 1e-8)

        summary_text = (
            f"Slice Metrics\n\n"
            f"Dice:  {slice_dice:.3f}\n"
            f"TP:    {tp_count}\n"
            f"FP:    {fp_count}\n"
            f"FN:    {fn_count}\n\n"
            f"Models: {n_models}\n"
            f"Threshold: {confidence_threshold}"
        )
        ax_metrics.text(0.5, 0.5, summary_text, transform=ax_metrics.transAxes,
                        fontsize=11, color='white', verticalalignment='center',
                        horizontalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d2d', alpha=0.8))

    # Dynamic legend
    legend_elements = []
    for i, name in enumerate(model_names):
        display_name = DISPLAY_NAMES.get(name, name)
        legend_elements.append(mpatches.Patch(facecolor=colors[i % len(colors)], label=display_name))
    legend_elements.extend([
        mpatches.Patch(facecolor='#fbbf24', label='True Positive'),
        mpatches.Patch(facecolor='#ef4444', label='False Positive'),
        mpatches.Patch(facecolor='#10b981', label='False Negative'),
    ])
    fig.legend(handles=legend_elements, loc='lower center', ncol=min(len(legend_elements), 9),
               fontsize=8, facecolor='#1a1a2e', edgecolor='none',
               labelcolor='white', framealpha=0)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


# ============================================================================
# REPORT GENERATION
# ============================================================================

# Module-level RAG cache (lazy-loaded once)
_rag_retriever = None
_rag_embedder = None
_rag_available = None


def _init_rag():
    """Lazy-initialize the RAG retriever. Returns True if v2 corpus is available."""
    global _rag_retriever, _rag_embedder, _rag_available
    if _rag_available is not None:
        return _rag_available

    v2_db_path = Path(__file__).parent.parent / "outputs" / "rag" / "chromadb_v2"
    bm25_path = Path(__file__).parent.parent / "outputs" / "rag" / "bm25_index.pkl"

    if v2_db_path.exists():
        try:
            from src.rag.embeddings import BiomedCLIPEmbedder
            from src.rag.retrieval import HybridRetriever
            _rag_embedder = BiomedCLIPEmbedder()
            _rag_retriever = HybridRetriever(
                db_path=str(v2_db_path),
                embedder=_rag_embedder,
                bm25_index_path=str(bm25_path) if bm25_path.exists() else None,
                collection_name="literature_chunks_v2",
            )
            _rag_available = True
            return True
        except Exception as e:
            print(f"Warning: Could not initialize v2 RAG: {e}")

    _rag_available = False
    return False


def retrieve_rag_evidence(gt_features, pred_features, mri_volume=None):
    """
    Retrieve literature evidence using v2 hybrid retrieval.
    Returns list of RetrievalResult objects with paper metadata.
    Falls back to old collections if v2 not available.
    """
    import time as _time
    retrieval_start = _time.time()

    num_lesions = gt_features['num_lesions']

    # Build clinical query from lesion characteristics
    if num_lesions == 0:
        query = "brain MRI no metastases surveillance imaging follow-up"
    elif num_lesions == 1:
        vol = gt_features['max_lesion_volume']
        if vol < 500:
            query = "small brain metastasis stereotactic radiosurgery treatment outcomes"
        elif vol < 5000:
            query = "solitary brain metastasis treatment surgery vs radiosurgery"
        else:
            query = "large brain metastasis surgical resection decompression"
    elif num_lesions <= 4:
        query = "oligometastatic brain disease stereotactic radiosurgery local control"
    else:
        query = "multiple brain metastases whole brain radiation therapy immunotherapy"

    results = []

    if _init_rag():
        try:
            # Text-based hybrid retrieval
            text_results = _rag_retriever.retrieve(query, k=8)
            results.extend(text_results)

            # Cross-modal retrieval if MRI volume provided
            if mri_volume is not None and _rag_embedder is not None:
                try:
                    img_embedding = _rag_embedder.embed_mri_volume(mri_volume)
                    img_results = _rag_retriever.retrieve_by_image(img_embedding, k=4)
                    results.extend(img_results)
                except Exception:
                    pass

            # Deduplicate by paper_id (keep highest score per paper)
            seen_papers = {}
            for r in results:
                paper_id = r.chunk_id.rsplit('_', 2)[0]  # Extract PMID from chunk_id
                if paper_id not in seen_papers or r.score > seen_papers[paper_id].score:
                    seen_papers[paper_id] = r
            results = sorted(seen_papers.values(), key=lambda x: x.score, reverse=True)[:8]

        except Exception as e:
            print(f"v2 retrieval error: {e}")
            results = []

    # Fallback to old collections
    if not results:
        try:
            import chromadb
            db_path = Path(__file__).parent.parent / "outputs" / "rag" / "chromadb"
            if db_path.exists():
                client = chromadb.PersistentClient(path=str(db_path))
                try:
                    kb = client.get_collection("medical_knowledge")
                    kb_results = kb.query(query_texts=[query], n_results=3)
                    for i, doc in enumerate(kb_results['documents'][0]):
                        # Create a minimal result-like object
                        results.append(type('FallbackResult', (), {
                            'chunk_id': f'kb_{i}', 'text': doc, 'score': 0.5,
                            'section': 'knowledge_base', 'title': 'Curated Knowledge',
                            'authors': 'BrainMetScan KB', 'journal': '', 'year': '',
                            'doi': '', 'mesh_terms': [], 'retrieval_method': 'legacy',
                        })())
                except Exception:
                    pass
        except Exception:
            pass

    retrieval_time = _time.time() - retrieval_start
    return results, retrieval_time


def generate_clinical_summary(gt_features, pred_features, dice_score, retrieval_results=None, mri_meta=None):
    """
    Generate a detailed clinical summary with inline citations [1][2].
    Includes MRI acquisition details, lesion characterization, and clinical context.
    """
    num_lesions = gt_features['num_lesions']
    total_volume = gt_features['total_volume']
    max_volume = gt_features['max_lesion_volume']

    # Build citation map
    citation_map = {}
    if retrieval_results:
        for i, r in enumerate(retrieval_results, 1):
            paper_id = r.chunk_id.rsplit('_', 2)[0]
            if paper_id not in citation_map:
                citation_map[paper_id] = i

    def _cite(result):
        """Generate citation tag for a result."""
        paper_id = result.chunk_id.rsplit('_', 2)[0]
        num = citation_map.get(paper_id, 0)
        if num > 0:
            return f' <sup style="color: #0C4DA2; font-weight: 600;">[{num}]</sup>'
        return ''

    # ---- MRI Acquisition Details ----
    mri_meta = mri_meta or {}
    dims = mri_meta.get('dimensions', None)
    seq_list = mri_meta.get('sequences', [])

    # Map sequence keys to clinical names
    seq_display = {
        't1_pre': 'T1-weighted pre-contrast',
        't1_gd': 'T1-weighted post-gadolinium',
        'flair': 'FLAIR',
        't2': 'T2-weighted',
        'fourth': 'T2-weighted',
    }
    seq_names = []
    seen = set()
    for s in seq_list:
        display = seq_display.get(s, s)
        if display not in seen:
            seq_names.append(display)
            seen.add(display)

    if dims:
        mri_text = f"Multi-sequence brain MRI was acquired and reconstructed at {dims[0]}&times;{dims[1]}&times;{dims[2]} voxels"
        if seq_names:
            mri_text += f", comprising {', '.join(seq_names[:-1])}, and {seq_names[-1]} sequences" if len(seq_names) > 1 else f", comprising a {seq_names[0]} sequence"
        mri_text += ". T1 post-gadolinium was used as the primary sequence for lesion detection, with FLAIR providing complementary evaluation of perilesional edema."
    else:
        mri_text = "Multi-sequence brain MRI was analyzed using T1 post-gadolinium as the primary detection sequence."

    # ---- Lesion Characterization ----
    # Spatial distribution analysis from centroid data
    spatial_text = ""
    if num_lesions > 1 and gt_features.get('lesion_centroids'):
        centroids = np.array(gt_features['lesion_centroids'])
        z_spread = centroids[:, 2].max() - centroids[:, 2].min()
        if dims:
            z_fraction = z_spread / dims[2]
            if z_fraction > 0.5:
                spatial_text = " Lesions are distributed across a wide axial range, spanning multiple brain regions."
            elif z_fraction > 0.2:
                spatial_text = " Lesions are distributed across several adjacent axial slices."
            else:
                spatial_text = " Lesions are clustered within a relatively focal axial region."

    # Volume heterogeneity
    volume_text = ""
    if num_lesions > 1 and gt_features.get('lesion_volumes'):
        volumes = gt_features['lesion_volumes']
        vol_ratio = max(volumes) / (min(volumes) + 1e-8)
        if vol_ratio > 10:
            volume_text = f" There is marked size heterogeneity among the lesions, with the largest measuring {max_volume:,.0f} voxels and the smallest {min(volumes):,.0f} voxels (ratio {vol_ratio:.0f}:1)."
        elif vol_ratio > 3:
            volume_text = f" Moderate size heterogeneity is present, with the largest lesion ({max_volume:,.0f} voxels) approximately {vol_ratio:.1f}x the size of the smallest ({min(volumes):,.0f} voxels)."

    # ---- Findings ----
    if num_lesions == 0:
        finding_text = f"{mri_text} Automated AI segmentation analysis did not identify any enhancing lesions suspicious for brain metastases."
        impact_text = "This is a favorable finding, suggesting no current evidence of intracranial metastatic disease. No regions of abnormal contrast enhancement or suspicious signal abnormality were detected across all evaluated sequences."
        next_steps = "Continued surveillance imaging is recommended per institutional protocol, particularly if the patient has a known primary malignancy with high CNS tropism (e.g., lung, breast, melanoma). A follow-up MRI in 3-6 months is advisable."

    elif num_lesions == 1:
        centroid = gt_features['lesion_centroids'][0] if gt_features.get('lesion_centroids') else None

        if max_volume < 500:
            size_desc = "small"
            size_approx = "sub-centimeter"
            treatment_note = "Given the small size, this lesion may be an excellent candidate for stereotactic radiosurgery (SRS), which offers precise single-fraction targeting with minimal damage to surrounding brain parenchyma."
        elif max_volume < 5000:
            size_desc = "moderate-sized"
            size_approx = "approximately 1-2 cm"
            treatment_note = "At this size, the lesion may be amenable to either stereotactic radiosurgery (SRS) or surgical resection, depending on location accessibility and patient performance status."
        else:
            size_desc = "large"
            size_approx = "exceeding 2 cm"
            treatment_note = "Given the size, surgical resection should be considered to achieve rapid decompression, alleviate mass effect, and obtain tissue for histopathological confirmation, potentially followed by adjuvant radiation to the resection cavity."

        location_text = ""
        if centroid and dims:
            # Approximate anatomical location from normalized coordinates
            z_frac = centroid[2] / dims[2]
            if z_frac > 0.7:
                location_text = " in the superior cerebral convexity"
            elif z_frac > 0.4:
                location_text = " at the level of the lateral ventricles"
            elif z_frac > 0.2:
                location_text = " in the infratentorial region"
            else:
                location_text = " in the posterior fossa"

        finding_text = f"{mri_text} The analysis identified a single {size_desc} enhancing lesion{location_text}, measuring {max_volume:,.0f} voxels ({size_approx} estimated equivalent diameter) on T1 post-gadolinium imaging."
        impact_text = "A solitary brain metastasis generally carries a more favorable prognosis compared to multiple lesions. Focal neurological symptoms may be present depending on the anatomical location and degree of perilesional edema visible on FLAIR."

        if retrieval_results and len(retrieval_results) >= 2:
            cite1 = _cite(retrieval_results[0])
            cite2 = _cite(retrieval_results[1])
            treatment_note += f"{cite1}"
            impact_text += f"{cite2}"

        next_steps = f"{treatment_note} A multidisciplinary tumor board discussion is recommended to determine the optimal therapeutic approach, integrating imaging findings with the patient's overall oncologic status."

    else:  # Multiple lesions
        if num_lesions <= 4:
            burden_desc = "oligometastatic"
            burden_detail = f"The oligometastatic burden ({num_lesions} lesions) suggests that individual lesion-directed therapy remains feasible."
            treatment_approach = "For oligometastatic disease, stereotactic radiosurgery (SRS) to each lesion is often preferred, providing good local control rates while preserving neurocognitive function compared to whole-brain radiation."
        elif num_lesions <= 10:
            burden_desc = "moderate metastatic"
            burden_detail = f"The {num_lesions} identified lesions represent a moderate intracranial disease burden."
            treatment_approach = "With this lesion count, a combination approach may be considered: SRS to larger/symptomatic lesions combined with systemic therapy for smaller lesions, or whole-brain radiation with hippocampal avoidance."
        else:
            burden_desc = "extensive metastatic"
            burden_detail = f"The extensive intracranial disease burden ({num_lesions} lesions) indicates advanced CNS involvement."
            treatment_approach = "With extensive brain metastases, whole-brain radiation therapy (WBRT) or hippocampal-sparing WBRT with memantine is typically indicated. Immunotherapy and targeted systemic agents with CNS penetration should also be evaluated."

        finding_text = f"{mri_text} The analysis identified {num_lesions} enhancing lesions consistent with {burden_desc} disease, with a total tumor burden of {total_volume:,.0f} voxels.{spatial_text}{volume_text}"
        impact_text = f"{burden_detail} Multiple brain metastases may present with headaches, seizures, focal neurological deficits, or progressive cognitive decline, depending on lesion size and location."

        if retrieval_results:
            cites = ''.join(_cite(r) for r in retrieval_results[:3] if _cite(r))
            treatment_approach += cites

        next_steps = f"{treatment_approach} Systemic therapy optimization should also be considered in coordination with the treating oncology team."

    # ---- Model Performance Context ----
    num_models = _model_info.get('num_models', 2)
    model_names = _model_info.get('model_names', [])
    model_desc = f"{num_models}-model ensemble ({', '.join(model_names)})" if model_names else "ensemble model"

    if dice_score >= 0.7:
        confidence_note = f"The {model_desc} achieved a Dice coefficient of {dice_score:.1%}, indicating high concordance between the AI segmentation and the reference annotation. These results support reliable lesion detection for this case."
    elif dice_score >= 0.5:
        confidence_note = f"The {model_desc} achieved a Dice coefficient of {dice_score:.1%}, suggesting moderate segmentation agreement. Expert radiologist verification is recommended, particularly for smaller lesions near the detection threshold."
    else:
        confidence_note = f"The {model_desc} achieved a Dice coefficient of {dice_score:.1%}, indicating limited segmentation agreement for this case. This may reflect small lesion sizes, atypical enhancement patterns, or challenging anatomy. Expert radiologist review is strongly recommended."

    summary = f"""
    <p style="margin-bottom: 14px; text-align: justify; color: #e8e8e6;">
        <strong style="color: #ffffff;">Imaging &amp; Findings:</strong> {finding_text}
    </p>
    <p style="margin-bottom: 14px; text-align: justify; color: #e8e8e6;">
        <strong style="color: #ffffff;">Clinical Significance:</strong> {impact_text}
    </p>
    <p style="margin-bottom: 14px; text-align: justify; color: #e8e8e6;">
        <strong style="color: #ffffff;">Management Considerations:</strong> {next_steps}
    </p>
    <p style="margin-bottom: 0; text-align: justify; color: #c0c0bc;">
        <strong style="color: #a8a8a3;">Segmentation Quality:</strong> {confidence_note}
    </p>
    """

    return summary


def generate_clinical_report(case_name, gt_features, pred_features, dice_score, confidence_threshold=0.5, is_ensemble=False, retrieval_results=None, retrieval_time=0.0, mri_meta=None):
    """Generate a professional clinical analysis report with RAG citations and evidence panel."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine performance level
    if dice_score >= 0.7:
        perf_level = "Excellent"
        perf_color = "#10b981"
    elif dice_score >= 0.5:
        perf_level = "Good"
        perf_color = "#3b82f6"
    elif dice_score >= 0.3:
        perf_level = "Moderate"
        perf_color = "#f59e0b"
    else:
        perf_level = "Limited"
        perf_color = "#ef4444"

    # Size classification
    total_vol = gt_features['total_volume']
    if total_vol < 500:
        size_class = "Tiny (<500 voxels)"
    elif total_vol < 5000:
        size_class = "Medium (500-5000 voxels)"
    else:
        size_class = "Large (>5000 voxels)"

    # Model info
    num_models = _model_info.get('num_models', 2)
    model_type = f"{num_models}-Model Ensemble" if is_ensemble else "Single Model"
    model_badge_color = "#10b981" if is_ensemble else "#878681"

    # RAG badge
    rag_badge = ""
    if retrieval_results:
        rag_badge = '<span style="background: #7c3aed; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600;">RAG-Enhanced</span>'

    report = f"""
<div style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.7; color: #0a0a0a; -webkit-font-smoothing: antialiased;">

<div style="background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%); padding: 32px; border-radius: 16px; margin-bottom: 32px; border-bottom: 3px solid #0C4DA2;">
    <p style="font-size: 0.75em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #a8a8a3; margin: 0 0 8px 0;">Clinical Analysis Report</p>
    <h2 style="margin: 0; font-size: 2em; font-weight: 600; letter-spacing: -0.02em; color: #ffffff;">{case_name}</h2>
    <p style="margin: 8px 0 0 0; color: #878681; font-size: 0.95em;">Generated {timestamp}</p>
    <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
        <span style="background: {model_badge_color}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600;">{model_type}</span>
        <span style="background: #0C4DA2; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600;">Threshold: {confidence_threshold}</span>
        {rag_badge}
    </div>
</div>

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px;">
    <div style="background: linear-gradient(180deg, #ffffff 0%, #e8e8e6 100%); padding: 24px 20px; border-radius: 16px; text-align: center; border: 1px solid #a8a8a3;">
        <div style="font-size: 2.2em; font-weight: 600; color: #0a0a0a; letter-spacing: -0.02em;">{gt_features['num_lesions']}</div>
        <div style="font-size: 0.85em; color: #878681; margin-top: 4px; font-weight: 500;">Lesions Detected</div>
    </div>
    <div style="background: linear-gradient(180deg, #ffffff 0%, #e8e8e6 100%); padding: 24px 20px; border-radius: 16px; text-align: center; border: 1px solid #a8a8a3;">
        <div style="font-size: 2.2em; font-weight: 600; color: #0a0a0a; letter-spacing: -0.02em;">{gt_features['total_volume']:,.0f}</div>
        <div style="font-size: 0.85em; color: #878681; margin-top: 4px; font-weight: 500;">Total Volume</div>
    </div>
    <div style="background: linear-gradient(180deg, #0C4DA2 0%, #083A7A 100%); padding: 24px 20px; border-radius: 16px; text-align: center;">
        <div style="font-size: 2.2em; font-weight: 600; color: #ffffff; letter-spacing: -0.02em;">{dice_score:.1%}</div>
        <div style="font-size: 0.85em; color: rgba(255,255,255,0.8); margin-top: 4px; font-weight: 500;">Dice Score</div>
    </div>
    <div style="background: linear-gradient(180deg, #ffffff 0%, #e8e8e6 100%); padding: 24px 20px; border-radius: 16px; text-align: center; border: 2px solid {perf_color};">
        <div style="font-size: 1.5em; font-weight: 600; color: {perf_color}; letter-spacing: -0.02em;">{perf_level}</div>
        <div style="font-size: 0.85em; color: #878681; margin-top: 4px; font-weight: 500;">Performance</div>
    </div>
</div>

<div style="background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%); border-radius: 16px; padding: 24px; margin-bottom: 24px; border-left: 4px solid #0C4DA2;">
    <h3 style="color: #ffffff; margin: 0 0 16px 0; font-size: 1.1em; font-weight: 600; letter-spacing: -0.01em;">
        Clinical Summary
    </h3>
    <div style="color: #e8e8e6;">
    {generate_clinical_summary(gt_features, pred_features, dice_score, retrieval_results=retrieval_results, mri_meta=mri_meta)}
    </div>
</div>
"""

    # ---- Literature Evidence Panel (NEW) ----
    if retrieval_results:
        report += """
<div style="background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%); border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #5c5c58;">
    <h3 style="color: #ffffff; margin: 0 0 4px 0; font-size: 1.1em; font-weight: 600; letter-spacing: -0.01em;">Literature Evidence</h3>
    <p style="color: #878681; font-size: 0.8em; margin: 0 0 16px 0;">Retrieved from PubMed corpus via hybrid BiomedCLIP + BM25 search</p>
"""
        for i, r in enumerate(retrieval_results, 1):
            # Score color
            if r.score > 0.008:
                score_class = "evidence-score-high"
                score_pct = min(100, r.score * 5000)
            elif r.score > 0.005:
                score_class = "evidence-score-med"
                score_pct = min(100, r.score * 5000)
            else:
                score_class = "evidence-score-low"
                score_pct = min(100, r.score * 5000)

            # Author truncation
            authors_short = r.authors[:60] + '...' if len(r.authors) > 60 else r.authors
            journal_year = f"{r.journal} {r.year}".strip() if r.journal else r.year

            # Section label
            section_label = r.section.replace('_', ' ').title() if r.section else 'Abstract'

            # Truncate text for display
            display_text = r.text[:400] + '...' if len(r.text) > 400 else r.text

            report += f"""
    <details class="evidence-card">
        <summary>
            <span style="background: #0C4DA2; color: white; padding: 2px 8px; border-radius: 8px; font-size: 0.75em; font-weight: 700; min-width: 24px; text-align: center;">{i}</span>
            <span style="flex: 1;">{r.title[:80]}{'...' if len(r.title) > 80 else ''}</span>
            <span style="color: #878681; font-size: 0.8em; white-space: nowrap;">{section_label}</span>
        </summary>
        <div class="evidence-body">
            <p style="color: #878681; font-size: 0.8em; margin: 12px 0 8px 0;">{authors_short} &middot; {journal_year} &middot; {r.retrieval_method}</p>
            <div style="background: rgba(92, 92, 88, 0.15); border-radius: 8px; padding: 12px; margin: 8px 0;">
                <p style="margin: 0; font-style: italic;">{display_text}</p>
            </div>
            <div class="evidence-score-bar {score_class}" style="width: {score_pct}%;"></div>
        </div>
    </details>
"""
        report += "</div>"

    # ---- Segmentation Results Table ----
    report += f"""
<div style="background: #ffffff; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #a8a8a3;">
    <h3 style="color: #0a0a0a; margin: 0 0 20px 0; font-size: 1.1em; font-weight: 600; letter-spacing: -0.01em;">Segmentation Results</h3>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.95em;">
        <tr>
            <th style="padding: 14px 12px; text-align: left; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em;">Metric</th>
            <th style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em;">Ground Truth</th>
            <th style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em;">Prediction</th>
            <th style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em;">Difference</th>
        </tr>
        <tr>
            <td style="padding: 14px 12px; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">Number of Lesions</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); font-weight: 600; color: #1d1d1f;">{gt_features['num_lesions']}</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); font-weight: 600; color: #1d1d1f;">{pred_features['num_lesions']}</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #86868b;">{pred_features['num_lesions'] - gt_features['num_lesions']:+d}</td>
        </tr>
        <tr>
            <td style="padding: 14px 12px; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">Total Volume</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{gt_features['total_volume']:,.0f}</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{pred_features['total_volume']:,.0f}</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #86868b;">{pred_features['total_volume'] - gt_features['total_volume']:+,.0f}</td>
        </tr>
        <tr>
            <td style="padding: 14px 12px; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">Mean Lesion Volume</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{gt_features['mean_lesion_volume']:,.0f}</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{pred_features['mean_lesion_volume']:,.0f}</td>
            <td style="padding: 14px 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #86868b;">-</td>
        </tr>
        <tr>
            <td style="padding: 14px 12px; color: #1d1d1f;">Largest Lesion</td>
            <td style="padding: 14px 12px; text-align: center; font-weight: 600; color: #1d1d1f;">{gt_features['max_lesion_volume']:,.0f}</td>
            <td style="padding: 14px 12px; text-align: center; font-weight: 600; color: #1d1d1f;">{pred_features['max_lesion_volume']:,.0f}</td>
            <td style="padding: 14px 12px; text-align: center; color: #86868b;">-</td>
        </tr>
    </table>
</div>

<div style="background: #ffffff; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #a8a8a3;">
    <h3 style="color: #0a0a0a; margin: 0 0 20px 0; font-size: 1.1em; font-weight: 600; letter-spacing: -0.01em;">Clinical Findings</h3>
"""

    # Findings based on lesion count
    num_lesions = gt_features['num_lesions']

    if num_lesions == 0:
        report += """
    <div style="background: #ffffff; padding: 20px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid #34c759; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="width: 10px; height: 10px; background: #34c759; border-radius: 50%;"></div>
            <strong style="color: #1a1a1a; font-size: 1.05em;">No Metastatic Lesions Detected</strong>
        </div>
        <p style="margin: 0; color: #3a3a3a; padding-left: 20px;">The scan shows no evidence of brain metastases on post-gadolinium imaging.</p>
    </div>
"""
    elif num_lesions == 1:
        centroid = gt_features['lesion_centroids'][0] if gt_features['lesion_centroids'] else (0, 0, 0)
        report += f"""
    <div style="background: #ffffff; padding: 20px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid #ff9f0a; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="width: 10px; height: 10px; background: #ff9f0a; border-radius: 50%;"></div>
            <strong style="color: #1a1a1a; font-size: 1.05em;">Single Metastatic Lesion Identified</strong>
        </div>
        <ul style="margin: 0; color: #3a3a3a; padding-left: 20px; list-style: none;">
            <li style="margin-bottom: 6px;">Volume: <strong style="color: #1a1a1a;">{gt_features['max_lesion_volume']:,.0f} voxels</strong></li>
            <li style="margin-bottom: 6px;">Size Classification: <strong style="color: #1a1a1a;">{size_class}</strong></li>
            <li>Approximate Location: <strong style="color: #1a1a1a;">({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})</strong></li>
        </ul>
    </div>
"""
    else:
        report += f"""
    <div style="background: #ffffff; padding: 20px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid #ff3b30; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="width: 10px; height: 10px; background: #ff3b30; border-radius: 50%;"></div>
            <strong style="color: #1a1a1a; font-size: 1.05em;">Multiple Metastatic Lesions Identified ({num_lesions})</strong>
        </div>
        <ul style="margin: 0; color: #000000; padding-left: 20px; list-style: none;">
            <li style="margin-bottom: 6px;">Total Tumor Burden: <strong style="color: #000000;">{gt_features['total_volume']:,.0f} voxels</strong></li>
            <li style="margin-bottom: 6px;">Largest Lesion: <strong style="color: #000000;">{gt_features['max_lesion_volume']:,.0f} voxels</strong></li>
            <li>Mean Lesion Size: <strong style="color: #000000;">{gt_features['mean_lesion_volume']:,.0f} voxels</strong></li>
        </ul>
    </div>
"""

    # Lesion locations
    if gt_features['lesion_slices']:
        report += """
    <h4 style="color: #86868b; font-size: 0.8em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px 0;">Lesion Locations</h4>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.95em;">
        <tr>
            <th style="padding: 12px; text-align: left; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Lesion</th>
            <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Slice Range</th>
            <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Center Slice</th>
        </tr>
"""
        for i, (z_min, z_max, z_center) in enumerate(gt_features['lesion_slices']):
            report += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">Lesion {i+1}</td>
            <td style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{int(z_min)+1} - {int(z_max)+1}</td>
            <td style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f; font-weight: 600;">{int(z_center)+1}</td>
        </tr>
"""
        report += "    </table>"

    report += "</div>"

    # Recommendations (with citation references if available)
    report += """
<div style="background: #ffffff; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #d0d0cd; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
    <h3 style="color: #000000; margin: 0 0 20px 0; font-size: 1.1em; font-weight: 600; letter-spacing: -0.01em;">Clinical Recommendations</h3>
    <ol style="padding-left: 20px; color: #000000; line-height: 1.9;">
"""

    if num_lesions > 0:
        report += """
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">Clinical Correlation:</strong> <span style="color: #000000;">Correlate imaging findings with patient history, known primary malignancy, and current systemic disease status</span></li>
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">MDT Review:</strong> <span style="color: #000000;">Recommend multidisciplinary tumor board discussion for integrated treatment planning (neuro-oncology, radiation oncology, neurosurgery)</span></li>
"""
        if num_lesions == 1 and total_vol > 1000:
            report += """
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">Treatment Options:</strong> <span style="color: #000000;">Single lesion may be amenable to surgical resection or stereotactic radiosurgery (SRS); decision depends on location, size, and patient performance status</span></li>
"""
        elif num_lesions <= 4:
            report += """
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">Treatment Options:</strong> <span style="color: #000000;">Oligometastatic disease pattern supports stereotactic radiosurgery (SRS) to individual lesions; systemic therapy with CNS penetration should also be evaluated</span></li>
"""
        else:
            report += """
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">Treatment Options:</strong> <span style="color: #000000;">Consider whole-brain radiation therapy (WBRT) with hippocampal avoidance, or systemic therapy with demonstrated CNS activity; SRS to dominant lesions may be considered in combination</span></li>
"""
        report += """
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">Follow-up Imaging:</strong> <span style="color: #000000;">Recommend surveillance MRI with contrast in 6-8 weeks to assess treatment response per RANO-BM criteria</span></li>
        <li style="margin-bottom: 0;"><strong style="color: #000000;">Supportive Care:</strong> <span style="color: #000000;">Assess for symptoms of raised intracranial pressure; consider corticosteroids if significant perilesional edema is present on FLAIR</span></li>
"""
    else:
        report += """
        <li style="margin-bottom: 10px;"><strong style="color: #000000;">Surveillance:</strong> <span style="color: #000000;">Continue routine surveillance imaging per institutional protocol, with interval dependent on primary tumor type and systemic disease burden</span></li>
        <li style="margin-bottom: 0;"><strong style="color: #000000;">Clinical Correlation:</strong> <span style="color: #000000;">Correlate negative imaging findings with clinical presentation; consider repeat imaging if new neurological symptoms develop</span></li>
"""

    report += """
    </ol>
</div>
"""

    # Confidence threshold explanation
    threshold_explanation = ""
    if confidence_threshold < 0.4:
        threshold_explanation = "Using a <strong>low threshold</strong> increases sensitivity but may include more false positives."
    elif confidence_threshold > 0.7:
        threshold_explanation = "Using a <strong>high threshold</strong> increases specificity but may miss smaller or less certain lesions."
    else:
        threshold_explanation = "Using a <strong>balanced threshold</strong> provides a tradeoff between sensitivity and specificity."

    report += f"""
<div style="background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); border-radius: 12px; padding: 20px; margin-bottom: 24px; border: 1px solid #5c5c58;">
    <h4 style="color: #ffffff; margin: 0 0 12px 0; font-size: 0.95em; font-weight: 600;">Confidence Threshold: {confidence_threshold}</h4>
    <p style="color: #a8a8a3; font-size: 0.85em; margin: 0;">{threshold_explanation}</p>
</div>
"""

    # ---- Retrieval Pipeline Info (NEW) ----
    if retrieval_results:
        # Get corpus stats
        corpus_size = "?"
        try:
            stats_path = Path(__file__).parent.parent / "outputs" / "rag" / "build_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                corpus_size = f"{stats.get('total_papers', '?'):,}"
        except Exception:
            pass

        retrieval_method = retrieval_results[0].retrieval_method if retrieval_results else 'unknown'
        report += f"""
<div style="background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%); border-radius: 12px; padding: 16px 20px; margin-bottom: 24px; border: 1px solid #5c5c58;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <span style="color: #878681; font-size: 0.8em;">Retrieved from corpus of <strong style="color: #a8a8a3;">{corpus_size} papers</strong></span>
        <span style="color: #878681; font-size: 0.8em;">Hybrid retrieval: <strong style="color: #a8a8a3;">BiomedCLIP dense + BM25 sparse</strong></span>
        <span style="color: #878681; font-size: 0.8em;">RRF fusion &middot; <strong style="color: #a8a8a3;">{retrieval_time:.2f}s</strong></span>
    </div>
</div>
"""

    # Model info footer
    if is_ensemble:
        num_models = _model_info.get('num_models', 2)
        model_names = _model_info.get('model_names', [])
        if model_names:
            model_desc = f"{num_models}-Model Ensemble ({', '.join(model_names)})"
        else:
            model_desc = f"{num_models}-Model Ensemble"
    else:
        model_desc = f"3D U-Net &middot; Epoch {_model_info.get('epoch', '?')}"

    report += f"""
<div style="background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%); border-radius: 12px; padding: 24px; text-align: center; margin-top: 32px; border-top: 3px solid #0C4DA2;">
    <p style="color: #a8a8a3; font-size: 0.85em; margin: 0 0 8px 0;">
        This report is AI-generated and requires expert radiologist review.
    </p>
    <p style="color: #878681; font-size: 0.8em; margin: 0;">Model: {model_desc} &middot; Validation Dice: {_model_info.get("val_dice", 0):.1%}</p>
</div>

</div>
"""

    return report


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_case(case_name, slice_pct, view_mode, confidence_threshold=0.5, slice_axis="Axial", progress=gr.Progress()):
    """Main processing function with ensemble support"""
    global _prediction_cache, _probability_cache

    if not case_name:
        return None, None, None, "Please select a case", ""

    try:
        # Try to load stacking/ensemble first, fall back to single model
        progress(0.1, desc="Loading stacking classifier...")
        ensemble = load_stacking_model_cached()
        is_ensemble = ensemble is not None

        if not is_ensemble:
            progress(0.1, desc="Loading model...")
            model = load_model()

        progress(0.2, desc="Loading MRI data...")
        sequences, ground_truth = load_case_data(case_name)

        H, W, D = sequences['t1_gd'].shape

        # Calculate slice index for the selected axis from the slider
        if slice_axis == "Sagittal":
            sag_idx = int((slice_pct / 100) * (H - 1))
            ax_idx = int(np.argmax(np.sum(ground_truth, axis=(0, 1)))) if np.any(ground_truth > 0.5) else D // 2
            cor_idx = int(np.argmax(np.sum(ground_truth, axis=(0, 2)))) if np.any(ground_truth > 0.5) else W // 2
        elif slice_axis == "Coronal":
            cor_idx = int((slice_pct / 100) * (W - 1))
            ax_idx = int(np.argmax(np.sum(ground_truth, axis=(0, 1)))) if np.any(ground_truth > 0.5) else D // 2
            sag_idx = int(np.argmax(np.sum(ground_truth, axis=(1, 2)))) if np.any(ground_truth > 0.5) else H // 2
        else:  # Axial
            ax_idx = int((slice_pct / 100) * (D - 1))
            gt_sum_sag = np.sum(ground_truth, axis=(1, 2))
            gt_sum_cor = np.sum(ground_truth, axis=(0, 2))
            sag_idx = int(np.argmax(gt_sum_sag)) if np.max(gt_sum_sag) > 0 else H // 2
            cor_idx = int(np.argmax(gt_sum_cor)) if np.max(gt_sum_cor) > 0 else W // 2

        # Run inference - cache probability maps
        cache_key = f"{case_name}_ensemble" if is_ensemble else case_name
        if cache_key not in _probability_cache:
            progress(0.3, desc="Running AI segmentation...")
            if is_ensemble:
                ensemble_result = run_stacking_demo_inference(ensemble, sequences, case_name=case_name)
                _probability_cache[cache_key] = ensemble_result
                prediction_volume = ensemble_result['fused']
            else:
                prediction_volume = run_full_inference(model, sequences)
                _probability_cache[cache_key] = {'fused': prediction_volume, 'individual': {}, 'agreement': None}
        else:
            ensemble_result = _probability_cache[cache_key]
            prediction_volume = ensemble_result['fused']

        # Resize prediction to match MRI if shapes differ (e.g. cache at 256^3, MRI at 256x256x150)
        mri_shape = sequences['t1_gd'].shape
        if prediction_volume.shape != mri_shape:
            from scipy.ndimage import zoom as _zoom
            factors = [t / s for t, s in zip(mri_shape, prediction_volume.shape)]
            prediction_volume = _zoom(prediction_volume.astype(np.float32), factors, order=1)
            # Also resize individual predictions and agreement in the cached result
            ensemble_result = _probability_cache[cache_key]
            ensemble_result['fused'] = prediction_volume
            if ensemble_result.get('agreement') is not None and ensemble_result['agreement'].shape != mri_shape:
                ensemble_result['agreement'] = _zoom(ensemble_result['agreement'].astype(np.float32), factors, order=0)
            for mname in list(ensemble_result.get('individual', {}).keys()):
                pred = ensemble_result['individual'][mname]
                if pred.shape != mri_shape:
                    ensemble_result['individual'][mname] = _zoom(pred.astype(np.float32), factors, order=1)

        # Also store in old cache for compatibility
        _prediction_cache[case_name] = prediction_volume

        progress(0.6, desc="Extracting features...")
        gt_features = extract_lesion_features(ground_truth, sequences['t1_gd'])
        # Extract features at current threshold
        pred_features = extract_lesion_features((prediction_volume > confidence_threshold).astype(np.float32), sequences['t1_gd'])

        # Calculate Dice and Relaxed Dice @2 at current threshold
        gt_binary = (ground_truth > 0.5).astype(np.float32)
        pred_binary = (prediction_volume > confidence_threshold).astype(np.float32)
        intersection = np.sum(gt_binary * pred_binary)
        dice_score = (2 * intersection) / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-8)
        relaxed_dice = compute_relaxed_dice(pred_binary, gt_binary, margin=2)

        # RAG evidence retrieval
        progress(0.65, desc="Retrieving literature evidence...")
        try:
            retrieval_results, retrieval_time = retrieve_rag_evidence(
                gt_features, pred_features, mri_volume=sequences.get('t1_gd')
            )
        except Exception:
            retrieval_results, retrieval_time = [], 0.0

        progress(0.7, desc="Creating visualizations...")

        # Create visualizations
        show_overlay = view_mode != "MRI Only"
        main_img = create_multiview_visualization(
            sequences, prediction_volume, ground_truth,
            (ax_idx, sag_idx, cor_idx), show_overlay, confidence_threshold,
            slice_axis=slice_axis
        )

        nav_img = create_slice_navigator(ground_truth, prediction_volume, ax_idx, confidence_threshold)
        metrics_img = create_metrics_chart(gt_features, pred_features, dice_score, relaxed_dice=relaxed_dice)

        progress(0.9, desc="Generating report...")
        mri_meta = {'dimensions': (H, W, D), 'sequences': list(sequences.keys())}
        report = generate_clinical_report(
            case_name, gt_features, pred_features, dice_score, confidence_threshold, is_ensemble,
            retrieval_results=retrieval_results, retrieval_time=retrieval_time, mri_meta=mri_meta
        )

        # Slice info with ensemble status and RAG info
        num_models = _model_info.get('num_models', 2)
        model_status = f"{num_models}-Model Ensemble" if is_ensemble else "Single Model"
        rag_status = f"{len(retrieval_results)} refs" if retrieval_results else "N/A"
        # Show the active axis prominently
        axis_info = {
            "Axial": f"**Axial Slice:** {ax_idx + 1} of {D}",
            "Sagittal": f"**Sagittal Slice:** {sag_idx + 1} of {H}",
            "Coronal": f"**Coronal Slice:** {cor_idx + 1} of {W}",
        }
        slice_info = f"""
### Model
- **Type:** {model_status}
- **Threshold:** {confidence_threshold}

### Current View ({slice_axis})
- {axis_info[slice_axis]}

### Volume Statistics
- **Dice Score:** {dice_score:.1%}
- **GT Lesions:** {gt_features['num_lesions']}
- **Pred Lesions:** {pred_features['num_lesions']}

### RAG Evidence
- **Citations:** {rag_status}
"""

        progress(1.0, desc="Complete!")

        return main_img, nav_img, metrics_img, slice_info, report

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error: {str(e)}", ""


def find_best_slice(case_name):
    """Find slice with most ground truth content"""
    if not case_name:
        return 50
    try:
        sequences, ground_truth = load_case_data(case_name)
        slice_sums = np.sum(ground_truth, axis=(0, 1))
        if np.max(slice_sums) > 0:
            best_slice = np.argmax(slice_sums)
            depth = ground_truth.shape[2]
            return int((best_slice / (depth - 1)) * 100)
        return 50
    except Exception:
        return 50


def clear_cache():
    """Clear all caches, then reload precomputed results"""
    global _prediction_cache, _case_cache, _probability_cache
    _prediction_cache = {}
    _case_cache = {}
    _probability_cache = {}
    n = _load_precomputed_cache()
    if n > 0:
        return f"### Cache Cleared\nReloaded {n} precomputed results."
    return "### Cache Cleared\nAll cached data has been removed."


def export_report(case_name, slice_pct):
    """Export the current analysis"""
    if not case_name:
        return None

    try:
        sequences, ground_truth = load_case_data(case_name)
        prediction_volume = _prediction_cache.get(case_name)

        if prediction_volume is None:
            return None

        gt_features = extract_lesion_features(ground_truth)
        pred_features = extract_lesion_features(prediction_volume)

        gt_binary = (ground_truth > 0.5).astype(np.float32)
        pred_binary = (prediction_volume > 0.5).astype(np.float32)
        intersection = np.sum(gt_binary * pred_binary)
        dice_score = (2 * intersection) / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-8)

        export_data = {
            "case_name": case_name,
            "timestamp": datetime.now().isoformat(),
            "dice_score": float(dice_score),
            "ground_truth": {
                "num_lesions": int(gt_features['num_lesions']),
                "total_volume": float(gt_features['total_volume']),
                "mean_volume": float(gt_features['mean_lesion_volume']),
                "max_volume": float(gt_features['max_lesion_volume']),
            },
            "prediction": {
                "num_lesions": int(pred_features['num_lesions']),
                "total_volume": float(pred_features['total_volume']),
                "mean_volume": float(pred_features['mean_lesion_volume']),
                "max_volume": float(pred_features['max_lesion_volume']),
            },
            "model_info": {
                "epoch": _model_info.get('epoch', 'unknown'),
                "validation_dice": float(_model_info.get('val_dice', 0))
            }
        }

        # Save to file
        export_path = Path(__file__).parent / f"{case_name}_analysis.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return str(export_path)

    except Exception:
        return None


# ============================================================================
# RECIST LONGITUDINAL COMPARISON
# ============================================================================

def run_longitudinal_comparison(baseline_case, followup_case, confidence_threshold=0.5, progress=gr.Progress()):
    """
    Run segmentation on two cases and compare using RECIST 1.1.
    Returns (recist_image, recist_report_html).
    """
    if not baseline_case or not followup_case:
        return None, "<p style='color: #878681;'>Please select both baseline and follow-up cases.</p>"

    if baseline_case == followup_case:
        return None, "<p style='color: #878681;'>Please select two different cases for comparison.</p>"

    try:
        from src.segmentation.longitudinal import LongitudinalTracker
        from scipy import ndimage as _ndimage

        progress(0.1, desc="Loading baseline case...")
        baseline_seqs, baseline_gt = load_case_data(baseline_case)

        progress(0.2, desc="Loading follow-up case...")
        followup_seqs, followup_gt = load_case_data(followup_case)

        # Run segmentation on both (or use ground truth for demo)
        progress(0.3, desc="Segmenting baseline...")
        ensemble = load_ensemble_model()
        if ensemble is not None:
            baseline_pred_raw = run_ensemble_inference(ensemble, baseline_seqs)
            baseline_pred = (baseline_pred_raw['fused'] > confidence_threshold).astype(np.uint8)
        else:
            model = load_model()
            baseline_pred = (run_full_inference(model, baseline_seqs) > confidence_threshold).astype(np.uint8)

        progress(0.5, desc="Segmenting follow-up...")
        if ensemble is not None:
            followup_pred_raw = run_ensemble_inference(ensemble, followup_seqs)
            followup_pred = (followup_pred_raw['fused'] > confidence_threshold).astype(np.uint8)
        else:
            followup_pred = (run_full_inference(model, followup_seqs) > confidence_threshold).astype(np.uint8)

        # Build lesion_details for both timepoints
        def _build_lesion_details(binary_mask):
            labeled, n = _ndimage.label(binary_mask > 0)
            details = []
            for i in range(1, n + 1):
                coords = np.argwhere(labeled == i)
                centroid = coords.mean(axis=0).tolist()
                volume_mm3 = float(coords.shape[0])  # Assuming 1mm voxels for demo
                details.append({
                    "id": i,
                    "centroid": centroid,
                    "volume_mm3": volume_mm3,
                })
            return details

        progress(0.65, desc="Comparing timepoints (RECIST 1.1)...")
        baseline_result = {
            "binary_mask": baseline_pred,
            "lesion_details": _build_lesion_details(baseline_pred),
        }
        followup_result = {
            "binary_mask": followup_pred,
            "lesion_details": _build_lesion_details(followup_pred),
        }

        tracker = LongitudinalTracker()
        comparison = tracker.compare_timepoints(baseline_result, followup_result)

        progress(0.8, desc="Creating RECIST visualization...")
        recist_img = create_recist_visualization(
            baseline_seqs, followup_seqs, baseline_pred, followup_pred, comparison,
            baseline_case, followup_case
        )

        progress(0.9, desc="Generating RECIST report...")
        recist_html = generate_recist_report(comparison, baseline_case, followup_case)

        progress(1.0, desc="Complete!")
        return recist_img, recist_html

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"<p style='color: #ef4444;'>Error: {str(e)}</p>"


def create_recist_visualization(baseline_seqs, followup_seqs, baseline_pred, followup_pred,
                                comparison, baseline_name, followup_name):
    """Generate a matplotlib figure for RECIST comparison."""
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')

    # Row 1: Side-by-side axial slices
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)

    # Find best slice for baseline
    b_sums = np.sum(baseline_pred, axis=(0, 1))
    b_slice = np.argmax(b_sums) if np.max(b_sums) > 0 else baseline_pred.shape[2] // 2

    # Find best slice for follow-up
    f_sums = np.sum(followup_pred, axis=(0, 1))
    f_slice = np.argmax(f_sums) if np.max(f_sums) > 0 else followup_pred.shape[2] // 2

    # Baseline MRI + overlay
    b_img = baseline_seqs['t1_gd'][:, :, b_slice]
    b_img_norm = (b_img - b_img.min()) / (b_img.max() - b_img.min() + 1e-8)
    ax1.imshow(b_img_norm, cmap='gray', aspect='equal')
    b_mask = baseline_pred[:, :, b_slice]
    overlay = np.zeros((*b_mask.shape, 4))
    overlay[b_mask > 0] = [1, 0.3, 0.3, 0.5]
    ax1.imshow(overlay, aspect='equal')
    ax1.set_title(f'Baseline: {baseline_name}', color='white', fontsize=10, fontweight='600')
    ax1.axis('off')

    # Follow-up MRI + overlay
    f_img = followup_seqs['t1_gd'][:, :, f_slice]
    f_img_norm = (f_img - f_img.min()) / (f_img.max() - f_img.min() + 1e-8)
    ax2.imshow(f_img_norm, cmap='gray', aspect='equal')
    f_mask = followup_pred[:, :, f_slice]
    overlay2 = np.zeros((*f_mask.shape, 4))
    overlay2[f_mask > 0] = [0.3, 0.5, 1, 0.5]
    ax2.imshow(overlay2, aspect='equal')
    ax2.set_title(f'Follow-up: {followup_name}', color='white', fontsize=10, fontweight='600')
    ax2.axis('off')

    # RECIST response badge
    response = comparison['response_category']
    response_colors = {'CR': '#10b981', 'PR': '#3b82f6', 'SD': '#f59e0b', 'PD': '#ef4444'}
    response_labels = {
        'CR': 'Complete Response', 'PR': 'Partial Response',
        'SD': 'Stable Disease', 'PD': 'Progressive Disease'
    }
    color = response_colors.get(response, '#878681')

    ax3.set_facecolor('#1a1a2e')
    ax3.text(0.5, 0.55, response, fontsize=48, fontweight='bold', color=color,
             ha='center', va='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.3, response_labels.get(response, ''), fontsize=14, color='#a8a8a3',
             ha='center', va='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.15, 'RECIST 1.1', fontsize=10, color='#5c5c58',
             ha='center', va='center', transform=ax3.transAxes)
    ax3.axis('off')

    # Row 2: Volume changes bar chart + summary stats
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#1a1a2e')

    matched = comparison['matched_lesions']
    if matched:
        lesion_ids = [f"L{m['baseline_id']}" for m in matched]
        changes = [m['volume_change_percent'] for m in matched]
        colors = ['#10b981' if c < 0 else '#ef4444' for c in changes]
        bars = ax4.barh(range(len(lesion_ids)), changes, color=colors, height=0.6)
        ax4.set_yticks(range(len(lesion_ids)))
        ax4.set_yticklabels(lesion_ids, color='white', fontsize=9)
        ax4.set_xlabel('Volume Change (%)', color='#a8a8a3', fontsize=9)
        ax4.axvline(x=0, color='#5c5c58', linewidth=0.8)
        ax4.tick_params(colors='#a8a8a3')
        ax4.set_title('Matched Lesion Changes', color='white', fontsize=10, fontweight='600')
        for spine in ax4.spines.values():
            spine.set_color('#5c5c58')
    else:
        ax4.text(0.5, 0.5, 'No matched\nlesions', fontsize=14, color='#878681',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')

    # Summary statistics
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#1a1a2e')
    ax5.axis('off')

    stats_text = [
        f"Matched Lesions: {len(matched)}",
        f"New Lesions: {comparison['new_lesions']}",
        f"Resolved Lesions: {comparison['resolved_lesions']}",
        "",
        f"Baseline SoD: {comparison['sum_of_diameters_baseline_mm']:.1f} mm",
        f"Follow-up SoD: {comparison['sum_of_diameters_followup_mm']:.1f} mm",
    ]

    for i, line in enumerate(stats_text):
        y = 0.85 - i * 0.12
        ax5.text(0.1, y, line, fontsize=11, color='#e8e8e6' if line else '#5c5c58',
                 transform=ax5.transAxes, fontweight='500')

    ax5.set_title('Summary Statistics', color='white', fontsize=10, fontweight='600')

    # SoD comparison
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#1a1a2e')

    sod_baseline = comparison['sum_of_diameters_baseline_mm']
    sod_followup = comparison['sum_of_diameters_followup_mm']

    bars = ax6.bar(['Baseline', 'Follow-up'], [sod_baseline, sod_followup],
                   color=['#0C4DA2', color], width=0.5, edgecolor='none')
    ax6.set_ylabel('Sum of Diameters (mm)', color='#a8a8a3', fontsize=9)
    ax6.set_title('RECIST Sum of Diameters', color='white', fontsize=10, fontweight='600')
    ax6.tick_params(colors='#a8a8a3')
    for spine in ax6.spines.values():
        spine.set_color('#5c5c58')

    plt.tight_layout(pad=2.0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def generate_recist_report(comparison, baseline_name, followup_name):
    """Generate an HTML report for RECIST comparison."""
    response = comparison['response_category']
    response_colors = {'CR': '#10b981', 'PR': '#3b82f6', 'SD': '#f59e0b', 'PD': '#ef4444'}
    response_labels = {
        'CR': 'Complete Response', 'PR': 'Partial Response',
        'SD': 'Stable Disease', 'PD': 'Progressive Disease'
    }
    response_descriptions = {
        'CR': 'All target lesions have disappeared. This represents the best possible treatment response.',
        'PR': 'At least 30% decrease in the sum of diameters of target lesions, indicating meaningful tumor shrinkage.',
        'SD': 'Neither sufficient shrinkage for PR nor sufficient increase for PD. Disease is controlled but not regressing significantly.',
        'PD': 'At least 20% increase in the sum of diameters of target lesions, or appearance of new lesions, indicating disease progression.'
    }

    color = response_colors.get(response, '#878681')
    label = response_labels.get(response, response)
    description = response_descriptions.get(response, '')

    sod_baseline = comparison['sum_of_diameters_baseline_mm']
    sod_followup = comparison['sum_of_diameters_followup_mm']
    sod_change = ((sod_followup - sod_baseline) / (sod_baseline + 1e-8)) * 100

    html = f"""
<div style="font-family: 'Inter', -apple-system, sans-serif; line-height: 1.7; -webkit-font-smoothing: antialiased;">

<div style="background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%); padding: 28px; border-radius: 16px; margin-bottom: 24px; border-bottom: 3px solid {color}; text-align: center;">
    <p style="font-size: 0.75em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #a8a8a3; margin: 0 0 12px 0;">RECIST 1.1 Treatment Response</p>
    <div style="display: inline-flex; align-items: center; justify-content: center; padding: 12px 32px; border-radius: 16px; background: {color}; margin-bottom: 12px;">
        <span style="color: white; font-size: 2em; font-weight: 700; letter-spacing: 0.05em;">{response}</span>
    </div>
    <p style="color: #e8e8e6; font-size: 1.1em; font-weight: 500; margin: 8px 0 0 0;">{label}</p>
    <p style="color: #878681; font-size: 0.9em; margin: 4px 0 0 0;">{baseline_name} &rarr; {followup_name}</p>
</div>

<div style="background: #ffffff; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #a8a8a3;">
    <h3 style="color: #0a0a0a; margin: 0 0 12px 0; font-size: 1.1em; font-weight: 600;">Response Interpretation</h3>
    <p style="color: #1d1d1f; margin: 0;">{description}</p>
</div>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
    <div style="background: linear-gradient(180deg, #ffffff 0%, #e8e8e6 100%); padding: 20px; border-radius: 16px; text-align: center; border: 1px solid #a8a8a3;">
        <div style="font-size: 1.8em; font-weight: 600; color: #0a0a0a;">{len(comparison['matched_lesions'])}</div>
        <div style="font-size: 0.85em; color: #878681; margin-top: 4px;">Matched Lesions</div>
    </div>
    <div style="background: linear-gradient(180deg, #ffffff 0%, #e8e8e6 100%); padding: 20px; border-radius: 16px; text-align: center; border: 1px solid #a8a8a3;">
        <div style="font-size: 1.8em; font-weight: 600; color: {'#ef4444' if comparison['new_lesions'] > 0 else '#0a0a0a'};">{comparison['new_lesions']}</div>
        <div style="font-size: 0.85em; color: #878681; margin-top: 4px;">New Lesions</div>
    </div>
    <div style="background: linear-gradient(180deg, #ffffff 0%, #e8e8e6 100%); padding: 20px; border-radius: 16px; text-align: center; border: 1px solid #a8a8a3;">
        <div style="font-size: 1.8em; font-weight: 600; color: {'#10b981' if comparison['resolved_lesions'] > 0 else '#0a0a0a'};">{comparison['resolved_lesions']}</div>
        <div style="font-size: 0.85em; color: #878681; margin-top: 4px;">Resolved Lesions</div>
    </div>
</div>

<div style="background: #ffffff; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #a8a8a3;">
    <h3 style="color: #0a0a0a; margin: 0 0 16px 0; font-size: 1.1em; font-weight: 600;">Sum of Diameters (RECIST 1.1)</h3>
    <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center;">
        <div style="text-align: center;">
            <div style="font-size: 0.8em; color: #878681; text-transform: uppercase; letter-spacing: 0.05em;">Baseline</div>
            <div style="font-size: 1.8em; font-weight: 600; color: #0C4DA2;">{sod_baseline:.1f} mm</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.4em; color: {color}; font-weight: 700;">{sod_change:+.1f}%</div>
            <div style="font-size: 1.2em; color: #878681;">&rarr;</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.8em; color: #878681; text-transform: uppercase; letter-spacing: 0.05em;">Follow-up</div>
            <div style="font-size: 1.8em; font-weight: 600; color: {color};">{sod_followup:.1f} mm</div>
        </div>
    </div>
</div>
"""

    # Matched lesion table
    if comparison['matched_lesions']:
        html += """
<div style="background: #ffffff; border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid #a8a8a3;">
    <h3 style="color: #0a0a0a; margin: 0 0 16px 0; font-size: 1.1em; font-weight: 600;">Matched Lesion Details</h3>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.95em;">
        <tr>
            <th style="padding: 12px; text-align: left; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Baseline ID</th>
            <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Follow-up ID</th>
            <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Baseline Vol</th>
            <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Follow-up Vol</th>
            <th style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.1); color: #86868b; font-weight: 500; font-size: 0.85em;">Change</th>
        </tr>
"""
        for m in comparison['matched_lesions']:
            change_pct = m['volume_change_percent']
            change_color = '#10b981' if change_pct < 0 else '#ef4444' if change_pct > 0 else '#878681'
            arrow = '&#9660;' if change_pct < 0 else '&#9650;' if change_pct > 0 else '&#9654;'
            html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">Lesion {m['baseline_id']}</td>
            <td style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">Lesion {m['followup_id']}</td>
            <td style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{m['baseline_volume_mm3']:,.0f} mm&sup3;</td>
            <td style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: #1d1d1f;">{m['followup_volume_mm3']:,.0f} mm&sup3;</td>
            <td style="padding: 12px; text-align: center; border-bottom: 1px solid rgba(0,0,0,0.05); color: {change_color}; font-weight: 600;">{arrow} {change_pct:+.1f}%</td>
        </tr>
"""
        html += "    </table>\n</div>"

    # Footer disclaimer
    html += """
<div style="background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%); border-radius: 12px; padding: 20px; text-align: center; border-top: 3px solid #0C4DA2;">
    <p style="color: #a8a8a3; font-size: 0.85em; margin: 0 0 4px 0;">
        RECIST 1.1 assessment is AI-generated and requires expert oncologist review.
    </p>
    <p style="color: #878681; font-size: 0.8em; margin: 0;">
        Longitudinal comparison uses centroid-based lesion matching with Hungarian algorithm.
    </p>
</div>

</div>
"""
    return html


# ============================================================================
# CORPUS STATS
# ============================================================================

def get_corpus_stats_html():
    """Get HTML display of corpus statistics for the sidebar."""
    v2_db_path = Path(__file__).parent.parent / "outputs" / "rag" / "chromadb_v2"
    stats_path = Path(__file__).parent.parent / "outputs" / "rag" / "build_stats.json"

    if not v2_db_path.exists():
        return """
        <div class="corpus-stats-card">
            <strong>Knowledge Base</strong><br>
            <span style="color: #878681;">Corpus not built yet.</span><br>
            <span style="font-size: 0.8em; color: #5c5c58;">Run: python scripts/build_corpus.py</span>
        </div>
        """

    total_papers = "?"
    total_chunks = "?"
    last_updated = "Unknown"

    try:
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            total_papers = f"{stats.get('total_papers', '?'):,}"
            total_chunks = f"{stats.get('total_chunks', '?'):,}"
    except Exception:
        pass

    try:
        import os
        mtime = os.path.getmtime(str(stats_path))
        last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        pass

    return f"""
    <div class="corpus-stats-card">
        <strong>Knowledge Base</strong>
        <hr style="border: none; border-top: 1px solid #5c5c58; margin: 8px 0;">
        <strong>{total_papers}</strong> papers &nbsp;|&nbsp; <strong>{total_chunks}</strong> chunks<br>
        Last updated: {last_updated}<br>
        BiomedCLIP embeddings (512-dim)<br>
        Hybrid retrieval: Dense + BM25
    </div>
    """


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_demo():
    """Create the professional Gradio demo interface"""

    available_cases = get_available_cases()

    with gr.Blocks(title="Brain Metastasis Segmentation") as demo:

        # Header
        gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">Brain Metastasis Segmentation</h1>
            <p class="header-subtitle">AI-Powered Medical Image Analysis</p>
            <div style="display: flex; gap: 10px; justify-content: center; margin-top: 20px; flex-wrap: wrap;">
                <div class="header-badge">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                    </svg>
                    Stacking Classifier (~78% Dice)
                </div>
                <div class="header-badge" style="background: #7c3aed; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                    </svg>
                    RAG-Enhanced Reports
                </div>
            </div>
        </div>
        """)

        with gr.Row():
            # Left Sidebar
            with gr.Column(scale=1, min_width=280):

                gr.Markdown("### Patient Selection")
                case_dropdown = gr.Dropdown(
                    choices=available_cases,
                    value=None,
                    label="Select Case",
                    info=f"{len(available_cases)} test cases available"
                )

                gr.Markdown("### Navigation")
                slice_axis = gr.Radio(
                    choices=["Axial", "Sagittal", "Coronal"],
                    value="Axial",
                    label="Slice Axis",
                    info="Select which anatomical plane to view"
                )
                slice_slider = gr.Slider(
                    minimum=0, maximum=100, value=50, step=1,
                    label="Slice Position",
                    info="Drag to navigate through slices"
                )

                gr.Markdown("### View Options")
                view_mode = gr.Radio(
                    choices=["Full Analysis", "MRI Only"],
                    value="Full Analysis",
                    label="Display Mode"
                )

                gr.Markdown("### Confidence Threshold")
                confidence_slider = gr.Slider(
                    minimum=0.3, maximum=0.99, value=0.9, step=0.05,
                    label="Detection Threshold",
                    info="Stacking optimal: 0.9. Lower = more sensitive, Higher = more specific"
                )

                with gr.Row():
                    run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
                    find_btn = gr.Button("Find Lesion", variant="secondary")

                # Quick Stats - initially hidden
                quick_stats_section = gr.Group(visible=False)
                with quick_stats_section:
                    gr.Markdown("### Quick Stats")
                    slice_info = gr.Markdown("*Select a case and run analysis*")

                # Corpus stats - always visible
                gr.HTML(value=get_corpus_stats_html())

                with gr.Accordion("Advanced Options", open=False):
                    export_btn = gr.Button("Export Results", size="sm")
                    clear_btn = gr.Button("Clear Cache", variant="stop", size="sm")

            # Main Content
            with gr.Column(scale=3):

                # Welcome placeholder - shown initially
                welcome_placeholder = gr.HTML(visible=True, value="""
                <div style="
                    background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
                    border: 2px dashed #878681;
                    border-radius: 24px;
                    padding: 80px 40px;
                    text-align: center;
                    min-height: 500px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                ">
                    <div style="
                        width: 100px;
                        height: 100px;
                        background: linear-gradient(180deg, #0C4DA2 0%, #083A7A 100%);
                        border-radius: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-bottom: 32px;
                        box-shadow: 0 8px 32px rgba(12, 77, 162, 0.4);
                    ">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.5">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                    </div>
                    <h2 style="color: #ffffff; font-size: 1.8em; font-weight: 600; margin: 0 0 12px 0; letter-spacing: -0.02em;">
                        Ready for Analysis
                    </h2>
                    <p style="color: #878681; font-size: 1.1em; margin: 0 0 32px 0; max-width: 400px; line-height: 1.6;">
                        Select a test case from the dropdown menu, or upload your own MRI scan for AI-powered brain metastasis segmentation.
                    </p>
                    <div style="display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;">
                        <div style="
                            background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
                            border: 1px solid #5c5c58;
                            border-radius: 12px;
                            padding: 20px 24px;
                            text-align: left;
                            min-width: 200px;
                        ">
                            <div style="color: #0C4DA2; font-weight: 600; margin-bottom: 8px; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em;">Option 1</div>
                            <div style="color: #ffffff; font-weight: 500;">Select Test Case</div>
                            <div style="color: #878681; font-size: 0.9em; margin-top: 4px;">Choose from 105 pre-loaded cases</div>
                        </div>
                        <div style="
                            background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
                            border: 1px solid #5c5c58;
                            border-radius: 12px;
                            padding: 20px 24px;
                            text-align: left;
                            min-width: 200px;
                            opacity: 0.5;
                        ">
                            <div style="color: #878681; font-weight: 600; margin-bottom: 8px; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em;">Option 2</div>
                            <div style="color: #a8a8a3; font-weight: 500;">Upload MRI Scan</div>
                            <div style="color: #5c5c58; font-size: 0.9em; margin-top: 4px;">Coming soon</div>
                        </div>
                    </div>
                </div>
                """)

                # Analysis content - hidden initially
                analysis_content = gr.Group(visible=False)
                with analysis_content:
                    with gr.Tabs():
                        with gr.Tab("Multi-View Visualization"):
                            main_image = gr.Image(
                                label="Axial | Sagittal | Coronal Views",
                                type="pil"
                            )

                            gr.HTML("""
                            <div style="display: flex; gap: 12px; justify-content: center; margin-top: 20px; flex-wrap: wrap;">
                                <div class="legend-item"><div class="legend-color" style="background: #fbbf24;"></div> True Positive</div>
                                <div class="legend-item"><div class="legend-color" style="background: #ef4444;"></div> False Positive</div>
                                <div class="legend-item"><div class="legend-color" style="background: #10b981;"></div> False Negative</div>
                                <div class="legend-item"><div class="legend-color" style="background: cyan;"></div> Threshold Contour</div>
                            </div>
                            """)

                        with gr.Tab("Slice Navigator"):
                            nav_image = gr.Image(
                                label="Lesion Distribution Across Slices",
                                type="pil"
                            )

                        with gr.Tab("Metrics Dashboard"):
                            metrics_image = gr.Image(
                                label="Performance Metrics",
                                type="pil"
                            )

                        with gr.Tab("Clinical Report"):
                            report_output = gr.HTML(label="AI-Generated Clinical Report")

        # Footer
        gr.HTML("""
        <div class="footer">
            <p style="font-weight: 600; color: #ffffff; margin-bottom: 8px; font-size: 1.1em;">BrainMetScan</p>
            <p style="color: #a8a8a3;">6-Model Stacking Classifier (~78% Dice)  ·  RAG-Enhanced Reports  ·  BiomedCLIP + BM25</p>
            <div style="display: flex; justify-content: center; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
                <span style="background: #2d2d2d; color: #a8a8a3; padding: 6px 12px; border-radius: 6px; font-size: 0.8em; border: 1px solid #5c5c58;">Slider: Navigate slices</span>
                <span style="background: #2d2d2d; color: #a8a8a3; padding: 6px 12px; border-radius: 6px; font-size: 0.8em; border: 1px solid #5c5c58;">Threshold: Adjust sensitivity</span>
            </div>
        </div>
        """)

        # Helper function to process and update visibility
        def process_and_show(case_name, slice_pct, view_mode, confidence_threshold, slice_axis_val):
            if not case_name:
                return None, None, None, "*Select a case and run analysis*", "", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

            result = process_case(case_name, slice_pct, view_mode, confidence_threshold, slice_axis=slice_axis_val)
            # result = (main_img, nav_img, metrics_img, slice_info, report)
            return result[0], result[1], result[2], result[3], result[4], gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

        # Event Handlers
        run_btn.click(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        find_btn.click(
            fn=find_best_slice,
            inputs=[case_dropdown],
            outputs=[slice_slider]
        ).then(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        case_dropdown.change(
            fn=find_best_slice,
            inputs=[case_dropdown],
            outputs=[slice_slider]
        ).then(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        slice_slider.release(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        view_mode.change(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        confidence_slider.release(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        slice_axis.change(
            fn=process_and_show,
            inputs=[case_dropdown, slice_slider, view_mode, confidence_slider, slice_axis],
            outputs=[main_image, nav_image, metrics_image, slice_info, report_output, welcome_placeholder, analysis_content, quick_stats_section]
        )

        export_btn.click(
            fn=export_report,
            inputs=[case_dropdown, slice_slider],
            outputs=[slice_info]
        )

        clear_btn.click(fn=clear_cache, outputs=[slice_info])

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Brain Metastasis Segmentation Demo - Ensemble Edition")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Data: {DATA_DIR}")
    print("=" * 60)

    # Try to load ensemble first
    ensemble = load_ensemble_model()
    if ensemble is None:
        print("Falling back to single model...")
        load_model()

    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
