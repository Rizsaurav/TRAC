"""
Single source of truth for all filesystem paths in this project.
Every module imports from here instead of hard-coding its own paths.
"""

import os

ROOT = os.path.dirname(__file__)

# ── model artefacts ────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(ROOT, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "gaze_autoencoder.pt")
SCALER_PATH  = os.path.join(MODELS_DIR, "gaze_scaler.pkl")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "dqn_agent")

# ── per-module output directories ──────────────────────────────────────────
OUTPUTS_DIR       = os.path.join(ROOT, "outputs")
SIMULATOR_OUT     = os.path.join(OUTPUTS_DIR, "simulator")
AUTOENCODER_OUT   = os.path.join(OUTPUTS_DIR, "autoencoder")
TRAINING_OUT      = os.path.join(OUTPUTS_DIR, "training")
EVALUATION_OUT    = os.path.join(OUTPUTS_DIR, "evaluation")
VISUALIZATION_OUT = os.path.join(OUTPUTS_DIR, "visualization")


def ensure(*dirs: str) -> None:
    """Create one or more directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
