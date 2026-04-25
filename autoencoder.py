"""
Gaze autoencoder for the Adaptive Proctoring Agent.

Architecture (Fuhl et al. 2021):
  Encoder:  6 → 32 → 16 → 8
  Decoder:  8 → 16 → 32 → 6

Trained exclusively on honest gaze windows. At inference, reconstruction
error (MSE) is the anomaly score — cheating windows reconstruct poorly
because the encoder has never seen that distribution.

Saved artefacts (written to ./models/):
  gaze_autoencoder.pt   – trained model weights
  gaze_scaler.pkl       – fitted StandardScaler (must travel with the model)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

from gaze_simulator import GazeSimulator, FEATURE_NAMES
from paths import MODEL_PATH, SCALER_PATH, MODELS_DIR, AUTOENCODER_OUT, ensure

N_FEATURES = len(FEATURE_NAMES)   # 6
LATENT_DIM = 8


# 
# Model definition


class GazeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_FEATURES, 32), nn.ReLU(),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, N_FEATURES),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


# Training

def train(
    n_honest: int = 10_000,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GazeAutoencoder, StandardScaler]:
    """Train the autoencoder on honest-only windows and save artefacts."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    sim = GazeSimulator(seed=seed)
    df_all = sim.generate_dataset(n_honest=n_honest, n_cheating=0)
    X_honest = df_all[FEATURE_NAMES].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_honest).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X_scaled))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GazeAutoencoder()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(X_scaled)
        history.append(epoch_loss)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}")

    ensure(MODELS_DIR)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if verbose:
        print(f"\nSaved model  → {MODEL_PATH}")
        print(f"Saved scaler → {SCALER_PATH}")

    return model, scaler, history


# Inference helpers

def load_model() -> tuple[GazeAutoencoder, StandardScaler]:
    """Load saved model and scaler from disk."""
    model = GazeAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def get_anomaly_score(
    window: np.ndarray,
    model: GazeAutoencoder,
    scaler: StandardScaler,
) -> float:
    """
    Return the MSE reconstruction error for a single 6-feature window.

    Parameters
    ----------
    window : 1-D numpy array of shape (6,)  — raw (unscaled) feature values
    """
    model.eval()
    x = scaler.transform(window.reshape(1, -1)).astype(np.float32)
    t = torch.from_numpy(x)
    with torch.no_grad():
        recon, _ = model(t)
    return float(nn.functional.mse_loss(recon, t).item())


def get_latent(
    window: np.ndarray,
    model: GazeAutoencoder,
    scaler: StandardScaler,
) -> np.ndarray:
    """Return the 8-dimensional latent code for a single window."""
    model.eval()
    x = scaler.transform(window.reshape(1, -1)).astype(np.float32)
    with torch.no_grad():
        z = model.encode(torch.from_numpy(x))
    return z.numpy().flatten()


# Validation

def validate(
    model: GazeAutoencoder,
    scaler: StandardScaler,
    n_honest: int = 2_000,
    n_cheat: int = 2_000,
    seed: int = 99,
) -> dict:
    """
    Score held-out honest and cheating windows.
    Returns a dict with scores and separation metrics, and saves a plot.
    """
    sim = GazeSimulator(seed=seed)
    df = sim.generate_dataset(n_honest=n_honest, n_cheating=n_cheat, seed_offset=seed)

    honest_scores = np.array([
        get_anomaly_score(row, model, scaler)
        for row in df.loc[df.label == 0, FEATURE_NAMES].values
    ])
    cheat_scores = np.array([
        get_anomaly_score(row, model, scaler)
        for row in df.loc[df.label == 1, FEATURE_NAMES].values
    ])

    # Threshold sweep for best F1
    all_scores = np.concatenate([honest_scores, cheat_scores])
    all_labels = np.array([0] * len(honest_scores) + [1] * len(cheat_scores))
    thresholds = np.percentile(all_scores, np.linspace(1, 99, 200))
    best_f1, best_thresh = 0.0, 0.0
    for t in thresholds:
        preds = (all_scores >= t).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = (all_scores >= best_thresh).astype(int)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    tn = ((preds == 0) & (all_labels == 0)).sum()

    result = {
        "honest_mean":  honest_scores.mean(),
        "honest_std":   honest_scores.std(),
        "cheat_mean":   cheat_scores.mean(),
        "cheat_std":    cheat_scores.std(),
        "best_f1":      best_f1,
        "best_thresh":  best_thresh,
        "tpr":          tp / (tp + fn + 1e-9),
        "fpr":          fp / (fp + tn + 1e-9),
    }

    _plot_separation(honest_scores, cheat_scores, best_thresh, result)
    return result


def _plot_separation(honest, cheat, threshold, metrics):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(honest, bins=60, alpha=0.65, color="steelblue",
            label=f"Honest  μ={metrics['honest_mean']:.4f}", density=True)
    ax.hist(cheat,  bins=60, alpha=0.65, color="tomato",
            label=f"Cheating μ={metrics['cheat_mean']:.4f}", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"Best thresh={threshold:.4f}  F1={metrics['best_f1']:.3f}")
    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder Anomaly Score — Honest vs. Cheating")
    ax.legend()
    ensure(AUTOENCODER_OUT)
    out = os.path.join(AUTOENCODER_OUT, "anomaly_separation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# Entry point

if __name__ == "__main__":
    print("=== Training autoencoder ===")
    model, scaler, history = train(n_honest=10_000, epochs=60, verbose=True)

    print("\n=== Validation ===")
    metrics = validate(model, scaler)
    print(f"  Honest   error: {metrics['honest_mean']:.4f} ± {metrics['honest_std']:.4f}")
    print(f"  Cheating error: {metrics['cheat_mean']:.4f} ± {metrics['cheat_std']:.4f}")
    print(f"  Best F1:  {metrics['best_f1']:.3f}  at threshold {metrics['best_thresh']:.4f}")
    print(f"  TPR: {metrics['tpr']:.3f}   FPR: {metrics['fpr']:.3f}")

    # Training loss curve
    ensure(AUTOENCODER_OUT)
    loss_path = os.path.join(AUTOENCODER_OUT, "autoencoder_loss.png")
    plt.figure(figsize=(7, 3))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder training loss")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=120)
    print(f"Saved {loss_path}")
