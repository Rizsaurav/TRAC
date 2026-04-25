"""
Synthetic gaze signal generator for the Adaptive Proctoring Agent.

Feature set per 500ms epoch follows Lim et al. (2021):
  0  fixation_duration_ms       – mean fixation length in the window
  1  saccade_amplitude_deg      – mean saccade amplitude in degrees
  2  off_screen_ratio           – fraction of samples outside screen bounds
  3  lateral_deviation_px       – mean absolute horizontal offset from centre
  4  blink_rate_per_min         – blink events scaled to per-minute rate
  5  gaze_entropy               – Shannon entropy over 9 screen regions

Honest vs. cheating distributions are grounded in Table 1 / Section 4 of
Lim et al. (2021). Individual variation is modelled by a per-subject offset
sampled once per generate_sequence() call.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Distribution parameters
# ---------------------------------------------------------------------------

@dataclass
class _GazeParams:
    fixation_duration_ms: tuple   # (mean, std)
    saccade_amplitude_deg: tuple
    off_screen_ratio: tuple       # Beta(a, b) → rescaled to [0,1]
    lateral_deviation_px: tuple
    blink_rate_per_min: tuple
    gaze_entropy: tuple


HONEST_PARAMS = _GazeParams(
    fixation_duration_ms    = (280.0,  45.0),
    saccade_amplitude_deg   = (3.5,    1.2),
    off_screen_ratio        = (0.04,   0.03),   # very rarely off-screen
    lateral_deviation_px    = (55.0,   20.0),
    blink_rate_per_min      = (17.0,   4.0),
    gaze_entropy            = (1.8,    0.25),
)

CHEATING_PARAMS = _GazeParams(
    fixation_duration_ms    = (160.0,  50.0),   # shorter, more erratic
    saccade_amplitude_deg   = (9.0,    3.0),    # larger sweeping saccades
    off_screen_ratio        = (0.22,   0.08),   # frequently off-screen
    lateral_deviation_px    = (180.0,  55.0),   # looking far to sides
    blink_rate_per_min      = (24.0,   6.0),    # increased blink under stress
    gaze_entropy            = (2.4,    0.30),   # gaze spread over more regions
)

FEATURE_NAMES = [
    "fixation_duration_ms",
    "saccade_amplitude_deg",
    "off_screen_ratio",
    "lateral_deviation_px",
    "blink_rate_per_min",
    "gaze_entropy",
]

# Hard clamp ranges (physiologically plausible)
CLAMP_RANGES = {
    "fixation_duration_ms":   (50.0,  800.0),
    "saccade_amplitude_deg":  (0.5,   30.0),
    "off_screen_ratio":       (0.0,   1.0),
    "lateral_deviation_px":   (0.0,   500.0),
    "blink_rate_per_min":     (2.0,   50.0),
    "gaze_entropy":           (0.0,   3.5),
}


# Core simulator

class GazeSimulator:
    """
    Generates labelled windows of synthetic gaze features.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    subject_variance_scale : float
        Controls how much per-subject offset is added (0 = none).
    """

    def __init__(self, seed: Optional[int] = 42, subject_variance_scale: float = 0.15):
        self.rng = np.random.default_rng(seed)
        self.subject_variance_scale = subject_variance_scale

    # Public API

    def generate_window(self, label: int) -> np.ndarray:
        """
        Return a single 6-feature window (numpy 1-D array).

        label : 0 = honest, 1 = cheating
        """
        params = HONEST_PARAMS if label == 0 else CHEATING_PARAMS
        return self._clamp(self._sample_window(params))

    def generate_sequence(
        self,
        n_steps: int = 120,
        cheat_start: Optional[int] = None,
        cheat_duration: int = 5,
    ) -> pd.DataFrame:
        """
        Generate a full exam sequence with an optional injected cheating block.

        Parameters
        ----------
        n_steps      : total windows (120 = 60-min exam at 30-sec granularity)
        cheat_start  : step index where cheating begins (None = no cheating)
        cheat_duration : how many consecutive steps the cheating lasts

        Returns
        -------
        pd.DataFrame  columns = FEATURE_NAMES + ['label']
        """
        # Per-subject offset: scaled to each feature's natural std so the
        # shift is proportionally small regardless of feature magnitude.
        honest_stds = np.array([
            HONEST_PARAMS.fixation_duration_ms[1],
            HONEST_PARAMS.saccade_amplitude_deg[1],
            HONEST_PARAMS.off_screen_ratio[1],
            HONEST_PARAMS.lateral_deviation_px[1],
            HONEST_PARAMS.blink_rate_per_min[1],
            HONEST_PARAMS.gaze_entropy[1],
        ])
        subject_offset = self.rng.normal(
            0, self.subject_variance_scale * honest_stds
        )

        labels = np.zeros(n_steps, dtype=int)
        if cheat_start is not None:
            end = min(cheat_start + cheat_duration, n_steps)
            labels[cheat_start:end] = 1

        rows = []
        for t in range(n_steps):
            params = HONEST_PARAMS if labels[t] == 0 else CHEATING_PARAMS
            window = self._sample_window(params) + subject_offset
            window = self._clamp(window)
            rows.append(window)

        df = pd.DataFrame(rows, columns=FEATURE_NAMES)
        df["label"] = labels
        return df

    def generate_dataset(
        self,
        n_honest: int = 10_000,
        n_cheating: int = 2_000,
        seed_offset: int = 0,
    ) -> pd.DataFrame:
        """
        Generate a flat labelled dataset of individual windows.
        Used to train/evaluate the autoencoder.
        """
        honest_rows = [
            self._sample_window(HONEST_PARAMS)
            for _ in range(n_honest)
        ]
        cheat_rows = [
            self._sample_window(CHEATING_PARAMS)
            for _ in range(n_cheating)
        ]

        df_honest = pd.DataFrame(honest_rows, columns=FEATURE_NAMES)
        df_honest["label"] = 0

        if cheat_rows:
            df_cheat = pd.DataFrame(cheat_rows, columns=FEATURE_NAMES)
            df_cheat["label"] = 1
            df = pd.concat([df_honest, df_cheat], ignore_index=True)
        else:
            df = df_honest.copy()
        df[FEATURE_NAMES] = df[FEATURE_NAMES].apply(
            lambda col: col.clip(*CLAMP_RANGES[col.name])
        )
        return df.sample(frac=1, random_state=seed_offset).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_window(self, params: _GazeParams) -> np.ndarray:
        rng = self.rng

        fixation = rng.normal(params.fixation_duration_ms[0],  params.fixation_duration_ms[1])
        saccade  = rng.normal(params.saccade_amplitude_deg[0], params.saccade_amplitude_deg[1])
        lateral  = rng.normal(params.lateral_deviation_px[0],  params.lateral_deviation_px[1])
        blink    = rng.normal(params.blink_rate_per_min[0],    params.blink_rate_per_min[1])
        entropy  = rng.normal(params.gaze_entropy[0],          params.gaze_entropy[1])

        # off_screen_ratio is bounded [0,1] and right-skewed — Beta fits better than Normal
        mu, sd = params.off_screen_ratio
        mu = np.clip(mu, 1e-4, 1 - 1e-4)
        variance = min(sd ** 2, mu * (1 - mu) - 1e-4)  # Beta validity constraint
        scale = mu * (1 - mu) / variance - 1
        a, b = mu * scale, (1 - mu) * scale
        off_screen = rng.beta(a, b)

        return np.array([fixation, saccade, off_screen, lateral, blink, entropy])

    def _clamp(self, window: np.ndarray) -> np.ndarray:
        out = window.copy()
        for i, name in enumerate(FEATURE_NAMES):
            lo, hi = CLAMP_RANGES[name]
            out[i] = np.clip(out[i], lo, hi)
        return out


# Quick smoke-test / demo

def _demo():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paths import SIMULATOR_OUT, ensure

    ensure(SIMULATOR_OUT)
    sim = GazeSimulator(seed=0)

    # --- flat dataset separation check ---
    df = sim.generate_dataset(n_honest=3000, n_cheating=3000)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Honest vs. Cheating Gaze Feature Distributions", fontsize=13)
    for ax, feat in zip(axes.flat, FEATURE_NAMES):
        for lbl, color, name in [(0, "steelblue", "Honest"), (1, "tomato", "Cheating")]:
            vals = df.loc[df.label == lbl, feat]
            ax.hist(vals, bins=40, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(SIMULATOR_OUT, "gaze_distributions.png")
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")

    # --- single exam sequence ---
    seq = sim.generate_sequence(n_steps=120, cheat_start=55, cheat_duration=6)
    print("\nSequence shape:", seq.shape)
    print("Cheating steps:", seq.index[seq.label == 1].tolist())
    print("\nHonest means:\n", seq[seq.label == 0][FEATURE_NAMES].mean().round(2))
    print("\nCheating means:\n", seq[seq.label == 1][FEATURE_NAMES].mean().round(2))


if __name__ == "__main__":
    _demo()
