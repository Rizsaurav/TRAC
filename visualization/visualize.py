"""
Phase 6 — Episode visualization for the Adaptive Proctoring Agent.
Produces a 4-panel figure showing one full exam episode:
  Panel 1: raw gaze features over time (normalised)
  Panel 2: anomaly score + rolling mean
  Panel 3: agent action at each step (color-coded)
  Panel 4: ground truth cheating indicator

Output: outputs/visualization/episode_visualization.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import DQN

from environment import ExamProctoringEnv
from environment.exam_env import EPISODE_LEN
from autoencoder import load_model, get_anomaly_score
from gaze_simulator import FEATURE_NAMES
from paths import VISUALIZATION_OUT, ensure

APA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "apa_dqn.zip")

ACTION_COLORS = {0: "#378ADD", 1: "#EF9F27", 2: "#E24B4A"}
ACTION_LABELS = {0: "Passive",  1: "Active",  2: "Alert"}


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(seed: int = 7) -> dict:
    """Roll out one deterministic episode with the trained APA and collect all data."""
    env      = ExamProctoringEnv(cheat_prob=1.0, sim_seed=seed)
    model    = DQN.load(APA_MODEL_PATH)
    ae_model, scaler = load_model()

    obs, _ = env.reset(seed=seed)

    steps, features, scores, rolling_means, actions, labels = [], [], [], [], [], []
    done = False
    t    = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # capture raw window before stepping so index stays in sync
        window = env._sequence[FEATURE_NAMES].iloc[env._step].values.astype("float32")
        score  = get_anomaly_score(window, ae_model, scaler)
        label  = int(env._labels[env._step])

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        steps.append(t)
        features.append(window)
        scores.append(score)
        rolling_means.append(float(obs[9]))   # obs[9] = 5-step rolling mean
        actions.append(action)
        labels.append(label)
        t += 1

    return {
        "steps":         np.array(steps),
        "features":      np.array(features),          # (T, 6)
        "scores":        np.array(scores),             # (T,)
        "rolling_means": np.array(rolling_means),      # (T,)
        "actions":       np.array(actions),            # (T,)
        "labels":        np.array(labels),             # (T,)
    }


# ---------------------------------------------------------------------------
# 4-panel figure
# ---------------------------------------------------------------------------

def plot_episode(data: dict, save_path: str) -> None:
    steps   = data["steps"]
    labels  = data["labels"]
    actions = data["actions"]
    scores  = data["scores"]
    rolling = data["rolling_means"]
    feats   = data["features"]

    cheat_mask = labels == 1

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Adaptive Proctoring Agent — Single Exam Episode", fontsize=13, y=0.995)

    # shared cheating span helper
    def _shade(ax):
        if cheat_mask.any():
            ax.axvspan(
                steps[cheat_mask][0] - 0.5,
                steps[cheat_mask][-1] + 0.5,
                alpha=0.13, color="red", zorder=0,
            )

    # ── Panel 1: gaze features ────────────────────────────────────────────
    ax = axes[0]
    feat_range = feats.max(axis=0) - feats.min(axis=0)
    feat_norm  = (feats - feats.min(axis=0)) / (feat_range + 1e-9)
    colors     = plt.cm.tab10(np.linspace(0, 0.6, len(FEATURE_NAMES)))
    for i, (fname, col) in enumerate(zip(FEATURE_NAMES, colors)):
        ax.plot(steps, feat_norm[:, i], alpha=0.75, linewidth=1.1,
                label=fname.replace("_", " "), color=col)
    _shade(ax)
    ax.set_ylabel("Normalised value")
    ax.set_title("Gaze features (min-max normalised per feature)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    # ── Panel 2: anomaly score ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(steps, scores,  color="tomato",  linewidth=1.2, alpha=0.8,
            label="Anomaly score")
    ax.plot(steps, rolling, color="darkred", linewidth=1.8,
            label="Rolling mean (5 steps)")
    _shade(ax)
    ax.set_ylabel("MSE reconstruction error")
    ax.set_title("Autoencoder anomaly score")
    ax.legend(fontsize=8)

    # ── Panel 3: agent actions ────────────────────────────────────────────
    ax = axes[2]
    for t, act in zip(steps, actions):
        ax.bar(t, 1, color=ACTION_COLORS[act], width=1.0, align="edge", linewidth=0)
    _shade(ax)
    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a])
               for a in [0, 1, 2]]
    ax.set_yticks([])
    ax.set_ylabel("Action")
    ax.set_title("Agent actions")
    ax.legend(handles=patches, fontsize=8, loc="upper left")

    # ── Panel 4: ground truth ─────────────────────────────────────────────
    ax = axes[3]
    ax.fill_between(steps, labels, alpha=0.55, color="red",
                    label="Cheating (ground truth)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Honest", "Cheating"])
    ax.set_ylabel("Label")
    ax.set_xlabel("Exam step  (1 step = 30 sec)")
    ax.set_title("Ground truth")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure(VISUALIZATION_OUT)
    out = os.path.join(VISUALIZATION_OUT, "episode_visualization.png")

    print("Rolling out episode …")
    data = run_episode(seed=7)

    cheat_steps = int(data["labels"].sum())
    alert_steps = int((data["actions"] == 2).sum())
    correct     = (data["actions"] == 2) & (data["labels"] == 1)
    missed      = (data["actions"] != 2) & (data["labels"] == 1)

    print(f"  Episode length   : {len(data['steps'])} steps")
    print(f"  Cheating steps   : {cheat_steps}")
    print(f"  Alerts fired     : {alert_steps}")
    print(f"  Correct alerts   : {correct.sum()}")
    print(f"  Missed detections: {missed.sum()}")
    print(f"  False positives  : {((data['actions'] == 2) & (data['labels'] == 0)).sum()}")

    plot_episode(data, out)
