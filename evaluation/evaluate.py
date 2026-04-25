"""
Evaluation and ablation for the Adaptive Proctoring Agent.

Policies evaluated over 500 episodes each:
  1. Trained APA (DQN)
  2. Static threshold baseline — alert if anomaly_score > 0.0003
  3. Random policy

Ablation:
  4. APA retrained WITHOUT the privacy penalty (honest+active reward = 0.0 instead
     of -0.1). Shows that the penalty is what drives low privacy cost.

Outputs (all in outputs/evaluation/):
  evaluation_results.png    — bar charts for policies 1–3
  ablation_comparison.png   — DQN vs no-penalty side-by-side
  eval_summary.csv          — full numeric table
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment import ExamProctoringEnv
from environment.exam_env import _REWARD_TABLE
from paths import MODELS_DIR, EVALUATION_OUT, ensure

N_EVAL_EPISODES = 500
APA_MODEL_PATH      = os.path.join(MODELS_DIR, "apa_dqn.zip")
ABLATION_MODEL_PATH = os.path.join(MODELS_DIR, "apa_dqn_noprivacy")  # SB3 appends .zip on save

# Reward table with the privacy penalty removed (honest+active = 0.0)
_NO_PRIVACY_TABLE = {**_REWARD_TABLE, (0, 1): 0.0}


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------

def rollout(policy_fn, n_episodes: int = N_EVAL_EPISODES, seed: int = 200) -> dict:
    """
    Run n_episodes with policy_fn(obs) → int action.
    Returns aggregated metrics dict.
    """
    env = ExamProctoringEnv(cheat_prob=0.5, sim_seed=seed)
    tp = fp = fn = tn = 0
    high_cost   = 0
    total_steps = 0
    ep_rewards  = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            label = info["true_label"]
            act   = info["action"]
            ep_reward   += reward
            total_steps += 1

            if   act == 2 and label == 1: tp += 1
            elif act == 2 and label == 0: fp += 1
            elif act != 2 and label == 1: fn += 1
            else:                         tn += 1

            if act in (1, 2):
                high_cost += 1

        ep_rewards.append(ep_reward)

    tpr  = tp / (tp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1   = 2 * prec * tpr / (prec + tpr + 1e-9)
    cost = high_cost / (total_steps + 1e-9)

    return {
        "mean_reward":  float(np.mean(ep_rewards)),
        "tpr":          tpr,
        "fpr":          fpr,
        "f1":           f1,
        "privacy_cost": cost,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------------

def make_apa_policy(path: str = APA_MODEL_PATH):
    model = DQN.load(path)
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def make_static_threshold_policy(threshold: float = 0.0003):
    def policy(obs):
        return 2 if float(obs[8]) >= threshold else 0
    return policy


def random_policy(obs):
    return int(np.random.randint(0, 3))


# ---------------------------------------------------------------------------
# Ablation training (runs only if the ablation model doesn't exist yet)
# ---------------------------------------------------------------------------

def _train_ablation(total_timesteps: int = 200_000) -> None:
    print("  Ablation model not found — training now …")
    train_env = Monitor(ExamProctoringEnv(
        cheat_prob=0.5, sim_seed=0, reward_table=_NO_PRIVACY_TABLE
    ))
    model = DQN(
        policy                 = "MlpPolicy",
        env                    = train_env,
        learning_rate          = 1e-3,
        buffer_size            = 50_000,
        batch_size             = 64,
        gamma                  = 0.95,
        exploration_fraction   = 0.25,
        exploration_final_eps  = 0.05,
        target_update_interval = 500,
        train_freq             = 4,
        verbose                = 0,
    )
    model.learn(total_timesteps=total_timesteps)
    ensure(MODELS_DIR)
    model.save(ABLATION_MODEL_PATH)
    print(f"  Saved ablation model → {ABLATION_MODEL_PATH}.zip")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_metrics(name: str, m: dict) -> None:
    print(f"  {name}")
    print(f"    TPR:          {m['tpr']:.3f}")
    print(f"    FPR:          {m['fpr']:.3f}")
    print(f"    F1:           {m['f1']:.3f}")
    print(f"    Privacy cost: {m['privacy_cost']:.3f}")
    print(f"    Mean reward:  {m['mean_reward']:.2f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_comparison(results: dict) -> None:
    ensure(EVALUATION_OUT)
    policies = list(results.keys())
    metrics  = ["tpr", "fpr", "f1", "privacy_cost"]
    titles   = ["TPR (detection rate)", "FPR (false positives)",
                 "F1 score", "Privacy cost"]
    colors   = ["steelblue", "tomato", "silver"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("APA Policy Comparison — 500 episodes", fontsize=12)

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [results[p][metric] for p in policies]
        bars = ax.bar(policies, vals, color=colors, width=0.5)
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9,
            )
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=15, ha="right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(EVALUATION_OUT, "evaluation_results.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def _plot_ablation(apa: dict, no_priv: dict) -> None:
    ensure(EVALUATION_OUT)
    labels  = ["APA (with penalty)", "APA (no penalty)"]
    metrics = ["tpr", "fpr", "privacy_cost"]
    titles  = ["TPR", "FPR", "Privacy cost"]
    vals_a  = [apa[m]    for m in metrics]
    vals_b  = [no_priv[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width/2, vals_a, width, label=labels[0], color="steelblue")
    b2 = ax.bar(x + width/2, vals_b, width, label=labels[1], color="tomato")
    ax.set_title("Ablation: Effect of Privacy Penalty on Agent Behaviour")
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylim(0, 1.15)
    ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=9,
        )
    plt.tight_layout()
    out = os.path.join(EVALUATION_OUT, "ablation_comparison.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation() -> dict:
    ensure(EVALUATION_OUT)
    np.random.seed(42)

    print("=== Policy evaluation — 500 episodes each ===\n")

    print("1. Trained APA (DQN) …")
    apa = rollout(make_apa_policy())
    _print_metrics("APA (DQN)", apa)

    print("\n2. Static threshold baseline (threshold=0.0003) …")
    static = rollout(make_static_threshold_policy(0.0003))
    _print_metrics("Static threshold", static)

    print("\n3. Random policy …")
    random = rollout(random_policy)
    _print_metrics("Random", random)

    results = {
        "APA (DQN)":        apa,
        "Static threshold": static,
        "Random":           random,
    }

    df = pd.DataFrame(results).T[["tpr", "fpr", "f1", "privacy_cost", "mean_reward"]]
    csv_path = os.path.join(EVALUATION_OUT, "eval_summary.csv")
    df.to_csv(csv_path)
    print(f"\nSaved {csv_path}")

    _plot_comparison(results)

    # --- Ablation ---
    print("\n=== Ablation: no-privacy-penalty model ===\n")
    if not os.path.exists(ABLATION_MODEL_PATH + ".zip"):
        _train_ablation(total_timesteps=200_000)

    print("4. APA without privacy penalty …")
    no_priv = rollout(make_apa_policy(ABLATION_MODEL_PATH))
    _print_metrics("APA (no penalty)", no_priv)
    _plot_ablation(apa, no_priv)

    return results


if __name__ == "__main__":
    run_evaluation()
