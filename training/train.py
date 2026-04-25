"""
Train the Adaptive Proctoring Agent using DQN (stable-baselines3).

Saves:
  models/apa_dqn.zip              — final trained policy
  models/apa_dqn_best.zip         — best checkpoint by eval reward
  outputs/training/training_log.csv
  outputs/training/training_curves.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment import ExamProctoringEnv
from paths import (
    MODELS_DIR, CHECKPOINT_PATH,
    TRAINING_OUT, ensure,
)

FINAL_PATH = os.path.join(MODELS_DIR, "apa_dqn")
BEST_PATH  = os.path.join(MODELS_DIR, "apa_dqn_best")
LOG_PATH   = os.path.join(TRAINING_OUT, "training_log.csv")


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """
    At each eval interval rolls out n_eval_episodes and logs:
      mean_reward, TPR, FPR, privacy_cost (fraction of high-cost actions).
    """

    def __init__(self, eval_env, eval_freq: int = 5_000,
                 n_eval_episodes: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.eval_freq        = eval_freq
        self.n_eval_episodes  = n_eval_episodes
        self.records          = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        tp = fp = fn = tn = 0
        high_cost = 0
        total_steps = 0
        ep_rewards = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
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
        cost = high_cost / (total_steps + 1e-9)
        mean_reward = float(np.mean(ep_rewards))

        self.records.append({
            "timestep":     self.num_timesteps,
            "mean_reward":  mean_reward,
            "tpr":          tpr,
            "fpr":          fpr,
            "privacy_cost": cost,
        })

        if self.verbose:
            print(
                f"  [{self.num_timesteps:>7d}]  "
                f"reward={mean_reward:6.2f}  "
                f"TPR={tpr:.3f}  FPR={fpr:.3f}  "
                f"cost={cost:.3f}"
            )
        return True

    def save_log(self) -> pd.DataFrame:
        ensure(TRAINING_OUT)
        df = pd.DataFrame(self.records)
        df.to_csv(LOG_PATH, index=False)
        print(f"Saved training log → {LOG_PATH}")
        return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 200_000) -> tuple:
    ensure(MODELS_DIR, TRAINING_OUT)

    train_env = Monitor(ExamProctoringEnv(cheat_prob=0.5, sim_seed=0))
    eval_env  = Monitor(ExamProctoringEnv(cheat_prob=0.5, sim_seed=99))

    metrics_cb = MetricsCallback(
        eval_env=eval_env,
        eval_freq=5_000,
        n_eval_episodes=50,
        verbose=1,
    )

    # EvalCallback saves the best checkpoint by mean eval reward
    eval_cb = EvalCallback(
        eval_env=Monitor(ExamProctoringEnv(cheat_prob=0.5, sim_seed=77)),
        best_model_save_path=BEST_PATH,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    model = DQN(
        policy                  = "MlpPolicy",
        env                     = train_env,
        learning_rate           = 1e-3,
        buffer_size             = 50_000,
        batch_size              = 64,
        gamma                   = 0.95,
        exploration_fraction    = 0.25,
        exploration_final_eps   = 0.05,
        target_update_interval  = 500,
        train_freq              = 4,
        verbose                 = 0,
    )

    print(f"Training for {total_timesteps:,} timesteps …")
    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb, eval_cb])

    model.save(FINAL_PATH)
    print(f"\nSaved final model  → {FINAL_PATH}.zip")
    print(f"Saved best model   → {BEST_PATH}/best_model.zip")

    df = metrics_cb.save_log()
    _plot_training(df)
    return model, df


# ---------------------------------------------------------------------------
# Training curves plot
# ---------------------------------------------------------------------------

def _plot_training(df: pd.DataFrame) -> None:
    if df.empty:
        print("No eval records yet — skipping training curves plot.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Adaptive Proctoring Agent — Training Curves", fontsize=13)

    axes[0, 0].plot(df.timestep, df.mean_reward)
    axes[0, 0].set_title("Mean episode reward")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Reward")

    axes[0, 1].plot(df.timestep, df.tpr, label="TPR", color="steelblue")
    axes[0, 1].plot(df.timestep, df.fpr, label="FPR", color="tomato")
    axes[0, 1].set_title("Detection rates")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].legend()

    axes[1, 0].plot(df.timestep, df.privacy_cost, color="darkorange")
    axes[1, 0].set_title("Privacy cost (fraction of high-cost actions)")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Cost")

    sc = axes[1, 1].scatter(df.fpr, df.tpr, c=df.timestep, cmap="viridis", s=20)
    axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[1, 1].set_title("TPR vs FPR over training")
    axes[1, 1].set_xlabel("FPR")
    axes[1, 1].set_ylabel("TPR")
    plt.colorbar(sc, ax=axes[1, 1], label="Timestep")

    plt.tight_layout()
    out = os.path.join(TRAINING_OUT, "training_curves.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, df = train(total_timesteps=200_000)
    print("\nFinal metrics (last eval):")
    print(df.iloc[-1].to_string())
