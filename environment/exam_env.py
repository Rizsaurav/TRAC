"""
ExamProctoringEnv — custom Gymnasium environment for the Adaptive Proctoring Agent.

Observation (10-D float32):
  [0:8]  8-dim autoencoder latent code of the current gaze window
  [8]    current anomaly score (MSE reconstruction error)
  [9]    rolling mean of the last ROLLING_WINDOW anomaly scores

Action space: Discrete(3)
  0  passive  — minimal signal collection, lowest privacy cost
  1  active   — full signal collection, medium privacy cost
  2  alert    — flag session; highest scrutiny, creates a record

Reward (Yang et al. 2024 framing):
  true_cheating  + alert    →  +2.0   (true positive)
  true_honest    + alert    →  -2.0   (false positive)
  true_cheating  + passive  →  -1.0   (missed detection)
  true_cheating  + active   →  -1.0   (missed detection)
  true_honest    + passive  →  +0.5   (correct low-cost decision)
  true_honest    + active   →  -0.1   (unnecessary data collection)

Episode length: 120 steps (60-min exam at 30-sec granularity).
Each episode either contains one cheating block (random start, random length
3–8 steps) with probability cheat_prob, or is fully honest.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gaze_simulator import GazeSimulator, FEATURE_NAMES
from autoencoder import load_model, get_anomaly_score, get_latent, LATENT_DIM

# 
# Constants
# 

EPISODE_LEN    = 120
ROLLING_WINDOW = 5
OBS_DIM        = LATENT_DIM + 2   # 8 latent + anomaly_score + rolling_mean = 10
N_ACTIONS      = 3

# (true_label, action) → reward
_REWARD_TABLE = {
    (1, 2): +2.0,
    (0, 2): -2.0,
    (1, 0): -1.0,
    (1, 1): -1.0,
    (0, 0): +0.5,
    (0, 1): -0.1,
}


# 
# Environment

class ExamProctoringEnv(gym.Env):
    """
    Parameters
    ----------
    cheat_prob      : probability that any given episode contains a cheating event
    min_cheat_dur   : minimum length (steps) of a cheating block
    max_cheat_dur   : maximum length (steps) of a cheating block
    sim_seed        : seed for the GazeSimulator (None = non-deterministic)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cheat_prob: float = 0.5,
        min_cheat_dur: int = 3,
        max_cheat_dur: int = 8,
        sim_seed: Optional[int] = None,
    ):
        super().__init__()

        self.cheat_prob    = cheat_prob
        self.min_cheat_dur = min_cheat_dur
        self.max_cheat_dur = max_cheat_dur

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._ae_model, self._scaler = load_model()
        self._sim   = GazeSimulator(seed=sim_seed)
        self._rng   = np.random.default_rng(sim_seed)

        # Episode state — initialised properly in reset()
        self._step: int = 0
        self._labels: Optional[np.ndarray] = None
        self._sequence = None
        self._anomaly_buf: deque = deque([0.0] * ROLLING_WINDOW, maxlen=ROLLING_WINDOW)

    # Gymnasium API

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._sim = GazeSimulator(seed=seed)

        self._step = 0
        self._anomaly_buf = deque([0.0] * ROLLING_WINDOW, maxlen=ROLLING_WINDOW)

        self._sequence, self._labels = self._new_episode()
        obs = self._obs_at(self._step)
        return obs, {}

    def step(self, action: int):
        assert self._labels is not None, "Call reset() before step()"

        label  = int(self._labels[self._step])
        reward = _REWARD_TABLE.get((label, int(action)), 0.0)

        info = {
            "true_label": label,
            "action":     int(action),
            "step":       self._step,
        }

        self._step += 1
        terminated = self._step >= EPISODE_LEN

        obs = self._obs_at(self._step) if not terminated else np.zeros(OBS_DIM, dtype=np.float32)
        return obs, reward, terminated, False, info

    # Internal helpers

    def _new_episode(self):
        if self._rng.random() < self.cheat_prob:
            dur   = int(self._rng.integers(self.min_cheat_dur, self.max_cheat_dur + 1))
            start = int(self._rng.integers(0, EPISODE_LEN - dur))
            seq   = self._sim.generate_sequence(
                n_steps=EPISODE_LEN, cheat_start=start, cheat_duration=dur
            )
        else:
            seq = self._sim.generate_sequence(n_steps=EPISODE_LEN, cheat_start=None)
        return seq, seq["label"].values.astype(np.int32)

    def _obs_at(self, t: int) -> np.ndarray:
        window = self._sequence[FEATURE_NAMES].iloc[t].values.astype(np.float32)
        latent = get_latent(window, self._ae_model, self._scaler)
        score  = get_anomaly_score(window, self._ae_model, self._scaler)

        self._anomaly_buf.append(score)
        rolling_mean = float(np.mean(self._anomaly_buf))

        return np.concatenate([latent, [score, rolling_mean]], dtype=np.float32)


# Smoke-test

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("Building environment …")
    env = ExamProctoringEnv(cheat_prob=0.5, sim_seed=7)

    print("Running gymnasium env_checker …")
    check_env(env, warn=True, skip_render_check=True)
    print("  check_env passed")

    print("\nRolling out one episode with random policy:")
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    cheating_steps = 0
    alerts = 0

    for _ in range(EPISODE_LEN):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward    += reward
        cheating_steps  += info["true_label"]
        alerts          += int(info["action"] == 2)
        if terminated:
            break

    print(f"  Total reward   : {total_reward:.2f}")
    print(f"  Cheating steps : {cheating_steps}")
    print(f"  Alerts fired   : {alerts}")
    print(f"  Obs shape      : {obs.shape}")
    print(f"  Obs sample     : {obs.round(4)}")
