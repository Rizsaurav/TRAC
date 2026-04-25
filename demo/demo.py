"""
Real-time desktop demo for the Adaptive Proctoring Agent.

Two-panel OpenCV window:
  Left  : live webcam feed with MediaPipe iris landmarks and gaze vector
  Right : dashboard — anomaly score graph, action badge,
          session metrics, 3×3 gaze heatmap

Press Q to quit.

Honest limitation: MediaPipe runs at ~30fps with pixel-level accuracy,
vs. 500–1000Hz with sub-degree accuracy for a real eye tracker. The 6
features computed here are approximations of the simulator's distributions
the autoencoder was trained on. Anomaly score behaviour on live webcam
input vs. simulated data is itself an open research question.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import time
from collections import deque
from typing import Optional, List

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as _mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
from stable_baselines3 import DQN

_LANDMARK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_landmark_model(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading face landmarker model → {path} …")
    urllib.request.urlretrieve(_LANDMARK_MODEL_URL, path)
    print("  done.")

from autoencoder import load_model, get_anomaly_score, get_latent, LATENT_DIM
from gaze_simulator import FEATURE_NAMES, HONEST_PARAMS
from paths import MODELS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DQN_PATH         = os.path.join(MODELS_DIR, "apa_dqn.zip")
FRAME_BUFFER     = 15          # frames per 500ms epoch at ~30fps
ROLLING_WINDOW   = 5
BLINK_EAR_THRESH = 0.20
VIEWING_DIST_CM  = 60.0
SCREEN_DPI       = 96.0
PX_PER_CM        = SCREEN_DPI / 2.54
PANEL_W          = 480

# OpenCV uses BGR.  Convert hex #RRGGBB → (B, G, R).
ACTION_BGR = {
    0: (221, 138, 55),   # blue  — passive
    1: (39,  159, 239),  # orange — active
    2: (74,   75, 226),  # red   — alert
}
ACTION_LABELS = {0: "PASSIVE", 1: "ACTIVE", 2: "ALERT"}

# Simulator's honest-behavior center — features are remapped into this space
# after personal calibration so the autoencoder sees familiar input scale.
SIMULATOR_MEAN = np.array([p[0] for p in [
    HONEST_PARAMS.fixation_duration_ms,
    HONEST_PARAMS.saccade_amplitude_deg,
    HONEST_PARAMS.off_screen_ratio,
    HONEST_PARAMS.lateral_deviation_px,
    HONEST_PARAMS.blink_rate_per_min,
    HONEST_PARAMS.gaze_entropy,
]], dtype=np.float32)

SIMULATOR_STD = np.array([p[1] for p in [
    HONEST_PARAMS.fixation_duration_ms,
    HONEST_PARAMS.saccade_amplitude_deg,
    HONEST_PARAMS.off_screen_ratio,
    HONEST_PARAMS.lateral_deviation_px,
    HONEST_PARAMS.blink_rate_per_min,
    HONEST_PARAMS.gaze_entropy,
]], dtype=np.float32)

# MediaPipe FaceMesh iris indices (requires refine_landmarks=True)
_LEFT_IRIS     = [468, 469, 470, 471, 472]
_RIGHT_IRIS    = [473, 474, 475, 476, 477]
_LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class Calibrator:
    """Collects 30 s of feature windows to compute the user's personal baseline."""

    DURATION_SEC = 30.0

    def __init__(self):
        self._samples: List[np.ndarray] = []
        self._start_t: float = time.time()
        self.done: bool = False
        self.personal_mean: Optional[np.ndarray] = None
        self.personal_std:  Optional[np.ndarray] = None

    @property
    def countdown(self) -> float:
        return max(0.0, self.DURATION_SEC - (time.time() - self._start_t))

    def update(self, features: np.ndarray) -> None:
        if self.done:
            return
        self._samples.append(features.copy())
        if time.time() - self._start_t >= self.DURATION_SEC:
            self._finalize()

    def _finalize(self) -> None:
        arr = np.array(self._samples, dtype=np.float32)
        self.personal_mean = arr.mean(axis=0)
        raw_std = arr.std(axis=0)
        self.personal_std = np.where(raw_std > 1e-6, raw_std, 1.0).astype(np.float32)
        self.done = True
        print("Calibration complete — personal baseline locked.")

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Remap personal-scale features into the simulator's honest-behavior space."""
        z = (features - self.personal_mean) / self.personal_std
        return (z * SIMULATOR_STD + SIMULATOR_MEAN).astype(np.float32)


# ---------------------------------------------------------------------------
# LandmarkExtractor
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """Wraps MediaPipe FaceLandmarker (tasks API). Returns iris centroids and EAR per frame."""

    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "face_landmarker.task")
        _ensure_landmark_model(model_path)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=model_path),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, bgr: np.ndarray) -> Optional[dict]:
        """Return landmark dict or None when no face is detected."""
        h, w = bgr.shape[:2]
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]   # list of NormalizedLandmark

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        def centroid(indices):
            return np.stack([px(i) for i in indices]).mean(axis=0)

        def ear(eye_idx):
            p = [px(i) for i in eye_idx]
            return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
                2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6
            )

        left  = centroid(_LEFT_IRIS)
        right = centroid(_RIGHT_IRIS)
        return {
            "gaze":       (left + right) / 2.0,
            "left_iris":  left,
            "right_iris": right,
            "left_ear":   ear(_LEFT_EYE_EAR),
            "right_ear":  ear(_RIGHT_EYE_EAR),
            "frame_wh":   (w, h),
        }

    def draw(self, bgr: np.ndarray, lm: Optional[dict]) -> np.ndarray:
        """Annotate frame with iris circles, gaze point, and EAR readout."""
        vis = bgr.copy()
        if lm is None:
            cv2.putText(vis, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis

        for pt in (lm["left_iris"], lm["right_iris"]):
            cv2.circle(vis, tuple(pt.astype(int)), 7, (0, 255, 0), 2)
        cv2.circle(vis, tuple(lm["gaze"].astype(int)), 4, (255, 80, 0), -1)

        w, h = lm["frame_wh"]
        for side, ear_val, x in (("L", lm["left_ear"], 12),
                                   ("R", lm["right_ear"], w - 80)):
            colour = (0, 0, 220) if ear_val < BLINK_EAR_THRESH else (0, 220, 0)
            cv2.putText(vis, f"{side} EAR:{ear_val:.2f}", (x, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        return vis


# ---------------------------------------------------------------------------
# FeatureBuffer
# ---------------------------------------------------------------------------

class FeatureBuffer:
    """
    Rolling 30-frame buffer. Every FRAME_BUFFER updates, compute_features()
    returns the 6-element window matching the simulator's feature order.
    """

    def __init__(self, fps: float = 30.0):
        self.fps            = fps
        self.gaze_buf       = deque(maxlen=30)
        self._ear_buf       = deque(maxlen=30)
        self._blink_count   = 0
        self._prev_blink    = False
        self._frame_count   = 0
        self._last_epoch_t  = time.time()   # wall-clock epoch trigger
        self.last_features  = np.zeros(6, dtype=np.float32)

    def update(self, lm: Optional[dict]) -> None:
        if lm is None:
            return
        self.gaze_buf.append(lm["gaze"].copy())
        ear = min(lm["left_ear"], lm["right_ear"])
        self._ear_buf.append(ear)
        is_blink = ear < BLINK_EAR_THRESH
        if is_blink and not self._prev_blink:
            self._blink_count += 1
        self._prev_blink = is_blink
        self._frame_count += 1

    def epoch_ready(self) -> bool:
        if self._frame_count == 0:
            return False
        if time.time() - self._last_epoch_t >= 0.5:
            self._last_epoch_t = time.time()
            return True
        return False

    def compute_features(self, frame_w: int = 640, frame_h: int = 480) -> np.ndarray:
        if len(self.gaze_buf) < 2:
            return self.last_features

        gaze = np.array(self.gaze_buf)   # (N, 2)

        # fixation duration: longest run within 15px radius (ms)
        max_run = cur_run = 1
        for i in range(1, len(gaze)):
            if np.linalg.norm(gaze[i] - gaze[i - 1]) < 15.0:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        fixation_ms = max_run / self.fps * 1000.0

        # saccade amplitude (degrees of visual angle)
        px_per_deg = PX_PER_CM * math.tan(math.radians(1)) * VIEWING_DIST_CM
        diffs      = np.linalg.norm(np.diff(gaze, axis=0), axis=1)
        saccade_deg = float(diffs.mean() / (px_per_deg + 1e-6))

        # off-screen ratio: gaze outside central 80% of frame
        mx, my = frame_w * 0.10, frame_h * 0.10
        off = (
            (gaze[:, 0] < mx) | (gaze[:, 0] > frame_w - mx) |
            (gaze[:, 1] < my) | (gaze[:, 1] > frame_h - my)
        )
        off_ratio = float(off.mean())

        # lateral deviation: mean |x - center| in pixels
        lateral = float(np.abs(gaze[:, 0] - frame_w / 2.0).mean())

        # blink rate per minute
        buf_sec    = len(self.gaze_buf) / self.fps
        blink_rate = self._blink_count / (buf_sec + 1e-6) * 60.0
        self._blink_count = 0  # reset each epoch

        # gaze entropy over 3x3 screen grid
        col   = np.clip((gaze[:, 0] / frame_w * 3).astype(int), 0, 2)
        row   = np.clip((gaze[:, 1] / frame_h * 3).astype(int), 0, 2)
        cells = row * 3 + col
        probs = (np.bincount(cells, minlength=9).astype(float) + 1e-9)
        probs /= probs.sum()
        entropy = float(-np.sum(probs * np.log(probs)))

        self.last_features = np.array(
            [fixation_ms, saccade_deg, off_ratio, lateral, blink_rate, entropy],
            dtype=np.float32,
        )
        return self.last_features


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------

class AgentSession:
    """Runs autoencoder + DQN pipeline; accumulates live session metrics."""

    def __init__(self):
        self._ae_model, self._scaler = load_model()
        self._dqn         = DQN.load(DQN_PATH)
        self._anomaly_buf = deque([0.0] * ROLLING_WINDOW, maxlen=ROLLING_WINDOW)

        self.n_steps: int    = 0
        self.n_alerts: int   = 0
        self.n_active: int   = 0
        self.score_history: List[float] = []

    def step(self, window: np.ndarray):
        """Return (anomaly_score, latent_8d, action_int)."""
        score  = get_anomaly_score(window, self._ae_model, self._scaler)
        latent = get_latent(window, self._ae_model, self._scaler)

        self._anomaly_buf.append(score)
        rolling = float(np.mean(self._anomaly_buf))
        obs     = np.concatenate([latent, [score, rolling]], dtype=np.float32)
        action  = int(self._dqn.predict(obs, deterministic=True)[0])

        self.n_steps  += 1
        self.n_alerts += int(action == 2)
        self.n_active += int(action == 1)
        self.score_history.append(score)
        return score, latent, action

    @property
    def alert_rate(self)   -> float: return self.n_alerts / (self.n_steps + 1e-9)
    @property
    def privacy_cost(self) -> float: return (self.n_alerts + self.n_active) / (self.n_steps + 1e-9)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    """Pure drawing — no inference. Produces the right-hand panel."""

    _GRAPH_H = 130
    _BADGE_H = 80
    _FONT    = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_B  = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, frame_h: int = 480):
        self.frame_h = frame_h

    def render(self, score_history: List[float], action: int,
               alert_rate: float, privacy_cost: float,
               gaze_buf: deque,
               calibrating: bool = False, countdown: float = 0.0) -> np.ndarray:
        panel = np.full((self.frame_h, PANEL_W, 3), 22, dtype=np.uint8)
        y = 8
        y = self._score_graph(panel, score_history, y)
        if calibrating:
            y = self._calibration_badge(panel, countdown, y)
        else:
            y = self._action_badge(panel, action, y)
        y = self._metrics(panel, alert_rate, privacy_cost, y)
        self._heatmap(panel, gaze_buf, y)
        return panel

    # ── sub-panels ──────────────────────────────────────────────────────

    def _score_graph(self, p, history, y0):
        W, H = PANEL_W - 20, self._GRAPH_H
        x0   = 10
        cv2.rectangle(p, (x0, y0), (x0 + W, y0 + H), (38, 38, 38), -1)
        cv2.putText(p, "Anomaly score", (x0 + 6, y0 + 16),
                    self._FONT, 0.44, (180, 180, 180), 1)

        if len(history) > 1:
            hist  = np.array(history[-(W):], dtype=float)
            vmax  = max(hist.max() * 1.3, 1e-5)
            step  = max(1, W // len(hist))
            pts   = []
            for i, v in enumerate(hist):
                px_ = x0 + i * W // len(hist)
                py_ = y0 + H - int(v / vmax * (H - 22)) - 4
                pts.append((px_, py_))
            for i in range(1, len(pts)):
                cv2.line(p, pts[i - 1], pts[i], (80, 200, 80), 1)
            # current value label
            cv2.putText(p, f"{hist[-1]:.5f}", (x0 + W - 90, y0 + 16),
                        self._FONT, 0.40, (80, 200, 80), 1)

        return y0 + H + 6

    def _calibration_badge(self, p, countdown: float, y0: int) -> int:
        W, H = PANEL_W - 20, self._BADGE_H
        x0   = 10
        cv2.rectangle(p, (x0, y0), (x0 + W, y0 + H), (30, 120, 30), -1)
        label = "CALIBRATING"
        tw  = cv2.getTextSize(label, self._FONT_B, 0.9, 2)[0][0]
        cv2.putText(p, label, (x0 + (W - tw) // 2, y0 + H // 2 - 4),
                    self._FONT_B, 0.9, (255, 255, 255), 2)
        timer = f"{int(countdown)}s remaining"
        tw2 = cv2.getTextSize(timer, self._FONT, 0.5, 1)[0][0]
        cv2.putText(p, timer, (x0 + (W - tw2) // 2, y0 + H // 2 + 18),
                    self._FONT, 0.5, (180, 230, 180), 1)
        return y0 + H + 6

    def _action_badge(self, p, action, y0):
        W, H = PANEL_W - 20, self._BADGE_H
        x0   = 10
        cv2.rectangle(p, (x0, y0), (x0 + W, y0 + H), ACTION_BGR[action], -1)
        label         = ACTION_LABELS[action]
        (tw, th), _   = cv2.getTextSize(label, self._FONT_B, 1.3, 2)
        cv2.putText(p, label, (x0 + (W - tw) // 2, y0 + (H + th) // 2),
                    self._FONT_B, 1.3, (255, 255, 255), 2)
        return y0 + H + 6

    def _metrics(self, p, alert_rate, privacy_cost, y0):
        for txt in (f"Alerts:        {alert_rate:.3f}",
                    f"Privacy cost:  {privacy_cost:.3f}"):
            cv2.putText(p, txt, (14, y0 + 20), self._FONT, 0.52, (210, 210, 210), 1)
            y0 += 26
        return y0 + 10

    def _heatmap(self, p, gaze_buf, y0):
        H_avail = self.frame_h - y0 - 10
        if H_avail < 40:
            return
        W = PANEL_W - 20
        H = min(130, H_avail)
        x0 = 10
        cv2.putText(p, "Gaze heatmap (3x3)", (x0 + 4, y0 + 14),
                    self._FONT, 0.44, (180, 180, 180), 1)
        y0 += 18

        grid = np.zeros((3, 3), dtype=float)
        for gp in gaze_buf:
            c = int(np.clip(gp[0] / 640 * 3, 0, 2))
            r = int(np.clip(gp[1] / 480 * 3, 0, 2))
            grid[r, c] += 1

        total  = grid.sum() + 1e-9
        cw, ch = W // 3, (H - 18) // 3
        for r in range(3):
            for c in range(3):
                intensity = int(grid[r, c] / total * 255)
                fill = (intensity // 5, intensity // 2, intensity)
                cx_, cy_ = x0 + c * cw, y0 + r * ch
                cv2.rectangle(p, (cx_, cy_), (cx_ + cw, cy_ + ch), fill, -1)
                cv2.rectangle(p, (cx_, cy_), (cx_ + cw, cy_ + ch), (70, 70, 70), 1)
                cv2.putText(p, f"{grid[r,c]/total*100:.0f}%",
                            (cx_ + 4, cy_ + ch - 5),
                            self._FONT, 0.34, (230, 230, 230), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def list_cameras(max_test: int = 5) -> None:
    """Print which device indices have a working camera."""
    print("Scanning camera indices 0 –", max_test - 1)
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [{i}]  {w}×{h} @ {fps:.0f}fps")
            cap.release()
        else:
            print(f"  [{i}]  (not available)")


def run_demo(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at device index {camera_index}. "
                           "Run with --list to see available cameras.")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {frame_w}×{frame_h} @ {fps:.1f}fps")
    print("Loading models …")

    extractor  = LandmarkExtractor()
    buf        = FeatureBuffer(fps=fps)
    session    = AgentSession()
    dash       = Dashboard(frame_h=frame_h)
    calibrator = Calibrator()

    current_action = 0
    print(f"Calibrating for {int(Calibrator.DURATION_SEC)}s — sit still and look at the screen normally.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = extractor.process(frame)
        buf.update(lm)

        if buf.epoch_ready():
            features = buf.compute_features(frame_w, frame_h)
            if not calibrator.done:
                calibrator.update(features)
            else:
                normalized = calibrator.normalize(features)
                _, _, current_action = session.step(normalized)

        calibrating = not calibrator.done
        annotated = extractor.draw(frame, lm)
        panel     = dash.render(
            score_history = session.score_history,
            action        = current_action,
            alert_rate    = session.alert_rate,
            privacy_cost  = session.privacy_cost,
            gaze_buf      = buf.gaze_buf,
            calibrating   = calibrating,
            countdown     = calibrator.countdown,
        )

        # Resize panel height to match camera frame if needed
        if panel.shape[0] != frame_h:
            panel = cv2.resize(panel, (PANEL_W, frame_h))

        cv2.imshow("Adaptive Proctoring Agent", np.hstack([annotated, panel]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Adaptive Proctoring Agent — live demo")
    ap.add_argument("--camera", type=int, default=0,
                    help="OpenCV camera device index (default: 0)")
    ap.add_argument("--list", action="store_true",
                    help="List available camera indices and exit")
    args = ap.parse_args()

    if args.list:
        list_cameras()
    else:
        run_demo(camera_index=args.camera)
