# Step counter - shared by the board (main.py) and the laptop (laptop/train_steps.py). Only needs numpy.
#
# The network looks at the last 2 s of all IMU channels and answers ONE number: how many steps per
# second you're taking right now. The board asks it every 0.5 s and adds rate x 0.5 s to the count.
#
# Training labels come from heel strikes: the sensor is on ONE ankle, so every swing of that leg
# (a big gyro peak) is one stride = 2 steps (the other foot lands halfway between). Recordings
# where you hand-counted your steps are used to check those labels and to test the network.
import json
from pathlib import Path
import numpy as np

RATE = 50                     # samples per second (same as the gait trainer)
WIN = 2 * RATE                # the network sees 2 s
HOP = RATE // 2               # the board asks it every 0.5 s
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "|a|", "|g|"]   # every sensor + the two magnitudes
NO_STEP_ACTIVITIES = {"standing", "biking"}                     # pedaling swings the leg but isn't a step


def channels(data):
    """N x 6 raw IMU (g, deg/s) -> N x 8 network channels."""
    a = np.linalg.norm(data[:, :3], axis=1, keepdims=True)
    g = np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    return np.hstack([data[:, :6], a, g]).astype(np.float32)


# ---------------------------------------------------------------- labels: when did each step happen?
def _smooth(x, n=7):
    return np.convolve(x, np.ones(n) / n, "same")


def stride_peaks(data):
    """Sample index of each swing of the instrumented leg (one per stride)."""
    g = _smooth(np.linalg.norm(data[:, 3:6], axis=1))
    thr = max(60.0, 0.45 * np.percentile(g, 98))          # ignore small wiggles
    cand = np.flatnonzero((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]) & (g[1:-1] > thr)) + 1
    keep = []
    for i in cand[np.argsort(-g[cand])]:                   # biggest first, at least 0.5 s apart
        if all(abs(i - j) >= RATE // 2 for j in keep):
            keep.append(i)
    return np.sort(np.array(keep, dtype=int))


def step_times(data, activity):
    """Sample index of every step (both feet). Empty for standing/biking."""
    if activity in NO_STEP_ACTIVITIES:
        return np.array([], dtype=float)
    p = stride_peaks(data).astype(float)
    if len(p) == 0:
        return p
    other = (p[:-1] + p[1:]) / 2                           # the other foot lands halfway between
    return np.sort(np.concatenate([p, other]))


def rate_target(steps, n, sigma=0.1 * RATE):
    """Steps per second at every sample: each step is a little bump with area 1."""
    t = np.arange(n)
    r = np.zeros(n)
    for s in steps:
        r += np.exp(-0.5 * ((t - s) / sigma) ** 2)
    r /= sigma * np.sqrt(2 * np.pi)                       # area 1 per step (in samples)
    return r * RATE                                       # -> per second


# ---------------------------------------------------------------- the network, in plain numpy
# Same layers as StepNet in laptop/train_steps.py. The laptop trains in PyTorch, saves the weights
# as JSON, and this runs them on the board.
def _conv1d(x, w, b):
    """x: C x T, w: O x C x K (padding keeps T)."""
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (k // 2, k // 2)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, k, axis=1)    # C x T x K
    return np.einsum("ctk,ock->ot", cols, w) + b[:, None]


def _pool(x):
    t = x.shape[1] // 2 * 2
    return x[:, :t].reshape(x.shape[0], -1, 2).max(axis=2)


class StepModel:
    """An ensemble of small 1-D conv nets. Several nets = an answer plus how sure they are."""

    def __init__(self, d):
        self.mean = np.array(d["mean"], np.float32)
        self.std = np.array(d["std"], np.float32)
        self.nets = [{k: np.array(v, np.float32) for k, v in net.items()} for net in d["nets"]]
        self.info = d.get("info", {})

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def rates(self, win):
        """win: WIN x 6 raw IMU -> steps/s from each net in the ensemble."""
        x = ((channels(win) - self.mean) / self.std).T          # 8 x WIN
        out = []
        for n in self.nets:
            h = _pool(np.maximum(_conv1d(x, n["c1.w"], n["c1.b"]), 0))
            h = _pool(np.maximum(_conv1d(h, n["c2.w"], n["c2.b"]), 0))
            h = np.maximum(_conv1d(h, n["c3.w"], n["c3.b"]), 0).mean(axis=1)
            h = np.maximum(n["f1.w"] @ h + n["f1.b"], 0)
            z = float((n["f2.w"] @ h + n["f2.b"])[0])
            out.append(np.log1p(np.exp(-abs(z))) + max(z, 0))     # softplus: rate can't go negative
        return np.array(out)


# ---------------------------------------------------------------- distance
class StrideLength:
    """Meters per step, from walks over a known distance. Faster steps are longer steps."""

    def __init__(self, a=0.0, b=0.0, default=0.70, n=0, err=None):
        self.a, self.b, self.default, self.n, self.err = a, b, default, n, err

    def meters_per_step(self, cadence_spm):
        if self.n == 0:
            return self.default
        if self.b == 0.0:
            return self.a
        return float(np.clip(self.a + self.b * cadence_spm, 0.3, 2.0))

    @classmethod
    def fit(cls, rows):
        """rows: (steps per minute, meters per step) from known-distance recordings."""
        if not rows:
            return cls()
        c, m = np.array(rows).T
        if len(rows) < 3 or np.ptp(c) < 15:          # not enough spread in speed -> just the average
            return cls(a=float(m.mean()), n=len(rows))
        b, a = np.polyfit(c, m, 1)
        return cls(a=float(a), b=float(b), n=len(rows))

    def to_dict(self):
        return {"a": self.a, "b": self.b, "default": self.default, "n": self.n, "err": self.err}

    @classmethod
    def from_dict(cls, d):
        return cls(d["a"], d["b"], d.get("default", 0.70), d.get("n", 0), d.get("err"))


# ---------------------------------------------------------------- recordings + their labels
def meta_path(csv):
    return Path(csv).with_suffix(".json")


def load_meta(csv):
    """What you typed after a recording: {"steps": counted steps, "meters": distance}."""
    p = meta_path(csv)
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}
