# Gait models - shared by the board (main.py) and the laptop (laptop/analyze.py). Only needs numpy.
#
# Data: 50 samples/s of [ax, ay, az] in g and [gx, gy, gz] in deg/s from the MPU-6050.
# Every 0.5 s we look at the last 2.5 s (a "window") and call it one of ACTIVITIES.
#
# Model 1 ("peak"): ONE number per window = the biggest acceleration along ONE direction
#   (e.g. +x). Activities are sorted on a number line and split by 3 thresholds.
# Model 2 ("tree"): 40 numbers per window from ALL 6 IMU channels (means, spreads, peaks,
#   rhythm) fed to a small decision tree (a flowchart of yes/no questions). No neural nets.
import json
import numpy as np

ACTIVITIES = ["standing", "walking", "running", "biking"]
RATE = 50                    # samples per second (must match SAMPLE_MS in the sketch)
WINDOW = int(2.5 * RATE)     # samples per window
STEP = RATE // 2             # new prediction every 0.5 s
TRIM = 2 * RATE              # ignore the first/last 2 s of each recording (getting started / stopping)

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
# Model 1 candidates: each accel axis, either way (+x = peak of ax, -x = peak of -ax)
DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]


# ---------------------------------------------------------------- features
def peak_in_direction(win, direction):
    """Model 1's only input: biggest acceleration (g) along one direction in the window."""
    axis = "xyz".index(direction[1])
    sign = 1.0 if direction[0] == "+" else -1.0
    return float(np.max(sign * win[:, axis]))


def cadence(signal):
    """Rhythm of a signal in cycles/min (one leg = strides/min) from autocorrelation, 0 if none."""
    s = signal - signal.mean()
    if s.std() < 1e-6:
        return 0.0
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac /= ac[0]
    lo, hi = int(0.3 * RATE), int(2.0 * RATE)        # periods 0.3-2 s = 30-200 per minute
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 * RATE / lag if ac[lag] > 0.3 else 0.0


def features(win):
    """Model 2's inputs: a dict of numbers describing one window (WINDOW x 6 array)."""
    f = {}
    for i, ch in enumerate(CHANNELS):
        x = win[:, i]
        f[f"{ch}_mean"] = x.mean()     # accel means = which way gravity points = leg tilt
        f[f"{ch}_std"] = x.std()       # how much it moves
        f[f"{ch}_max"] = x.max()
        f[f"{ch}_min"] = x.min()
        f[f"{ch}_range"] = x.max() - x.min()
    acc = np.linalg.norm(win[:, :3], axis=1)
    gyr = np.linalg.norm(win[:, 3:], axis=1)
    for name, m in (("acc", acc), ("gyro", gyr)):  # magnitudes don't care how the pouch is turned
        f[f"{name}_mag_mean"] = m.mean()
        f[f"{name}_mag_std"] = m.std()
        f[f"{name}_mag_max"] = m.max()
        f[f"{name}_mag_p90"] = np.percentile(m, 90)
    f["acc_jerk"] = np.abs(np.diff(acc)).mean() * RATE   # how sharp the impacts are (g/s)
    f["cadence"] = cadence(gyr)
    return {k: float(v) for k, v in f.items()}


NICE = {
    "acc_jerk": "impact sharpness (g/s)", "cadence": "rhythm (per min)",
    "acc_mag_mean": "avg total accel (g)", "acc_mag_std": "total accel wobble (g)",
    "acc_mag_max": "max total accel (g)", "acc_mag_p90": "big total accel (g)",
    "gyro_mag_mean": "avg spin (deg/s)", "gyro_mag_std": "spin wobble (deg/s)",
    "gyro_mag_max": "max spin (deg/s)", "gyro_mag_p90": "big spin (deg/s)",
}


def describe(feature):
    """Plain-English name for a feature, e.g. ax_mean -> 'avg accel x (g) = tilt'."""
    if feature in NICE:
        return NICE[feature]
    ch, stat = feature.split("_")
    what, unit = (f"accel {ch[1]}", "g") if ch[0] == "a" else (f"spin {ch[1]}", "deg/s")
    s = {"mean": "avg", "std": "wobble", "max": "max", "min": "min", "range": "range"}[stat]
    return f"{s} {what} ({unit})" + (" = tilt" if ch[0] == "a" and stat == "mean" else "")


def is_dead(win):
    """True if most samples are exactly 0 (the MPU-6050 reset itself - a real ankle always feels gravity)."""
    return bool(np.mean(np.all(win == 0, axis=1)) > 0.5)


def windows(data):
    """Slide a WINDOW over a recording (N x 6). Yields (start_index, window)."""
    for s in range(0, len(data) - WINDOW + 1, STEP):
        yield s, data[s:s + WINDOW]


# ---------------------------------------------------------------- model 1
class PeakModel:
    """Sort the activities by their typical peak; put a threshold between each neighbor pair."""

    def __init__(self, direction="+x", order=None, thresholds=None):
        self.direction = direction
        self.order = order or list(ACTIVITIES)     # activities from lowest to highest peak
        self.thresholds = thresholds or []         # len(order) - 1 cut points (g)

    def fit(self, X, y):
        """X = 1-D peaks, y = labels."""
        X, y = np.asarray(X), np.asarray(y)
        present = [a for a in ACTIVITIES if np.any(y == a)]
        self.order = sorted(present, key=lambda a: np.median(X[y == a]))
        self.thresholds = []
        for lo, hi in zip(self.order, self.order[1:]):
            a, b = X[y == lo], X[y == hi]
            # try every cut point (halfway between neighboring values) between the two medians,
            # keep the one that mislabels fewest
            vals = np.unique(np.concatenate([a, b]))
            cands = (vals[:-1] + vals[1:]) / 2
            cands = cands[(cands >= np.median(a)) & (cands <= np.median(b))]
            if len(cands) == 0:
                cands = np.array([(np.median(a) + np.median(b)) / 2])
            errs = [np.sum(a > c) + np.sum(b <= c) for c in cands]
            best = np.flatnonzero(errs == np.min(errs))
            self.thresholds.append(float(np.mean(cands[best])))   # middle of the tied best cuts
        self.thresholds = [float(t) for t in np.maximum.accumulate(self.thresholds)] if self.thresholds else []
        return self

    def predict_value(self, peak):
        i = int(np.searchsorted(self.thresholds, peak, side="left"))
        return self.order[min(i, len(self.order) - 1)]

    def predict(self, X):
        return [self.predict_value(v) for v in X]

    def to_dict(self):
        return {"direction": self.direction, "order": self.order, "thresholds": self.thresholds}

    @classmethod
    def from_dict(cls, d):
        return cls(d["direction"], d["order"], d["thresholds"])


# ---------------------------------------------------------------- model 2
class TreeModel:
    """Tiny CART decision tree (Gini impurity), written out so it's easy to explain and draw."""

    def __init__(self, max_depth=3, min_leaf=8, root=None, feature_names=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root = root
        self.feature_names = feature_names or []

    @staticmethod
    def _gini(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p * p)

    def _leaf(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return {"leaf": str(labels[np.argmax(counts)]),
                "counts": {str(l): int(c) for l, c in zip(labels, counts)}}

    def _best_split(self, X, y):
        best = None
        n = len(y)
        parent = self._gini(y)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            xs, ys = X[order, j], y[order]
            # running class counts make every split point cheap to score
            classes = np.unique(y)
            onehot = (ys[:, None] == classes[None, :]).astype(float)
            left = np.cumsum(onehot, axis=0)[:-1]
            right = onehot.sum(0) - left
            nl = np.arange(1, n)
            nr = n - nl
            gl = 1 - np.sum((left / nl[:, None]) ** 2, axis=1)
            gr = 1 - np.sum((right / nr[:, None]) ** 2, axis=1)
            score = (nl * gl + nr * gr) / n
            ok = (xs[1:] > xs[:-1]) & (nl >= self.min_leaf) & (nr >= self.min_leaf)
            if not ok.any():
                continue
            score[~ok] = np.inf
            i = int(np.argmin(score))
            if best is None or score[i] < best[0]:
                best = (score[i], j, float((xs[i] + xs[i + 1]) / 2))
        if best is None or best[0] >= parent - 1e-9:
            return None
        return best[1], best[2]

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2 * self.min_leaf:
            return self._leaf(y)
        split = self._best_split(X, y)
        if split is None:
            return self._leaf(y)
        j, thr = split
        m = X[:, j] <= thr
        node = {"feature": self.feature_names[j], "threshold": thr,
                "yes": self._grow(X[m], y[m], depth + 1),
                "no": self._grow(X[~m], y[~m], depth + 1)}
        # both sides say the same thing -> just a leaf
        if "leaf" in node["yes"] and "leaf" in node["no"] and node["yes"]["leaf"] == node["no"]["leaf"]:
            return self._leaf(y)
        return node

    def fit(self, F, y):
        """F = list of feature dicts, y = labels."""
        self.feature_names = sorted(F[0])
        X = np.array([[f[k] for k in self.feature_names] for f in F])
        self.root = self._grow(X, np.asarray(y), 0)
        return self

    def predict_one(self, f):
        """Returns (label, path) where path is the list of yes/no answers taken."""
        node, path = self.root, []
        while "leaf" not in node:
            ans = "yes" if f[node["feature"]] <= node["threshold"] else "no"
            path.append(ans)
            node = node[ans]
        return node["leaf"], path

    def predict(self, F):
        return [self.predict_one(f)[0] for f in F]

    def used_features(self, node=None):
        node = node or self.root
        if "leaf" in node:
            return set()
        return {node["feature"]} | self.used_features(node["yes"]) | self.used_features(node["no"])

    def to_text(self, node=None, indent=""):
        node = node or self.root
        if "leaf" in node:
            return f"{indent}-> {node['leaf']}  {node['counts']}\n"
        s = f"{indent}{node['feature']} <= {node['threshold']:.2f}?\n"
        s += f"{indent}  yes:\n" + self.to_text(node["yes"], indent + "    ")
        s += f"{indent}  no:\n" + self.to_text(node["no"], indent + "    ")
        return s

    def to_dict(self):
        return {"max_depth": self.max_depth, "min_leaf": self.min_leaf, "root": self.root}

    @classmethod
    def from_dict(cls, d):
        return cls(d["max_depth"], d["min_leaf"], d["root"])


# ---------------------------------------------------------------- training
def load_csv(path):
    """Recording file -> N x 6 array (header row: ax,ay,az,gx,gy,gz)."""
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def parse_name(path):
    """data/<activity>__<person>__<time>.csv -> (activity, person)"""
    from pathlib import Path
    parts = Path(path).stem.split("__")
    return parts[0], parts[1] if len(parts) > 1 else "me"


def make_dataset(paths):
    """Cut every recording into windows. Returns rows of (activity, person, recording, block, window)."""
    rows = []
    for rec, p in enumerate(paths):
        activity, person = parse_name(p)
        if activity not in ACTIVITIES:
            continue
        data = load_csv(p)
        data = data[TRIM:len(data) - TRIM]
        n = len(data)
        for s, w in windows(data):
            if is_dead(w):
                continue                                         # sensor was sending zeros
            block = min(4, int(5 * (s + WINDOW / 2) / max(n, 1)))   # which fifth of the recording
            rows.append((activity, person, rec, block, w))
    return rows


def fit_both(rows, direction=None):
    """Train both models on rows from make_dataset. direction=None -> pick the best one."""
    y = [r[0] for r in rows]
    wins = [r[4] for r in rows]
    if direction is None:
        direction = max(DIRECTIONS, key=lambda d: _peak_cv_score(rows, d))
    m1 = PeakModel(direction).fit([peak_in_direction(w, direction) for w in wins], y)
    m2 = TreeModel().fit([features(w) for w in wins], y)
    return m1, m2


def _peak_cv_score(rows, direction):
    folds = _folds(rows)
    X = np.array([peak_in_direction(r[4], direction) for r in rows])
    y = np.array([r[0] for r in rows])
    right = 0
    for test in folds:
        train = ~test & ~_neighbors(rows, test)
        if len(set(y[train])) < 2:
            continue
        m = PeakModel(direction).fit(X[train], y[train])
        right += sum(p == t for p, t in zip(m.predict(X[test]), y[test]))
    return right


def _folds(rows):
    """Test masks for cross-validation.
    2+ people recorded -> leave one PERSON out (the honest test for your 3 peers).
    Otherwise -> hold out the same fifth of every recording, 5 times."""
    people = sorted({r[1] for r in rows})
    if len(people) >= 2:
        return [np.array([r[1] == p for r in rows]) for p in people]
    return [np.array([r[3] == b for r in rows]) for b in range(5)]


def _neighbors(rows, test):
    """Windows that overlap a test window in time (same recording, next block) leak - drop them."""
    out = np.zeros(len(rows), bool)
    tested = {(r[2], r[3]) for r, t in zip(rows, test) if t}
    for i, r in enumerate(rows):
        if (r[2], r[3] - 1) in tested or (r[2], r[3] + 1) in tested:
            out[i] = True
    return out & ~test


def cross_validate(rows, direction=None):
    """Accuracy of both models on data they were NOT trained on, + confusion matrices."""
    y = np.array([r[0] for r in rows])
    p1, p2, truth = [], [], []
    for test in _folds(rows):
        train = ~test & ~_neighbors(rows, test)
        sub = [r for r, t in zip(rows, train) if t]
        if len(set(y[train])) < 2:
            continue
        m1, m2 = fit_both(sub, direction)
        tw = [r[4] for r, t in zip(rows, test) if t]
        p1 += m1.predict([peak_in_direction(w, m1.direction) for w in tw])
        p2 += m2.predict([features(w) for w in tw])
        truth += list(y[test])
    labels = [a for a in ACTIVITIES if a in set(y)]

    def report(pred):
        cm = [[sum(1 for t, p in zip(truth, pred) if t == a and p == b) for b in labels] for a in labels]
        acc = sum(t == p for t, p in zip(truth, pred)) / max(len(truth), 1)
        return {"accuracy": acc, "confusion": cm}

    people = sorted({r[1] for r in rows})
    return {"labels": labels, "scheme": "leave-one-person-out" if len(people) >= 2 else "held-out fifths",
            "people": people, "windows": len(truth), "peak": report(p1), "tree": report(p2)}


def save(path, m1, m2, cv=None):
    with open(path, "w") as fh:
        json.dump({"peak": m1.to_dict(), "tree": m2.to_dict(), "cv": cv}, fh, indent=1)


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return PeakModel.from_dict(d["peak"]), TreeModel.from_dict(d["tree"]), d.get("cv")
