# Pictures of both models for the write-up, from the recordings in laptop/data/
# (get them with pull_data.py first). Saves PNGs in laptop/figures/. Run from the ArduinoApps folder:
#   ~/AIRobotics/.venv/bin/python gait_trainer/laptop/analyze.py
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import gait

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
COLORS = {"standing": "#5b7fa6", "walking": "#3f9b5d", "running": "#d9622b", "biking": "#8a5cc2"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

paths = sorted(DATA.glob("*.csv"))
if not paths:
    sys.exit(f"No recordings in {DATA} - run pull_data.py first")
rows = gait.make_dataset(paths)
acts = [a for a in gait.ACTIVITIES if any(r[0] == a for r in rows)]
y = np.array([r[0] for r in rows])
print(f"{len(paths)} recordings -> {len(rows)} windows: " +
      ", ".join(f"{a} {np.sum(y == a)}" for a in acts))

m1, m2 = gait.fit_both(rows)
cv = gait.cross_validate(rows, m1.direction)
print(f"Model 1 (peak along {m1.direction}): {cv['peak']['accuracy']:.0%}   "
      f"Model 2 (tree): {cv['tree']['accuracy']:.0%}   [{cv['scheme']}]")
print(m2.to_text())
(HERE / "data").mkdir(exist_ok=True)
gait.save(HERE / "data" / "model_laptop.json", m1, m2, cv)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)
    print("wrote", FIGS / name)


# 1. What the raw IMU looks like for each activity (5 s from the middle of the first recording)
fig, axes = plt.subplots(len(acts), 2, figsize=(11, 2.2 * len(acts)), sharex=True, squeeze=False)
for i, a in enumerate(acts):
    p = next(p for p in paths if gait.parse_name(p)[0] == a)
    d = gait.load_csv(p)
    mid = len(d) // 2
    d = d[mid - 125:mid + 125]
    t = np.arange(len(d)) / gait.RATE
    for k, lab in enumerate("xyz"):
        axes[i, 0].plot(t, d[:, k], label=f"a{lab}", lw=1.2)
        axes[i, 1].plot(t, d[:, 3 + k], label=f"g{lab}", lw=1.2)
    axes[i, 0].set_ylabel(f"{a}\naccel (g)", color=COLORS[a])
    axes[i, 1].set_ylabel("gyro (°/s)")
axes[0, 0].legend(ncol=3, fontsize=9, loc="upper right")
axes[0, 1].legend(ncol=3, fontsize=9, loc="upper right")
axes[-1, 0].set_xlabel("seconds")
axes[-1, 1].set_xlabel("seconds")
fig.suptitle("Raw MPU-6050 on the ankle")
save(fig, "1_raw_signals.png")

# 2. Model 1: the number line (each dot = one 2.5 s window) + thresholds
peaks = np.array([gait.peak_in_direction(r[4], m1.direction) for r in rows])
fig, ax = plt.subplots(figsize=(10, 3.6))
rng = np.random.default_rng(0)
for i, a in enumerate(m1.order):
    v = peaks[y == a]
    ax.scatter(v, i + rng.uniform(-0.25, 0.25, len(v)), s=10, alpha=0.5, color=COLORS[a])
for t in m1.thresholds:
    ax.axvline(t, color="k", ls="--", lw=1)
    ax.text(t, -0.75, f"{t:.2f} g", ha="center", fontsize=10,
            bbox=dict(fc="white", ec="none", pad=1))
ax.set_yticks(range(len(m1.order)), m1.order)
ax.set_ylim(-1, len(m1.order) - 0.5)
ax.set_xlabel(f"peak acceleration along {m1.direction} in a 2.5 s window (g)")
ax.set_title(f"Model 1: one number, three thresholds  ({cv['peak']['accuracy']:.0%} on held-out data)")
save(fig, "2_model1_number_line.png")

# 3. Model 1: why this direction? held-out accuracy for every direction
scores = {}
for d in gait.DIRECTIONS:
    scores[d] = gait.cross_validate(rows, d)["peak"]["accuracy"]
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(list(scores), [100 * s for s in scores.values()],
       color=["#1d1d1b" if d == m1.direction else "#bbbbb5" for d in scores])
ax.set_ylabel("held-out accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Model 1: which direction to watch")
save(fig, "3_model1_directions.png")
print("direction accuracies:", {d: f"{s:.0%}" for d, s in scores.items()})


# 4. Model 2: the decision tree as a flowchart
def layout(node, depth=0, pos=None, x=[0]):
    """Leaves get x = 0, 1, 2...; a question sits above the middle of its children."""
    pos = {} if pos is None else pos
    if "leaf" in node:
        pos[id(node)] = (x[0], -depth)
        x[0] += 1
    else:
        layout(node["yes"], depth + 1, pos, x)
        layout(node["no"], depth + 1, pos, x)
        pos[id(node)] = ((pos[id(node["yes"])][0] + pos[id(node["no"])][0]) / 2, -depth)
    return pos


leaves = [0]
pos = layout(m2.root, x=leaves)
fig, ax = plt.subplots(figsize=(max(8, 2.2 * leaves[0]), 5))


def draw(node):
    x0, y0 = pos[id(node)]
    if "leaf" in node:
        n = sum(node["counts"].values())
        ax.text(x0, y0, f"{node['leaf']}\n{node['counts'].get(node['leaf'], 0)}/{n} windows", ha="center",
                va="center", color="white", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc=COLORS.get(node["leaf"], "gray"), ec="none"))
        return
    for ans in ("yes", "no"):
        x1, y1 = pos[id(node[ans])]
        ax.plot([x0, x1], [y0, y1], color="#999", lw=1, zorder=0)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, ans, fontsize=9, color="#666", ha="center",
                bbox=dict(fc="white", ec="none", pad=1))
        draw(node[ans])
    ax.text(x0, y0, f"{gait.describe(node['feature'])}\n≤ {node['threshold']:.2f} ?", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#333"))


draw(m2.root)
ax.axis("off")
ax.set_title(f"Model 2: decision tree on all IMU channels  ({cv['tree']['accuracy']:.0%} on held-out data)")
save(fig, "4_model2_tree.png")

# 5. Both confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, key, title in ((axes[0], "peak", f"Model 1 (peak {m1.direction})"), (axes[1], "tree", "Model 2 (tree)")):
    cm = np.array(cv[key]["confusion"])
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Greens", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)
    ax.set_xticks(range(len(cv["labels"])), cv["labels"], rotation=30)
    ax.set_yticks(range(len(cv["labels"])), cv["labels"])
    ax.set_xlabel("model said")
    ax.set_ylabel("really was")
    ax.set_title(f"{title}: {cv[key]['accuracy']:.0%}")
fig.suptitle(f"Tested on windows the model never trained on ({cv['scheme']})")
save(fig, "5_confusion.png")

# 6. Model 2's first two questions as a 2-D map
used = []
stack = [m2.root]
while stack and len(used) < 2:
    n = stack.pop(0)
    if "leaf" not in n:
        if n["feature"] not in used:
            used.append(n["feature"])
        stack += [n["yes"], n["no"]]
if len(used) == 2:
    F = [gait.features(r[4]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    for a in acts:
        pts = np.array([[f[used[0]], f[used[1]]] for f, l in zip(F, y) if l == a])
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, color=COLORS[a], label=a)
    ax.set_xlabel(gait.describe(used[0]))
    ax.set_ylabel(gait.describe(used[1]))
    ax.legend()
    ax.set_title("Model 2's top two questions: each dot is one 2.5 s window")
    save(fig, "6_model2_top_features.png")
