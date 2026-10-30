# Train the step-counting neural net in PyTorch, test it, and export it for the UNO Q.
# Run from the ArduinoApps folder (after pull_data.py):
#   ~/AIRobotics/.venv/bin/python step_counter/laptop/train_steps.py
#
# What it does:
#   1. Labels every recording with step times (heel strikes from the gyro, 0 for standing/biking)
#      and checks those labels against the recordings where you hand-counted your steps.
#   2. Leave-one-recording-out test: for every recording, train a net WITHOUT it, then count the
#      steps in it exactly like the board would. This is the honest "how good is it" number.
#   3. Trains the final ensemble (5 nets) on everything and saves it as data/steps_model.json.
#   4. Fits meters-per-step from recordings where you typed a distance.
#   5. Saves figures for the write-up in laptop/figures/.
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import gait
import steps as S

DATA = HERE / "data"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
ENSEMBLE = 5
EPOCHS = 40
SUSPECT = 0.15                     # hand count more than 15% away from the sensor's swings -> flagged
TRAIN_HOP = 10                     # training windows every 0.2 s (lots of slightly shifted examples)
torch.set_num_threads(4)


# ---------------------------------------------------------------- the network
class StepNet(nn.Module):
    """2 s of 8 IMU channels -> steps per second. Same layers as StepModel in python/steps.py."""

    def __init__(self, ch=len(S.CHANNELS), h=16):
        super().__init__()
        self.c1 = nn.Conv1d(ch, h, 9, padding=4)    # 16 pattern detectors, each 9 samples (0.18 s) wide
        self.c2 = nn.Conv1d(h, h, 9, padding=4)     # patterns of patterns
        self.c3 = nn.Conv1d(h, h, 9, padding=4)
        self.f1 = nn.Linear(h, h)
        self.f2 = nn.Linear(h, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):                            # x: batch x 8 x 100
        h = self.pool(torch.relu(self.c1(x)))
        h = self.pool(torch.relu(self.c2(h)))
        h = torch.relu(self.c3(h)).mean(dim=2)      # average over time -> 16 numbers
        h = torch.relu(self.f1(h))
        return nn.functional.softplus(self.f2(h)).squeeze(1)   # steps/s, never negative


# ---------------------------------------------------------------- data
def load_recordings():
    recs = []
    for p in sorted(DATA.glob("*.csv")):
        activity, person = gait.parse_name(p)
        data = gait.load_csv(p)
        if gait.is_dead(data):
            continue
        meta = S.load_meta(p)
        st = S.step_times(data, activity)
        counted = meta.get("steps")
        # a hand count far from the leg swings the sensor saw is probably a miscount
        suspect = bool(counted) and abs(len(st) - counted) > max(3, SUSPECT * counted)
        recs.append({"name": p.stem, "activity": activity, "data": data, "steps": st,
                     "labeled": len(st), "counted": counted, "suspect": suspect, "meters": meta.get("meters"),
                     "rate": S.rate_target(st, len(data))})
    # Your hand counts are the truth: scale each counted recording's labels to match your count, and
    # scale the uncounted ones by the average correction (the leg-swing labels run a little high).
    ratios = [r["counted"] / r["labeled"] for r in recs if r["counted"] and not r["suspect"] and r["labeled"]]
    fix = float(np.median(ratios)) if ratios else 1.0
    for r in recs:
        k = r["counted"] / r["labeled"] if (r["counted"] and not r["suspect"] and r["labeled"]) else fix
        r["rate"] = r["rate"] * k
        r["scale"] = k
    print(f"\nLeg-swing labels x {fix:.3f} to match your hand counts (from {len(ratios)} counted recordings)")
    return recs


def windows(rec, hop):
    ch = S.channels(rec["data"])
    X, y = [], []
    for e in range(S.WIN, len(ch) + 1, hop):
        X.append(ch[e - S.WIN:e].T)
        y.append(rec["rate"][e - S.WIN:e].mean())
    return np.array(X, np.float32), np.array(y, np.float32)


def train_net(recs, mean, std, seed, epochs=EPOCHS, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = zip(*(windows(r, TRAIN_HOP) for r in recs))
    X = (np.concatenate(X) - mean[None, :, None]) / std[None, :, None]
    y = np.concatenate(y)
    X, y = torch.tensor(X), torch.tensor(y)
    net = StepNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            xb = X[b] * torch.empty(len(b), 1, 1).uniform_(0.9, 1.1)       # a bit stronger/weaker
            xb = xb + 0.05 * torch.randn_like(xb)                            # a bit of sensor noise
            loss = nn.functional.mse_loss(net(xb), y[b])                     # how wrong, squared
            opt.zero_grad()
            loss.backward()                                                  # back-propagation
            opt.step()                                                       # nudge every weight downhill
            total += loss.item() * len(b)
        sched.step()
        if log is not None:
            log.append(total / len(X))
    return net.eval()


def export(net):
    return {k.replace(".weight", ".w").replace(".bias", ".b"): np.round(v.detach().numpy(), 6).tolist()
            for k, v in net.state_dict().items()}


def count_like_board(model_rates, data):
    """Walk through a recording 0.5 s at a time like main.py does. Returns (count, std, rate trace)."""
    pad_a = np.repeat(data[:1], S.WIN, 0)            # still before and after, so every step
    pad_b = np.repeat(data[-1:], S.WIN, 0)           # gets seen by 4 windows like on the board
    d = np.vstack([pad_a, data, pad_b])
    count, var, trace = 0.0, 0.0, []
    for e in range(S.WIN, len(d) + 1, S.HOP):
        r = model_rates(d[e - S.WIN:e])
        count += r.mean() * S.HOP / S.RATE
        var += (r.std() * S.HOP / S.RATE) ** 2 if len(r) > 1 else 0
        trace.append(r.mean())
    return count, np.sqrt(var), np.array(trace)


def torch_rates(nets, mean, std):
    def f(win):
        x = torch.tensor(((S.channels(win) - mean) / std).T[None])
        with torch.no_grad():
            return np.array([float(n(x)) for n in nets])
    return f


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    recs = load_recordings()
    if not recs:
        sys.exit(f"No recordings in {DATA} - run pull_data.py first")
    allch = np.concatenate([S.channels(r["data"]) for r in recs])
    mean, std = allch.mean(0), allch.std(0) + 1e-6

    print("\n1. Labels (heel strikes x 2) vs your hand counts")
    lab_err = []
    for r in recs:
        c = r["counted"]
        note = ""
        if c:
            if not r["suspect"]:
                lab_err.append((r["labeled"] - c) / c)
            note = f"  you counted {c}  ({100 * (r['labeled'] - c) / c:+.1f}%)" + ("  <- SUSPECT, testing vs sensor" if r["suspect"] else "")
        print(f"   {r['name']:40s} {r['activity']:9s} labeled {r['labeled']:4d}{note}")

    print("\n2. Leave-one-recording-out test (train without it, then count it like the board)")
    results = []
    for i, r in enumerate(recs):
        others = [q for j, q in enumerate(recs) if j != i]
        net = train_net(others, mean, std, seed=i)
        pred, _, trace = count_like_board(torch_rates([net], mean, std), r["data"])
        hand = bool(r["counted"]) and not r["suspect"]
        truth = r["counted"] if hand else round(r["labeled"] * r["scale"])
        results.append({"name": r["name"], "activity": r["activity"], "truth": truth,
                        "hand_counted": hand, "suspect": r["suspect"], "counted": r["counted"], "pred": pred,
                        "seconds": len(r["data"]) / S.RATE, "trace": trace.tolist()})
        print(f"   {r['name']:40s} truth {truth:5.0f}{'*' if hand else ' '}  net {pred:6.1f}  "
              f"error {pred - truth:+6.1f}")
    moving = [q for q in results if q["truth"] > 0]
    pct = np.array([(q["pred"] - q["truth"]) / q["truth"] for q in moving]) if moving else np.array([0.0])
    still = [q for q in results if q["truth"] == 0]
    false_per_min = (np.mean([q["pred"] / q["seconds"] * 60 for q in still]) if still else 0.0)
    summary = {"recordings": len(results), "mean_pct_error": float(np.mean(pct) * 100),
               "mean_abs_pct_error": float(np.mean(np.abs(pct)) * 100), "std_pct_error": float(np.std(pct) * 100),
               "false_steps_per_min_still": float(false_per_min),
               "hand_counted": sum(q["hand_counted"] for q in results),
               "suspect": [q["name"] for q in results if q["suspect"]],
               "hand_only_mean_abs_pct_error": float(np.mean([abs(q["pred"] - q["truth"]) / q["truth"] * 100
                                                    for q in results if q["hand_counted"] and q["truth"] > 0] or [0])),
               "label_vs_hand_pct": [float(e * 100) for e in lab_err]}
    print(f"   walking/running: off by {summary['mean_abs_pct_error']:.1f}% on average "
          f"(bias {summary['mean_pct_error']:+.1f}%, spread {summary['std_pct_error']:.1f}%)")
    print(f"   only your trusted hand counts: off by {summary['hand_only_mean_abs_pct_error']:.1f}% on average")
    print(f"   standing/biking: {false_per_min:.1f} phantom steps per minute")

    print(f"\n3. Final ensemble of {ENSEMBLE} nets on all {len(recs)} recordings")
    losses = []
    nets = []
    for k in range(ENSEMBLE):
        log = []
        nets.append(train_net(recs, mean, std, seed=100 + k, log=log))
        losses.append(log)
    # check the numpy copy on the board gives the same answer as PyTorch
    model = S.StepModel({"mean": mean.tolist(), "std": std.tolist(), "nets": [export(n) for n in nets]})
    win = recs[0]["data"][:S.WIN]
    diff = np.abs(model.rates(win) - torch_rates(nets, mean, std)(win)).max()
    print(f"   numpy vs PyTorch max difference: {diff:.2e}")
    assert diff < 1e-3, "numpy forward pass doesn't match PyTorch"

    print("\n4. Distance (from recordings where you typed meters)")
    rows, dist_rows = [], []
    for r in recs:
        if r["meters"]:
            n = r["counted"] if r["counted"] and not r["suspect"] else r["labeled"]
            dur = len(r["data"]) / S.RATE
            rows.append((n / dur * 60, r["meters"] / n))
            dist_rows.append((r, n, dur))
            print(f"   {r['name']:40s} {r['meters']} m / {n} steps = {r['meters'] / n:.2f} m per step "
                  f"at {n / dur * 60:.0f} steps/min")
    stride = S.StrideLength.fit(rows)
    if len(rows) >= 2:      # leave-one-out distance error
        errs = []
        for i, (r, n, dur) in enumerate(dist_rows):
            sl = S.StrideLength.fit([x for j, x in enumerate(rows) if j != i])
            pred_steps = next(q["pred"] for q in results if q["name"] == r["name"])
            est = pred_steps * sl.meters_per_step(pred_steps / dur * 60)
            errs.append((est - r["meters"]) / r["meters"])
            print(f"   left out {r['name']}: estimated {est:.1f} m vs real {r['meters']} m ({100 * errs[-1]:+.1f}%)")
        stride.err = {"mean_abs_pct": float(np.mean(np.abs(errs)) * 100), "n": len(errs)}
    print(f"   meters per step = {stride.a:.3f} + {stride.b:.5f} x steps/min  (n={stride.n})"
          if stride.n else "   no distance recordings yet -> using 0.70 m per step")

    info = {"trained": time.strftime("%Y-%m-%d %H:%M"), "test": summary, "stride": stride.to_dict(),
            "recordings": [r["name"] for r in recs]}
    out = DATA / "steps_model.json"
    with open(out, "w") as fh:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "nets": [export(n) for n in nets],
                   "stride": stride.to_dict(), "info": info}, fh)
    with open(FIGS / "test_results.json", "w") as fh:
        json.dump({"summary": summary, "results": results, "losses": losses}, fh)
    print(f"\nSaved {out} ({out.stat().st_size // 1024} KB) in {time.time() - t0:.0f} s")
    figures(recs, results, losses, summary, nets, mean, std)


def figures(recs, results, losses, summary, nets, mean, std):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    col = {"standing": "#5b7fa6", "walking": "#3f9b5d", "running": "#d9622b", "biking": "#8a5cc2"}

    # learning curves
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for l in losses:
        ax.plot(range(1, len(l) + 1), l, alpha=0.8)
    ax.set_xlabel("epoch (one pass through all the training windows)")
    ax.set_ylabel("loss (squared error, (steps/s)²)")
    ax.set_yscale("log")
    ax.set_title("Training: each line is one net in the ensemble")
    fig.tight_layout(); fig.savefig(FIGS / "1_learning_curves.png", dpi=160); plt.close(fig)

    # true vs counted
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hi = max(max(q["truth"], q["pred"]) for q in results) * 1.1 + 1
    ax.plot([0, hi], [0, hi], color="#999", lw=1)
    for q in results:
        ax.scatter(q["truth"], q["pred"], s=50, color=col.get(q["activity"], "k"),
                   marker="o" if q["hand_counted"] else "s", edgecolor="k", lw=0.5)
    ax.set_xlabel("real steps (● hand-counted, ■ heel-strike labels)")
    ax.set_ylabel("steps the net counted")
    ax.set_title(f"Each recording, counted by a net that never saw it\n"
                 f"walking/running off by {summary['mean_abs_pct_error']:.1f}% on average")
    fig.tight_layout(); fig.savefig(FIGS / "2_counted_vs_real.png", dpi=160); plt.close(fig)

    # one walk: what the net outputs over time
    r = next((r for r in recs if r["activity"] == "walking"), recs[0])
    q = next(q for q in results if q["name"] == r["name"])
    t = np.arange(len(q["trace"])) * S.HOP / S.RATE - S.WIN / S.RATE
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    tt = np.arange(len(r["data"])) / S.RATE
    axes[0].plot(tt, np.linalg.norm(r["data"][:, 3:], axis=1), color="#666", lw=0.8)
    for s in r["steps"]:
        axes[0].axvline(s / S.RATE, color=col["walking"], lw=0.6, alpha=0.6)
    axes[0].set_ylabel("spin (°/s)")
    axes[0].set_title("A walk: gyro, with the labeled steps as green lines")
    axes[1].plot(t, q["trace"], color="k", label="net's answer (steps/s)")
    axes[1].set_ylabel("steps per second")
    axes[1].set_xlabel("seconds")
    axes[1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "3_example_walk.png", dpi=160); plt.close(fig)
    print("Figures in", FIGS)


if __name__ == "__main__":
    main()
