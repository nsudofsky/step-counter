# Step counter - UNO Q Linux side (deploy: python3 tools/deploy.py step_counter)
#
# Built on the gait trainer: the sketch sends 50 IMU samples a second ("imu"). Every 0.5 s:
#   - the step net (steps.py, trained in PyTorch on the laptop) says how many steps per second
#     you're taking; count += rate x 0.5 s, distance += steps x meters-per-step
#   - the gait models from the last project still say standing / walking / running / biking
# The LED matrix shows the step count. Web page: http://<board-ip>:7000
#
# Recordings (data/*.csv) can get a hand-counted step total and a distance (data/*.json) for
# training and testing on the laptop. The trained net is data/steps_model.json.
from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import re, threading, time, traceback
import numpy as np

import gait
import steps as S

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
MODEL_FILE = DATA / "model.json"
STEP_FILE = DATA / "steps_model.json"

ACC_SCALE = 2048.0       # counts per g at +-16 g (sketch ACCEL_CONFIG)
GYRO_SCALE = 16.4        # counts per deg/s at +-2000 deg/s (sketch GYRO_CONFIG)
SMOOTH = 3               # majority vote over the last 3 predictions (1.5 s)
COUNTDOWN = 5            # seconds to put the phone away before recording starts
ICON = {"standing": 0, "walking": 1, "running": 2, "biking": 3, None: 4}
CHECK_ICON = 5

def matrix(*args):
    """Draw on the LED matrix without ever freezing the app if the MCU is slow to answer."""
    try:
        Bridge.call(*args, timeout=1)
    except Exception as e:
        print("Matrix call failed:", args, e)


lock = threading.Lock()
buf = deque(maxlen=gait.RATE * 10)       # last 10 s of samples
last_sample = 0.0
n_samples = 0
recording = None                         # list of samples while recording
rec_info = {}
state = "live"                           # live / countdown / recording
matrix_model = "steps"                   # what the matrix shows: steps / peak / tree
m1 = m2 = cv = None
hist1, hist2 = deque(maxlen=SMOOTH), deque(maxlen=SMOOTH)

step_net = stride = None
step_mtime = 0
counter = {"steps": 0.0, "var": 0.0, "meters": 0.0, "since": time.time()}


def load_step_net():
    """(Re)load the step net whenever the laptop pushes a new data/steps_model.json."""
    global step_net, stride, step_mtime
    if not STEP_FILE.exists():
        return
    m = STEP_FILE.stat().st_mtime
    if m == step_mtime:
        return
    try:
        import json
        with open(STEP_FILE) as fh:
            d = json.load(fh)
        step_net = S.StepModel(d)
        stride = S.StrideLength.from_dict(d.get("stride", {"a": 0, "b": 0}))
        step_mtime = m
        t = step_net.info.get("test", {})
        print(f"Loaded step net: {len(step_net.nets)} nets, trained {step_net.info.get('trained')}, "
              f"tested off by {t.get('mean_abs_pct_error', 0):.1f}%")
    except Exception as e:
        print("Couldn't load step net:", e)


def reset_counter():
    counter.update(steps=0.0, var=0.0, meters=0.0, since=time.time())


def on_imu(ax, ay, az, gx, gy, gz):
    global last_sample, n_samples
    s = (ax / ACC_SCALE, ay / ACC_SCALE, az / ACC_SCALE, gx / GYRO_SCALE, gy / GYRO_SCALE, gz / GYRO_SCALE)
    with lock:
        buf.append(s)
        if recording is not None:
            recording.append(s)
        last_sample = time.time()
        n_samples += 1


Bridge.provide("imu", on_imu)


def load_models():
    global m1, m2, cv
    if MODEL_FILE.exists():
        m1, m2, cv = gait.load(MODEL_FILE)
        print(f"Loaded models: peak along {m1.direction} {m1.thresholds}, tree uses {sorted(m2.used_features())}")
    else:
        m1 = m2 = cv = None
        print("No model yet - record some data and press Train on the web page")


def recordings():
    out = []
    for p in sorted(DATA.glob("*.csv")):
        activity, person = gait.parse_name(p)
        d = gait.load_csv(p)
        out.append({"file": p.name, "activity": activity, "person": person,
                    "seconds": round(len(d) / gait.RATE), "dead": bool(gait.is_dead(d)),
                    **{k: v for k, v in S.load_meta(p).items() if k in ("steps", "meters")}})
    return out


def model_info():
    if m1 is None:
        return None
    return {"peak": m1.to_dict(), "tree": m2.to_dict(), "cv": cv}


def step_info():
    if step_net is None:
        return None
    return {"nets": len(step_net.nets), **step_net.info, "stride": stride.to_dict()}


def send_status(sid=None):
    ui.send_message("status", {"models": model_info(), "step_net": step_info(), "recordings": recordings(),
                               "matrix_model": matrix_model, "state": state,
                               "activities": gait.ACTIVITIES})


# ---------------------------------------------------------------- web page
ui = WebUI()


def on_hello(sid, data):
    send_status(sid)


def on_record(sid, data):
    global state
    if state != "live":
        return
    activity = data.get("activity")
    person = re.sub(r"[^a-z0-9]", "", str(data.get("person", "me")).lower()) or "me"
    seconds = int(data.get("seconds") or 0)
    seconds = 600 if seconds <= 0 else max(10, min(600, seconds))     # 0 = until you press Stop
    if activity not in gait.ACTIVITIES:
        return
    state = "countdown"
    threading.Thread(target=record_thread, args=(activity, person, seconds), daemon=True).start()


def record_thread(activity, person, seconds):
    global state, recording
    for n in range(COUNTDOWN, 0, -1):
        ui.send_message("record", {"phase": "countdown", "left": n})
        matrix("number", n)
        time.sleep(1)
        if state != "countdown":         # cancelled
            return finish_record(None)
    with lock:
        recording = []
    state = "recording"
    matrix("show", ICON[activity])
    t0 = time.time()
    while state == "recording" and time.time() - t0 < seconds:
        ui.send_message("record", {"phase": "recording", "left": round(seconds - (time.time() - t0)),
                                   "elapsed": round(time.time() - t0), "samples": len(recording)})
        time.sleep(0.5)
    with lock:
        samples, recording = recording, None
    finish_record(samples if state == "recording" or len(samples) > 10 * gait.RATE else None,
                  activity, person)


def finish_record(samples, activity=None, person=None):
    global state
    name = None
    dead_msg = None
    if samples and gait.is_dead(np.array(samples)):
        print("Not saving: the sensor sent zeros (loose wire?)")
        dead_msg = "Not saved: the sensor was sending zeros. Check the wires and record again."
        samples = None
    if samples:
        name = f"{activity}__{person}__{datetime.now():%Y%m%d-%H%M%S}.csv"
        np.savetxt(DATA / name, np.array(samples), delimiter=",", fmt="%.4f",
                   header="ax,ay,az,gx,gy,gz", comments="")
        print(f"Saved {name} ({len(samples)} samples)")
        matrix("show", CHECK_ICON)
    state = "live"
    hist1.clear()
    hist2.clear()
    ui.send_message("record", {"phase": "done", "file": name, "error": None if name else dead_msg})
    send_status()


def on_stop(sid, data):
    global state
    if state == "countdown":
        state = "cancel"
    elif state == "recording":
        state = "stopping"


def on_delete(sid, data):
    name = Path(str(data.get("file", ""))).name
    p = DATA / name
    if p.suffix == ".csv" and p.exists():
        p.unlink()
        S.meta_path(p).unlink(missing_ok=True)
        print("Deleted", name)
    send_status()


def on_label(sid, data):
    """After a recording: how many steps you counted and how far you went (either can be blank)."""
    import json
    p = DATA / Path(str(data.get("file", ""))).name
    if p.suffix != ".csv" or not p.exists():
        return
    meta = S.load_meta(p)
    for key, cast in (("steps", int), ("meters", float)):
        v = data.get(key)
        if v not in (None, ""):
            meta[key] = cast(float(v))
    with open(S.meta_path(p), "w") as fh:
        json.dump(meta, fh)
    print(f"Labeled {p.name}: {meta}")
    send_status()


def on_reset(sid, data):
    reset_counter()
    send_status()


def on_train(sid, data):
    threading.Thread(target=train, args=(data.get("direction"),), daemon=True).start()


def train(direction):
    global m1, m2, cv
    if direction not in gait.DIRECTIONS:
        direction = None
    ui.send_message("train", {"phase": "training"})
    try:
        rows = gait.make_dataset(sorted(DATA.glob("*.csv")))
        if len({r[0] for r in rows}) < 2:
            raise ValueError("Record at least 2 activities first")
        t0 = time.time()
        new1, new2 = gait.fit_both(rows, direction)
        new_cv = gait.cross_validate(rows, new1.direction)   # same direction as the final model
        gait.save(MODEL_FILE, new1, new2, new_cv)
        m1, m2, cv = new1, new2, new_cv
        print(f"Trained on {len(rows)} windows in {time.time() - t0:.1f} s: "
              f"peak {cv['peak']['accuracy']:.0%}, tree {cv['tree']['accuracy']:.0%}")
        print(m2.to_text())
        ui.send_message("train", {"phase": "done"})
    except Exception as e:
        print("Training failed:", e)
        ui.send_message("train", {"phase": "error", "error": str(e)})
    send_status()


dataset_cache = {"key": None, "msg": None}
EXPLAIN_EXTRA = {"acc_jerk", "acc_mag_mean", "gyro_mag_mean", "cadence"}   # extra scatter axes on the explain page


def on_dataset(sid, data):
    """For the explain page: every training window's numbers, so it can plot what the models learned."""
    threading.Thread(target=send_dataset, daemon=True).start()


def send_dataset():
    try:
        _send_dataset()
    except Exception as e:
        print("Dataset failed (will retry next time):", e)


def _send_dataset():
    paths = sorted(DATA.glob("*.csv"))
    key = (tuple(p.name for p in paths), MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else 0)
    if dataset_cache["key"] != key:
        rows = gait.make_dataset(paths)
        direction = m1.direction if m1 else "+x"
        names = sorted(m2.used_features() | EXPLAIN_EXTRA) if m2 else []
        windows = []
        for activity, person, rec, block, w in rows:
            f = gait.features(w)
            windows.append({"a": activity, "p": person, "r": rec, "peak": round(gait.peak_in_direction(w, direction), 3),
                            "f": {k: round(f[k], 3) for k in names}})
        examples = {}                     # one 2.5 s snippet per activity, from the middle of a recording
        for path in paths:
            activity, _ = gait.parse_name(path)
            if activity in examples or activity not in gait.ACTIVITIES:
                continue
            d = gait.load_csv(path)
            if gait.is_dead(d):
                continue
            mid = len(d) // 2
            w = d[mid - gait.WINDOW // 2: mid + gait.WINDOW // 2]
            axis, sign = "xyz".index(direction[1]), 1 if direction[0] == "+" else -1
            examples[activity] = {"axis": np.round(sign * w[:, axis], 2).tolist(),
                                  "mag": np.round(np.linalg.norm(w[:, :3], axis=1), 2).tolist()}
        dataset_cache.update(key=key, msg={"windows": windows, "examples": examples,
                                           "files": [p.name for p in paths],
                                           "describe": {k: gait.describe(k) for k in names}})
    ui.send_message("dataset", dataset_cache["msg"])


def on_matrix_model(sid, data):
    global matrix_model
    if data.get("model") in ("peak", "tree", "steps"):
        matrix_model = data["model"]
        hist1.clear()
        hist2.clear()
    send_status()


ui.on_message("hello", on_hello)
ui.on_message("record", on_record)
ui.on_message("stop", on_stop)
ui.on_message("delete", on_delete)
ui.on_message("train", on_train)
ui.on_message("matrix_model", on_matrix_model)
ui.on_message("dataset", on_dataset)
ui.on_message("label", on_label)
ui.on_message("reset", on_reset)

load_models()
load_step_net()


# ---------------------------------------------------------------- live loop
def vote(hist, label):
    hist.append(label)
    return Counter(hist).most_common(1)[0][0]


def count_steps(win):
    """Ask every net in the ensemble for steps/s, add 0.5 s worth to the count."""
    r = step_net.rates(win)
    dt = S.HOP / S.RATE
    rate, spread = float(r.mean()), float(r.std())
    counter["steps"] += rate * dt
    counter["var"] += (spread * dt) ** 2                      # the nets disagreeing
    counter["meters"] += rate * dt * stride.meters_per_step(rate * 60)
    test = step_net.info.get("test", {}).get("mean_abs_pct_error", 10.0) / 100
    n = counter["steps"]
    unc = float(np.sqrt(counter["var"] + (test * n) ** 2))   # + how far off it was in testing
    d_err = (stride.err or {}).get("mean_abs_pct", 15.0) / 100 if stride.n else 0.25
    return {"count": round(n, 1), "unc": round(unc, 1), "rate": round(rate, 2), "spread": round(spread, 2),
            "per_net": np.round(r, 2).tolist(), "cadence": round(rate * 60),
            "meters": round(counter["meters"], 1), "meters_unc": round(counter["meters"] * d_err, 1),
            "m_per_step": round(stride.meters_per_step(rate * 60), 2), "calibrated": stride.n > 0,
            "since": counter["since"]}


next_tick = 0.0
shown = None
imu_was_ok = None
last_rate_check, last_count = time.time(), 0


def loop():
    try:
        tick()
    except Exception:
        print("Live loop error (keeps running):")
        traceback.print_exc()
        time.sleep(0.5)


def tick():
    global next_tick, shown, last_rate_check, last_count, imu_was_ok
    now = time.time()
    if now < next_tick:
        time.sleep(0.02)
        return
    next_tick = now + gait.STEP / gait.RATE

    with lock:
        win = np.array(list(buf)[-gait.WINDOW:]) if len(buf) >= gait.WINDOW else None
        age = now - last_sample
        rate = (n_samples - last_count) / max(now - last_rate_check, 1e-3)
        last_rate_check, last_count = now, n_samples

    dead = win is not None and gait.is_dead(win)
    msg = {"state": state, "rate": round(rate), "imu": age < 2 and not dead, "dead": dead}
    if msg["imu"] != imu_was_ok:
        print(f"IMU streaming at {rate:.0f} Hz" if msg["imu"] else
              "IMU sending zeros (MPU-6050 reset itself - loose wire?)" if dead else
              "No IMU data (X on the matrix = MPU-6050 not answering, check wiring)")
        imu_was_ok = msg["imu"]
    if int(now) % 5 == 0:
        load_step_net()                   # picks up a newly pushed net without restarting
    if win is not None and age < 2 and not dead and step_net is not None:
        msg["steps"] = count_steps(win[-S.WIN:])
    if win is not None and age < 2 and not dead:
        f = gait.features(win)
        msg["cadence"] = round(f["cadence"])
        msg["trace_mag"] = np.round(np.linalg.norm(win[:, :3], axis=1)[::2], 2).tolist()
        if m1 is not None:
            peak = gait.peak_in_direction(win, m1.direction)
            label1 = vote(hist1, m1.predict_value(peak))
            label2_raw, path = m2.predict_one(f)
            label2 = vote(hist2, label2_raw)
            axis = "xyz".index(m1.direction[1])
            sign = 1 if m1.direction[0] == "+" else -1
            msg.update({"peak": round(peak, 2), "peak_label": label1,
                        "tree_label": label2, "tree_path": path,
                        "tree_values": {k: round(f[k], 2) for k in m2.used_features() | EXPLAIN_EXTRA},
                        "trace": np.round(sign * win[:, axis][::2], 2).tolist()})
            if state == "live" and matrix_model != "steps":
                want = ICON[label1 if matrix_model == "peak" else label2]
                if want != shown:
                    matrix("show", want)
                    shown = want
    elif state == "live" and shown != ICON[None]:
        matrix("show", ICON[None])
        shown = ICON[None]
    if state == "live" and matrix_model == "steps" and "steps" in msg:
        want = ("n", int(round(msg["steps"]["count"])) % 1000)
        if want != shown:
            matrix("number", want[1])
            shown = want
    if state != "live":
        shown = None                      # redraw after recording
    ui.send_message("live", msg)


App.run(user_loop=loop)
