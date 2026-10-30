# Copy the recordings, labels + models from the UNO Q to laptop/data/ (run from the ArduinoApps folder):
#   python3 step_counter/laptop/pull_data.py
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
from deploy import find_board, REMOTE_DIR, SSH_OPTS   # same USB-then-Wi-Fi lookup as deploy.py

remote = f"{REMOTE_DIR}/step_counter/data"
local = HERE / "data"
local.mkdir(exist_ok=True)

kind, target = find_board()
if kind == "adb":
    subprocess.run(["adb", "pull", f"{remote}/.", str(local)], check=True)
else:
    subprocess.run(["scp", *SSH_OPTS, "-r", "-q", f"{target}:{remote}/.", str(local)], check=True)
print(f"Now in {local}:", ", ".join(sorted(p.name for p in local.iterdir())))
