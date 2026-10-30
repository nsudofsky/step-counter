# Send the trained step net (laptop/data/steps_model.json) to the UNO Q. The app picks it up within
# a few seconds, no restart needed. Run from the ArduinoApps folder:
#   python3 step_counter/laptop/push_model.py           just the net
#   python3 step_counter/laptop/push_model.py --all     also every recording + label (first time setup)
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
from deploy import find_board, remote, REMOTE_DIR, SSH_OPTS

data = HERE / "data"
files = [data / "steps_model.json"]
if "--all" in sys.argv:
    files += sorted(data.glob("*.csv")) + sorted(p for p in data.glob("*.json") if p.name != "steps_model.json")
files = [str(f) for f in files if f.exists()]
dest = f"{REMOTE_DIR}/step_counter/data"

board = find_board()
remote(board, f"mkdir -p {dest}")
kind, target = board
if kind == "adb":
    for f in files:
        subprocess.run(["adb", "push", f, f"{dest}/"], check=True, capture_output=True)
else:
    subprocess.run(["scp", *SSH_OPTS, "-q", *files, f"{target}:{dest}/"], check=True)
print(f"Sent {len(files)} file(s) to {dest}")
