# Step Counter

The Gait Trainer's ankle pouch (UNO Q + MPU-6050), extended to **count steps with a neural net**
and **estimate how far you walked**. The gait models from the last project still run alongside.

This repo is the `step_counter` app from my UNO Q apps folder (`~/AIRobotics/ArduinoApps/step_counter`),
which is where `deploy.py` and the laptop scripts run from. `laptop/animation/` is the Claude-made
animation of the whole system (open `index.html`).

## How it works

- **Input:** the last 2 s of every IMU channel: ax ay az, gx gy gz, plus total acceleration and
  total spin (8 channels × 100 samples).
- **Network:** a small 1-D convolutional net (3 conv layers → average over time → 2 dense layers)
  outputs one number: **steps per second right now**.
- **Training:** PyTorch on the laptop (`laptop/train_steps.py`).
- **Running on the board:** the weights are exported to JSON, and the board runs the same math in
  numpy (`python/steps.py`), so no PyTorch on the UNO Q. The script checks that both give the same answer.
- **Counting:** every 0.5 s, count += rate × 0.5 s.
- **Uncertainty:** 5 nets are trained separately (an ensemble). The ± combines how much they disagree
  with how far off the count was in testing.
- **Labels:** the sensor is on one ankle, so every leg swing (a big gyro peak) is a stride = 2 steps,
  and the other foot lands halfway between. Standing and biking are labeled 0 steps. The net has to
  learn that pedaling isn't walking. Recordings where you **hand-count** your steps check the labels
  and test the net.
- **Distance:** steps × meters per step. Meters per step is fitted from walks over a measured
  distance, and longer steps go with a faster pace.

## Workflow

```
cd ~/AIRobotics/ArduinoApps
python3 tools/deploy.py step_counter
python3 step_counter/laptop/push_model.py --all
```

The second line sends your gait recordings and the first net to the board, and is only needed once.
Open `http://<board-ip>:7000` and record walks and runs. Press Stop, then type the steps you counted
(and the meters, if you measured the path).

```
python3 step_counter/laptop/pull_data.py
~/AIRobotics/.venv/bin/python step_counter/laptop/train_steps.py
python3 step_counter/laptop/push_model.py
```

Training prints the leave-one-recording-out test (every recording counted by a net that never saw it),
saves figures to `laptop/figures/`, and the board picks up the new net within seconds.

## Files

- `python/steps.py`: labels, the numpy network, meters-per-step model (shared by board + laptop)
- `python/main.py`: live counting, recording, labeling, web page messages
- `python/gait.py`: the gait trainer's models (unchanged)
- `laptop/train_steps.py`: PyTorch training, testing, export, figures
- `assets/index.html`: the step counter page; `explain.html` / `gait.html` are the gait trainer's pages
- `sketch/sketch.ino`: reads the MPU-6050 at 50 Hz and shows the count (up to 999) on the LED matrix
