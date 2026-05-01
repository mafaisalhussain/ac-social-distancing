# Social Distancing Detector — Jetson Nano

A real-time social distancing violation detector running on **NVIDIA Jetson Nano 4GB** using **poseNet** from Dusty's `jetson-inference` library. The system detects people through a live camera feed, estimates their body pose, computes the distance between them, and flags violations with an **Assassin's Creed Eagle Vision** themed HUD.

---

## Demo

> *(Add your screenshot here)*

---

## How It Works

1. Camera feed is captured using `jetson.utils.videoSource`
2. **poseNet (resnet18-body)** detects body keypoints for every person in the frame
3. The **midpoint between each person's left hip and right hip** keypoints is used as their location
4. **Euclidean distance** is calculated between every pair of people:

```
distance = sqrt((x1 - x2)² + (y1 - y2)²)
```

5. If the distance is below the threshold → **violation flagged**
6. Violating pairs are highlighted in **red** with a dashed line and pixel distance label
7. Safe persons are highlighted in **green**

---

## Features

- Real-time multi-person pose estimation using poseNet (TensorRT accelerated)
- Hip-center based person localization
- Euclidean distance computation between all detected pairs
- Color-coded skeletons — green = safe, red = violation
- Dashed red line with pixel distance between violating pairs
- Flashing BREACH DETECTED alert banner
- Live event log with timestamps
- Subject analysis panel
- FPS counter and session timer
- Screenshot capture with `S` key
- Eagle Vision dark tint toggle with `E` key

---

## Hardware & Software

| Component | Details |
|---|---|
| Board | NVIDIA Jetson Nano 4GB |
| Camera | Logitech C270 HD Webcam |
| OS | Ubuntu 18.04 (JetPack) |
| Python | 3.8 |
| Inference | jetson-inference (Dusty's repo) |
| Model | poseNet resnet18-body |
| UI | OpenCV |

---

## Installation

### 1. Install jetson-inference

```bash
cd ~/Desktop
git clone --depth=1 https://github.com/dusty-nv/jetson-inference
cd jetson-inference
git submodule update --init --depth=1
mkdir build && cd build
cmake ../
make -j4
sudo make install
sudo ldconfig
```

During `cmake`, select **PyTorch 2.0** when prompted. When the model downloader opens, select **resnet18-body** under Pose Estimation.

### 2. Clone this repo

```bash
cd ~/Desktop
git clone https://github.com/YOUR_USERNAME/ac-social-distancing.git
cd ac-social-distancing
```

### 3. Run

```bash
python3 main.py
```

---

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `E` | Toggle Eagle Vision tint on/off |
| `S` | Save screenshot to current folder |
| `R` | Clear event log |

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Camera device index |
| `DISTANCE_THRESHOLD` | `350` | Pixel distance that triggers a violation |
| `KEYPOINT_THRESHOLD` | `0.15` | Minimum keypoint confidence |
| `POSENET_MODEL` | `resnet18-body` | Pose model to use |

---

## Project Structure

```
ac-social-distancing/
├── main.py        — Entry point, camera loop, key controls
├── distance.py    — Hip center extraction + Euclidean distance checks
├── overlay.py     — Assassin's Creed Eagle Vision HUD (OpenCV)
├── config.py      — All settings and constants
└── README.md      — This file
```

---

## Results

| Metric | Value |
|---|---|
| Model | poseNet resnet18-body (TensorRT FP16) |
| FPS on Jetson Nano | ~15–20 FPS |
| Distance metric | Euclidean distance in pixel space |
| Violation threshold | 350px (configurable) |
| Center point | Midpoint of left hip + right hip keypoints |

---

## UI Theme

The HUD is inspired by **Eagle Vision** from the Assassin's Creed franchise:

- 🟢 **Green skeleton** — safe distance maintained
- 🔴 **Red skeleton** — too close, violation flagged
- **Dashed red line** — connects the violating pair with distance in pixels
- **Gold panels** — subject list, event log, session stats

---

## License

MIT License — free to use, modify, and distribute.
