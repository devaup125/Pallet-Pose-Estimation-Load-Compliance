# Pallet Pose Estimation & Load Compliance

Computer vision pipeline for warehouse pallet pose estimation and load compliance checking — YOLOv8s detection, PnP pose estimation, SOP-PAL-03 checks. Built for Delhivery AI/ML assignment.

---

## Overview

A warehouse staging area holds loaded pallets in numbered floor slots. A camera mounted at 1.2m height, tilted 20 degrees down, captures images. This pipeline:

1. **Detects** pallets using YOLOv8s (CNN-based object detector)
2. **Estimates metric pose** (x, y, theta) using PnP with camera calibration
3. **Checks load compliance** against 8 SOP-PAL-03 rules
4. **Outputs** a structured JSON assessment per pallet

No fiducial markers (ArUco, AprilTags) are used — pose comes from the pallet's own geometry via Perspective-n-Point.

---

## Pipeline

```
Image → Detection (YOLOv8s) → Pose Estimation (PnP IPPE) → SOP Checks → JSON Output
         403ms                  7ms                        0.5ms
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/DevanshuRanjanUpadhyay/Pallet-Pose-Estimation-Load-Compliance.git
cd Pallet-Pose-Estimation-Load-Compliance
pip install -r requirements.txt
```

### 1. Train the Detector (Google Colab Recommended)

Open `notebooks/train_yolov8_colab.ipynb` in Google Colab and run all cells. This downloads the pallet dataset from Roboflow, trains YOLOv8s on a Tesla T4 GPU, exports to ONNX, and downloads `best.pt` and `best.onnx` to your machine.

Place the downloaded weights in `models/best.pt` and `models/best.onnx`.

### 2. Calibrate the Camera

Print a checkerboard pattern (7x4 inner corners, 35mm squares), take 15-30 photos at varied angles, and save them to `calibration/images/`.

```bash
python src/pose/calibration.py
```

Update `config/default.yaml` with your checkerboard parameters:
```yaml
camera:
  checkerboard_cols: 7
  checkerboard_rows: 4
  square_size_mm: 35.0
```

### 3. Run the Full Pipeline

```bash
python src/pipeline.py --image test_pallet.jpg
```

### 4. Evaluate

```bash
# Detection evaluation
python src/detection/evaluate.py

# Pose evaluation (synthetic ground truth)
python src/pose/evaluate_pose.py

# Deployment benchmark
python src/deployment/benchmark.py
```

---

## Project Structure

```
Pallet-Pose-Estimation-Load-Compliance/
├── README.md                          # This file
├── DATASET.md                         # Dataset documentation
├── Pallet_Pose_Solution.pdf           # Full technical solution (8 pages)
├── Pallet_Pose_Solution.docx          # Same solution in Word format
├── requirements.txt                   # Python dependencies
├── .gitignore
├── config/
│   └── default.yaml                   # All parameters in one place
├── src/
│   ├── pipeline.py                    # End-to-end pipeline (run this)
│   ├── detection/                     # Section 1: Dataset & Detection (30%)
│   │   ├── predict.py                 # YOLOv8 inference + corner estimation
│   │   ├── prepare_dataset.py         # Dataset merge + split
│   │   └── evaluate.py               # Detection accuracy evaluation
│   ├── pose/                          # Section 2: Pose Estimation (35%)
│   │   ├── calibration.py            # Camera calibration (checkerboard)
│   │   ├── pnp_pose.py               # PnP pose solver + floor transform
│   │   └── evaluate_pose.py          # Synthetic GT eval + error distributions
│   ├── load_analysis/                # Section 3: Load Analysis (25%)
│   │   └── sop_checks.py             # 8 SOP-PAL-03 checks with triage
│   └── deployment/                   # Section 4: Deployment (10%)
│       └── benchmark.py              # Latency, ONNX, failure analysis
├── notebooks/
│   └── train_yolov8_colab.ipynb      # Colab training notebook
├── models/                           # Trained weights (not in git)
│   ├── best.pt                       # YOLOv8s weights (21.5 MB)
│   └── best.onnx                     # ONNX export (42.7 MB)
├── calibration/                      # Camera intrinsics
│   ├── camera_matrix.npy             # 3x3 intrinsics
│   ├── distortion.npy                # Distortion coefficients
│   ├── calibration_report.png        # Visual summary
│   └── images/                       # Checkerboard photos (not in git)
├── results/                          # Evaluation outputs
│   ├── detection/
│   │   └── det_test_pallet.jpg       # Annotated detection image
│   ├── pose/
│   │   ├── pose_error_distributions.png
│   │   ├── sensitivity_analysis.png
│   │   └── usable_envelope.png
│   ├── pipeline/
│   │   └── assessment_test_pallet.json
│   └── deployment/
│       └── deployment_report.json
└── docs/
    └── labelling_guideline.md        # One-page labelling guide
```

---

## Results

### Camera Calibration

| Metric | Value | Target |
|--------|-------|--------|
| Reprojection error | **0.4376 pixels** | < 0.5 px |
| Images used | 41 | 15+ |
| Checkerboard | 7x4, 35mm | - |

### Pose Estimation (Synthetic Evaluation, 200 poses)

| Metric | Mean | Median | P90 | P95 | % within target |
|--------|------|--------|-----|-----|-----------------|
| Translation (cm) | 3.97 | 2.41 | 9.41 | 11.82 | 40.1% within +/-2cm |
| Rotation (deg) | 0.39 | 0.27 | 0.85 | 1.12 | **100% within +/-3 deg** |

### Deployment Benchmark

| Stage | Mean | P50 | P95 |
|-------|------|-----|-----|
| Detection | 403.2ms | 331.9ms | 384.1ms |
| Pose | 7.3ms | 8.0ms | - |
| SOP | 0.5ms | 0.0ms | - |
| **Total** | **411.0ms** | **339.0ms** | **390.7ms** |

- **FPS: 2.4** on Intel i7-6700HQ (CPU only)
- **ONNX speedup: 1.87x** (248ms vs 465ms)
- **Estimated on Jetson Orin Nano: 14-20 FPS** (reasoning, not measured)

### SOP Checks (Sample Output)

| Check | Verdict | Confidence |
|-------|---------|------------|
| overhang | fail | 0.47 |
| load_height | pass | 0.33 |
| column_alignment | manual_inspection | 0.30 |
| size_inversion | manual_inspection | 0.20 |
| stretch_wrap | manual_inspection | 0.30 |
| box_damage | manual_inspection | 0.00 |
| load_centred | pass | 0.47 |
| pallet_damage | manual_inspection | 0.00 |

**Overall: MANUAL_INSPECTION (confidence: 0.50)**

---

## Output Format

Each pallet assessment is a JSON object:

```json
{
  "pallet_id": "pallet_0",
  "pose": {
    "x_m": 12.195,
    "y_m": -5.231,
    "theta_deg": -58.5,
    "uncertainty": {"x_cm": 11.4, "y_cm": 10.7, "theta_deg": 18.1},
    "reproj_error_px": 0.52
  },
  "sop_checks": [
    {"check": "overhang", "verdict": "fail", "confidence": 0.47},
    {"check": "load_height", "verdict": "pass", "confidence": 0.33}
  ],
  "overall_verdict": "MANUAL_INSPECTION",
  "overall_confidence": 0.50,
  "detection_confidence": 0.56
}
```

---

## Key Technical Decisions

### Why YOLOv8s (not Vision Transformer)?

YOLOv8s is a CNN-based single-stage detector optimized for real-time inference. It's fast, easy to train and export (ONNX, TensorRT), and supports keypoint detection. Vision Transformers are typically slower at inference and harder to deploy on edge hardware like Jetson.

### Why SOLVEPNP_IPPE?

The pallet top surface is a flat plane. IPPE (Infinitesimal Plane-Based Pose Estimation) is specifically designed for planar targets — it's more accurate than generic PnP for flat objects and returns two candidate solutions (we pick the one with lower reprojection error).

### Why Synthetic Ground Truth for Evaluation?

No physical pallet pose measurements were available. Synthetic evaluation generates known 3D poses, projects corners to 2D with realistic noise, runs PnP, and compares to the known truth — allowing exact error measurement.

### Bbox-to-Corner Fallback

The dataset only has bounding box annotations (no keypoints). When no keypoints are detected, 4 pallet corners are estimated from the bounding box (5% inset). This is the single biggest accuracy limitation — a YOLOv8-pose model with keypoint annotations would significantly reduce pose error.

---

## SOP-PAL-03 Triage

| # | Rule | Verifiable? | Reason |
|---|------|-------------|--------|
| 1 | No box overhangs >3cm | Partially | Visible from one side, not all 4 |
| 2 | Load height <=1.8m | Partially | Requires accurate pose |
| 3 | Boxes aligned <15 deg | Partially | Only visible faces checkable |
| 4 | Larger boxes below | Not verifiable | Can't see all layers |
| 5 | Stretch-wrapped | Partially | Texture heuristic, unreliable |
| 6 | No damaged boxes | Not verifiable | No damage classifier trained |
| 7 | Load centred <10cm | Partially | One-side view only |
| 8 | Pallet undamaged | Not verifiable | Bottom not visible |

---

## Failure Analysis — Three Worst Cases

### Case 1: Bbox-derived corners produce high pose uncertainty
- **What:** Pose uncertainty is +/-11.4cm (target: +/-2cm)
- **Root cause:** Model trained for bbox only, no keypoint annotations. Corners estimated from bbox don't capture true pallet geometry at oblique angles.
- **Fix:** Annotate 4 corners as keypoints, retrain with YOLOv8-pose.

### Case 2: IPPE PnP 180-degree ambiguity
- **What:** SOLVEPNP_IPPE sometimes returns a 180-degree flipped rotation solution.
- **Root cause:** IPPE returns two candidate solutions for planar targets. Code picks the first, not the better one.
- **Fix:** Compare both solutions by reprojection error and select the lower.

### Case 3: Detection latency far exceeds 15 FPS target
- **What:** Pipeline runs at 2.4 FPS on laptop (target: 15 FPS).
- **Root cause:** PyTorch CPU inference on older i7-6700HQ. GTX 950M not supported by modern CUDA.
- **Fix:** ONNX export provides 1.87x speedup. On Jetson Orin Nano with TensorRT INT8, estimated 14-20 FPS.

---

## What Couldn't Be Finished and Why

- **YOLOv8-pose keypoint training:** Dataset only had bbox labels. Annotating 4 corners for 519 images would take 4-6 hours. Bbox-to-corner fallback implemented instead.
- **TensorRT INT8 quantisation:** No Jetson hardware available. ONNX export measured (1.87x); TensorRT impact documented as reasoning.
- **SOP checks 6 and 8:** Box damage and pallet damage — would need separate classifiers and training data. Marked as `manual_inspection`.
- **Real pose ground truth:** No physical measurements taken. Synthetic ground truth used instead.
- **Multi-frame temporal consistency:** Strategy documented but not implemented (single-image processing).

---

## AI Tool Usage

- **Tool:** AI assistant (Claude-based) — used for project scaffolding, code generation, debugging OpenCV type errors, explaining PnP geometry, and reviewing the coordinate frame transform.
- **What it got wrong that I caught:**
  1. `calibration.py` had a `cv2.norm()` type mismatch (CV_32FC1 vs CV_32FC2) — caught at runtime, fixed by reshaping arrays.
  2. Synthetic pose evaluation had a coordinate axis swap and didn't handle IPPE 180-degree ambiguity — caught by noticing 645cm mean translation error, fixed.
  3. Benchmark script referenced a non-existent `data_splits` config key — caught at runtime, fixed with a fallback path.
  4. Initial project plan assumed the dataset had keypoint annotations — it didn't, so a bbox-corner fallback was needed.
- **Lesson:** Always validate generated code by running it — type errors and logic bugs in geometric code are easy to miss in review.

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Detection | YOLOv8s (Ultralytics) | CNN-based, fast, easy to export |
| Camera calibration | OpenCV calibrateCamera | Industry standard |
| Pose estimation | OpenCV solvePnP (IPPE) | Designed for planar targets |
| Evaluation | Synthetic ground truth | No real pose measurements available |
| Export | ONNX | Cross-platform, Jetson-compatible |
| Training | Google Colab (T4 GPU) | Free GPU access |
| Language | Python 3.11 | Standard CV ecosystem |

---

## Model Weights

The trained weights are excluded from this repo due to size. Download from:

- **Google Drive:** [Add your Drive link here for best.pt]
- **Or retrain:** Run `notebooks/train_yolov8_colab.ipynb` in Colab

Place `best.pt` in `models/best.pt` and `best.onnx` in `models/best.onnx`.

---

## Dataset

Sourced from Roboflow Universe — 519 pallet images with bounding box annotations.

- **URL:** https://universe.roboflow.com/david-akhihero-pvxdr/pallet-dezmj
- **Format:** YOLOv8
- **Classes:** 1 (`pallet`)
- **Split:** 70/15/15 (random)

See `DATASET.md` for full documentation including biases, gaps, and accuracy ceiling estimate.

---

## Citations

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- OpenCV: https://opencv.org/
- Roboflow pallet dataset: https://universe.roboflow.com/david-akhihero-pvxdr/pallet-dezmj

---
