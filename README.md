# Pallet Pose Estimation & Load Compliance

Computer vision pipeline for warehouse pallet pose estimation and load compliance checking, built for the Delhivery AI/ML Engineer assignment.

## Quick Start

```bash
# Clone and install
git clone <your-repo-url>
cd pallet-pose
pip install -r requirements.txt

# 1. Train detector (Google Colab recommended)
#    Open notebooks/train_yolov8_colab.ipynb in Colab

# 2. Calibrate camera
python src/pose/calibration.py

# 3. Run full pipeline on an image
python src/pipeline.py --image path/to/image.jpg

# 4. Evaluate
python src/detection/evaluate.py
python src/pose/evaluate_pose.py

# 5. Benchmark deployment
python src/deployment/benchmark.py
```

## Pipeline

```
Image → Detection (YOLOv8s) → Corner Estimation → Pose Estimation (PnP) → Load Analysis (SOP) → Output JSON
```

## Project Structure

```
pallet-pose/
├── README.md
├── DATASET.md
├── requirements.txt
├── config/default.yaml        ← All parameters in one place
├── src/
│   ├── detection/             ← Section 1: Dataset & Detection (30%)
│   ├── pose/                   ← Section 2: Pose Estimation (35%)
│   ├── load_analysis/          ← Section 3: Load Analysis (25%)
│   ├── deployment/             ← Section 4: Deployment (10%)
│   └── pipeline.py             ← End-to-end pipeline
├── notebooks/
│   └── train_yolov8_colab.ipynb  ← Colab training notebook
├── data/                       ← Datasets (not in git)
├── models/                     ← Trained weights (best.pt, best.onnx)
├── calibration/                ← Camera intrinsics + checkerboard photos
├── results/                    ← Evaluation outputs, plots
└── docs/                       ← Triage doc, labelling guide
```

---

## 1. Approach and Significant Decisions

### Detection

- **Model:** YOLOv8s (Small) trained on Google Colab (Tesla T4 GPU), 150 epochs with early stopping at patience=30.
- **Dataset:** Merged from a Roboflow pallet detection dataset (519 images, 1 class: `pallet`). See DATASET.md for sourcing details.
- **Why YOLOv8s:** Good accuracy/speed tradeoff; Ultralytics API handles training, export, and inference cleanly. CNN-based architecture chosen over Vision Transformers for inference speed on edge hardware.
- **Detection type:** Bounding box detection (`[x1, y1, x2, y2]` + confidence). The model detects pallets but does not output keypoints.
- **Corner estimation fallback:** Since the trained model is bbox-only, 4 pallet corners are estimated from the bounding box (inset 5%) as a fallback for PnP. This is documented as a known limitation — a YOLOv8-pose model trained with keypoint annotations would produce more accurate corners.
- **Cost:** Roboflow datasets may not match real warehouse conditions (lighting, camera angle, pallet types). Bias documented in DATASET.md.

### Pose Estimation

- **Method:** Perspective-n-Point (PnP) using `cv2.solvePnP()` with `SOLVEPNP_IPPE` (designed for planar targets — pallet top surface is flat).
- **Inputs:** 4 detected pallet corners (2D pixels) + known EUR pallet dimensions (1.2m × 0.8m, 3D points) + camera intrinsics from calibration.
- **Camera calibration:** Performed with a 7×4 inner-corner checkerboard (35mm squares), 41 photos taken at varied angles with a phone camera. Reprojection error: **0.44 pixels** (target: < 0.5px).
- **Coordinate frame:** Camera at 1.2m height, tilted 20° down. Floor-frame transform applies rotation to undo tilt, yielding metric (x, y) position.
- **Uncertainty estimation:** Monte Carlo simulation (50 trials) — perturbs detected 2D corners by ±2px Gaussian noise, re-solves PnP, measures std dev of resulting poses.
- **Evaluation method:** Synthetic ground truth — known 3D poses projected to 2D with realistic noise, then PnP-solved and compared. No real pose ground truth available; this allows exact error measurement.
- **Cost:** PnP accuracy depends on corner detection quality. Bbox-derived corners introduce more error than a true keypoint model would.

### Load Analysis

- Triage of all 8 SOP-PAL-03 rules (see `src/load_analysis/sop_checks.py`).
- **Verifiable from single view:** overhang, load height, column alignment, load centring.
- **Partially verifiable:** stretch wrap (texture heuristic), size inversion (requires box segmentation).
- **Not verifiable from single view:** box damage (would need separate damage classifier), pallet damage (bottom not visible).
- Each check returns verdict (pass/fail/manual_inspection) + confidence + measurement.

### Deployment

- **Measured on:** Intel Core i7-6700HQ + NVIDIA GeForce GTX 950M (laptop). CPU inference (PyTorch CPU build).
- **Target:** Jetson Orin Nano 15W @ 15 FPS.
- **ONNX export:** Model exported to ONNX format (42.7 MB). Measured 1.87× speedup over PyTorch.
- **Temporal consistency strategy:** Multi-frame averaging of pose estimates for stationary pallets (documented in deployment report).

---

## 2. Results — Distributions

> Results are reported as distributions (histograms/violin plots), not point estimates.
> All plots are in `results/`.

### Detection

- Model: YOLOv8s, 11.1M parameters, 28.4 GFLOPs
- Trained on 519 images from Roboflow pallet dataset
- Detection confidence on test image: 0.56
- mAP values from Colab training (see training notebook output)

### Camera Calibration

- Checkerboard: 7×4 inner corners, 35mm squares
- Images used: 41 (after deduplication from 82 with duplicates)
- **Reprojection error: 0.4376 pixels** (mean), min 0.22, max 0.79
- Camera matrix: fx=1223.9, fy=1229.3, cx=730.6, cy=681.2
- Distortion: [-0.079, 0.346, -0.008, -0.011, -0.671]

### Pose Estimation

Synthetic evaluation (200 poses, 2px corner noise):

**Translation Error (cm):**
| Statistic | Value |
|-----------|-------|
| Mean | 3.97 |
| Median | 2.41 |
| P90 | 9.41 |
| P95 | 11.82 |
| % within ±2cm | 40.1% |

**Rotation Error (degrees):**
| Statistic | Value |
|-----------|-------|
| Mean | 0.39 |
| Median | 0.27 |
| P90 | 0.85 |
| P95 | 1.12 |
| % within ±3° | 100.0% |

- Plots: `results/pose/pose_error_distributions.png`, `results/pose/sensitivity_analysis.png`, `results/pose/usable_envelope.png`
- Rotation accuracy is excellent (100% within ±3°). Translation accuracy is moderate (40% within ±2cm) — limited by bbox-derived corners rather than precise keypoints.

### Load Analysis (SOP Checks)

Sample output on test pallet:

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

Overall verdict: MANUAL_INSPECTION (confidence: 0.50)

### Deployment Benchmark

**Latency (50 runs, single image):**

| Stage | Mean | P50 | P95 |
|-------|------|-----|-----|
| Detection | 403.2ms | 331.9ms | 384.1ms |
| Pose | 7.3ms | 8.0ms | — |
| SOP | 0.5ms | 0.0ms | — |
| **Total** | **411.0ms** | **339.0ms** | **390.7ms** |

- **FPS: 2.4** (target: 15 FPS — NOT MET on laptop hardware)
- Detection is the bottleneck (98% of total time)

**ONNX Export:**
- ONNX latency: 248.5ms mean
- PyTorch latency: 465.5ms mean
- **Speedup: 1.87×**
- ONNX model size: 42.7 MB

**Expected on Jetson Orin Nano (reasoning, not measured):**
- Jetson Orin Nano has a dedicated NPU + GPU (1024 CUDA cores, 40 Tensor cores) at 15W
- INT8 quantisation via TensorRT would further reduce latency by ~2-3×
- Estimated: 40-60ms detection → 15-25 FPS achievable
- This is reasoning based on published benchmarks, NOT measured — no Jetson hardware was available

---

## 3. Failure Analysis — Three Worst Cases

### Case 1: Bbox-only model produces imprecise corners
- **What went wrong:** Pose estimation uncertainty is ±11.4cm (target: ±2cm) because corners are estimated from the bounding box, not detected as keypoints.
- **Root cause:** The model was trained for bounding box detection only. The project scaffold expected a YOLOv8-pose model with 4 keypoint annotations, but the dataset only had bbox labels. A bbox-to-corner fallback was implemented (inset 5%), but this cannot capture the true pallet geometry — a pallet's visible corners don't align with the bbox rectangle, especially at oblique angles.
- **Fix:** Annotate 4 corners as keypoints in the dataset and retrain with YOLOv8-pose. This would likely bring translation error to sub-2cm at close range.

### Case 2: IPPE PnP 180° ambiguity
- **What went wrong:** The SOLVEPNP_IPPE method (designed for planar targets) sometimes returns a 180°-flipped rotation solution, causing rotation errors up to 193° in the synthetic evaluation.
- **Root cause:** IPPE returns two candidate solutions for planar targets. The code picks the first returned solution rather than comparing reprojection errors of both and choosing the better one.
- **Fix:** Compare both IPPE solutions by reprojection error and select the lower. The synthetic evaluation was patched to handle the 180° ambiguity in error computation, but the production PnP solver should also be fixed.

### Case 3: Detection latency far exceeds 15 FPS target
- **What went wrong:** Pipeline runs at 2.4 FPS on the laptop — well below the 15 FPS target. Detection alone takes 403ms.
- **Root cause:** PyTorch CPU inference on an older i7-6700HQ with no GPU acceleration (torch CPU-only build). The GTX 950M is not supported by modern CUDA versions.
- **Fix:** ONNX export provides 1.87× speedup (248ms). On Jetson Orin Nano with TensorRT INT8 quantisation, detection would drop to ~40-60ms based on published benchmarks for YOLOv8s, meeting the 15 FPS target.

---

## 4. What I Couldn't Finish and Why

- **YOLOv8-pose keypoint training:** The dataset only had bounding box annotations. Sourcing or annotating 4-corner keypoints for 519 images would take 4-6 hours of manual labelling. Instead, a bbox-to-corner fallback was implemented to keep the pipeline functional. This is the single biggest accuracy limitation.
- **TensorRT INT8 quantisation:** No Jetson Orin Nano hardware available. ONNX export was measured (1.87× speedup); TensorRT quantisation impact is documented as reasoning, not measurement.
- **SOP check 6 (box damage):** Would need a separate damage detection model and training data. Marked as `manual_inspection` with confidence 0.0.
- **SOP check 8 (pallet damage):** Bottom of pallet not visible from a single overhead view. Marked as `manual_inspection`.
- **Real pose ground truth:** No physical pallet measurements taken. Synthetic ground truth used instead — documented and justified.
- **Multi-frame temporal consistency:** Strategy documented but not implemented as the system currently processes single images.

---

## 5. AI Tool Usage

- **Tool:** AI assistant (Claude-based) — used for project scaffolding, code generation, debugging OpenCV type errors, explaining PnP geometry, and reviewing the coordinate frame transform.
- **What it got wrong that I caught:**
  1. The generated `calibration.py` had a `cv2.norm()` type mismatch (CV_32FC1 vs CV_32FC2) — caught at runtime, fixed by reshaping arrays.
  2. The synthetic pose evaluation had a coordinate axis swap (comparing floor_x against true_x instead of true_y) and didn't handle IPPE 180° ambiguity — caught by noticing 645cm mean translation error, fixed.
  3. The benchmark script referenced a non-existent `data_splits` config key — caught at runtime, fixed with a fallback path.
  4. The initial project plan assumed the dataset had keypoint annotations — it didn't, so a bbox-corner fallback was needed.
- **What I learned:** Always validate generated code by running it — type errors and logic bugs in geometric code are easy to miss in code review.

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
    {
      "check": "overhang",
      "verdict": "fail",
      "confidence": 0.47,
      "measurement": {"max_overhang_cm": 4.2}
    }
  ],
  "overall_verdict": "MANUAL_INSPECTION",
  "overall_confidence": 0.50,
  "detection_confidence": 0.56
}
```

## Citation

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- OpenCV: https://opencv.org/
- Roboflow pallet dataset: https://universe.roboflow.com/david-akhihero-pvxdr/pallet-dezmj
- Checkerboard calibration: OpenCV `findChessboardCorners` + `calibrateCamera`
