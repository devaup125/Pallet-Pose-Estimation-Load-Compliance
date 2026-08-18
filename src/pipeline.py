"""
End-to-end pipeline — runs the full pallet assessment pipeline on an image.

Usage:
    python src/pipeline.py --image path/to/image.jpg
    python src/pipeline.py --image-dir path/to/images/
    python src/pipeline.py --image path/to/image.jpg --output results/assessment.json

Pipeline stages:
    1. Detection (YOLOv8) → finds pallets + corners
    2. Pose Estimation (PnP) → metric (x, y, theta) with uncertainty
    3. Load Analysis (SOP checks) → per-check verdicts
    4. Output JSON → one assessment per pallet

This is what you'd run in the 5-min screen recording.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(image_path, model, config, camera_matrix, dist_coeffs, verbose=True):
    """
    Run the full pipeline on a single image.

    Returns a list of per-pallet assessment dicts (the required output format).
    """
    from src.detection.predict import detect
    from src.pose.pnp_pose import (
        get_pallet_corners_3d, solve_pose, transform_to_floor,
        estimate_pose_uncertainty,
    )
    from src.load_analysis.sop_checks import run_sop_checks

    timings = {}

    # ── Stage 1: Detection ──────────────────────────────────────────────
    t0 = time.time()
    detections = detect(model, image_path, conf_threshold=config["pose"]["min_confidence"])
    timings["detection_ms"] = (time.time() - t0) * 1000

    # Load image for shape + annotation
    image = cv2.imread(str(image_path))
    image_shape = image.shape if image is not None else None

    # ── Stage 2: Pose Estimation ────────────────────────────────────────
    t0 = time.time()
    corners_3d = get_pallet_corners_3d(config)

    pallet_assessments = []

    for i, det in enumerate(detections):
        if det["class_name"] != "pallet":
            continue

        pallet_id = f"pallet_{i}"

        # ── Pose ────────────────────────────────────────────────────────
        pose_result = None
        if det.get("keypoints") and len(det["keypoints"]) >= 4:
            corners_2d = [[k[0], k[1]] for k in det["keypoints"][:4]]
            pnp_result = solve_pose(corners_2d, corners_3d, camera_matrix, dist_coeffs)

            if pnp_result["success"]:
                rvec = np.array(pnp_result["rvec"])
                tvec = np.array(pnp_result["tvec"])
                x_m, y_m = transform_to_floor(
                    tvec,
                    config["camera"]["height_m"],
                    config["camera"]["tilt_deg"],
                )
                uncertainty = estimate_pose_uncertainty(
                    rvec, tvec, camera_matrix, dist_coeffs, corners_3d,
                )
                pose_result = {
                    "x_m": x_m,
                    "y_m": y_m,
                    "theta_deg": pnp_result["theta_deg"],
                    "uncertainty": uncertainty,
                    "reproj_error_px": pnp_result["reproj_error_px"],
                }
        timings_pose = (time.time() - t0) * 1000

        # ── Stage 3: SOP Checks ─────────────────────────────────────────
        t1 = time.time()
        sop_result = run_sop_checks(det, pose_result, config, image_shape)
        timings_sop = (time.time() - t1) * 1000

        # ── Assemble assessment ─────────────────────────────────────────
        assessment = {
            "pallet_id": pallet_id,
            "pose": pose_result,
            "sop_checks": sop_result["checks"],
            "overall_verdict": sop_result["overall"]["verdict"],
            "overall_confidence": sop_result["overall"]["confidence"],
            "overall_reasoning": sop_result["overall"]["reasoning"],
            "detection_confidence": det["confidence"],
        }

        if pose_result is None:
            assessment["pose"] = None
            assessment["pose_failure_reason"] = "Insufficient keypoints or PnP failure"

        pallet_assessments.append(assessment)

    timings["pose_ms"] = timings_pose if detections else 0
    timings["sop_ms"] = timings_sop if detections else 0
    timings["total_ms"] = timings["detection_ms"] + timings["pose_ms"] + timings["sop_ms"]

    if verbose:
        print(f"\n  Timings: detection={timings['detection_ms']:.1f}ms  "
              f"pose={timings['pose_ms']:.1f}ms  "
              f"sop={timings['sop_ms']:.1f}ms  "
              f"total={timings['total_ms']:.1f}ms")

    return pallet_assessments, timings


def main():
    parser = argparse.ArgumentParser(description="Full pallet assessment pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Path to YOLOv8 weights")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/pipeline/)")
    args = parser.parse_args()

    config = load_config()

    # ── Load model ───────────────────────────────────────────────────────
    from ultralytics import YOLO
    weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print("Train the model first using the Colab notebook.")
        return

    model = YOLO(str(weights_path))

    # ── Load calibration ─────────────────────────────────────────────────
    calib_dir = PROJECT_ROOT / config["paths"]["calibration_dir"]
    cam_matrix_path = calib_dir / "camera_matrix.npy"
    dist_path = calib_dir / "distortion.npy"

    if not cam_matrix_path.exists():
        print(f"ERROR: Camera not calibrated. Run: python src/pose/calibration.py")
        return

    camera_matrix = np.load(cam_matrix_path)
    dist_coeffs = np.load(dist_path)

    # ── Run pipeline ─────────────────────────────────────────────────────
    print("=" * 60)
    print("PALLET POSE — Full Assessment Pipeline")
    print("=" * 60)
    print(f"  Image: {args.image}")
    print(f"  Weights: {weights_path}")

    assessments, timings = run_pipeline(
        args.image, model, config, camera_matrix, dist_coeffs, verbose=True
    )

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Found {len(assessments)} pallet(s)")
    print(f"{'='*60}")

    for a in assessments:
        print(f"\n  {a['pallet_id']}:")
        if a["pose"]:
            print(f"    Pose: x={a['pose']['x_m']:.3f}m  y={a['pose']['y_m']:.3f}m  "
                  f"θ={a['pose']['theta_deg']:.1f}°")
            u = a["pose"]["uncertainty"]
            print(f"    Uncertainty: ±{u['x_cm']:.1f}cm ±{u['y_cm']:.1f}cm "
                  f"±{u['theta_deg']:.1f}°")
        else:
            print(f"    Pose: FAILED — {a.get('pose_failure_reason', 'unknown')}")

        print(f"    SOP Checks:")
        for check in a["sop_checks"]:
            print(f"      {check['check']:20s} {check['verdict']:20s} "
                  f"conf={check['confidence']:.2f}")

        print(f"\n    OVERALL: {a['overall_verdict'].upper()} "
              f"(confidence: {a['overall_confidence']:.2f})")

    # ── Save output ─────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "results" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = output_dir / f"assessment_{Path(args.image).stem}.json"

    output = {
        "image": str(args.image),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": config["deployment"]["measured_hardware"],
        "timings_ms": timings,
        "pallets": assessments,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
