

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def benchmark_latency(model, config, camera_matrix, dist_coeffs, test_images, n_runs=50):
    
    from src.pipeline import run_pipeline

    all_timings = {
        "detection_ms": [],
        "pose_ms": [],
        "sop_ms": [],
        "total_ms": [],
    }

    for img_path in test_images:
        for _ in range(n_runs // len(test_images)):
            _, timings = run_pipeline(img_path, model, config, camera_matrix, dist_coeffs, verbose=False)
            for key in all_timings:
                all_timings[key].append(timings[key])

    # Compute statistics
    stats = {}
    for key, values in all_timings.items():
        arr = np.array(values)
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    # Throughput
    total_mean = stats["total_ms"]["mean"]
    fps = 1000.0 / total_mean if total_mean > 0 else 0

    return stats, fps


def benchmark_onnx(weights_path, test_images, config):
    
    
    try:
        from ultralytics import YOLO
        import onnxruntime as ort
    except ImportError:
        return {
            "status": "skipped",
            "reason": "onnxruntime not installed. Run: pip install onnxruntime",
        }

    onnx_path = str(weights_path).replace(".pt", ".onnx")

    # Export to ONNX
    model = YOLO(str(weights_path))
    print("  Exporting to ONNX...")
    model.export(format="onnx", imgsz=config["training"]["imgsz"])

    if not Path(onnx_path).exists():
        return {"status": "failed", "reason": "ONNX export failed"}

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Measure ONNX latency
    onnx_times = []
    for img_path in test_images[:20]:
        image = cv2.imread(str(img_path))
        image_resized = cv2.resize(image, (640, 640))
        input_tensor = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = input_tensor[np.newaxis]

        t0 = time.time()
        session.run(None, {"images": input_tensor})
        onnx_times.append((time.time() - t0) * 1000)

    # Compare with PyTorch
    pt_times = []
    for img_path in test_images[:20]:
        t0 = time.time()
        model(str(img_path), verbose=False)
        pt_times.append((time.time() - t0) * 1000)

    return {
        "status": "success",
        "onnx_latency_ms": {
            "mean": float(np.mean(onnx_times)),
            "p50": float(np.percentile(onnx_times, 50)),
            "p95": float(np.percentile(onnx_times, 95)),
        },
        "pytorch_latency_ms": {
            "mean": float(np.mean(pt_times)),
            "p50": float(np.percentile(pt_times, 50)),
            "p95": float(np.percentile(pt_times, 95)),
        },
        "speedup": float(np.mean(pt_times) / np.mean(onnx_times)) if np.mean(onnx_times) > 0 else 0,
        "onnx_path": onnx_path,
    }


def main():
    config = load_config()
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"] / "deployment"
    results_dir.mkdir(parents=True, exist_ok=True)

    weights_path = PROJECT_ROOT / "models" / "best.pt"
    if not weights_path.exists():
        print("ERROR: Weights not found. Train first.")
        return

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # Load calibration
    calib_dir = PROJECT_ROOT / config["paths"]["calibration_dir"]
    camera_matrix = np.load(calib_dir / "camera_matrix.npy")
    dist_coeffs = np.load(calib_dir / "distortion.npy")

    # Get test images
       # Use local test images if available, otherwise use test_pallet.jpg
    test_img_dir = PROJECT_ROOT / "data" / "test_images"
    if test_img_dir.exists():
        test_images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.jpeg")) + list(test_img_dir.glob("*.png")))[:20]
    else:
        # Fallback: use test_pallet.jpg in project root
        fallback = PROJECT_ROOT / "test_pallet.jpg"
        if not fallback.exists():
            print("ERROR: No test images found. Add images to data/test_images/ or test_pallet.jpg")
            return
        test_images = [fallback]


    if not test_images:
        print("ERROR: No test images found.")
        return

    hardware = config["deployment"]["measured_hardware"]
    target = config["deployment"]["target_hardware"]
    target_fps = config["deployment"]["target_fps"]

    print("=" * 60)
    print("PALLET POSE — Deployment Benchmark")
    print("=" * 60)
    print(f"  Measured on: {hardware}")
    print(f"  Target: {target} @ {target_fps} FPS")

    # ── 1. Latency benchmark ────────────────────────────────────────────
    print(f"\n[1/3] Latency benchmark ({len(test_images)} images, 50 runs)...")
    stats, fps = benchmark_latency(model, config, camera_matrix, dist_coeffs, test_images)

    print(f"\n  Detection:  mean={stats['detection_ms']['mean']:.1f}ms  "
          f"p50={stats['detection_ms']['p50']:.1f}ms  "
          f"p95={stats['detection_ms']['p95']:.1f}ms")
    print(f"  Pose:       mean={stats['pose_ms']['mean']:.1f}ms  "
          f"p50={stats['pose_ms']['p50']:.1f}ms")
    print(f"  SOP:        mean={stats['sop_ms']['mean']:.1f}ms  "
          f"p50={stats['sop_ms']['p50']:.1f}ms")
    print(f"  TOTAL:      mean={stats['total_ms']['mean']:.1f}ms  "
          f"p50={stats['total_ms']['p50']:.1f}ms  "
          f"p95={stats['total_ms']['p95']:.1f}ms")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Target:     {target_fps} FPS — {'MET' if fps >= target_fps else 'NOT MET'}")

    # 2. ONNX / Quantisation
    print("\n[2/3] ONNX export & latency comparison...")
    onnx_results = benchmark_onnx(weights_path, test_images, config)

    if onnx_results["status"] == "success":
        print(f"  ONNX latency: mean={onnx_results['onnx_latency_ms']['mean']:.1f}ms")
        print(f"  PyTorch:      mean={onnx_results['pytorch_latency_ms']['mean']:.1f}ms")
        print(f"  Speedup:      {onnx_results['speedup']:.2f}x")
    else:
        print(f"  {onnx_results['status']}: {onnx_results.get('reason', '')}")

    #  3. Failure behaviour + temporal consistency
    print("\n[3/3] Failure behaviour analysis...")
    failure_analysis = {
        "failure_modes": [
            {
                "mode": "No pallet detected",
                "cause": "Pallet occluded, out of frame, or below confidence threshold",
                "system_output": "Empty assessment list — no pallets found",
                "consumer_signal": "Zero pallets in output JSON",
            },
            {
                "mode": "Pallet detected but no keypoints",
                "cause": "Corners not visible (heavy occlusion, lighting, angle)",
                "system_output": "pose: null, reason: 'No keypoints detected'",
                "consumer_signal": "pose field is null with reason string",
            },
            {
                "mode": "PnP fails to converge",
                "cause": "Degenerate corner geometry (collinear, too few points)",
                "system_output": "pose: null, reason: 'PnP did not converge'",
                "consumer_signal": "pose field is null with reason string",
            },
            {
                "mode": "High reprojection error",
                "cause": "Poor corner localisation or wrong pallet dimensions",
                "system_output": "pose with high uncertainty values",
                "consumer_signal": "uncertainty values exceed tolerance",
            },
        ],
        "temporal_consistency": {
            "strategy": "Multi-frame averaging — pallet stays in same slot across many frames",
            "implementation": [
                "Track pallet across N consecutive frames using IoU matching",
                "Collect pose estimates from each frame",
                "Reject outliers (beyond 2-sigma from median)",
                "Average remaining poses for final estimate",
                "Increase confidence with more frames (up to a cap)",
            ],
            "expected_benefit": "Reduces random pose error by ~sqrt(N) for N frames",
            "caveat": "Only valid if pallet is stationary; must detect movement to reset",
        },
        "jetson_expectation": {
            "target": target,
            "reasoning": (
                "Jetson Orin Nano has a 40 TOPS NPU (INT8). YOLOv8s on Orin Nano "
                "typically achieves 30-60 FPS with TensorRT INT8, well above the "
                "15 FPS target. PnP and SOP checks are CPU-bound and negligible. "
                "Main change: model export to TensorRT engine with INT8 calibration. "
                "Expected accuracy drop: <2% mAP for INT8 vs FP32 on YOLOv8s."
            ),
            "measured_on": hardware,
            "note": "These are reasoned expectations, NOT measured numbers. "
                    "Actual deployment requires Jetson hardware to verify.",
        },
    }

    # ── Save results ────────────────────────────────────────────────────
    report = {
        "hardware": hardware,
        "target": target,
        "latency_stats": stats,
        "throughput_fps": fps,
        "meets_target_fps": fps >= target_fps,
        "onnx_results": onnx_results,
        "failure_analysis": failure_analysis,
    }

    report_path = results_dir / "deployment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {report_path}")
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
