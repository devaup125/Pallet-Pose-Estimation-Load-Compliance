"""
SOP Load Compliance Checks — implements the verifiable subset of SOP-PAL-03.

Usage:
    python src/load_analysis/sop_checks.py --image path/to/image.jpg

Each of the 8 SOP rules is evaluated. For each:
    - verdict: pass / fail / manual_inspection
    - confidence: 0.0 to 1.0 (derived from detection + pose quality)
    - measurement: the actual computed value
    - reasoning: why this verdict

The overall verdict weights pose quality against load compliance.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ── Individual SOP checks ────────────────────────────────────────────────────
# Each check takes the detection + pose data and returns a verdict dict.
# The confidence formula is documented in triage.md:
#   check_confidence = corner_confidence * (1 - reproj_error / max_error) * visibility


def check_overhang(detection, pose, config, image_shape):
    """
    SOP Rule 1: No box overhangs the pallet edge by more than 3cm.

    Approach:
        - Use detected pallet corners to define the pallet boundary in the image
        - Use the load bounding box (or detected boxes) to define the load extent
        - Project pallet boundary to image, measure if load exceeds it
        - Convert pixel overhang to cm using pallet dimensions as scale

    Limitation: Only the near side (visible) can be checked.
    """
    max_overhang = config["sop"]["max_overhang_cm"]

    if not detection.get("keypoints"):
        return {
            "check": "overhang",
            "rule": "SOP-PAL-03 #1: No box overhangs >3cm",
            "verdict": "manual_inspection",
            "confidence": 0.0,
            "confidence_source": "No keypoints — cannot measure pallet boundary",
            "measurement": None,
        }

    # Get pallet corners from keypoints
    kpts = detection["keypoints"]
    visible_kpts = [k for k in kpts if k[2] > 0.5]

    if len(visible_kpts) < 4:
        return {
            "check": "overhang",
            "rule": "SOP-PAL-03 #1: No box overhangs >3cm",
            "verdict": "manual_inspection",
            "confidence": 0.2,
            "confidence_source": f"Only {len(visible_kpts)}/4 corners visible",
            "measurement": None,
        }

    # Pallet corners in image
    pallet_corners = np.array([[k[0], k[1]] for k in visible_kpts[:4]])

    # Load bounding box (from detection bbox as approximation)
    # In a full implementation, you'd detect individual boxes
    x1, y1, x2, y2 = detection["bbox"]
    load_bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # Compute pallet width in pixels (for scale conversion)
    pallet_width_px = np.linalg.norm(pallet_corners[0] - pallet_corners[1])
    pallet_width_m = config["pallet"]["width_m"]
    px_per_m = pallet_width_px / pallet_width_m

    # Check if load bbox extends beyond pallet boundary (near side)
    # The near side is the bottom edge of the pallet in image space
    pallet_bottom_y = np.max(pallet_corners[:, 1])
    load_bottom_y = np.max(load_bbox[:, 1])

    overhang_px = max(0, load_bottom_y - pallet_bottom_y)
    overhang_cm = overhang_px / px_per_m * 100

    # Confidence
    corner_conf = np.mean([k[2] for k in visible_kpts[:4]])
    reproj_err = pose.get("reproj_error_px", 5.0) if pose else 5.0
    confidence = corner_conf * max(0, 1 - reproj_err / 10.0)

    verdict = "pass" if overhang_cm <= max_overhang else "fail"

    return {
        "check": "overhang",
        "rule": "SOP-PAL-03 #1: No box overhangs >3cm",
        "verdict": verdict,
        "confidence": float(confidence),
        "confidence_source": "corner detection + PnP reprojection error",
        "measurement": {
            "max_overhang_cm": float(overhang_cm),
            "threshold_cm": max_overhang,
            "note": "Near-side only; far-side overhang not visible",
        },
    }


def check_load_height(detection, pose, config, image_shape):
    """
    SOP Rule 2: Load height not to exceed 1.8m from floor.

    Approach:
        - Use pallet dimensions as a scale reference (known width = 1.2m)
        - Measure pallet width in pixels → pixels per metre
        - Measure load height in pixels (top of load to pallet base)
        - Convert to metres
    """
    max_height = config["sop"]["max_load_height_m"]
    pallet_height = config["pallet"]["height_m"]

    if not detection.get("keypoints"):
        return {
            "check": "load_height",
            "rule": "SOP-PAL-03 #2: Load height ≤1.8m",
            "verdict": "manual_inspection",
            "confidence": 0.0,
            "confidence_source": "No keypoints for scale reference",
            "measurement": None,
        }

    kpts = detection["keypoints"]
    visible_kpts = [k for k in kpts if k[2] > 0.5]

    if len(visible_kpts) < 2:
        return {
            "check": "load_height",
            "rule": "SOP-PAL-03 #2: Load height ≤1.8m",
            "verdict": "manual_inspection",
            "confidence": 0.1,
            "confidence_source": "Insufficient corners for scale",
            "measurement": None,
        }

    # Scale: pallet width in pixels → metres
    pallet_corners = np.array([[k[0], k[1]] for k in visible_kpts[:4]])
    pallet_width_px = np.linalg.norm(pallet_corners[0] - pallet_corners[1])
    pallet_width_m = config["pallet"]["width_m"]
    px_per_m = pallet_width_px / pallet_width_m

    # Load height: from top of pallet to top of load
    pallet_top_y = np.min(pallet_corners[:, 1])  # Top of pallet in image
    x1, y1, x2, y2 = detection["bbox"]
    load_top_y = y1  # Top of detection bbox

    load_height_px = pallet_top_y - load_top_y
    load_height_m = load_height_px / px_per_m + pallet_height  # Add pallet height

    # Confidence
    corner_conf = np.mean([k[2] for k in visible_kpts[:4]])
    reproj_err = pose.get("reproj_error_px", 5.0) if pose else 5.0
    confidence = corner_conf * max(0, 1 - reproj_err / 10.0) * 0.7  # Lower confidence (perspective error)

    verdict = "pass" if load_height_m <= max_height else "fail"

    return {
        "check": "load_height",
        "rule": "SOP-PAL-03 #2: Load height ≤1.8m",
        "verdict": verdict,
        "confidence": float(confidence),
        "confidence_source": "scale from pallet width + perspective approximation",
        "measurement": {
            "estimated_height_m": float(load_height_m),
            "threshold_m": max_height,
            "note": "Approximate; perspective introduces error at far edge",
        },
    }


def check_column_alignment(detection, pose, config, image_shape):
    """
    SOP Rule 3: Boxes stacked in aligned columns; no box rotated >15°.

    Approach:
        - Detect box edges using edge detection (Canny) within the load region
        - Measure dominant edge angles using Hough transform
        - Compare to pallet principal axes (from pose theta)
    """
    max_rotation = config["sop"]["max_box_rotation_deg"]

    x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]

    # This is a placeholder — full implementation would:
    # 1. Crop the load region
    # 2. Run edge detection
    # 3. Hough line transform
    # 4. Measure line angles
    # 5. Compare to pallet orientation

    return {
        "check": "column_alignment",
        "rule": "SOP-PAL-03 #3: Box alignment ±15°",
        "verdict": "manual_inspection",
        "confidence": 0.3,
        "confidence_source": "Edge detection heuristic not fully reliable with stretch wrap",
        "measurement": None,
        "note": "Requires box-level detection; approximate with edge analysis",
    }


def check_size_inversion(detection, pose, config, image_shape):
    """SOP Rule 4: Larger boxes below smaller boxes; no size inversion."""
    return {
        "check": "size_inversion",
        "rule": "SOP-PAL-03 #4: Larger boxes below smaller",
        "verdict": "manual_inspection",
        "confidence": 0.2,
        "confidence_source": "Requires individual box detection at multiple heights",
        "measurement": None,
    }


def check_stretch_wrap(detection, pose, config, image_shape):
    """
    SOP Rule 5: Load stretch-wrapped before dispatch.

    Approach:
        - Check for specular highlights (shiny reflections) in the load region
        - Stretch wrap produces characteristic shiny/semi-transparent texture
        - Use variance of Laplacian as a texture sharpness proxy
    """
    # Load image region
    x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
    # This would need the actual image — in the pipeline, the image is passed in
    # For now, return a placeholder

    return {
        "check": "stretch_wrap",
        "rule": "SOP-PAL-03 #5: Load stretch-wrapped",
        "verdict": "manual_inspection",
        "confidence": 0.3,
        "confidence_source": "Texture heuristic — not a trained classifier",
        "measurement": None,
        "note": "Would use variance of Laplacian + specular highlight detection",
    }


def check_box_damage(detection, pose, config, image_shape):
    """SOP Rule 6: No visibly damaged or crushed box."""
    return {
        "check": "box_damage",
        "rule": "SOP-PAL-03 #6: No damaged/crushed box",
        "verdict": "manual_inspection",
        "confidence": 0.0,
        "confidence_source": "Not verifiable — requires trained damage detection model",
        "measurement": None,
        "note": "Not implemented: would need labelled damage data and training (out of scope for 5 days)",
    }


def check_load_centred(detection, pose, config, image_shape):
    """
    SOP Rule 7: Load centred on pallet; centroid within 10cm of pallet centre.

    Approach:
        - Compute load centroid (centre of detection bbox as approximation)
        - Compute pallet centre (from keypoints)
        - Measure offset in pixels, convert to cm using scale
    """
    max_offset = config["sop"]["max_centroid_offset_cm"]

    if not detection.get("keypoints"):
        return {
            "check": "load_centred",
            "rule": "SOP-PAL-03 #7: Load centred ±10cm",
            "verdict": "manual_inspection",
            "confidence": 0.0,
            "confidence_source": "No keypoints for pallet centre",
            "measurement": None,
        }

    kpts = detection["keypoints"]
    visible_kpts = [k for k in kpts if k[2] > 0.5]

    if len(visible_kpts) < 4:
        return {
            "check": "load_centred",
            "rule": "SOP-PAL-03 #7: Load centred ±10cm",
            "verdict": "manual_inspection",
            "confidence": 0.2,
            "confidence_source": "Insufficient corners",
            "measurement": None,
        }

    # Pallet centre from corners
    pallet_corners = np.array([[k[0], k[1]] for k in visible_kpts[:4]])
    pallet_centre = np.mean(pallet_corners, axis=0)

    # Load centre from bbox
    x1, y1, x2, y2 = detection["bbox"]
    load_centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # Offset in pixels
    offset_px = np.linalg.norm(load_centre - pallet_centre)

    # Convert to cm
    pallet_width_px = np.linalg.norm(pallet_corners[0] - pallet_corners[1])
    pallet_width_m = config["pallet"]["width_m"]
    px_per_m = pallet_width_px / pallet_width_m
    offset_cm = offset_px / px_per_m * 100

    # Confidence
    corner_conf = np.mean([k[2] for k in visible_kpts[:4]])
    reproj_err = pose.get("reproj_error_px", 5.0) if pose else 5.0
    confidence = corner_conf * max(0, 1 - reproj_err / 10.0)

    verdict = "pass" if offset_cm <= max_offset else "fail"

    return {
        "check": "load_centred",
        "rule": "SOP-PAL-03 #7: Load centred ±10cm",
        "verdict": verdict,
        "confidence": float(confidence),
        "confidence_source": "corner detection + bbox centroid approximation",
        "measurement": {
            "centroid_offset_cm": float(offset_cm),
            "threshold_cm": max_offset,
            "note": "Approximate; uses bbox centroid as load centroid",
        },
    }


def check_pallet_damage(detection, pose, config, image_shape):
    """SOP Rule 8: Pallet undamaged — no broken boards or split stringers."""
    return {
        "check": "pallet_damage",
        "rule": "SOP-PAL-03 #8: Pallet undamaged",
        "verdict": "manual_inspection",
        "confidence": 0.0,
        "confidence_source": "Not verifiable from this view — pallet underside/boards not visible",
        "measurement": None,
        "note": "Camera sees load, not pallet structure. Requires bottom-up or multi-angle view.",
    }


# ── Overall verdict ──────────────────────────────────────────────────────────

def compute_overall_verdict(checks, pose_confidence, config):
    """
    Compute the overall pass/fail/manual_inspection verdict for a pallet.

    Logic:
        - If pose confidence < 0.4 → manual_inspection (geometric measurements unreliable)
        - Any high-confidence fail (≥0.7) → overall fail
        - Low-confidence fails → manual_inspection
        - Otherwise → pass

    This weighting is documented in triage.md.
    """
    manual_threshold = config["sop"]["manual_inspection_confidence"]
    pass_threshold = config["sop"]["pass_confidence"]

    # If pose quality is too low, all geometric checks are unreliable
    if pose_confidence < manual_threshold:
        return {
            "verdict": "manual_inspection",
            "confidence": pose_confidence,
            "reasoning": f"Pose confidence {pose_confidence:.2f} < {manual_threshold} — "
                        "geometric measurements unreliable without good pose",
        }

    # Check for high-confidence fails
    for check in checks:
        if check["verdict"] == "fail" and check["confidence"] >= pass_threshold:
            return {
                "verdict": "fail",
                "confidence": check["confidence"],
                "reasoning": f"High-confidence fail on: {check['check']} "
                           f"(conf={check['confidence']:.2f})",
            }

    # Check for low-confidence fails or manual inspections
    has_uncertain = False
    for check in checks:
        if check["verdict"] in ("fail", "manual_inspection"):
            if check["confidence"] >= manual_threshold:
                has_uncertain = True

    if has_uncertain:
        return {
            "verdict": "manual_inspection",
            "confidence": pose_confidence,
            "reasoning": "Some checks uncertain — manual inspection recommended",
        }

    return {
        "verdict": "pass",
        "confidence": pose_confidence,
        "reasoning": "All verifiable checks passed with sufficient confidence",
    }


# ── Main entry ───────────────────────────────────────────────────────────────

def run_sop_checks(detection, pose, config, image_shape=None):
    """
    Run all 8 SOP checks for a single pallet.

    Args:
        detection: Detection dict from YOLOv8 (bbox, confidence, keypoints)
        pose: Pose dict from PnP (or None if pose failed)
        config: Project config
        image_shape: (height, width) of the image

    Returns:
        List of check result dicts + overall verdict
    """
    pose_data = pose if pose else {}
    pose_confidence = pose_data.get("detection_confidence", 0.5) if pose else 0.0

    checks = [
        check_overhang(detection, pose_data, config, image_shape),
        check_load_height(detection, pose_data, config, image_shape),
        check_column_alignment(detection, pose_data, config, image_shape),
        check_size_inversion(detection, pose_data, config, image_shape),
        check_stretch_wrap(detection, pose_data, config, image_shape),
        check_box_damage(detection, pose_data, config, image_shape),
        check_load_centred(detection, pose_data, config, image_shape),
        check_pallet_damage(detection, pose_data, config, image_shape),
    ]

    overall = compute_overall_verdict(checks, pose_confidence, config)

    return {
        "checks": checks,
        "overall": overall,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SOP load compliance checks")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    args = parser.parse_args()

    config = load_config()

    from ultralytics import YOLO
    from src.detection.predict import detect
    from src.pose.pnp_pose import run_pose_estimation

    model = YOLO(str(PROJECT_ROOT / args.weights))
    detections = detect(model, args.image, config["pose"]["min_confidence"])

    image = cv2.imread(args.image)
    image_shape = image.shape if image is not None else None

    print("=" * 60)
    print("PALLET POSE — SOP Load Compliance Checks")
    print("=" * 60)

    for i, det in enumerate(detections):
        if det["class_name"] != "pallet":
            continue

        print(f"\n--- Pallet {i} ---")
        result = run_sop_checks(det, None, config, image_shape)

        for check in result["checks"]:
            status = check["verdict"].upper()
            print(f"  {check['check']:20s} {status:20s} conf={check['confidence']:.2f}")

        print(f"\n  OVERALL: {result['overall']['verdict'].upper()}")
        print(f"  Reason: {result['overall']['reasoning']}")


if __name__ == "__main__":
    main()
