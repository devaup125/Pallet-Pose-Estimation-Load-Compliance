

import argparse
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


def load_calibration(calibration_dir):
   
    calibration_dir = Path(calibration_dir)
    K = np.load(calibration_dir / "camera_matrix.npy")
    D = np.load(calibration_dir / "distortion.npy")
    return K, D


def get_pallet_corners_3d(config):
 
    corners = config["pallet"]["corners_3d"]
    return np.array(corners, dtype=np.float64)


def solve_pose(image_corners_2d, object_points_3d, camera_matrix, dist_coeffs):

    if len(image_corners_2d) < 4:
        return {"success": False, "reason": "need at least 4 corners"}

    # Ensure correct shape (N, 1, 2) for OpenCV
    img_pts = np.array(image_corners_2d, dtype=np.float64).reshape(-1, 1, 2)
    obj_pts = np.array(object_points_3d, dtype=np.float64).reshape(-1, 1, 3)

   
    try:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error as e:
        return {"success": False, "reason": f"PnP failed: {e}"}

    if not success:
        return {"success": False, "reason": "PnP did not converge"}

    # --- Compute reprojection error ---
    # Re-project 3D points back to 2D and measure pixel error
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_error = np.mean(np.linalg.norm(projected.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1))

    tx, ty, tz = tvec.ravel()

    # Convert camera-frame to floor-frame
    # This is a simplified version — the full transform uses the camera
    # extrinsic (rotation due to tilt + translation due to height).
    # For a camera at height h tilted down by angle alpha:
    #   floor_x = tz * cos(alpha) - ty * sin(alpha)  (along ground)
    #   floor_y = tx  (perpendicular, along the floor)
    #   depth = tz * sin(alpha) + ty * cos(alpha)   (distance from camera)
    #
    # But for initial testing, we return the raw camera-frame values.
    # The full transform is applied in transform_to_floor() below.

    # Extract rotation angle (theta) from rvec
    # rvec is a Rodrigues vector; convert to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # The pallet's orientation is the yaw angle (rotation around Z-axis in floor frame)
    # Extract from rotation matrix: atan2(R[1,0], R[0,0])
    theta_rad = np.arctan2(R[1, 0], R[0, 0])
    theta_deg = np.degrees(theta_rad)

    return {
        "success": True,
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "reproj_error_px": float(reproj_error),
        "theta_deg": float(theta_deg),
        # Raw camera-frame position (before floor transform)
        "camera_frame": {"tx": float(tx), "ty": float(ty), "tz": float(tz)},
    }


def transform_to_floor(tvec, camera_height_m, camera_tilt_deg):

    tx, ty, tz = tvec.ravel()
    alpha = np.radians(camera_tilt_deg)

   
    # Floor x (forward, along camera viewing direction projected on floor):
    floor_x = tz * np.cos(alpha) - ty * np.sin(alpha)
    # Floor y (perpendicular, sideways):
    floor_y = tx

    return float(floor_x), float(floor_y)


def estimate_pose_uncertainty(rvec, tvec, camera_matrix, dist_coeffs, object_points_3d,
                              corner_pixel_noise=2.0):

    # Re-project reference pose to get reference 2D corners
    ref_pts_2d, _ = cv2.projectPoints(object_points_3d.reshape(-1, 1, 3),
                                       rvec, tvec, camera_matrix, dist_coeffs)
    ref_pts_2d = ref_pts_2d.reshape(-1, 2)

    # Monte Carlo: perturb corners and re-solve
    n_trials = 50
    translations = []
    rotations = []

    for _ in range(n_trials):
        noise = np.random.normal(0, corner_pixel_noise, ref_pts_2d.shape)
        noisy_pts = (ref_pts_2d + noise).reshape(-1, 1, 2).astype(np.float64)
        obj_pts = object_points_3d.reshape(-1, 1, 3).astype(np.float64)

        try:
            ok, rv, tv = cv2.solvePnP(
                obj_pts, noisy_pts, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok:
                translations.append(tv.ravel())
                R, _ = cv2.Rodrigues(rv)
                rotations.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        except cv2.error:
            continue

    if len(translations) < 10:
        return {"x_cm": 5.0, "y_cm": 5.0, "theta_deg": 3.0}  # Conservative fallback

    translations = np.array(translations)
    rotations = np.array(rotations)

    return {
        "x_cm": float(np.std(translations[:, 0]) * 100),
        "y_cm": float(np.std(translations[:, 1]) * 100),
        "theta_deg": float(np.std(rotations)),
    }


def run_pose_estimation(image_path, model, config, camera_matrix=None, dist_coeffs=None):

    from src.detection.predict import detect

    # Load calibration if not provided
    if camera_matrix is None or dist_coeffs is None:
        calib_dir = PROJECT_ROOT / config["paths"]["calibration_dir"]
        camera_matrix, dist_coeffs = load_calibration(calib_dir)

    # Get 3D pallet corners
    corners_3d = get_pallet_corners_3d(config)

    # Run detection
    detections = detect(model, image_path, conf_threshold=config["pose"]["min_confidence"])

    pallet_poses = []

    for i, det in enumerate(detections):
        if det["class_name"] != "pallet":
            continue

        # Check if we have keypoints (corners)
        if not det["keypoints"]:
            pallet_poses.append({
                "pallet_id": f"pallet_{i}",
                "pose": None,
                "reason": "No keypoints detected — cannot estimate pose",
                "detection_confidence": det["confidence"],
            })
            continue

        # Extract 2D corner positions (only visible ones)
        kpts = det["keypoints"]
        corners_2d = []
        for kp in kpts:
            kx, ky, kconf = kp
            corners_2d.append([kx, ky])

        # Need exactly 4 corners for PnP
        if len(corners_2d) < 4:
            pallet_poses.append({
                "pallet_id": f"pallet_{i}",
                "pose": None,
                "reason": f"Only {len(corners_2d)}/4 corners visible",
                "detection_confidence": det["confidence"],
            })
            continue

        # Solve PnP
        result = solve_pose(
            corners_2d[:4], corners_3d, camera_matrix, dist_coeffs
        )

        if not result["success"]:
            pallet_poses.append({
                "pallet_id": f"pallet_{i}",
                "pose": None,
                "reason": result["reason"],
                "detection_confidence": det["confidence"],
            })
            continue

        # Transform to floor frame
        rvec = np.array(result["rvec"])
        tvec = np.array(result["tvec"])
        x_m, y_m = transform_to_floor(
            tvec,
            config["camera"]["height_m"],
            config["camera"]["tilt_deg"],
        )

        # Estimate uncertainty
        uncertainty = estimate_pose_uncertainty(
            rvec, tvec, camera_matrix, dist_coeffs, corners_3d,
            corner_pixel_noise=2.0,
        )

        pallet_poses.append({
            "pallet_id": f"pallet_{i}",
            "pose": {
                "x_m": x_m,
                "y_m": y_m,
                "theta_deg": result["theta_deg"],
                "uncertainty": uncertainty,
                "reproj_error_px": result["reproj_error_px"],
            },
            "detection_confidence": det["confidence"],
            "raw_rvec": result["rvec"],
            "raw_tvec": result["tvec"],
        })

    return pallet_poses


def main():
    parser = argparse.ArgumentParser(description="Pallet pose estimation via PnP")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Path to trained YOLOv8 weights")
    args = parser.parse_args()

    config = load_config()
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / args.weights))

    print("=" * 60)
    print("PALLET POSE — Pose Estimation")
    print("=" * 60)

    poses = run_pose_estimation(args.image, model, config)

    print(f"\nFound {len(poses)} pallet(s):")
    for p in poses:
        if p["pose"]:
            print(f"\n  {p['pallet_id']}:")
            print(f"    Position:  x={p['pose']['x_m']:.3f}m, y={p['pose']['y_m']:.3f}m")
            print(f"    Orientation: {p['pose']['theta_deg']:.1f}°")
            print(f"    Uncertainty: ±{p['pose']['uncertainty']['x_cm']:.1f}cm, "
                  f"±{p['pose']['uncertainty']['y_cm']:.1f}cm, "
                  f"±{p['pose']['uncertainty']['theta_deg']:.1f}°")
            print(f"    Reproj error: {p['pose']['reproj_error_px']:.2f} px")
        else:
            print(f"\n  {p['pallet_id']}: POSE FAILED — {p['reason']}")

    # Save results
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"] / "pose"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"pose_{Path(args.image).stem}.json"

    # Strip raw arrays before saving
    for p in poses:
        p.pop("raw_rvec", None)
        p.pop("raw_tvec", None)

    with open(out_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
