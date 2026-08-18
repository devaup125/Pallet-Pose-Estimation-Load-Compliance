"""
Camera calibration — computes camera intrinsics using a checkerboard pattern.

Usage:
    python src/pose/calibration.py                          # interactive
    python src/pose/calibration.py --images calibration/images/  # batch
    python src/pose/calibration.py --single calibration/images/IMG_001.jpg

Prerequisites:
    1. Print a checkerboard pattern (9x6 or 7x5 inner corners)
    2. Take 20-30 photos of it at different angles with your camera/phone
    3. Save them in calibration/images/

Output:
    - calibration/camera_matrix.npy  (3x3 intrinsics)
    - calibration/distortion.npy    (distortion coefficients)
    - calibration/reprojection_error.txt
    - calibration/calibration_report.png  (visual summary)

WHY THIS MATTERS:
    Camera calibration gives us the camera matrix (K) and distortion coefficients (D).
    K tells us the focal length and optical centre — essential for PnP pose estimation.
    Without calibration, we CANNOT convert from pixels to real-world metres.
    The reprojection error tells us how accurate our calibration is.
    Target: < 0.5 pixels for good calibration.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def calibrate_from_images(image_dir, checker_cols, checker_rows, square_size_mm, output_dir):
    """
    Calibrate camera from a set of checkerboard images.

    This uses OpenCV's findChessboardCorners + calibrateCamera.

    Args:
        image_dir: Directory containing checkerboard photos
        checker_cols: Number of inner corners horizontally (e.g. 9)
        checker_rows: Number of inner corners vertically (e.g. 6)
        square_size_mm: Physical size of each checkerboard square in mm
        output_dir: Where to save calibration results

    Returns:
        camera_matrix (3x3), distortion coefficients, reprojection error
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare the 3D points of the checkerboard ---
    # In the real world, the checkerboard corners form a grid.
    # Each square is square_size_mm millimetres apart.
    # Z=0 because the board is flat.
    objp = np.zeros((checker_cols * checker_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_cols, 0:checker_rows].T.reshape(-1, 2)
    objp *= square_size_mm  # Scale to real-world millimetres

    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image

    # --- Termination criteria for corner sub-pixel refinement ---
    # This makes corner detection more precise than just pixel-level.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- Process each image ---
    image_files = sorted(
    list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) +
    list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG")) +
    list(image_dir.glob("*.JPEG"))
    )


    if len(image_files) == 0:
        print(f"ERROR: No images found in {image_dir}")
        print("Please add checkerboard photos to calibration/images/")
        return None, None, None

    successful = 0
    failed = 0
    image_shape = None

    print(f"Found {len(image_files)} images. Processing...")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP (unreadable): {img_path.name}")
            failed += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape[::-1]  # (width, height)

        # --- Find checkerboard corners ---
        # This searches the image for the chessboard pattern.
        # Returns (found, corners) where found is True/False.
        found, corners = cv2.findChessboardCorners(gray, (checker_cols, checker_rows), None)

        if found:
            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            successful += 1
            print(f"  OK: {img_path.name}")
        else:
            print(f"  FAIL (no pattern): {img_path.name}")
            failed += 1

    print(f"\nProcessed: {successful} successful, {failed} failed")

    if successful < 5:
        print("ERROR: Need at least 5 successful images for calibration.")
        return None, None, None

    # --- Run calibration ---
    # This solves for the camera matrix (K) and distortion coefficients (D)
    # that best explain the relationship between 3D points and 2D image points.
    print("\nRunning calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None
    )

    # --- Compute reprojection error ---
    # We re-project the 3D points back to 2D using our calibration,
    # and measure how far off they are from the actual detected points.
    # Lower = better. Target < 0.5 pixels.
    total_error = 0
    per_image_errors = []

    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i].reshape(-1, 2), projected.reshape(-1, 2), cv2.NORM_L2) / len(projected)

        per_image_errors.append(error)
        total_error += error

    mean_error = total_error / len(obj_points)
    print(f"Reprojection error: {mean_error:.4f} pixels (mean per image)")
    print(f"  Min: {min(per_image_errors):.4f}  Max: {max(per_image_errors):.4f}")

    # --- Save results ---
    np.save(output_dir / "camera_matrix.npy", camera_matrix)
    np.save(output_dir / "distortion.npy", dist_coeffs)

    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump({
            "reprojection_error_px": float(mean_error),
            "per_image_errors_px": [float(e) for e in per_image_errors],
            "num_images_used": successful,
            "image_size": [image_shape[0], image_shape[1]],
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coeffs": dist_coeffs.tolist(),
            "checkerboard": {
                "cols": checker_cols,
                "rows": checker_rows,
                "square_size_mm": square_size_mm,
            },
        }, f, indent=2)

    # --- Plot visual summary ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Reprojection error per image
    axes[0].bar(range(len(per_image_errors)), per_image_errors, color="steelblue", edgecolor="black")
    axes[0].axhline(0.5, color="red", linestyle="--", label="0.5 px target")
    axes[0].axhline(mean_error, color="green", linestyle="--", label=f"Mean: {mean_error:.3f}")
    axes[0].set_xlabel("Image Index")
    axes[0].set_ylabel("Reprojection Error (px)")
    axes[0].set_title("Per-Image Reprojection Error")
    axes[0].legend()

    # Plot 2: Camera matrix visualisation
    axes[1].imshow(camera_matrix, cmap="viridis")
    axes[1].set_title("Camera Matrix (K)")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{camera_matrix[i, j]:.1f}", ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_report.png", dpi=150)
    plt.close()

    print(f"\nCalibration results saved to: {output_dir}")
    print(f"  camera_matrix.npy — 3x3 intrinsics")
    print(f"  distortion.npy — distortion coefficients")
    print(f"  calibration_report.png — visual summary")

    return camera_matrix, dist_coeffs, mean_error


def main():
    config = load_config()
    calib_config = config["camera"]
    calib_dir = PROJECT_ROOT / config["paths"]["calibration_dir"]
    images_dir = calib_dir / "images"

    parser = argparse.ArgumentParser(description="Camera calibration using checkerboard")
    parser.add_argument("--images", type=str, default=str(images_dir),
                        help="Directory with checkerboard photos")
    parser.add_argument("--single", type=str, default=None,
                        help="Single image to test (creates a temporary dir)")
    args = parser.parse_args()

    image_dir = args.images

    # Handle single-image mode
    if args.single:
        img_path = Path(args.single)
        tmp_dir = Path("/tmp/calib_single")
        tmp_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(img_path, tmp_dir / img_path.name)
        image_dir = str(tmp_dir)

    print("=" * 60)
    print("PALLET POSE — Camera Calibration")
    print("=" * 60)
    print(f"  Checkerboard: {calib_config['checkerboard_cols']}x{calib_config['checkerboard_rows']}")
    print(f"  Square size: {calib_config['square_size_mm']}mm")
    print(f"  Image directory: {image_dir}")

    if not Path(image_dir).exists():
        print(f"\n  Directory does not exist: {image_dir}")
        print("  Create it and add checkerboard photos:")
        print(f"    mkdir -p {image_dir}")
        print(f"    # Copy your checkerboard photos there")
        return

    print()
    camera_matrix, dist_coeffs, error = calibrate_from_images(
        image_dir,
        calib_config["checkerboard_cols"],
        calib_config["checkerboard_rows"],
        calib_config["square_size_mm"],
        calib_dir,
    )

    if camera_matrix is not None:
        print("\n" + "=" * 60)
        print("Calibration successful!")
        print(f"  Reprojection error: {error:.4f} pixels")
        print(f"  Camera matrix:\n{camera_matrix}")
        print(f"  Distortion: {dist_coeffs.ravel()}")
        print()
        print("Next: update config/default.yaml with these values,")
        print("or they will be loaded automatically in pnp_pose.py")
        print("=" * 60)


if __name__ == "__main__":
    main()
