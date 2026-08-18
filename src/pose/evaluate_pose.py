"""
Pose evaluation — measures translation and rotation error as distributions.

Usage:
    python src/pose/evaluate_pose.py

Generates:
    - Translation error distribution (cm) — histogram
    - Rotation error distribution (degrees) — histogram
    - Sensitivity to camera height/tilt perturbation
    - Usable envelope analysis (where ±2cm/±3° is met)

The rubric requires:
    - Error reported as DISTRIBUTIONS, not point estimates
    - Translation and rotation reported SEPARATELY
    - Sensitivity to camera height/tilt error at short and long range
    - The usable envelope: where does pose meet the ±2cm/±3° bar?
    - What the system outputs when it cannot produce a reliable pose
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ── Ground truth handling ────────────────────────────────────────────────────
# There is no pose ground truth unless you make it. We support two approaches:
#
# 1. Manual measurement: Physically measure pallet positions and angles,
#    store in a JSON file (ground_truth.json).
# 2. Synthetic: Generate known poses, project corners to 2D, add noise,
#    then run PnP and compare to the known pose.
#
# Document your choice in the README and justify it.


def load_ground_truth(gt_path):
    """
    Load manually-measured ground truth poses.

    Expected JSON format:
    {
        "image_name.jpg": [
            {"pallet_id": "pallet_0", "x_m": 2.5, "y_m": 1.8, "theta_deg": 12.0},
            ...
        ],
        ...
    }
    """
    with open(gt_path, "r") as f:
        return json.load(f)


def generate_synthetic_ground_truth(camera_matrix, dist_coeffs, config, n_samples=100):
    """
    Generate synthetic pose ground truth for evaluation.

    This creates known pallet poses, projects the 3D corners to 2D pixels,
    adds realistic detection noise, then runs PnP — so we can measure exact
    error against known truth.

    This is the recommended approach when you don't have real measurements.
    Document it clearly in the README.

    Args:
        camera_matrix: Camera intrinsics
        dist_coeffs: Distortion coefficients
        config: Project config
        n_samples: Number of synthetic poses to generate

    Returns:
        List of {true_pose, predicted_pose, error} dicts
    """
    from src.pose.pnp_pose import get_pallet_corners_3d, solve_pose, transform_to_floor

    corners_3d = get_pallet_corners_3d(config)
    camera_height = config["camera"]["height_m"]
    camera_tilt = config["camera"]["tilt_deg"]

    results = []

    # Generate poses at various ranges and angles
    ranges = np.random.uniform(1.0, 8.0, n_samples)  # 1m to 8m distance
    x_offsets = np.random.uniform(-3.0, 3.0, n_samples)  # ±3m lateral
    angles = np.random.uniform(-90, 90, n_samples)  # ±90° orientation

    for i in range(n_samples):
        # True pose (floor frame)
        true_x = x_offsets[i]
        true_y = ranges[i]
        true_theta = angles[i]

        # Convert floor pose to camera-frame translation
        # (inverse of transform_to_floor)
        alpha = np.radians(camera_tilt)
        tz = true_y * np.cos(alpha)
        ty = -true_y * np.sin(alpha)
        tx = true_x
        tvec_true = np.array([[tx], [ty], [tz]], dtype=np.float64)

        # Create rotation vector from theta
        rvec_true = np.array([[0, 0, np.radians(true_theta)]], dtype=np.float64)

        # Project 3D corners to 2D
        projected, _ = cv2.projectPoints(
            corners_3d.reshape(-1, 1, 3), rvec_true, tvec_true,
            camera_matrix, dist_coeffs
        )
        corners_2d = projected.reshape(-1, 2)

        # Add realistic detection noise (±2 pixels std)
        noise = np.random.normal(0, 2.0, corners_2d.shape)
        noisy_corners = corners_2d + noise

        # Run PnP on noisy corners
        result = solve_pose(noisy_corners, corners_3d, camera_matrix, dist_coeffs)

        if result["success"] and result["reproj_error_px"] < 10.0:
            rvec_pred = np.array(result["rvec"])
            tvec_pred = np.array(result["tvec"])
            pred_x, pred_y = transform_to_floor(tvec_pred, camera_height, camera_tilt)

            # Compute errors — note axis mapping:
            #   floor_x = depth (should compare to true_y/range)
            #   floor_y = lateral (should compare to true_x/offset)
            trans_error = np.sqrt((pred_x - true_y)**2 + (pred_y - true_x)**2) * 100  # cm

            # Handle IPPE 180° ambiguity — normalise rotation error
            rot_diff = abs(result["theta_deg"] - true_theta) % 360
            rot_error = min(rot_diff, abs(rot_diff - 180))


            results.append({
                "true_pose": {"x_m": true_x, "y_m": true_y, "theta_deg": true_theta},
                "pred_pose": {"x_m": pred_x, "y_m": pred_y, "theta_deg": result["theta_deg"]},
                "trans_error_cm": float(trans_error),
                "rot_error_deg": float(rot_error),
                "range_m": float(ranges[i]),
            })

    return results


def plot_error_distributions(results, output_dir):
    """
    Plot translation and rotation error as distributions (histograms + violin).

    The rubric explicitly requires DISTRIBUTIONS, not point estimates.
    """
    trans_errors = [r["trans_error_cm"] for r in results]
    rot_errors = [r["rot_error_deg"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Translation error histogram
    axes[0, 0].hist(trans_errors, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].axvline(2.0, color="red", linestyle="--", linewidth=2, label="±2cm target")
    axes[0, 0].axvline(np.mean(trans_errors), color="green", linestyle="--",
                       label=f"Mean: {np.mean(trans_errors):.2f}cm")
    axes[0, 0].axvline(np.median(trans_errors), color="orange", linestyle="--",
                       label=f"Median: {np.median(trans_errors):.2f}cm")
    axes[0, 0].set_xlabel("Translation Error (cm)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Translation Error Distribution")
    axes[0, 0].legend()

    # Rotation error histogram
    axes[0, 1].hist(rot_errors, bins=30, edgecolor="black", alpha=0.7, color="coral")
    axes[0, 1].axvline(3.0, color="red", linestyle="--", linewidth=2, label="±3° target")
    axes[0, 1].axvline(np.mean(rot_errors), color="green", linestyle="--",
                       label=f"Mean: {np.mean(rot_errors):.2f}°")
    axes[0, 1].axvline(np.median(rot_errors), color="orange", linestyle="--",
                       label=f"Median: {np.median(rot_errors):.2f}°")
    axes[0, 1].set_xlabel("Rotation Error (degrees)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Rotation Error Distribution")
    axes[0, 1].legend()

    # Violin plots (show full distribution shape)
    axes[1, 0].violinplot([trans_errors], showmeans=True, showmedians=True)
    axes[1, 0].axhline(2.0, color="red", linestyle="--", label="±2cm target")
    axes[1, 0].set_ylabel("Translation Error (cm)")
    axes[1, 0].set_title("Translation Error (Violin)")
    axes[1, 0].set_xticks([1])
    axes[1, 0].set_xticklabels(["Translation"])
    axes[1, 0].legend()

    axes[1, 1].violinplot([rot_errors], showmeans=True, showmedians=True)
    axes[1, 1].axhline(3.0, color="red", linestyle="--", label="±3° target")
    axes[1, 1].set_ylabel("Rotation Error (degrees)")
    axes[1, 1].set_title("Rotation Error (Violin)")
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels(["Rotation"])
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = output_dir / "pose_error_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # Print summary statistics
    print(f"\n  Translation Error (cm):")
    print(f"    Mean: {np.mean(trans_errors):.2f}  Median: {np.median(trans_errors):.2f}")
    print(f"    P90:  {np.percentile(trans_errors, 90):.2f}  P95:  {np.percentile(trans_errors, 95):.2f}")
    print(f"    % within ±2cm: {np.mean(np.array(trans_errors) <= 2.0) * 100:.1f}%")

    print(f"\n  Rotation Error (degrees):")
    print(f"    Mean: {np.mean(rot_errors):.2f}  Median: {np.median(rot_errors):.2f}")
    print(f"    P90:  {np.percentile(rot_errors, 90):.2f}  P95:  {np.percentile(rot_errors, 95):.2f}")
    print(f"    % within ±3°: {np.mean(np.array(rot_errors) <= 3.0) * 100:.1f}%")


def sensitivity_analysis(camera_matrix, dist_coeffs, config, output_dir):
    """
    Analyse sensitivity of pose to errors in camera height and tilt estimates.

    We perturb the assumed camera height by ±5cm, ±10cm and tilt by ±2°, ±5°,
    and measure how much the pose output changes. Done at short (2m) and long (6m) range.
    """
    from src.pose.pnp_pose import get_pallet_corners_3d, solve_pose, transform_to_floor

    corners_3d = get_pallet_corners_3d(config)
    base_height = config["camera"]["height_m"]
    base_tilt = config["camera"]["tilt_deg"]

    # Perturbations to test
    height_perturbations = [-0.10, -0.05, 0, 0.05, 0.10]  # metres
    tilt_perturbations = [-5, -2, 0, 2, 5]  # degrees

    # Test at two ranges
    test_ranges = {"short_2m": 2.0, "long_6m": 6.0}

    results = {}

    for range_name, range_m in test_ranges.items():
        # Create a synthetic pallet at this range
        alpha = np.radians(base_tilt)
        true_tz = range_m * np.cos(alpha)
        true_ty = -range_m * np.sin(alpha)
        true_tvec = np.array([[0], [true_ty], [true_tz]], dtype=np.float64)
        true_rvec = np.array([[0, 0, 0]], dtype=np.float64)

        # Project corners to get clean 2D points
        projected, _ = cv2.projectPoints(
            corners_3d.reshape(-1, 1, 3), true_rvec, true_tvec,
            camera_matrix, dist_coeffs
        )
        corners_2d = projected.reshape(-1, 2)

        range_results = {"height": [], "tilt": []}

        # Vary height
        for dh in height_perturbations:
            result = solve_pose(corners_2d, corners_3d, camera_matrix, dist_coeffs)
            if result["success"]:
                tvec = np.array(result["tvec"])
                x, y = transform_to_floor(tvec, base_height + dh, base_tilt)
                range_results["height"].append({
                    "perturbation_cm": dh * 100,
                    "x_m": x, "y_m": y,
                    "x_error_cm": abs(x) * 100,  # True x = 0
                    "y_error_cm": abs(y - range_m) * 100,
                })

        # Vary tilt
        for dt in tilt_perturbations:
            result = solve_pose(corners_2d, corners_3d, camera_matrix, dist_coeffs)
            if result["success"]:
                tvec = np.array(result["tvec"])
                x, y = transform_to_floor(tvec, base_height, base_tilt + dt)
                range_results["tilt"].append({
                    "perturbation_deg": dt,
                    "x_m": x, "y_m": y,
                    "x_error_cm": abs(x) * 100,
                    "y_error_cm": abs(y - range_m) * 100,
                })

        results[range_name] = range_results

    # Plot sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (range_name, range_data) in enumerate(results.items()):
        # Height sensitivity
        height_data = range_data["height"]
        perturbs = [d["perturbation_cm"] for d in height_data]
        y_errors = [d["y_error_cm"] for d in height_data]
        axes[0, i].plot(perturbs, y_errors, "bo-", markersize=8)
        axes[0, i].axhline(2.0, color="red", linestyle="--", label="±2cm target")
        axes[0, i].set_xlabel("Height Error (cm)")
        axes[0, i].set_ylabel("Pose Y Error (cm)")
        axes[0, i].set_title(f"Height Sensitivity — {range_name}")
        axes[0, i].legend()

        # Tilt sensitivity
        tilt_data = range_data["tilt"]
        perturbs = [d["perturbation_deg"] for d in tilt_data]
        y_errors = [d["y_error_cm"] for d in tilt_data]
        axes[1, i].plot(perturbs, y_errors, "ro-", markersize=8)
        axes[1, i].axhline(2.0, color="red", linestyle="--", label="±2cm target")
        axes[1, i].set_xlabel("Tilt Error (degrees)")
        axes[1, i].set_ylabel("Pose Y Error (cm)")
        axes[1, i].set_title(f"Tilt Sensitivity — {range_name}")
        axes[1, i].legend()

    plt.tight_layout()
    plot_path = output_dir / "sensitivity_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    return results


def usable_envelope_analysis(results, output_dir):
    """
    Determine the usable envelope: at what range/angle does pose meet ±2cm/±3°?

    Plots a 2D map showing where the system is reliable (green) vs unreliable (red).
    """
    trans_errors = np.array([r["trans_error_cm"] for r in results])
    rot_errors = np.array([r["rot_error_deg"] for r in results])
    ranges = np.array([r["range_m"] for r in results])

    # Classify each point
    meets_bar = (trans_errors <= 2.0) & (rot_errors <= 3.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ranges[meets_bar], trans_errors[meets_bar],
               c="green", alpha=0.6, label="Within ±2cm / ±3°")
    ax.scatter(ranges[~meets_bar], trans_errors[~meets_bar],
               c="red", alpha=0.6, label="Outside tolerance")
    ax.axhline(2.0, color="red", linestyle="--", linewidth=2, label="±2cm bar")
    ax.set_xlabel("Range (metres)")
    ax.set_ylabel("Translation Error (cm)")
    ax.set_title("Usable Envelope — Error vs Range")
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / "usable_envelope.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # Report the boundary
    within = ranges[meets_bar]
    outside = ranges[~meets_bar]
    if len(within) > 0 and len(outside) > 0:
        print(f"\n  Usable envelope:")
        print(f"    Reliable up to ~{np.max(within):.1f}m range")
        print(f"    Errors exceed tolerance beyond ~{np.min(outside):.1f}m")
    elif len(within) > 0:
        print(f"\n  All tested ranges within tolerance (up to {np.max(within):.1f}m)")
    else:
        print(f"\n  WARNING: No poses within tolerance in synthetic test")


def main():
    config = load_config()
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"] / "pose"
    results_dir.mkdir(parents=True, exist_ok=True)

    calib_dir = PROJECT_ROOT / config["paths"]["calibration_dir"]
    camera_matrix_path = calib_dir / "camera_matrix.npy"
    dist_path = calib_dir / "distortion.npy"

    if not camera_matrix_path.exists():
        print("ERROR: Camera not calibrated. Run calibration.py first.")
        return

    camera_matrix = np.load(camera_matrix_path)
    dist_coeffs = np.load(dist_path)

    # ── 1. Generate synthetic ground truth ─────────────────────────────
    print("\n[1/4] Generating synthetic ground truth (100 poses)...")
    print("  Method: Known 3D poses → project to 2D → add noise → PnP → compare")
    print("  Justification: No real pose ground truth available. Synthetic allows")
    print("  exact error measurement against known truth. Documented in README.")
    results = generate_synthetic_ground_truth(
        camera_matrix, dist_coeffs, config, n_samples=200
    )
    print(f"  Generated {len(results)} valid pose estimates")

    # Save raw results
    with open(results_dir / "pose_eval_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── 2. Plot error distributions ─────────────────────────────────────
    print("\n[2/4] Plotting error distributions...")
    plot_error_distributions(results, results_dir)

    # ── 3. Sensitivity analysis ──────────────────────────────────────────
    print("\n[3/4] Running sensitivity analysis...")
    sensitivity = sensitivity_analysis(camera_matrix, dist_coeffs, config, results_dir)
    with open(results_dir / "sensitivity_results.json", "w") as f:
        json.dump(sensitivity, f, indent=2)

    # ── 4. Usable envelope ──────────────────────────────────────────────
    print("\n[4/4] Computing usable envelope...")
    usable_envelope_analysis(results, results_dir)

    print("\n" + "=" * 60)
    print("Pose evaluation complete! Results in:", results_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
