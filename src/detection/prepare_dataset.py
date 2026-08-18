"""
Dataset preparation — downloads from Roboflow, merges, and creates splits.

Usage:
    python src/detection/prepare_dataset.py

This script:
1. Downloads both Roboflow datasets in YOLOv8 format
2. Merges them into a single dataset
3. Creates train/val/test splits
4. Generates a data.yaml for YOLOv8 training
"""

import os
import sys
import shutil
import random
from pathlib import Path

import yaml

# ── Setup paths ──────────────────────────────────────────────────────────────
# Resolve project root (go up 2 levels: src/detection/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    """Load the YAML config file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def download_roboflow_dataset(api_key, dataset_id, version, location):
    """
    Download a dataset from Roboflow using their Python package.

    Args:
        api_key: Your Roboflow API key
        dataset_id: The dataset identifier (e.g. 'plh-c-1-veibf')
        version: Dataset version number (usually 1)
        location: Where to save the downloaded dataset

    Returns:
        Path to the downloaded dataset folder
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    # The workspace is 'rj-xvfw4' based on the URLs in the assignment
    workspace = "rj-xvfw4"
    project = rf.workspace(workspace).project(dataset_id)
    dataset = project.version(version).download("yolov8", location=location)

    return location


def merge_datasets(raw_dir, processed_dir, class_names):
    """
    Merge multiple YOLO-format datasets into one.

    Each Roboflow download contains:
        - train/images, train/labels
        - valid/images, valid/labels
        - test/images, test/labels
        - data.yaml

    We merge all images into a single pool, then re-split.
    """
    processed_dir = Path(processed_dir)
    all_images_dir = processed_dir / "all_images"
    all_labels_dir = processed_dir / "all_labels"
    all_images_dir.mkdir(parents=True, exist_ok=True)
    all_labels_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(raw_dir)

    # Track stats for DATASET.md
    stats = {"total_images": 0, "datasets": []}

    # Find all downloaded dataset folders
    dataset_folders = [d for d in raw_dir.iterdir() if d.is_dir()]

    for ds_folder in dataset_folders:
        ds_images_count = 0

        # Each dataset has train/valid/test subfolders
        for split in ["train", "valid", "test"]:
            img_dir = ds_folder / split / "images"
            lbl_dir = ds_folder / split / "labels"

            if not img_dir.exists():
                continue

            for img_file in img_dir.glob("*"):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                # Copy image with dataset prefix to avoid name collisions
                new_name = f"{ds_folder.name}_{img_file.name}"
                shutil.copy2(img_file, all_images_dir / new_name)

                # Copy corresponding label
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    shutil.copy2(lbl_file, all_labels_dir / new_name.replace(
                        img_file.suffix, ".txt"
                    ))

                ds_images_count += 1

        stats["datasets"].append({
            "name": ds_folder.name,
            "images": ds_images_count
        })
        stats["total_images"] += ds_images_count
        print(f"  Merged {ds_images_count} images from {ds_folder.name}")

    return stats


def create_splits(processed_dir, splits_dir, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Create train/val/test splits from the merged image pool.

    Splits are random (no location metadata available in source datasets).
    This is documented in DATASET.md as a known limitation.
    """
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)

    all_images_dir = processed_dir / "all_images"
    all_labels_dir = processed_dir / "all_labels"

    # Get all image filenames
    image_files = sorted([f.name for f in all_images_dir.iterdir()
                          if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if len(image_files) == 0:
        print("ERROR: No images found. Did you download the datasets first?")
        sys.exit(1)

    # Shuffle for random split
    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:],
    }

    # Create split directories
    for split_name, files in splits.items():
        for subdir in ["images", "labels"]:
            (splits_dir / split_name / subdir).mkdir(parents=True, exist_ok=True)

        for fname in files:
            # Copy image
            shutil.copy2(
                all_images_dir / fname,
                splits_dir / split_name / "images" / fname
            )
            # Copy label if exists
            lbl_name = fname.rsplit(".", 1)[0] + ".txt"
            lbl_path = all_labels_dir / lbl_name
            if lbl_path.exists():
                shutil.copy2(lbl_path, splits_dir / split_name / "labels" / lbl_name)

        print(f"  {split_name}: {len(files)} images")

    return splits


def write_data_yaml(splits_dir, class_names):
    """
    Write the data.yaml file that YOLOv8 needs for training.

    This file tells YOLOv8 where to find images and what classes exist.
    """
    splits_dir = Path(splits_dir).resolve()

    data_yaml = {
        "path": str(splits_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = splits_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"  Written: {yaml_path}")
    return yaml_path


def main():
    """Main entry point for dataset preparation."""
    config = load_config()

    print("=" * 60)
    print("PALLET POSE — Dataset Preparation")
    print("=" * 60)

    raw_dir = PROJECT_ROOT / config["paths"]["data_raw"]
    processed_dir = PROJECT_ROOT / config["paths"]["data_processed"]
    splits_dir = PROJECT_ROOT / config["paths"]["data_splits"]

    # ── Step 1: Download datasets ───────────────────────────────────────
    print("\n[1/4] Downloading datasets from Roboflow...")
    print("  NOTE: If you haven't already, install roboflow: pip install roboflow")
    print("  Get your API key from: https://app.roboflow.com/")
    print("  Go to Settings > API > Copy your private API key")
    print()

    api_key = config["dataset"]["roboflow_api_key"]
    if api_key == "YOUR_ROBOFLOW_API_KEY":
        print("  ERROR: Set your Roboflow API key in config/default.yaml first!")
        print("  Look for 'roboflow_api_key' in the config file.")
        print()
        print("  Alternatively, manually download the datasets from:")
        print(f"    1. https://app.roboflow.com/rj-xvfw4/{config['dataset']['dataset_1']}/1")
        print(f"    2. https://app.roboflow.com/rj-xvfw4/{config['dataset']['dataset_2']}/1")
        print("  Export as YOLOv8 format and extract into data/raw/")
        sys.exit(1)

    datasets_info = [
        (config["dataset"]["dataset_1"], 1, str(raw_dir / config["dataset"]["dataset_1"])),
        (config["dataset"]["dataset_2"], 1, str(raw_dir / config["dataset"]["dataset_2"])),
    ]

    for ds_id, version, location in datasets_info:
        print(f"  Downloading {ds_id}...")
        download_roboflow_dataset(api_key, ds_id, version, location)

    # ── Step 2: Merge datasets ──────────────────────────────────────────
    print("\n[2/4] Merging datasets...")
    stats = merge_datasets(raw_dir, processed_dir, config["dataset"]["classes"])

    print(f"\n  Total images merged: {stats['total_images']}")

    # Save stats for DATASET.md reference
    stats_path = processed_dir / "dataset_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")

    # ── Step 3: Create splits ───────────────────────────────────────────
    print("\n[3/4] Creating train/val/test splits...")
    splits = create_splits(
        processed_dir, splits_dir,
        config["dataset"]["train_ratio"],
        config["dataset"]["val_ratio"],
        config["dataset"]["test_ratio"],
    )

    # ── Step 4: Write data.yaml ─────────────────────────────────────────
    print("\n[4/4] Writing data.yaml for YOLOv8...")
    write_data_yaml(splits_dir, config["dataset"]["classes"])

    print("\n" + "=" * 60)
    print("DONE! Dataset ready for training.")
    print(f"  Splits in: {splits_dir}")
    print(f"  data.yaml: {splits_dir / 'data.yaml'}")
    print()
    print("Next steps:")
    print("  1. Open notebooks/train_yolov8_colab.ipynb in Google Colab")
    print("  2. Upload your data/splits/ folder to Colab")
    print("  3. Run the training notebook")
    print("=" * 60)


if __name__ == "__main__":
    main()
