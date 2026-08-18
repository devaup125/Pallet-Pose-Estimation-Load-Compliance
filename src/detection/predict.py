"""
Detection prediction — runs YOLOv8 on images and returns structured results.

Usage:
    python src/detection/predict.py --image path/to/image.jpg
    python src/detection/predict.py --image-dir path/to/images/
    python src/detection/predict.py --image path/to/image.jpg --weights models/best.pt --conf 0.5

Output:
    Prints detection results to console and saves annotated image.
    Returns structured detection data for the pipeline.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_model(weights_path):
    """
    Load a trained YOLOv8 model.

    Args:
        weights_path: Path to .pt file (e.g. models/best.pt)

    Returns:
        YOLO model object ready for inference
    """
    model = YOLO(str(weights_path))
    print(f"Loaded model from {weights_path}")
    return model


def detect(model, image_path, conf_threshold=0.5):
    """
    Run detection on a single image.

    If the model outputs keypoints (YOLOv8-pose), use them directly.
    Otherwise, estimate 4 pallet corners from the bounding box as a fallback.
    """
    results = model(image_path, conf=conf_threshold, verbose=False)

    detections = []

    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            confidence = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            class_name = model.names[cls_id]

            detection = {
                "bbox": bbox,
                "confidence": confidence,
                "class_name": class_name,
                "keypoints": None,
            }

            # If using YOLOv8-pose, extract real keypoints
            has_keypoints = (hasattr(result, "keypoints") 
                            and result.keypoints is not None 
                            and hasattr(result.keypoints, 'data')
                            and result.keypoints.data is not None)
            
            if has_keypoints:
                try:
                    kpts = result.keypoints.data[i].cpu().numpy()
                    detection["keypoints"] = kpts.tolist()
                except (IndexError, AttributeError):
                    has_keypoints = False

            # Fallback: estimate 4 corners from bounding box
            # if no keypoints are available (bbox-only model)
            if not has_keypoints or detection["keypoints"] is None:
                x1, y1, x2, y2 = bbox
                # Inset slightly (5%) to get pallet edges, not bbox edges
                w = x2 - x1
                h = y2 - y1
                inset_x = w * 0.05
                inset_y = h * 0.05
                
                # 4 corners: top-left, top-right, bottom-right, bottom-left
                # Matches the 3D corner order in config/default.yaml
                detection["keypoints"] = [
                    [x1 + inset_x, y1 + inset_y, confidence],       # top-left
                    [x2 - inset_x, y1 + inset_y, confidence],       # top-right
                    [x2 - inset_x, y2 - inset_y, confidence],       # bottom-right
                    [x1 + inset_x, y2 - inset_y, confidence],       # bottom-left
                ]
                detection["keypoints_source"] = "bbox_estimated"

            detections.append(detection)

    return detections



def draw_detections(image_path, detections, output_path=None):
    """
    Draw bounding boxes and keypoints on the image for visualisation.

    Args:
        image_path: Path to input image
        detections: List of detection dicts from detect()
        output_path: Where to save annotated image (optional)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not read image {image_path}")
        return

    colors = {
        "pallet": (0, 255, 0),      # Green
        "box": (0, 165, 255),        # Orange
    }

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = colors.get(det["class_name"], (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw keypoints if present
        if det["keypoints"]:
            for kp in det["keypoints"]:
                kx, ky, kconf = kp
                if kconf > 0.5:  # Only draw confident keypoints
                    cv2.circle(image, (int(kx), int(ky)), 5, (0, 0, 255), -1)
                    # Label the keypoint index
                    idx = det["keypoints"].index(kp)
                    cv2.putText(image, str(idx), (int(kx) + 8, int(ky)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw lines connecting corners (for pallet geometry visualisation)
            kpts = [k for k in det["keypoints"] if k[2] > 0.5]
            if len(kpts) == 4:
                pts = np.array([[k[0], k[1]] for k in kpts], dtype=np.int32)
                cv2.polylines(image, [pts], True, (255, 0, 0), 2)

    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Annotated image saved to {output_path}")
    else:
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run pallet detection on an image")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, help="Path to a directory of images")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                        help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--output-dir", type=str, default="results/detection",
                        help="Where to save annotated images")
    args = parser.parse_args()

    config = load_config()
    weights_path = PROJECT_ROOT / args.weights

    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print("Train the model first using the Colab notebook.")
        return

    model = load_model(weights_path)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images to process
    if args.image:
        images = [args.image]
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    else:
        print("ERROR: Provide --image or --image-dir")
        return

    # Run detection on each image
    all_results = []
    for img_path in images:
        print(f"\nProcessing: {img_path}")
        detections = detect(model, img_path, args.conf)

        print(f"  Found {len(detections)} detection(s)")
        for det in detections:
            print(f"    {det['class_name']}: conf={det['confidence']:.3f} "
                  f"bbox={[int(v) for v in det['bbox']]}")
            if det["keypoints"]:
                print(f"    keypoints: {len(det['keypoints'])} points")

        # Save annotated image
        out_path = output_dir / f"det_{Path(img_path).name}"
        draw_detections(img_path, detections, out_path)

        all_results.append({
            "image": str(img_path),
            "detections": detections,
        })

    # Save all results as JSON
    results_path = output_dir / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
