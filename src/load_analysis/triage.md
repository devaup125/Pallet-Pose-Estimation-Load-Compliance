# SOP-PAL-03 Load Compliance — Triage of Verifiability

This document triages all 8 SOP rules for verifiability from a single camera
view (1.2m height, 20° tilt, one side of the pallet visible).

## Classification scheme

- **Verifiable** — Can be checked directly from the image with reasonable confidence.
- **Partially verifiable** — Can be estimated under stated assumptions; confidence is moderate.
- **Not verifiable** — Cannot be checked from this camera view; requires additional sensor or manual inspection.

## Triage table

| # | SOP Rule | Classification | Justification |
|---|----------|---------------|---------------|
| 1 | No box overhangs >3cm | **Partially verifiable** | If we detect both the pallet boundary (corners) and the load boundary (topmost box edges), we can project the pallet footprint and measure overhang. Limitation: we only see one side, so overhang on the far side is invisible. Confidence derived from corner detection score and projection residual. |
| 2 | Load height ≤1.8m | **Partially verifiable** | We can estimate height using the known pallet dimensions as a scale reference. The pallet width (1.2m) in the image gives us pixels-per-metre; we then measure the load height in pixels and convert. Limitation: perspective and tilt introduce error. The far edge of the load is at a different depth than the near edge, so the scale factor varies. We use the pallet centre depth as the reference. |
| 3 | Box column alignment ±15° | **Partially verifiable** | If individual boxes are detectable (edge detection or a second detector), we can measure the angle of box edges relative to the pallet's principal axes (from the pose). Limitation: boxes wrapped in plastic film are hard to separate; edges may be occluded. |
| 4 | Larger boxes below smaller | **Partially verifiable** | Requires detecting individual box sizes at different heights. Feasible if boxes are visible and distinguishable. Often blocked by stretch wrap. |
| 5 | Stretch-wrapped | **Partially verifiable** | Stretch wrap has a distinctive shiny/semi-transparent texture. A simple texture classifier (variance of Laplacian, specular highlight detection) can give a moderate-confidence check. Not a trained model — just a heuristic. |
| 6 | No damaged/crushed box | **Not verifiable (in this scope)** | Would require a trained damage-detection model with sufficient labelled data of damaged boxes. Sourcing this data and training a reliable classifier is not feasible in 5 days. Verdict: manual_inspection. |
| 7 | Load centred ±10cm | **Partially verifiable** | Compare the centroid of the detected load region with the pallet centre (from pose). The load centroid in the image can be approximated from the bounding box of detected boxes. Limitation: load shape is irregular; centroid estimate is approximate. |
| 8 | Pallet undamaged (no broken boards) | **Not verifiable** | The camera sees the top of the pallet (covered by load) or one side. Broken boards or split stringers are typically on the bottom or hidden. Even the visible side is often obscured by the load. Verdict: manual_inspection. |

## Implemented subset

Based on the triage, the following checks are implemented:

1. **Overhang check** (Rule 1) — partial, near-side only
2. **Height estimate** (Rule 2) — using pallet dimensions as scale
3. **Column alignment** (Rule 3) — if box edges are detectable
4. **Load centring** (Rule 7) — centroid comparison

Rules 4, 5 are implemented as lightweight heuristics with low confidence.
Rules 6, 8 return `manual_inspection` with justification.

## Confidence derivation

Each check's confidence is derived from:
- **Corner detection score** (from YOLOv8 keypoint confidence)
- **PnP reprojection error** (lower = more reliable pose = more reliable geometric measurements)
- **Occlusion estimate** (how much of the pallet/load is visible)

Formula (documented, not arbitrary):
```
check_confidence = corner_confidence * (1 - reproj_error / max_error) * visibility_factor
```
Where:
- `corner_confidence`: mean keypoint confidence (0-1)
- `reproj_error`: PnP reprojection error in pixels (normalised)
- `visibility_factor`: fraction of pallet boundary visible (0-1)

## Weighting pose quality vs load compliance

The overall verdict considers:
- If pose confidence < 0.4 → overall verdict is `manual_inspection` regardless of load checks
  (because geometric measurements are unreliable without a good pose)
- If pose confidence ≥ 0.4, load checks are weighted by their individual confidence
- Any `fail` on a high-confidence check (≥0.7) → overall `fail`
- Low-confidence fails → overall `manual_inspection`
