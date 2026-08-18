# Pallet Labelling Guideline (One Page)

## Classes

| Class | Description |
|-------|-------------|
| `pallet` | The pallet deck (top surface + visible stringers) |
| `box` | Individual carton/case on the pallet (optional, for load analysis) |

## Pallet Bounding Box

- Draw tightly around the **entire visible pallet** — deck boards + visible stringers.
- Do not include the load (boxes/wrap) inside the pallet box.
- If the pallet is partially occluded, label the visible portion.

## Pallet Corners (Keypoints — for Pose Estimation)

Mark the 4 corners of the pallet **top surface** (the deck). Order is clockwise:

1. **Top-left** — furthest corner on the left
2. **Top-right** — furthest corner on the right
3. **Bottom-right** — nearest corner on the right
4. **Bottom-left** — nearest corner on the left

- If a corner is occluded, mark it visible=false.
- Be as precise as possible — corner precision directly affects pose accuracy.

## Quality Rules

- One labeller for consistency.
- Ambiguous cases (heavy occlusion, poor lighting) flagged with `ambiguous=true`.
- At least 10% of labels double-checked.

## Why This Matters

The 4 corners feed into PnP (Perspective-n-Point) pose estimation. A 2-pixel error in corner position can translate to several centimetres of pose error at range. Corner precision is the single biggest factor in pose accuracy.
