# react-native-ghost-guide: Academic Package Documentation

### 1. Package Overview (The "What")
The `react-native-ghost-guide` package provides real-time pose alignment and "ghost" skeleton overlay logic for React Native applications. Its primary responsibility is to take live pose landmark inputs, align a reference skeleton to the user's body proportions, and output a composite structure used by the UI to render a guided overlay and alignment status. In the broader workout-hacker system, this package serves as the pose guidance core: it translates raw landmark frames into a normalized ghost skeleton and alignment metrics, which are then rendered by the consuming app.

The package solves the problem of consistent, frame-by-frame pose alignment between a live user and a prerecorded exercise reference sequence. This includes scaling the reference pose to the user's size, applying rotational adjustments to match orientation, and producing a stable output representation suitable for rendering and alignment feedback.

### 2. Architectural Rationale (The "Why")
This package is implemented as a TypeScript-only library to maximize portability and simplify integration with React Native apps without native build requirements. The computational core is purely deterministic and stateless per frame, except for a small, explicit in-memory reference state managed through `loadReference`. This design favors predictability and ease of testing, while keeping data transformations transparent and auditable.

Key architectural choices evident in the source:
1. **Typed domain model:** The code defines explicit data types (`Point3D`, `Skeleton`, `RawLandmarkFrame`, `ReferenceData`, `ProcessResult`) to make the geometry pipeline and IO boundaries unambiguous. This is consistent with a goal of reproducibility and strict contract definition.
2. **Reference-as-state, processing-as-pure function:** The API separates reference loading (`loadReference`, `setReferenceFrameIndex`) from per-frame processing (`processFrame`, `processFrameWithReference`). This enables two usage modes: static reference with stateful frame indices or fully externalized reference frames for each call.
3. **Geometry-focused alignment pipeline:** The algorithm uses Euclidean distances, vector normalization, cross/dot products, and rotation matrices to scale and orient the reference skeleton to match the user, avoiding heavier optimization or ML-based alignment that would be harder to audit and maintain.

Limitations on provenance analysis: This workspace does not contain a `.git` directory, so commit history and PR discussion are unavailable. As a result, architectural intent is inferred from code structure, naming, and example usage rather than explicit commit-level rationale.

Trade-offs reflected in the implementation:
- **Performance vs. simplicity:** The pipeline performs per-point transforms with straightforward vector math instead of more complex pose fitting. This keeps the code readable and fast enough for real-time use, at the cost of not optimizing alignment across all joints.
- **Modularity vs. global state:** The `referenceData` module-level variable simplifies usage but introduces shared state. This trades explicit dependency injection for convenience in common app flows.
- **Robustness vs. precision:** Landmark values are defaulted to `0` when missing, and alignment uses a simple averaged distance threshold. This is robust to missing data but may under-report misalignment in some edge cases.

### 3. Internal Mechanics (The "How")
The package transforms inputs in a linear, geometry-centric pipeline:

1. **Input normalization**
   - Raw landmarks (`RawLandmarkFrame`) are converted to `Point3D[]` using `landmarksToPoints`, coercing missing values to `0`.
   - A `ReferenceData` object is normalized to ensure index and rep counters are defined.

2. **Reference scaling and alignment**
   - `scaleToUser` computes scaling factors based on shoulder width, hip width, and torso height using joint indices aligned with the MediaPipe pose model (e.g., shoulders at 11/12, hips at 23/24).
   - It rotates the reference torso around an axis derived from the cross product of reference and user shoulder vectors, with a damped rotation strength.
   - It applies a yaw correction based on the user’s facing direction, minimizing over-rotation when the user is nearly front-facing.
   - It scales the reference skeleton anisotropically (width and height) relative to the user’s torso center.
   - It performs arm-length corrections for upper-limb joints to better match user proportions.

3. **Pose-specific adjustments (optional)**
   - `buildGhostSkeleton` optionally applies reference pose joint-angle constraints using `applyShoulderAngle` and `applyElbowAngle`. These functions rotate upper arm and forearm segments in 2D to match reference joint angles while preserving 3D positions where possible.

4. **Alignment check**
   - `checkAlignment` measures average 3D distance between key joints and reports alignment when the mean is below `0.1` (unitless, normalized space), yielding a boolean `isAligned` for UI feedback.

5. **Result emission**
   - `processFrame` returns a `ProcessResult` including the scaled ghost skeleton, the current checkpoint index, alignment status, and rep count.
   - `processFrameWithReference` injects a `ghostSkeleton` derived from explicit reference frames, enabling on-demand playback synchronized to a UI-controlled frame index.

The library is synchronous and uses no background threads. The calling application is expected to manage any timing, streaming, and scheduling (e.g., 30 Hz frame updates), which is demonstrated in the example app.

**Geometry algorithm details (executable math with intent)**
This section expands each geometric primitive and composite transformation used by the library.

**Distance metrics**
- `dist2(a, b)` computes Euclidean distance in the image plane: `sqrt((ax-bx)^2 + (ay-by)^2)`. It is used for width-like measurements (shoulder width, hip width, arm segment lengths) where depth should not dominate scaling.
- `dist3(a, b)` computes Euclidean distance in 3D: `sqrt(dx^2 + dy^2 + dz^2)`. It is used for torso height and alignment checks where depth should matter.

**Vector normalization and degeneracy handling**
- `normalize3(v)` returns `v / ||v||`, where `||v|| = sqrt(x^2 + y^2 + z^2)`. If the length is near zero, it returns the zero vector to avoid divide-by-zero. This is used before cross/dot products to ensure rotation axes and directions are unit length.

**Cross and dot products (orientation and angle)**
- `cross3(a, b)` yields a vector orthogonal to the plane spanned by `a` and `b`. Its direction encodes the rotation axis needed to move `a` toward `b`.
- `dot3(a, b)` measures cosine similarity when both vectors are normalized: `dot = cos(theta)`. Clamping to `[-1, 1]` prevents numerical errors before `acos`.

**Rodrigues' rotation formula (3D rotation)**
- `rotate3D(origin, point, axis, angle)` uses Rodrigues' formula to rotate a point around an axis through `origin`:
  - Translate to origin: `v = point - origin`.
  - Rotate: `v' = v*cos(angle) + (k x v)*sin(angle) + k*(k·v)*(1-cos(angle))`, where `k` is the unit axis.
  - Translate back: `origin + v'`.
- This enables smooth, stable torso and yaw alignment without building a full rotation matrix.

**Angle normalization (yaw wrapping)**
- `normalizeAngle(angle)` constrains angular deltas to `[-pi, pi]`, preventing discontinuities when angles wrap around `2pi`.

**Scaling to user proportions (`scaleToUser`)**
- **Scale estimation:**
  - Shoulder scale = user shoulder width / reference shoulder width.
  - Hip scale = user hip width / reference hip width.
  - Height scale = user torso length / reference torso length, where torso length is the average of left and right shoulder-to-hip distances.
- **Centering:**
  - The torso center is computed as the midpoint between the shoulder center and hip center. Scaling is applied relative to this center to preserve pose orientation while resizing.
- **Orientation alignment (torso rotation):**
  - Compute shoulder direction vectors for user and reference (`userVec`, `ghostVec`).
  - The rotation axis is `cross(ghostVec, userVec)`; the rotation angle is `acos(dot(ghostVec, userVec))`, then damped by `rotationStrength = 0.6` to avoid overfitting to noisy landmarks.
  - The entire reference skeleton is rotated about the reference torso center using `rotate3D`.
- **Yaw correction (facing direction):**
  - Yaw is derived via `atan2(z, x)` from shoulder vectors. The yaw delta aligns reference to user around the vertical axis.
  - A thresholded strength factor prevents abrupt rotations when the user is nearly front-facing. This stabilizes overlays during minor heading changes.
- **Anisotropic scaling:**
  - For each point, compute its vertical interpolation `t` between shoulders and hips.
  - Width scale is interpolated between shoulder and hip scales: `widthScale = shoulderScale + (hipScale - shoulderScale) * t`.
  - Positions are scaled around the torso center: `x` and `z` with `widthScale`, `y` with `heightScale`.
- **Arm-length correction:**
  - Upper-limb lengths (shoulder->elbow->wrist) are measured in 2D; scale factors are computed separately for left and right arms.
  - Elbow and wrist points are scaled from the shoulder origin using `scaleFrom` to match user arm proportions without affecting torso scaling.

**2D angle computations for pose adjustment**
- `angleBetweenSigned(a, b)` returns a signed angle using `atan2(cross, dot)` in 2D, preserving direction (clockwise vs. counterclockwise). This is crucial for applying joint rotations with the correct orientation.
- `rotate2D(v, angle)` applies a planar rotation around the origin using the standard rotation matrix. The z component is preserved to keep depth information from the original 3D pose.

**Joint-specific pose correction**
- `applyShoulderAngle`:
  - Computes the desired shoulder angle from reference torso and upper-arm vectors.
  - Computes the current angle in the ghost pose.
  - Rotates the ghost upper arm and forearm around the shoulder to match the reference angle, using `rotate2D` on the XY plane.
  - This keeps the shoulder position fixed and adjusts downstream joints for pose fidelity.
- `applyElbowAngle`:
  - Computes the desired elbow angle from reference upper-arm and forearm vectors.
  - Rotates the ghost forearm around the elbow to match the reference angle, preserving the elbow joint position.

**Alignment test**
- `checkAlignment` computes the mean 3D distance across key joints `[11, 13, 15, 23, 25, 27]` (upper body and leg anchors) and reports alignment if the mean distance is below `0.1`. This effectively defines a simple tolerance band in normalized coordinate space.

### 4. Interface & Data Flow (Inputs and Outputs)

**Inputs**
- `RawLandmarkFrame` (`{ landmarks: RawLandmark[] }`) and landmark buffers (`number[]`):
  - These represent raw pose landmarks in normalized coordinates, typically originating from a pose estimation subsystem. In the example app, inputs come from `react-native-pose-landmarks` via `PoseLandmarks.getLandmarksBuffer()`, which yields a flat buffer of 33 landmarks × 4 values (x, y, z, visibility).
- `ReferenceData` or `RawLandmarkFrame[]`:
  - Reference frames are prerecorded pose sequences, typically captured from a canonical exercise (e.g., bicep curl). In the example app, these frames are loaded from JSON assets such as `example/assets/bicep_curl_frames.json`.
- `GhostPoseOptions` (`applyReferencePose?: boolean`, `frameIndex?: number`):
  - Options control whether the ghost skeleton should adopt the reference pose (angle adjustments) and which reference frame to use.

**Outputs**
- `ProcessResult`:
  - `ghostSkeleton: Skeleton` — the scaled and optionally pose-adjusted ghost skeleton.
  - `currentCheckpointIndex: number` — checkpoint index from reference data (useful for higher-level state machines).
  - `isAligned: boolean` — alignment flag used by the UI for feedback.
  - `repCount: number` — rep counter sourced from reference data state.

In the broader system, these outputs are primarily consumed by UI rendering code to draw overlay skeletons and show alignment indicators. They can also be fed into analytics or rep-counting logic if external systems update `ReferenceData` counters.

### 5. Integration & Usage

**Example 1: Initialize and process live landmarks with reference playback**
```ts
import {
  GhostGuideCore,
  ProcessResult,
} from 'react-native-ghost-guide'
import { PoseLandmarks } from 'react-native-pose-landmarks'

// Load reference frames at initialization
const reference = GhostGuideCore.createReferenceFromFrames(frames, 'bicep_curl')
GhostGuideCore.loadReference(reference)

// Periodic processing (e.g., every 100ms)
const buffer = PoseLandmarks.getLandmarksBuffer()
if (buffer.length === 33 * 4) {
  const result: ProcessResult | null =
    GhostGuideCore.processLandmarksBufferWithReference(buffer, frames, {
      applyReferencePose: true,
      frameIndex: currentFrameIndex,
    })
}
```

**Example 2: Stateless per-frame processing with a prebuilt skeleton**
```ts
import { GhostGuideCore } from 'react-native-ghost-guide'

const skeleton = GhostGuideCore.buildGhostSkeleton(userSkeleton, referenceFrames, {
  applyReferencePose: true,
  frameIndex: 10,
})

const result = GhostGuideCore.processFrameWithReference(
  userSkeleton,
  referenceFrames,
  { applyReferencePose: false }
)
```

### 6. Output
This report is written to `academic_doc.md` in the package root.
