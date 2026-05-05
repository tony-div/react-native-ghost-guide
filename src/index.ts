export type Point3D = {
  x: number
  y: number
  z: number
}

export type Skeleton = {
  points: Point3D[]
}

export type ProcessResult = {
  ghostSkeleton: Skeleton
  currentCheckpointIndex: number
  isAligned: boolean
  repCount: number
}

export type RawLandmark = {
  x?: number
  y?: number
  z?: number
}

export type RawLandmarkFrame = {
  landmarks: RawLandmark[]
}

export type ReferenceFrame = {
  points: Point3D[]
}

export type Checkpoint = {
  name: string
  frame_index: number
  target_skeleton: Skeleton
}

export type ReferenceData = {
  exercise: string
  frames: ReferenceFrame[]
  checkpoints: Checkpoint[]
  current_frame_index?: number
  current_checkpoint_idx?: number
  rep_count?: number
}

export type GhostPoseOptions = {
  applyReferencePose?: boolean
  frameIndex?: number
}

const LANDMARK_COUNT = 33

let referenceData: ReferenceData | null = null

const ensurePoint = (points: Point3D[], index: number): Point3D =>
  points[index] ?? { x: 0, y: 0, z: 0 }

const dist2 = (a: Point3D, b: Point3D) => {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.sqrt(dx * dx + dy * dy)
}

const dist3 = (a: Point3D, b: Point3D) => {
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = a.z - b.z
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

const normalize3 = (v: Point3D): Point3D => {
  const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
  if (len < 0.000001) return { x: 0, y: 0, z: 0 }
  return { x: v.x / len, y: v.y / len, z: v.z / len }
}

const cross3 = (a: Point3D, b: Point3D): Point3D => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
})

const dot3 = (a: Point3D, b: Point3D) => a.x * b.x + a.y * b.y + a.z * b.z

const rotate3D = (
  origin: Point3D,
  point: Point3D,
  axis: Point3D,
  angle: number
): Point3D => {
  const k = normalize3(axis)
  if (k.x === 0 && k.y === 0 && k.z === 0) return point

  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  const v = {
    x: point.x - origin.x,
    y: point.y - origin.y,
    z: point.z - origin.z,
  }

  const kv = dot3(k, v)
  const kCrossV = cross3(k, v)

  return {
    x:
      origin.x +
      v.x * cosA +
      kCrossV.x * sinA +
      k.x * kv * (1 - cosA),
    y:
      origin.y +
      v.y * cosA +
      kCrossV.y * sinA +
      k.y * kv * (1 - cosA),
    z:
      origin.z +
      v.z * cosA +
      kCrossV.z * sinA +
      k.z * kv * (1 - cosA),
  }
}

const normalizeAngle = (angle: number) => {
  const twoPi = Math.PI * 2
  let a = angle % twoPi
  if (a > Math.PI) a -= twoPi
  if (a < -Math.PI) a += twoPi
  return a
}

const scaleToUser = (user: Skeleton, reference: Skeleton): Skeleton => {
  const scaled: Skeleton = {
    points: reference.points.map(point => ({ ...point })),
  }

  const userLeft = ensurePoint(user.points, 11)
  const userRight = ensurePoint(user.points, 12)
  const ghostLeft = ensurePoint(scaled.points, 11)
  const ghostRight = ensurePoint(scaled.points, 12)

  const userHipLeft = ensurePoint(user.points, 23)
  const userHipRight = ensurePoint(user.points, 24)
  const ghostHipLeft = ensurePoint(scaled.points, 23)
  const ghostHipRight = ensurePoint(scaled.points, 24)

  const userShoulderWidth = dist2(userLeft, userRight)
  const ghostShoulderWidth = Math.max(dist2(ghostLeft, ghostRight), 0.001)
  const userHipWidth = dist2(userHipLeft, userHipRight)
  const ghostHipWidth = Math.max(dist2(ghostHipLeft, ghostHipRight), 0.001)

  const userTorso =
    (dist3(userLeft, userHipLeft) + dist3(userRight, userHipRight)) * 0.5
  const ghostTorso = Math.max(
    (dist3(ghostLeft, ghostHipLeft) + dist3(ghostRight, ghostHipRight)) * 0.5,
    0.001
  )

  const shoulderScale = userShoulderWidth / ghostShoulderWidth
  const hipScale = userHipWidth / ghostHipWidth
  const heightScale = userTorso / ghostTorso

  const userCenter: Point3D = {
    x: (userLeft.x + userRight.x) * 0.5,
    y: (userLeft.y + userRight.y) * 0.5,
    z: (userLeft.z + userRight.z) * 0.5,
  }

  const userHipCenter: Point3D = {
    x: (userHipLeft.x + userHipRight.x) * 0.5,
    y: (userHipLeft.y + userHipRight.y) * 0.5,
    z: (userHipLeft.z + userHipRight.z) * 0.5,
  }

  const ghostCenter: Point3D = {
    x: (ghostLeft.x + ghostRight.x) * 0.5,
    y: (ghostLeft.y + ghostRight.y) * 0.5,
    z: (ghostLeft.z + ghostRight.z) * 0.5,
  }

  const ghostHipCenter: Point3D = {
    x: (ghostHipLeft.x + ghostHipRight.x) * 0.5,
    y: (ghostHipLeft.y + ghostHipRight.y) * 0.5,
    z: (ghostHipLeft.z + ghostHipRight.z) * 0.5,
  }

  const userTorsoCenter: Point3D = {
    x: (userCenter.x + userHipCenter.x) * 0.5,
    y: (userCenter.y + userHipCenter.y) * 0.5,
    z: (userCenter.z + userHipCenter.z) * 0.5,
  }

  const ghostTorsoCenter: Point3D = {
    x: (ghostCenter.x + ghostHipCenter.x) * 0.5,
    y: (ghostCenter.y + ghostHipCenter.y) * 0.5,
    z: (ghostCenter.z + ghostHipCenter.z) * 0.5,
  }

  const userVec = normalize3({
    x: userRight.x - userLeft.x,
    y: userRight.y - userLeft.y,
    z: userRight.z - userLeft.z,
  })
  const ghostVec = normalize3({
    x: ghostRight.x - ghostLeft.x,
    y: ghostRight.y - ghostLeft.y,
    z: ghostRight.z - ghostLeft.z,
  })

  const axis = cross3(ghostVec, userVec)
  const dot = Math.max(-1, Math.min(1, dot3(ghostVec, userVec)))
  const rotationStrength = 0.6
  const rotation = Math.acos(dot) * rotationStrength

  scaled.points = scaled.points.map(point =>
    rotate3D(ghostTorsoCenter, point, axis, rotation)
  )

  const userYaw = Math.atan2(userVec.z, userVec.x)
  const ghostYaw = Math.atan2(ghostVec.z, ghostVec.x)
  const yawDelta = normalizeAngle(userYaw - ghostYaw)
  const yawAmount = Math.abs(userYaw)
  const yawThreshold = 0.25
  const yawStrength =
    yawAmount <= yawThreshold
      ? 0
      : Math.min(1, (yawAmount - yawThreshold) / (Math.PI / 2 - yawThreshold))

  if (yawStrength > 0) {
    const yawAxis = { x: 0, y: 1, z: 0 }
    scaled.points = scaled.points.map(point =>
      rotate3D(ghostTorsoCenter, point, yawAxis, yawDelta * yawStrength)
    )
  }

  const shoulderY = ghostCenter.y
  const hipY = ghostHipCenter.y
  const denom = Math.max(Math.abs(hipY - shoulderY), 0.000001)

  scaled.points = scaled.points.map(point => {
    const t = Math.max(
      0,
      Math.min(1, Math.abs(point.y - shoulderY) / denom)
    )
    const widthScale = shoulderScale + (hipScale - shoulderScale) * t
    const dx = point.x - ghostTorsoCenter.x
    const dy = point.y - ghostTorsoCenter.y
    const dz = point.z - ghostTorsoCenter.z

    return {
      x: userTorsoCenter.x + dx * widthScale,
      y: userTorsoCenter.y + dy * heightScale,
      z: userTorsoCenter.z + dz * widthScale,
    }
  })

  const userLeftArm =
    dist2(ensurePoint(user.points, 11), ensurePoint(user.points, 13)) +
    dist2(ensurePoint(user.points, 13), ensurePoint(user.points, 15))
  const userRightArm =
    dist2(ensurePoint(user.points, 12), ensurePoint(user.points, 14)) +
    dist2(ensurePoint(user.points, 14), ensurePoint(user.points, 16))

  const ghostLeftArm =
    dist2(ensurePoint(scaled.points, 11), ensurePoint(scaled.points, 13)) +
    dist2(ensurePoint(scaled.points, 13), ensurePoint(scaled.points, 15))
  const ghostRightArm =
    dist2(ensurePoint(scaled.points, 12), ensurePoint(scaled.points, 14)) +
    dist2(ensurePoint(scaled.points, 14), ensurePoint(scaled.points, 16))

  const leftArmScale = ghostLeftArm > 0.001 ? userLeftArm / ghostLeftArm : 1
  const rightArmScale = ghostRightArm > 0.001 ? userRightArm / ghostRightArm : 1

  const scaleFrom = (origin: Point3D, point: Point3D, s: number): Point3D => ({
    x: origin.x + (point.x - origin.x) * s,
    y: origin.y + (point.y - origin.y) * s,
    z: origin.z + (point.z - origin.z) * s,
  })

  const leftShoulder = ensurePoint(scaled.points, 11)
  const rightShoulder = ensurePoint(scaled.points, 12)

  scaled.points[13] = scaleFrom(
    leftShoulder,
    ensurePoint(scaled.points, 13),
    leftArmScale
  )
  scaled.points[15] = scaleFrom(
    leftShoulder,
    ensurePoint(scaled.points, 15),
    leftArmScale
  )
  scaled.points[14] = scaleFrom(
    rightShoulder,
    ensurePoint(scaled.points, 14),
    rightArmScale
  )
  scaled.points[16] = scaleFrom(
    rightShoulder,
    ensurePoint(scaled.points, 16),
    rightArmScale
  )

  return scaled
}

const checkAlignment = (user: Skeleton, ghost: Skeleton) => {
  const keyJoints = [11, 13, 15, 23, 25, 27]
  let totalDist = 0

  for (const idx of keyJoints) {
    const userPoint = ensurePoint(user.points, idx)
    const ghostPoint = ensurePoint(ghost.points, idx)
    totalDist += dist3(userPoint, ghostPoint)
  }

  return totalDist / keyJoints.length < 0.1
}

const normalizeReference = (data: ReferenceData): ReferenceData => ({
  ...data,
  current_frame_index: data.current_frame_index ?? 0,
  current_checkpoint_idx: data.current_checkpoint_idx ?? 0,
  rep_count: data.rep_count ?? 0,
})

const landmarksToPoints = (
  frame: RawLandmarkFrame,
  count = LANDMARK_COUNT
): Point3D[] =>
  frame.landmarks.slice(0, count).map(lm => ({
    x: lm.x ?? 0,
    y: lm.y ?? 0,
    z: lm.z ?? 0,
  }))

const angleBetweenSigned = (a: Point3D, b: Point3D) => {
  const cross = a.x * b.y - a.y * b.x
  const dot = a.x * b.x + a.y * b.y
  return Math.atan2(cross, dot)
}

const rotate2D = (v: Point3D, angle: number): Point3D => {
  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  return {
    x: v.x * cosA - v.y * sinA,
    y: v.x * sinA + v.y * cosA,
    z: v.z,
  }
}

const applyShoulderAngle = (
  ghostPoints: Point3D[],
  referencePoints: Point3D[],
  hipIndex: number,
  shoulderIndex: number,
  elbowIndex: number,
  wristIndex: number
) => {
  const gHip = ensurePoint(ghostPoints, hipIndex)
  const gShoulder = ensurePoint(ghostPoints, shoulderIndex)
  const gElbow = ensurePoint(ghostPoints, elbowIndex)
  const gWrist = ensurePoint(ghostPoints, wristIndex)

  const rHip = ensurePoint(referencePoints, hipIndex)
  const rShoulder = ensurePoint(referencePoints, shoulderIndex)
  const rElbow = ensurePoint(referencePoints, elbowIndex)

  const rTorso = { x: rHip.x - rShoulder.x, y: rHip.y - rShoulder.y, z: 0 }
  const rUpper = { x: rElbow.x - rShoulder.x, y: rElbow.y - rShoulder.y, z: 0 }
  const desiredAngle = angleBetweenSigned(rTorso, rUpper)

  const gTorso = { x: gHip.x - gShoulder.x, y: gHip.y - gShoulder.y, z: 0 }
  const gUpper = { x: gElbow.x - gShoulder.x, y: gElbow.y - gShoulder.y, z: 0 }
  const currentAngle = angleBetweenSigned(gTorso, gUpper)

  let delta = desiredAngle - currentAngle
  const upperArm = { x: gElbow.x - gShoulder.x, y: gElbow.y - gShoulder.y, z: gElbow.z - gShoulder.z }
  const rotatedUpper = rotate2D(upperArm, delta)

  const newElbow = {
    x: gShoulder.x + rotatedUpper.x,
    y: gShoulder.y + rotatedUpper.y,
    z: gShoulder.z + rotatedUpper.z,
  }
  ghostPoints[elbowIndex] = newElbow

  const forearm = { x: gWrist.x - gElbow.x, y: gWrist.y - gElbow.y, z: gWrist.z - gElbow.z }
  const rotatedFore = rotate2D(forearm, delta)

  ghostPoints[wristIndex] = {
    x: newElbow.x + rotatedFore.x,
    y: newElbow.y + rotatedFore.y,
    z: newElbow.z + rotatedFore.z,
  }
}

const applyElbowAngle = (
  ghostPoints: Point3D[],
  referencePoints: Point3D[],
  shoulderIndex: number,
  elbowIndex: number,
  wristIndex: number
) => {
  const gShoulder = ghostPoints[shoulderIndex]
  const gElbow = ghostPoints[elbowIndex]
  const gWrist = ghostPoints[wristIndex]
  const rShoulder = referencePoints[shoulderIndex]
  const rElbow = referencePoints[elbowIndex]
  const rWrist = referencePoints[wristIndex]

  if (!gShoulder || !gElbow || !gWrist || !rShoulder || !rElbow || !rWrist) {
    return
  }

  const gUpper = { x: gShoulder.x - gElbow.x, y: gShoulder.y - gElbow.y, z: 0 }
  const gFore = { x: gWrist.x - gElbow.x, y: gWrist.y - gElbow.y, z: 0 }
  const rUpper = { x: rShoulder.x - rElbow.x, y: rShoulder.y - rElbow.y, z: 0 }
  const rFore = { x: rWrist.x - rElbow.x, y: rWrist.y - rElbow.y, z: 0 }

  const currentAngle = angleBetweenSigned(gUpper, gFore)
  const desiredAngle = angleBetweenSigned(rUpper, rFore)
  const delta = desiredAngle - currentAngle
  const rotatedFore = rotate2D(gFore, delta)

  ghostPoints[wristIndex] = {
    x: gElbow.x + rotatedFore.x,
    y: gElbow.y + rotatedFore.y,
    z: gWrist.z,
  }
}

export const buildGhostSkeleton = (
  userSkeleton: Skeleton,
  referenceFrames: RawLandmarkFrame[],
  options: GhostPoseOptions = {}
): Skeleton => {
  const ghostPoints = userSkeleton.points.map(point => ({ ...point }))
  if (!options.applyReferencePose) return { points: ghostPoints }

  const frameIndex = options.frameIndex ?? 0
  const frame = referenceFrames[frameIndex]
  if (!frame) return { points: ghostPoints }

  const referencePoints = landmarksToPoints(frame, LANDMARK_COUNT)

  applyShoulderAngle(ghostPoints, referencePoints, 23, 11, 13, 15)
  applyElbowAngle(ghostPoints, referencePoints, 11, 13, 15)

  applyShoulderAngle(ghostPoints, referencePoints, 24, 12, 14, 16)
  applyElbowAngle(ghostPoints, referencePoints, 12, 14, 16)

  return { points: ghostPoints }
}

export const processFrameWithReference = (
  userSkeleton: Skeleton,
  referenceFrames: RawLandmarkFrame[],
  options: GhostPoseOptions = {}
): ProcessResult => {
  const result = processFrame(userSkeleton)
  result.ghostSkeleton = buildGhostSkeleton(
    userSkeleton,
    referenceFrames,
    options
  )
  return result
}

export const createReferenceFromFrames = (
  frames: RawLandmarkFrame[],
  exercise: string,
  checkpoints: Checkpoint[] = []
): ReferenceData => ({
  exercise,
  frames: frames.map(frame => ({
    points: frame.landmarks.slice(0, 33).map(lm => ({
      x: lm.x ?? 0,
      y: lm.y ?? 0,
      z: lm.z ?? 0,
    })) as Point3D[],
  })),
  checkpoints,
  current_frame_index: 0,
  current_checkpoint_idx: 0,
  rep_count: 0,
})

export const loadReference = (reference: ReferenceData | string) => {
  const data =
    typeof reference === 'string'
      ? (JSON.parse(reference) as ReferenceData)
      : reference
  referenceData = normalizeReference(data)
}

export const setReferenceFrameIndex = (index: number) => {
  if (!referenceData || referenceData.frames.length === 0) return

  const clampedIndex = Math.max(
    0,
    Math.min(index, referenceData.frames.length - 1)
  )

  referenceData.current_frame_index = clampedIndex
}

export const processFrame = (userSkeleton: Skeleton): ProcessResult => {
  if (!referenceData || referenceData.frames.length === 0) {
    return {
      ghostSkeleton: userSkeleton,
      currentCheckpointIndex: 0,
      isAligned: false,
      repCount: referenceData?.rep_count ?? 0,
    }
  }

  const frameIndex = referenceData.current_frame_index ?? 0
  const reference =
    referenceData.frames[frameIndex] ?? referenceData.frames[0] ?? { points: [] }
  const scaledGhost = scaleToUser(userSkeleton, reference)
  const isAligned = checkAlignment(userSkeleton, scaledGhost)

  return {
    ghostSkeleton: scaledGhost,
    currentCheckpointIndex: referenceData.current_checkpoint_idx ?? 0,
    isAligned,
    repCount: referenceData.rep_count ?? 0,
  }
}

export const GhostGuide = {
  loadReference,
  setReferenceFrameIndex,
  processFrame,
}

export const skeletonFromLandmarksBuffer = (
  buffer: number[],
  landmarkCount = 33
): Skeleton | null => {
  if (buffer.length !== landmarkCount * 4) return null

  return {
    points: Array.from({ length: landmarkCount }, (_, i) => ({
      x: buffer[i * 4] ?? 0,
      y: buffer[i * 4 + 1] ?? 0,
      z: buffer[i * 4 + 2] ?? 0,
    })),
  }
}

export const processLandmarksBuffer = (
  buffer: number[],
  landmarkCount = 33
): ProcessResult | null => {
  const skeleton = skeletonFromLandmarksBuffer(buffer, landmarkCount)
  if (!skeleton) return null

  return processFrame(skeleton)
}

export const processLandmarksBufferWithReference = (
  buffer: number[],
  referenceFrames: RawLandmarkFrame[],
  options: GhostPoseOptions = {},
  landmarkCount = 33
): ProcessResult | null => {
  const skeleton = skeletonFromLandmarksBuffer(buffer, landmarkCount)
  if (!skeleton) return null

  return processFrameWithReference(skeleton, referenceFrames, options)
}

export type GhostGuideAPI = {
  createReferenceFromFrames: typeof createReferenceFromFrames
  loadReference: typeof loadReference
  setReferenceFrameIndex: typeof setReferenceFrameIndex
  processFrame: typeof processFrame
  processFrameWithReference: typeof processFrameWithReference
  processLandmarksBuffer: typeof processLandmarksBuffer
  processLandmarksBufferWithReference: typeof processLandmarksBufferWithReference
  buildGhostSkeleton: typeof buildGhostSkeleton
}

export const GhostGuideCore: GhostGuideAPI = {
  createReferenceFromFrames,
  loadReference,
  setReferenceFrameIndex,
  processFrame,
  processFrameWithReference,
  processLandmarksBuffer,
  processLandmarksBufferWithReference,
  buildGhostSkeleton,
}
