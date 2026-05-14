# react-native-ghost-guide

Ghost Guide is a TypeScript-first library for driving pose-aligned ghost skeleton playback in React Native apps. It ships as pure TS/JS (no native code) and exposes a small, stable API for building reference-guided pose overlays.

## Install

```bash
npm install github:tony-div/react-native-ghost-guide
```

## API

```ts
import {
  createReferenceFromFrames,
  loadReference,
  processLandmarksBufferWithReference,
} from 'react-native-ghost-guide'
```

- `createReferenceFromFrames(frames, exercise, checkpoints?)`
  - Build reference data from pose landmark frames.
- `loadReference(reference)`
  - Store reference data for alignment + playback.
- `processLandmarksBufferWithReference(buffer, referenceFrames, options)`
  - Run live landmarks through the reference to produce ghost skeleton data.

## Example

See `example/` for a React Native app that consumes the library.
