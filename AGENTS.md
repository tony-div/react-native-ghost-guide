# AGENTS.md

## Commands
- `npm run specs` — regenerate native code from `src/specs/*.nitro.ts` (runs `tsc` then `nitrogen`)
- `npm run typescript` — compile TypeScript to `lib/`
- `npm run typecheck` — type-check without emitting
- `npm run lint` — lint and auto-fix (`eslint "**/*.{js,ts,tsx}" --fix`)
- `build_rust.sh` — builds Android Rust libs via `cargo ndk` (requires NDK + `cargo-ndk`)

## Example app
- `example/` is a React Native app that consumes this package via `react-native-ghost-guide: "file:.."`
- Metro config in `example/metro.config.js` watches the repo root and aliases `react-native-ghost-guide` to the local package
- Run app commands from `example/` (`npm run ios`, `npm run android`, `npm start`, `npm test`)

## Architecture
- **Specs**: `src/specs/*.nitro.ts` define HybridObjects; regenerate with `npm run specs`
- **Generated code**: `nitrogen/` is auto-generated and must be committed
- **C++**: `cpp/` implements HybridObjects (autolinking target: `HybridGhostGuide`)
- **iOS**: module name `GhostGuideNitro`, namespace `ghostguide`
- **Android**: namespace `ghostguide`, C++ lib name `GhostGuide`
- **Entry**: `src/index.ts` loads HybridObjects at runtime

## Conventions
- ESLint: `@react-native` config + prettier; single quotes; no semis
- TypeScript: `noUnusedLocals` and `noUnusedParameters` enabled — remove unused code
