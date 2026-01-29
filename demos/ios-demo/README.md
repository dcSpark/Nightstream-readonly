# iOS Demo (TestingWasm) — run in Simulator

## Prereqs
- Xcode (and iOS Simulator)
- Rust toolchain (`rustup`, `cargo`)
- `wasm-pack` (`cargo install wasm-pack`)

## Steps
1. `cd demos/ios-demo`

2. Build the WASM bundles (used by the app):
   - `./scripts/build_wasm.sh --release`
   - Optional (threads bundle for WKWebView): `./scripts/build_wasm.sh --release --threads`

3. Build the native XCFramework (enables the “Native” backend in the UI):
   - `./scripts/build_native.sh --release`

4. Open the Xcode project:
   - `open TestingWasm.xcodeproj`

5. In Xcode: select an iPhone Simulator device, then press **Run**.

## Quick build (CLI)
- Rebuild native + build the app: `./scripts/build.sh --native`
- Rebuild wasm + build the app: `./scripts/build.sh --wasm`
- Rebuild everything + build the app: `./scripts/build.sh --all`

## Troubleshooting
- UI still says “Native backend unavailable”: confirm `TestingWasm/Frameworks/NeoFoldFFI.xcframework` exists, then **Product → Clean Build Folder** in Xcode and build again.
- Linker complains about missing `x86_64`: rebuild the framework with `./scripts/build_native.sh --include-x86_64-sim` (only needed for Intel-simulator builds).
