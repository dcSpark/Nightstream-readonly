# Android Demo — run on emulator/device

This demo mirrors `demos/ios-demo`: it runs the `demos/wasm-demo` worker workflow inside an Android `WebView` (WASM backend), and can optionally run the same workflow natively via a Rust JNI library (Native backend).

## Prereqs
- Android Studio (or Android SDK + Gradle)
- Rust toolchain (`rustup`, `cargo`)
- `wasm-pack` (`cargo install wasm-pack`)

Optional (Native backend):
- Android NDK (set `ANDROID_NDK_HOME`)
- `cargo-ndk` (`cargo install cargo-ndk`)

Optional (WASM threads bundle):
- Rust nightly toolchain with `wasm32-unknown-unknown` + `rust-src` (see script output)

## Steps
1. `cd demos/android-demo`

2. Build the WASM bundle(s) into app assets:
   - `./scripts/build_wasm.sh --release`
   - Optional (threads bundle): `./scripts/build_wasm.sh --release --threads`

3. (Optional) Build the native JNI library into `app/src/main/jniLibs/`:
   - `export ANDROID_NDK_HOME=/path/to/ndk`
   - `./scripts/build_native.sh --release`
   - Optional Spartan2: `./scripts/build_native.sh --release --spartan`

4. Open the project in Android Studio:
   - Open folder `demos/android-demo`
   - Run on an emulator/device.

## Quick build (CLI)
- Rebuild native + build the app: `./scripts/build.sh --native`
- Rebuild wasm + build the app: `./scripts/build.sh --wasm`
- Rebuild everything + build the app: `./scripts/build.sh --all`

## Troubleshooting
- UI says “WASM: bundle missing”: run `./scripts/build_wasm.sh --release` and confirm files exist under `app/src/main/assets/web/pkg/`.
- Native backend is disabled: run `./scripts/build_native.sh --release` and confirm `app/src/main/jniLibs/<abi>/libneo_fold_jni.so` exists.
- Threads show as disabled: it’s still fine; the demo falls back to single-thread WASM unless the WebView is cross-origin isolated and `pkg_threads` is present.

