import org.gradle.api.tasks.Copy

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.midnight.neofold.demo"
    compileSdk = 34

    buildFeatures {
        buildConfig = true
    }

    defaultConfig {
        applicationId = "com.midnight.neofold.demo"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.webkit:webkit:1.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
}

val wasmDemoWebDir = rootProject.file("../wasm-demo/web")
val wasmDemoPkgDir = wasmDemoWebDir.resolve("pkg")
val wasmDemoPkgThreadsDir = wasmDemoWebDir.resolve("pkg_threads")

val assetsWebDir = projectDir.resolve("src/main/assets/web")
val assetsPkgDir = assetsWebDir.resolve("pkg")
val assetsPkgThreadsDir = assetsWebDir.resolve("pkg_threads")

val syncWasmDemoPkg by tasks.registering(Copy::class) {
    group = "demo"
    description = "Copies the prebuilt wasm-demo bundle (pkg) into Android assets if missing."
    from(wasmDemoPkgDir)
    into(assetsPkgDir)
    exclude(".gitignore")
    onlyIf {
        wasmDemoPkgDir.exists() && !assetsPkgDir.resolve("neo_fold_demo.js").exists()
    }
}

val syncWasmDemoPkgThreads by tasks.registering(Copy::class) {
    group = "demo"
    description = "Copies the prebuilt wasm-demo bundle (pkg_threads) into Android assets if missing."
    from(wasmDemoPkgThreadsDir)
    into(assetsPkgThreadsDir)
    exclude(".gitignore")
    onlyIf {
        wasmDemoPkgThreadsDir.exists() && !assetsPkgThreadsDir.resolve("neo_fold_demo.js").exists()
    }
}

val syncWasmAssets by tasks.registering {
    group = "demo"
    description = "Ensures the WASM bundles exist under app assets."
    dependsOn(syncWasmDemoPkg, syncWasmDemoPkgThreads)
}

tasks.named("preBuild") {
    dependsOn(syncWasmAssets)
}
