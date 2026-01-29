package com.midnight.neofold.demo

object NeoFoldNative {
    private var loaded: Boolean = false

    init {
        loaded =
            try {
                System.loadLibrary("neo_fold_jni")
                true
            } catch (_: UnsatisfiedLinkError) {
                false
            }
    }

    val isAvailable: Boolean
        get() = loaded

    @JvmStatic external fun runWasmDemoWorkflowJson(json: String, doSpartan: Boolean): String
}

