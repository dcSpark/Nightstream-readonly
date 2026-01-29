package com.midnight.neofold.demo

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

data class NativeRunResult(
    val raw: JSONObject,
    val verifyOk: Boolean?,
    val steps: Int?,
    val timingsMs: JSONObject?,
)

class NativeService {
    suspend fun runProveVerify(
        json: String,
        doSpartan: Boolean,
        onLog: (String) -> Unit,
    ): NativeRunResult {
        if (!NeoFoldNative.isAvailable) {
            throw IllegalStateException("Native backend unavailable. Build and copy libneo_fold_jni.so into jniLibs/ first.")
        }

        onLog("Running native neo-fold workflow…")
        onLog("Input JSON size: ${fmtBytes(json.toByteArray(Charsets.UTF_8).size)}")

        val outJson =
            withContext(Dispatchers.Default) {
                NeoFoldNative.runWasmDemoWorkflowJson(json, doSpartan)
            }

        val raw = JSONObject(outJson)
        val verifyOk =
            when (val v = raw.opt("verify_ok")) {
                is Boolean -> v
                is Number -> v.toInt() != 0
                else -> null
            }
        val steps = raw.optInt("steps", -1).takeIf { it >= 0 }
        val timings = raw.optJSONObject("timings_ms")

        if (timings != null) {
            onLog("Timings (ms): ${timings.toString(2)}")
        }

        onLog("\n\nRaw result:")
        onLog(raw.toString(2))

        return NativeRunResult(
            raw = raw,
            verifyOk = verifyOk,
            steps = steps,
            timingsMs = timings,
        )
    }

    private fun fmtBytes(bytes: Int): String {
        if (bytes < 1024) return "${bytes} B"
        val kb = bytes.toDouble() / 1024.0
        if (kb < 1024) return String.format("%.2f KB", kb)
        val mb = kb / 1024.0
        return String.format("%.2f MB", mb)
    }
}
