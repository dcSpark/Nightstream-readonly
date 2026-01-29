package com.midnight.neofold.demo

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var wasmService: WasmWebViewService
    private val nativeService = NativeService()

    private lateinit var wasmStatus: TextView
    private lateinit var threadsStatus: TextView
    private lateinit var backendGroup: RadioGroup
    private lateinit var backendWasm: RadioButton
    private lateinit var backendNative: RadioButton
    private lateinit var backendBoth: RadioButton
    private lateinit var nativeHint: TextView
    private lateinit var spartanSwitch: SwitchCompat
    private lateinit var circuitJsonInput: EditText
    private lateinit var runButton: Button
    private lateinit var runningProgress: ProgressBar
    private lateinit var outputText: TextView
    private lateinit var clearOutputButton: Button
    private lateinit var copyOutputButton: Button

    private var logText: StringBuilder = StringBuilder()
    private val maxLogChars: Int = 30_000

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        wasmStatus = findViewById(R.id.wasm_status)
        threadsStatus = findViewById(R.id.threads_status)
        backendGroup = findViewById(R.id.backend_group)
        backendWasm = findViewById(R.id.backend_wasm)
        backendNative = findViewById(R.id.backend_native)
        backendBoth = findViewById(R.id.backend_both)
        nativeHint = findViewById(R.id.native_hint)
        spartanSwitch = findViewById(R.id.spartan_switch)
        circuitJsonInput = findViewById(R.id.input_json)
        runButton = findViewById(R.id.btn_run)
        runningProgress = findViewById(R.id.progress_running)
        outputText = findViewById(R.id.output_text)
        clearOutputButton = findViewById(R.id.btn_clear_output)
        copyOutputButton = findViewById(R.id.btn_copy_output)

        val webView = findViewById<WebView>(R.id.wasm_webview)
        wasmService = WasmWebViewService(this, webView, lifecycleScope)

        clearOutputButton.setOnClickListener { clearOutput() }
        copyOutputButton.setOnClickListener { copyOutputToClipboard() }

        findViewById<Button>(R.id.btn_example_toy).setOnClickListener {
            loadExampleAsset("examples/toy_square.json", "Loaded toy circuit.")
        }
        findViewById<Button>(R.id.btn_example_toy_8).setOnClickListener {
            loadExampleAsset("examples/toy_square_folding_8_steps.json", "Loaded toy folding circuit (8 steps).")
        }
        findViewById<Button>(R.id.btn_example_poseidon2).setOnClickListener {
            loadExampleAsset("examples/poseidon2_ic_batch_1.json", "Loaded Poseidon2 IC circuit batch 1.")
        }

        runButton.setOnClickListener { runProveVerify() }

        val nativeAvailable = NeoFoldNative.isAvailable
        backendNative.isEnabled = nativeAvailable
        backendBoth.isEnabled = nativeAvailable
        nativeHint.visibility = if (nativeAvailable) View.GONE else View.VISIBLE
        if (nativeAvailable) {
            backendBoth.isChecked = true
        }

        lifecycleScope.launch {
            try {
                val caps = wasmService.ensureLoaded()
                if (!caps.singleBundlePresent) {
                    wasmStatus.text = "WASM: bundle missing (run scripts/build_wasm.sh)"
                } else {
                    wasmStatus.text = "WASM: loaded (WebView)"
                }
                threadsStatus.text = formatThreadsStatus(caps)
            } catch (e: Throwable) {
                wasmStatus.text = "WASM: error (${e.message ?: "unknown"})"
                threadsStatus.text = "Threads: unavailable"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        wasmService.destroy()
    }

    private fun runProveVerify() {
        val json = circuitJsonInput.text?.toString() ?: ""
        if (json.isBlank()) {
            toast("Paste circuit JSON first.")
            return
        }

        clearOutput()
        setRunning(true)

        val doSpartan = spartanSwitch.isChecked

        lifecycleScope.launch {
            try {
                when (backendGroup.checkedRadioButtonId) {
                    backendWasm.id -> runWasm(json, doSpartan)
                    backendNative.id -> runNative(json, doSpartan)
                    backendBoth.id -> {
                        appendLog("=== WASM (WebView) ===")
                        runWasm(json, doSpartan)
                        appendLog("\n\n=== Native (JNI) ===")
                        runNative(json, doSpartan)
                    }
                    else -> runWasm(json, doSpartan)
                }
            } catch (e: Throwable) {
                appendLog("ERROR: ${e.message ?: e.toString()}")
            } finally {
                setRunning(false)
            }
        }
    }

    private suspend fun runWasm(json: String, doSpartan: Boolean) {
        appendLog("Backend: WASM (WebView)")
        if (doSpartan) {
            appendLog("Spartan2: enabled")
        }

        val result =
            wasmService.runProveVerify(
                json = json,
                doSpartan = doSpartan,
                onLog = { line -> appendLog(line) },
            )
        appendLog("\nDone. verify_ok=${result.raw.optBoolean("verify_ok")} steps=${result.raw.optInt("steps")}")
    }

    private suspend fun runNative(json: String, doSpartan: Boolean) {
        appendLog("Backend: Native (JNI)")
        if (doSpartan) {
            appendLog("Spartan2: enabled")
        }

        val result =
            nativeService.runProveVerify(
                json = json,
                doSpartan = doSpartan,
                onLog = { line -> appendLog(line) },
            )
        appendLog("\nDone. verify_ok=${result.verifyOk ?: "?"} steps=${result.steps ?: "?"}")
    }

    private fun loadExampleAsset(path: String, successLog: String) {
        lifecycleScope.launch {
            try {
                val text =
                    withContext(Dispatchers.IO) {
                        assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
                    }
                circuitJsonInput.setText(text)
                appendLog(successLog)
            } catch (_: Throwable) {
                toast("Missing asset: $path. Run demos/android-demo/scripts/build_wasm.sh")
            }
        }
    }

    private fun formatThreadsStatus(caps: WasmCapabilities): String {
        if (!caps.crossOriginIsolated) return "Threads: disabled (crossOriginIsolated=false)"
        if (!caps.supportsThreadsRuntime) return "Threads: disabled (no SharedArrayBuffer/Atomics)"
        if (!caps.threadsBundlePresent) return "Threads: supported, bundle missing"
        if (caps.defaultThreads > 0) return "Threads: available (${caps.defaultThreads} workers)"
        return "Threads: available"
    }

    private fun clearOutput() {
        logText = StringBuilder()
        outputText.text = "Output will appear here."
    }

    private fun appendLog(line: String) {
        val normalized = line.trimEnd()
        if (normalized.startsWith("Threads: initialized (")) {
            threadsStatus.text = normalized.replace("Threads: initialized", "Threads: enabled")
        } else if (normalized.startsWith("Threads init failed;")) {
            threadsStatus.text = "Threads: disabled (init failed)"
        }

        if (logText.isNotEmpty()) {
            logText.append('\n')
        }
        logText.append(normalized)

        if (logText.length > maxLogChars) {
            val keep = maxLogChars - 80
            val tail = logText.substring(maxOf(0, logText.length - keep))
            logText = StringBuilder("… (output truncated) …\n").append(tail)
        }

        outputText.text = logText.toString()
    }

    private fun copyOutputToClipboard() {
        val text = outputText.text?.toString() ?: ""
        if (text.isBlank()) {
            toast("No output to copy.")
            return
        }

        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("NeoFold output", text))
        toast("Copied output.")
    }

    private fun setRunning(isRunning: Boolean) {
        runButton.isEnabled = !isRunning
        runningProgress.visibility = if (isRunning) View.VISIBLE else View.GONE
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
