package com.midnight.neofold.demo

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageInfo
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.widget.NestedScrollView
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private enum class ContentPanel {
        CIRCUIT,
        OUTPUT,
    }

    private data class RunTimings(
        val sessionCreateMs: Double?,
        val ajtaiSetupMs: Double?,
        val buildCcsMs: Double?,
        val sessionInitMs: Double?,
        val addStepsMs: Double?,
        val foldAndProveMs: Double?,
        val verifyMs: Double?,
        val totalMs: Double?,
        val spartanProveMs: Double?,
        val spartanVerifyMs: Double?,
        val spartanVerifyOk: Boolean?,
        val spartanSnarkBytes: Int?,
        val spartanVkBytes: Int?,
        val spartanVkAndSnarkBytes: Int?,
    )

    private data class RunSummary(
        val timings: RunTimings?,
        val verifyOk: Boolean?,
        val steps: Int?,
    )

    private data class SummaryRow(
        val phase: String,
        val wasm: String,
        val native: String,
    )

    private val stateContentPanelKey = "content_panel"
    private val minRequiredWebViewMajor: Int = 80

    private lateinit var wasmService: WasmWebViewService
    private val nativeService = NativeService()

    private lateinit var mainScroll: NestedScrollView

    private lateinit var contentToggle: MaterialButtonToggleGroup
    private lateinit var contentCircuit: MaterialButton
    private lateinit var contentOutput: MaterialButton
    private lateinit var examplesCard: View
    private lateinit var circuitCard: View
    private lateinit var outputCard: View

    private lateinit var wasmStatus: TextView
    private lateinit var threadsStatus: TextView
    private lateinit var webViewStatus: TextView
    private lateinit var backendGroup: MaterialButtonToggleGroup
    private lateinit var backendWasm: MaterialButton
    private lateinit var backendNative: MaterialButton
    private lateinit var backendBoth: MaterialButton
    private lateinit var nativeHint: TextView
    private lateinit var spartanSwitch: SwitchCompat
    private lateinit var circuitJsonInput: EditText
    private lateinit var runButton: Button
    private lateinit var runningProgress: ProgressBar
    private lateinit var outputText: TextView
    private lateinit var outputSummaryHeader: View
    private lateinit var outputSummaryBytes: TextView
    private lateinit var outputSummaryTable: TextView
    private lateinit var clearOutputButton: Button
    private lateinit var copyOutputButton: Button

    private var nativeAvailable: Boolean = false
    private var logText: StringBuilder = StringBuilder()
    private val maxLogChars: Int = 30_000
    private var contentPanel: ContentPanel = ContentPanel.CIRCUIT
    private var suppressContentToggleListener: Boolean = false
    private var wasmRuntimeOk: Boolean = true
    private var wasmAssetsOk: Boolean = true
    private var lastRunInputBytes: Int = 0
    private var lastWasmSummary: RunSummary? = null
    private var lastNativeSummary: RunSummary? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mainScroll = findViewById(R.id.main_scroll)

        contentToggle = findViewById(R.id.content_toggle)
        contentCircuit = findViewById(R.id.content_circuit)
        contentOutput = findViewById(R.id.content_output)
        examplesCard = findViewById(R.id.card_examples)
        circuitCard = findViewById(R.id.card_circuit)
        outputCard = findViewById(R.id.card_output)

        wasmStatus = findViewById(R.id.wasm_status)
        threadsStatus = findViewById(R.id.threads_status)
        webViewStatus = findViewById(R.id.webview_status)
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
        outputSummaryHeader = findViewById(R.id.output_summary_header)
        outputSummaryBytes = findViewById(R.id.output_summary_bytes)
        outputSummaryTable = findViewById(R.id.output_summary_table)
        clearOutputButton = findViewById(R.id.btn_clear_output)
        copyOutputButton = findViewById(R.id.btn_copy_output)

        val webView = findViewById<WebView>(R.id.wasm_webview)
        wasmService = WasmWebViewService(this, webView, lifecycleScope)

        clearOutputButton.setOnClickListener { clearOutput() }
        copyOutputButton.setOnClickListener { copyOutputToClipboard() }

        contentToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (suppressContentToggleListener || !isChecked) return@addOnButtonCheckedListener
            when (checkedId) {
                contentCircuit.id -> applyContentPanel(ContentPanel.CIRCUIT, scrollTo = circuitCard)
                contentOutput.id -> applyContentPanel(ContentPanel.OUTPUT, scrollTo = outputCard)
            }
        }

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

        val webViewInfo = getWebViewInfo()
        webViewStatus.text = formatWebViewStatus(webViewInfo)
        wasmRuntimeOk =
            webViewInfo?.major?.let { it >= minRequiredWebViewMajor } ?: true

        nativeAvailable = NeoFoldNative.isAvailable
        wasmAssetsOk = true
        updateBackendAvailability(nativeAvailable = nativeAvailable)

        val restoredPanel =
            savedInstanceState
                ?.getString(stateContentPanelKey)
                ?.let { runCatching { ContentPanel.valueOf(it) }.getOrNull() }
                ?: ContentPanel.CIRCUIT
        applyContentPanel(restoredPanel)

        lifecycleScope.launch {
            if (!wasmRuntimeOk) {
                wasmAssetsOk = false
                wasmStatus.text = "WASM: unavailable (update WebView)"
                threadsStatus.text = "Threads: unavailable"
                updateBackendAvailability(nativeAvailable = nativeAvailable)
                return@launch
            }

            try {
                val caps = wasmService.ensureLoaded()
                wasmAssetsOk = caps.singleBundlePresent
                wasmRuntimeOk = wasmRuntimeOk && caps.singleBundlePresent

                webViewStatus.text = formatWebViewStatus(webViewInfo, caps.userAgent)

                if (!caps.singleBundlePresent) {
                    wasmStatus.text = "WASM: bundle missing (run scripts/build_wasm.sh)"
                } else {
                    wasmStatus.text = "WASM: loaded (WebView)"
                }
                threadsStatus.text = formatThreadsStatus(caps)
                updateBackendAvailability(nativeAvailable = nativeAvailable)
            } catch (e: Throwable) {
                wasmRuntimeOk = false
                wasmAssetsOk = false
                wasmStatus.text = "WASM: unavailable (WebView unsupported)"
                threadsStatus.text = "Threads: unavailable"
                updateBackendAvailability(nativeAvailable = nativeAvailable)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        wasmService.destroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(stateContentPanelKey, contentPanel.name)
        super.onSaveInstanceState(outState)
    }

    private fun runProveVerify() {
        val json = circuitJsonInput.text?.toString() ?: ""
        if (json.isBlank()) {
            toast("Paste circuit JSON first.")
            return
        }

        if (isWasmSelected() && !isWasmBackendAvailable()) {
            toast("WASM unavailable. Update Android System WebView/Chrome and ensure the WASM bundle is present.")
            return
        }

        clearOutput()
        lastRunInputBytes = json.toByteArray(Charsets.UTF_8).size
        setRunning(true)
        applyContentPanel(ContentPanel.OUTPUT, scrollTo = outputCard)

        val doSpartan = spartanSwitch.isChecked

        lifecycleScope.launch {
            try {
                when (backendGroup.checkedButtonId) {
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
        lastWasmSummary = parseRunSummary(result.raw)
        updateOutputSummary()
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
        lastNativeSummary = parseRunSummary(result.raw)
        updateOutputSummary()
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
                applyContentPanel(ContentPanel.CIRCUIT)
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

    private fun isWasmBackendAvailable(): Boolean {
        return wasmRuntimeOk && wasmAssetsOk
    }

    private fun isWasmSelected(): Boolean {
        return backendGroup.checkedButtonId == backendWasm.id || backendGroup.checkedButtonId == backendBoth.id
    }

    private fun updateBackendAvailability(nativeAvailable: Boolean) {
        val wasmAvailable = isWasmBackendAvailable()

        backendWasm.isEnabled = wasmAvailable
        backendNative.isEnabled = nativeAvailable
        backendBoth.isEnabled = wasmAvailable && nativeAvailable
        nativeHint.visibility = if (nativeAvailable) View.GONE else View.VISIBLE

        val desiredChecked =
            when {
                backendBoth.isEnabled -> backendBoth.id
                backendWasm.isEnabled -> backendWasm.id
                backendNative.isEnabled -> backendNative.id
                else -> null
            }

        if (desiredChecked != null && backendGroup.checkedButtonId != desiredChecked) {
            backendGroup.check(desiredChecked)
        }

        runButton.isEnabled =
            !runningProgress.isShown &&
                when (backendGroup.checkedButtonId) {
                    backendWasm.id -> wasmAvailable
                    backendNative.id -> nativeAvailable
                    backendBoth.id -> wasmAvailable && nativeAvailable
                    else -> wasmAvailable
                }
    }

    private data class WebViewInfo(
        val packageName: String,
        val versionName: String?,
        val major: Int?,
    )

    private fun getWebViewInfo(): WebViewInfo? {
        val pkg: PackageInfo = WebView.getCurrentWebViewPackage() ?: return null
        val versionName = pkg.versionName
        val major =
            versionName
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
                ?.substringBefore('.', missingDelimiterValue = "")
                ?.toIntOrNull()
        return WebViewInfo(
            packageName = pkg.packageName,
            versionName = versionName,
            major = major,
        )
    }

    private fun formatWebViewStatus(
        info: WebViewInfo?,
        userAgent: String? = null,
    ): String {
        val versionName = info?.versionName
        val major = info?.major
        val label =
            when (info?.packageName) {
                null -> null
                "com.google.android.webview" -> "Android System WebView"
                "com.android.chrome" -> "Chrome"
                else -> info.packageName
            }

        val base =
            when {
                versionName != null && label != null -> "WebView: $label $versionName"
                versionName != null -> "WebView: $versionName"
                label != null -> "WebView: $label"
                else -> "WebView: unknown"
            }

        val tooOld = major != null && major < minRequiredWebViewMajor
        if (tooOld) {
            return "$base (update required)"
        }

        val uaMajor = userAgent?.let { extractChromeMajorFromUserAgent(it) }
        if (versionName == null && uaMajor != null) {
            return "$base (Chrome/$uaMajor)"
        }

        return base
    }

    private fun extractChromeMajorFromUserAgent(userAgent: String): Int? {
        val match =
            Regex("""\bChrome/(\d{2,3})\.""")
                .find(userAgent)
                ?.groupValues
                ?.getOrNull(1)
                ?: return null
        return match.toIntOrNull()
    }

    private fun applyContentPanel(panel: ContentPanel, scrollTo: View? = null) {
        contentPanel = panel
        val showCircuit = panel == ContentPanel.CIRCUIT
        examplesCard.visibility = if (showCircuit) View.VISIBLE else View.GONE
        circuitCard.visibility = if (showCircuit) View.VISIBLE else View.GONE
        outputCard.visibility = if (showCircuit) View.GONE else View.VISIBLE

        val desiredId = if (showCircuit) contentCircuit.id else contentOutput.id
        if (contentToggle.checkedButtonId != desiredId) {
            suppressContentToggleListener = true
            contentToggle.check(desiredId)
            suppressContentToggleListener = false
        }

        if (scrollTo != null) {
            mainScroll.post {
                val y = scrollTo.top
                mainScroll.smoothScrollTo(0, y)
            }
        }
    }

    private fun clearOutput() {
        logText = StringBuilder()
        outputText.text = "Output will appear here."
        lastRunInputBytes = 0
        lastWasmSummary = null
        lastNativeSummary = null
        outputSummaryHeader.visibility = View.GONE
        outputSummaryTable.visibility = View.GONE
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

    private fun parseRunSummary(raw: JSONObject): RunSummary {
        val verifyOk = parseJsonBool(raw.opt("verify_ok"))
        val steps = raw.optInt("steps", -1).takeIf { it >= 0 }

        val timings = raw.optJSONObject("timings_ms")
        val spartan = raw.optJSONObject("spartan")
        val parsedTimings =
            parseRunTimings(
                timings = timings,
                spartan = spartan,
            )

        return RunSummary(
            timings = parsedTimings,
            verifyOk = verifyOk,
            steps = steps,
        )
    }

    private fun parseRunTimings(
        timings: JSONObject?,
        spartan: JSONObject?,
    ): RunTimings? {
        if (timings == null && spartan == null) return null

        val total = optDouble(timings, "total")
        val add = optDouble(timings, "add_steps_total")
        val prove = optDouble(timings, "fold_and_prove")
        val verify = optDouble(timings, "verify")

        val sessionCreate =
            optDouble(timings, "session_create")
                ?: run {
                    if (total == null || add == null || prove == null || verify == null) return@run null
                    max(0.0, total - add - prove - verify)
                }

        return RunTimings(
            sessionCreateMs = sessionCreate,
            ajtaiSetupMs = optDouble(timings, "ajtai_setup"),
            buildCcsMs = optDouble(timings, "build_ccs"),
            sessionInitMs = optDouble(timings, "session_init"),
            addStepsMs = add,
            foldAndProveMs = prove,
            verifyMs = verify,
            totalMs = total,
            spartanProveMs = optDouble(spartan, "prove_ms"),
            spartanVerifyMs = optDouble(spartan, "verify_ms"),
            spartanVerifyOk = parseJsonBool(spartan?.opt("verify_ok")),
            spartanSnarkBytes = optInt(spartan, "snark_bytes"),
            spartanVkBytes = optInt(spartan, "vk_bytes"),
            spartanVkAndSnarkBytes = optInt(spartan, "vk_and_snark_bytes"),
        )
    }

    private fun updateOutputSummary() {
        val wasm = lastWasmSummary
        val native = lastNativeSummary
        if (wasm == null && native == null) {
            outputSummaryHeader.visibility = View.GONE
            outputSummaryTable.visibility = View.GONE
            return
        }

        outputSummaryHeader.visibility = View.VISIBLE
        outputSummaryTable.visibility = View.VISIBLE

        if (lastRunInputBytes > 0) {
            outputSummaryBytes.visibility = View.VISIBLE
            outputSummaryBytes.text = String.format(Locale.US, "%,d bytes", lastRunInputBytes)
        } else {
            outputSummaryBytes.visibility = View.GONE
        }

        val showSpartan =
            (wasm?.timings?.spartanProveMs != null) ||
                (native?.timings?.spartanProveMs != null)

        val rows =
            buildList {
                add(
                    SummaryRow(
                        phase = "Session ready",
                        wasm = fmtMs(wasm?.timings?.sessionCreateMs),
                        native = fmtMs(native?.timings?.sessionCreateMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Ajtai setup",
                        wasm = fmtMs(wasm?.timings?.ajtaiSetupMs),
                        native = fmtMs(native?.timings?.ajtaiSetupMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Build CCS",
                        wasm = fmtMs(wasm?.timings?.buildCcsMs),
                        native = fmtMs(native?.timings?.buildCcsMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Session init",
                        wasm = fmtMs(wasm?.timings?.sessionInitMs),
                        native = fmtMs(native?.timings?.sessionInitMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Add steps",
                        wasm = fmtMs(wasm?.timings?.addStepsMs),
                        native = fmtMs(native?.timings?.addStepsMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Fold+Prove",
                        wasm = fmtMs(wasm?.timings?.foldAndProveMs),
                        native = fmtMs(native?.timings?.foldAndProveMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Verify",
                        wasm = fmtMs(wasm?.timings?.verifyMs),
                        native = fmtMs(native?.timings?.verifyMs),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Total",
                        wasm = fmtMs(wasm?.timings?.totalMs),
                        native = fmtMs(native?.timings?.totalMs),
                    ),
                )

                if (showSpartan) {
                    add(
                        SummaryRow(
                            phase = "Spartan prove",
                            wasm = fmtMs(wasm?.timings?.spartanProveMs),
                            native = fmtMs(native?.timings?.spartanProveMs),
                        ),
                    )
                    add(
                        SummaryRow(
                            phase = "Spartan verify",
                            wasm = fmtMs(wasm?.timings?.spartanVerifyMs),
                            native = fmtMs(native?.timings?.spartanVerifyMs),
                        ),
                    )
                    add(
                        SummaryRow(
                            phase = "Spartan ok",
                            wasm = fmtBool(wasm?.timings?.spartanVerifyOk),
                            native = fmtBool(native?.timings?.spartanVerifyOk),
                        ),
                    )
                    add(
                        SummaryRow(
                            phase = "Spartan snark",
                            wasm = fmtBytes(wasm?.timings?.spartanSnarkBytes),
                            native = fmtBytes(native?.timings?.spartanSnarkBytes),
                        ),
                    )
                    add(
                        SummaryRow(
                            phase = "Spartan vk",
                            wasm = fmtBytes(wasm?.timings?.spartanVkBytes),
                            native = fmtBytes(native?.timings?.spartanVkBytes),
                        ),
                    )
                    add(
                        SummaryRow(
                            phase = "Spartan total",
                            wasm = fmtBytes(wasm?.timings?.spartanVkAndSnarkBytes),
                            native = fmtBytes(native?.timings?.spartanVkAndSnarkBytes),
                        ),
                    )
                }

                add(
                    SummaryRow(
                        phase = "Verify ok",
                        wasm = fmtBool(wasm?.verifyOk),
                        native = fmtBool(native?.verifyOk),
                    ),
                )
                add(
                    SummaryRow(
                        phase = "Steps",
                        wasm = fmtInt(wasm?.steps),
                        native = fmtInt(native?.steps),
                    ),
                )
            }

        outputSummaryTable.text = renderSummaryTable(rows)
    }

    private fun renderSummaryTable(rows: List<SummaryRow>): String {
        if (rows.isEmpty()) return ""

        val phaseHeader = "Phase"
        val wasmHeader = "WASM"
        val nativeHeader = "Native"

        val phaseWidth =
            max(
                phaseHeader.length,
                rows.maxOf { it.phase.length },
            )
        val wasmWidth =
            max(
                wasmHeader.length,
                rows.maxOf { it.wasm.length },
            )
        val nativeWidth =
            max(
                nativeHeader.length,
                rows.maxOf { it.native.length },
            )

        fun rightPad(
            value: String,
            width: Int,
        ): String = value.padEnd(width, ' ')

        fun leftPad(
            value: String,
            width: Int,
        ): String = value.padStart(width, ' ')

        val sb = StringBuilder()
        sb.append(rightPad(phaseHeader, phaseWidth))
            .append("  ")
            .append(leftPad(wasmHeader, wasmWidth))
            .append("  ")
            .append(leftPad(nativeHeader, nativeWidth))

        for (row in rows) {
            sb.append('\n')
            sb.append(rightPad(row.phase, phaseWidth))
                .append("  ")
                .append(leftPad(row.wasm, wasmWidth))
                .append("  ")
                .append(leftPad(row.native, nativeWidth))
        }

        return sb.toString()
    }

    private fun fmtMs(ms: Double?): String {
        if (ms == null || !ms.isFinite()) return "—"
        return String.format(Locale.US, "%.1f ms", ms)
    }

    private fun fmtBool(value: Boolean?): String {
        return when (value) {
            null -> "—"
            true -> "true"
            false -> "false"
        }
    }

    private fun fmtInt(value: Int?): String {
        return value?.toString() ?: "—"
    }

    private fun fmtBytes(bytes: Int?): String {
        if (bytes == null) return "—"
        if (bytes < 1024) {
            return "${bytes} B"
        }
        val kb = bytes.toDouble() / 1024.0
        if (kb < 1024.0) {
            return String.format(Locale.US, "%.2f KB", kb)
        }
        val mb = kb / 1024.0
        return String.format(Locale.US, "%.2f MB", mb)
    }

    private fun parseJsonBool(value: Any?): Boolean? {
        return when (value) {
            is Boolean -> value
            is Number -> value.toInt() != 0
            else -> null
        }
    }

    private fun optDouble(
        obj: JSONObject?,
        key: String,
    ): Double? {
        if (obj == null) return null
        val value = obj.opt(key)
        return (value as? Number)?.toDouble()
    }

    private fun optInt(
        obj: JSONObject?,
        key: String,
    ): Int? {
        if (obj == null) return null
        val value = obj.opt(key)
        return (value as? Number)?.toInt()
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
        runningProgress.visibility = if (isRunning) View.VISIBLE else View.GONE
        if (isRunning) {
            runButton.isEnabled = false
        } else {
            updateBackendAvailability(nativeAvailable = nativeAvailable)
        }
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
