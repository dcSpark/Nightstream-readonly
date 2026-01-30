package com.midnight.neofold.demo

import android.content.Context
import android.net.Uri
import android.os.Build
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.webkit.WebViewAssetLoader
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.json.JSONObject

data class WasmCapabilities(
    val userAgent: String,
    val crossOriginIsolated: Boolean,
    val supportsThreadsRuntime: Boolean,
    val singleBundlePresent: Boolean,
    val threadsBundlePresent: Boolean,
    val defaultThreads: Int,
)

data class WasmRunResult(
    val raw: JSONObject,
    val spartanFilename: String?,
    val spartanBytes: ByteArray?,
)

class WasmWebViewService(
    private val context: Context,
    private val webView: WebView,
    private val scope: CoroutineScope,
) {
    private val bridgeReadyTimeoutMs = 6_000L

    private val assetLoader =
        WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
            .build()

    private var started: Boolean = false
    private val readyDeferred = CompletableDeferred<WasmCapabilities>()
    private var capabilities: WasmCapabilities? = null

    private data class PendingRun(
        val onLog: (String) -> Unit,
        val deferred: CompletableDeferred<WasmRunResult>,
    )

    private val pendingRuns: MutableMap<Int, PendingRun> = mutableMapOf()
    private var nextRunId: Int = 1

    suspend fun ensureLoaded(): WasmCapabilities {
        if (capabilities != null) {
            return capabilities!!
        }

        startIfNeeded()

        val caps =
            withTimeout(bridgeReadyTimeoutMs) {
                readyDeferred.await()
            }
        capabilities = caps
        return caps
    }

    suspend fun runProveVerify(
        json: String,
        doSpartan: Boolean,
        onLog: (String) -> Unit,
    ): WasmRunResult {
        ensureLoaded()

        return withContext(Dispatchers.Main.immediate) {
            val id = nextRunId
            nextRunId += 1

            val deferred = CompletableDeferred<WasmRunResult>()
            pendingRuns[id] = PendingRun(onLog, deferred)

            try {
                val jsJson = JSONObject.quote(json)
                val jsDoSpartan = if (doSpartan) "true" else "false"
                val script = "window.__swift_neofold_startRun($jsJson, $jsDoSpartan, { id: $id });"
                webView.evaluateJavascript(script, null)
            } catch (e: Throwable) {
                pendingRuns.remove(id)
                deferred.completeExceptionally(e)
            }

            deferred.await()
        }
    }

    fun destroy() {
        webView.removeJavascriptInterface("neofoldAndroid")
        webView.stopLoading()
        webView.destroy()
    }

    private suspend fun startIfNeeded() {
        withContext(Dispatchers.Main.immediate) {
            if (started) return@withContext
            started = true

            if (BuildConfig.DEBUG) {
                WebView.setWebContentsDebuggingEnabled(true)
            }

            webView.settings.javaScriptEnabled = true
            webView.settings.domStorageEnabled = true
            webView.settings.cacheMode = WebSettings.LOAD_NO_CACHE

            webView.settings.allowFileAccess = false
            webView.settings.allowContentAccess = false

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                webView.settings.safeBrowsingEnabled = true
            }

            webView.webChromeClient = WebChromeClient()
            webView.addJavascriptInterface(AndroidBridge(), "neofoldAndroid")
            webView.webViewClient =
                object : WebViewClient() {
                    override fun shouldInterceptRequest(
                        view: WebView,
                        request: WebResourceRequest,
                    ): WebResourceResponse? {
                        val resp = assetLoader.shouldInterceptRequest(request.url) ?: return null

                        val headers = (resp.responseHeaders?.toMutableMap() ?: mutableMapOf()).apply {
                            put("Cross-Origin-Opener-Policy", "same-origin")
                            put("Cross-Origin-Embedder-Policy", "require-corp")
                            put("Cross-Origin-Resource-Policy", "same-origin")
                            put("Cache-Control", "no-store")
                        }
                        resp.responseHeaders = headers
                        return resp
                    }

                    override fun shouldInterceptRequest(
                        view: WebView,
                        url: String,
                    ): WebResourceResponse? {
                        // Deprecated, but needed for some older Chromium builds.
                        val resp = assetLoader.shouldInterceptRequest(Uri.parse(url)) ?: return null

                        val headers = (resp.responseHeaders?.toMutableMap() ?: mutableMapOf()).apply {
                            put("Cross-Origin-Opener-Policy", "same-origin")
                            put("Cross-Origin-Embedder-Policy", "require-corp")
                            put("Cross-Origin-Resource-Policy", "same-origin")
                            put("Cache-Control", "no-store")
                        }
                        resp.responseHeaders = headers
                        return resp
                    }
                }

            webView.loadUrl("https://appassets.androidplatform.net/assets/web/index.html")
        }
    }

    private inner class AndroidBridge {
        @JavascriptInterface
        fun postMessage(json: String) {
            scope.launch(Dispatchers.Main.immediate) { handleBridgeMessage(json) }
        }
    }

    private fun handleBridgeMessage(json: String) {
        val obj =
            try {
                JSONObject(json)
            } catch (e: Throwable) {
                if (!readyDeferred.isCompleted) {
                    readyDeferred.completeExceptionally(IllegalStateException("Invalid bridge message JSON"))
                }
                return
            }

        when (obj.optString("type")) {
            "ready" -> {
                val capObj = obj.optJSONObject("capabilities")
                if (capObj == null) {
                    readyDeferred.completeExceptionally(IllegalStateException("Bridge ready without capabilities"))
                    return
                }
                val caps = parseCapabilities(capObj)
                capabilities = caps
                if (!readyDeferred.isCompleted) {
                    readyDeferred.complete(caps)
                }
            }

            "log" -> {
                val id = obj.optInt("id", -1)
                val line = obj.optString("line", "")
                pendingRuns[id]?.onLog?.invoke(line)
            }

            "done" -> {
                val id = obj.optInt("id", -1)
                val pending = pendingRuns.remove(id) ?: return

                val raw = obj.optJSONObject("raw")
                if (raw == null) {
                    pending.deferred.completeExceptionally(IllegalStateException("WASM run missing raw result"))
                    return
                }

                val spartan = obj.optJSONObject("spartan")
                val spartanFilename = spartan?.optString("filename", null)
                val spartanBytes =
                    spartan
                        ?.optString("bytes_b64", null)
                        ?.let { Base64.decode(it, Base64.DEFAULT) }

                pending.deferred.complete(
                    WasmRunResult(
                        raw = raw,
                        spartanFilename = spartanFilename,
                        spartanBytes = spartanBytes,
                    ),
                )
            }

            "error" -> {
                val id = obj.optInt("id", -1)
                val message = obj.optString("error", "Unknown error")

                if (id == -1) {
                    if (!readyDeferred.isCompleted) {
                        readyDeferred.completeExceptionally(IllegalStateException(message))
                    }
                    return
                }

                val pending = pendingRuns.remove(id) ?: return
                pending.deferred.completeExceptionally(IllegalStateException(message))
            }
        }
    }

    private fun parseCapabilities(obj: JSONObject): WasmCapabilities {
        return WasmCapabilities(
            userAgent = obj.optString("userAgent", ""),
            crossOriginIsolated = obj.optBoolean("crossOriginIsolated", false),
            supportsThreadsRuntime = obj.optBoolean("supportsThreadsRuntime", false),
            singleBundlePresent = obj.optBoolean("singleBundlePresent", false),
            threadsBundlePresent = obj.optBoolean("threadsBundlePresent", false),
            defaultThreads = obj.optInt("defaultThreads", 0),
        )
    }
}
