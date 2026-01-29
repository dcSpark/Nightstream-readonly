import Foundation
import WebKit

struct NeoFoldWasmWebViewCapabilities {
    let userAgent: String
    let crossOriginIsolated: Bool
    let supportsThreadsRuntime: Bool
    let singleBundlePresent: Bool
    let threadsBundlePresent: Bool
    let defaultThreads: Int
}

enum NeoFoldWasmWebViewError: Error, LocalizedError {
    case webRootMissing
    case webViewNotInitialized
    case bridgeNotReady
    case invalidMessage
    case invalidStartResponse
    case runFailed(String)
    case missingRawResult

    var errorDescription: String? {
        switch self {
        case .webRootMissing:
            return "Missing web assets in app bundle (web/index.html). Run ./scripts/build_wasm.sh (or ./scripts/build.sh --wasm) to generate the WKWebView wasm bundles."
        case .webViewNotInitialized:
            return "WKWebView not initialized."
        case .bridgeNotReady:
            return "WASM web bridge not ready."
        case .invalidMessage:
            return "Invalid message from WASM web bridge."
        case .invalidStartResponse:
            return "Invalid response from WASM web bridge startRun()."
        case .runFailed(let message):
            return message
        case .missingRawResult:
            return "WASM run completed without a result payload."
        }
    }
}

@MainActor
final class NeoFoldWasmWebViewService: NSObject {
    static let shared = NeoFoldWasmWebViewService()

    private let bridgeReadyTimeoutNs: UInt64 = 6_000_000_000

    private var webView: WKWebView?
    private var server: LocalhostStaticServer?
    private var baseURL: URL?

    private var isReady: Bool = false
    private var readyContinuation: CheckedContinuation<Void, Error>?
    private var loadTask: Task<Void, Error>?

    private var capabilities: NeoFoldWasmWebViewCapabilities?

    private struct PendingRun {
        let log: @MainActor (String) -> Void
        let continuation: CheckedContinuation<NeoFoldRunResult, Error>
    }

    private var pendingRuns: [Int: PendingRun] = [:]
    private var nextRunId: Int = 1

    func makeOrGetWebView() throws -> WKWebView {
        if let webView { return webView }

        let controller = WKUserContentController()
        controller.add(self, name: "neofold")

        let config = WKWebViewConfiguration()
        config.userContentController = controller

        let view = WKWebView(frame: .zero, configuration: config)
        view.navigationDelegate = self

        webView = view
        return view
    }

    func ensureLoaded() async throws {
        if isReady { return }
        if let loadTask {
            try await loadTask.value
            return
        }

        let task = Task<Void, Error> {
            let webRoot: URL = {
                if let indexURL = Bundle.main.url(forResource: "index", withExtension: "html", subdirectory: "web") {
                    return indexURL.deletingLastPathComponent()
                }
                // Fallback for older builds where resources were flattened into the bundle root.
                if let indexURL = Bundle.main.url(forResource: "index", withExtension: "html") {
                    return indexURL.deletingLastPathComponent()
                }
                return URL(fileURLWithPath: "/")
            }()

            guard webRoot.path != "/" else {
                throw NeoFoldWasmWebViewError.webRootMissing
            }

            if server == nil {
                server = try LocalhostStaticServer(webRoot: webRoot)
            }

            if baseURL == nil, let server {
                baseURL = try await server.start()
            }

            let webView = try makeOrGetWebView()
            guard let baseURL else { throw NeoFoldWasmWebViewError.bridgeNotReady }

            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                readyContinuation = continuation
                isReady = false

                let url = baseURL.appendingPathComponent("index.html")
                let req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
                _ = webView.load(req)

                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: self.bridgeReadyTimeoutNs)
                    guard !self.isReady, let cont = self.readyContinuation else { return }
                    self.readyContinuation = nil
                    cont.resume(throwing: NeoFoldWasmWebViewError.bridgeNotReady)
                }
            }
        }

        loadTask = task
        do {
            try await task.value
        } catch {
            loadTask = nil
            throw error
        }
    }

    func getCapabilities() async throws -> NeoFoldWasmWebViewCapabilities {
        if let capabilities { return capabilities }
        try await ensureLoaded()
        guard let webView else { throw NeoFoldWasmWebViewError.webViewNotInitialized }

        let any = try await callJS(webView, body: "return window.__swift_neofold_getCapabilities()")
        guard let caps = Self.parseCapabilities(any) else {
            throw NeoFoldWasmWebViewError.invalidMessage
        }
        capabilities = caps
        return caps
    }

    func runProveVerify(
        json: String,
        doSpartan: Bool,
        log: @escaping @MainActor (String) -> Void
    ) async throws -> NeoFoldRunResult {
        try await ensureLoaded()
        guard isReady else { throw NeoFoldWasmWebViewError.bridgeNotReady }
        guard let webView else { throw NeoFoldWasmWebViewError.webViewNotInitialized }

        let runId = nextRunId
        nextRunId &+= 1

        return try await withCheckedThrowingContinuation { continuation in
            pendingRuns[runId] = PendingRun(log: log, continuation: continuation)

            Task {
                do {
                    _ = try await callJS(
                        webView,
                        body: "return window.__swift_neofold_startRun(json, doSpartan, opts)",
                        arguments: [
                            "json": json,
                            "doSpartan": doSpartan,
                            "opts": ["id": runId],
                        ]
                    )
                } catch {
                    if let pending = self.pendingRuns.removeValue(forKey: runId) {
                        pending.continuation.resume(throwing: error)
                    }
                }
            }
        }
    }

    private func handleBridgeMessage(_ body: Any) {
        guard let msg = body as? [String: Any], let type = msg["type"] as? String else {
            readyContinuation?.resume(throwing: NeoFoldWasmWebViewError.invalidMessage)
            readyContinuation = nil
            return
        }

        switch type {
        case "ready":
            if let caps = Self.parseCapabilities(msg["capabilities"]) {
                capabilities = caps
            }
            isReady = true
            readyContinuation?.resume(returning: ())
            readyContinuation = nil

        case "log":
            guard let id = Self.int(msg["id"]), let line = msg["line"] as? String else { return }
            pendingRuns[id]?.log(line)

        case "done":
            guard let id = Self.int(msg["id"]) else { return }
            guard let pending = pendingRuns.removeValue(forKey: id) else { return }

            do {
                let result = try Self.parseRunResult(raw: msg["raw"], spartan: msg["spartan"])
                pending.continuation.resume(returning: result)
            } catch {
                pending.continuation.resume(throwing: error)
            }

        case "error":
            let id = Self.int(msg["id"]) ?? -1
            let errString = (msg["error"] as? String) ?? "Unknown error"

            if id == -1 {
                readyContinuation?.resume(throwing: NeoFoldWasmWebViewError.runFailed(errString))
                readyContinuation = nil
                return
            }

            if let pending = pendingRuns.removeValue(forKey: id) {
                pending.continuation.resume(throwing: NeoFoldWasmWebViewError.runFailed(errString))
            }

        default:
            break
        }
    }

    private static func parseCapabilities(_ any: Any?) -> NeoFoldWasmWebViewCapabilities? {
        guard let dict = any as? [String: Any] else { return nil }
        guard let userAgent = dict["userAgent"] as? String else { return nil }

        let crossOriginIsolated = (dict["crossOriginIsolated"] as? Bool) ?? false
        let supportsThreadsRuntime = (dict["supportsThreadsRuntime"] as? Bool) ?? false
        let singleBundlePresent = (dict["singleBundlePresent"] as? Bool) ?? false
        let threadsBundlePresent = (dict["threadsBundlePresent"] as? Bool) ?? false
        let defaultThreads = int(dict["defaultThreads"]) ?? 0

        return NeoFoldWasmWebViewCapabilities(
            userAgent: userAgent,
            crossOriginIsolated: crossOriginIsolated,
            supportsThreadsRuntime: supportsThreadsRuntime,
            singleBundlePresent: singleBundlePresent,
            threadsBundlePresent: threadsBundlePresent,
            defaultThreads: defaultThreads
        )
    }

    private static func parseRunResult(raw: Any?, spartan: Any?) throws -> NeoFoldRunResult {
        guard let rawDict = raw as? [String: Any] else {
            throw NeoFoldWasmWebViewError.missingRawResult
        }

        let timings = parseTimings(rawDict)

        var out = NeoFoldRunResult()
        out.timings = timings

        if let spartanDict = spartan as? [String: Any],
           let filename = spartanDict["filename"] as? String,
           let b64 = spartanDict["bytes_b64"] as? String,
           let data = Data(base64Encoded: b64)
        {
            out.spartanFilename = filename
            out.spartanSnark = data
        }

        return out
    }

    private static func parseTimings(_ raw: [String: Any]) -> NeoFoldWasmTimings? {
        guard let timings = raw["timings_ms"] as? [String: Any] else { return nil }

        let ajtai = double(timings["ajtai_setup"]) ?? 0
        let buildCcs = double(timings["build_ccs"]) ?? 0
        let sessionInit = double(timings["session_init"]) ?? 0
        let sessionCreate = double(timings["session_create"]) ?? 0
        let addSteps = double(timings["add_steps_total"]) ?? 0
        let foldAndProve = double(timings["fold_and_prove"]) ?? 0
        let verify = double(timings["verify"]) ?? 0
        let total = double(timings["total"]) ?? 0

        let steps = int(raw["steps"]) ?? 0
        let verifyOk = (raw["verify_ok"] as? Bool) ?? false

        let spartanDict = raw["spartan"] as? [String: Any]
        let spartanProveMs = double(spartanDict?["prove_ms"])
        let spartanVerifyMs = double(spartanDict?["verify_ms"])
        let spartanVerifyOk = spartanDict?["verify_ok"] as? Bool
        let spartanSnarkBytes = int(spartanDict?["snark_bytes"])
        let spartanVkBytes = int(spartanDict?["vk_bytes"])
        let spartanVkAndSnarkBytes = int(spartanDict?["vk_and_snark_bytes"])

        return NeoFoldWasmTimings(
            ajtaiSetupMs: ajtai,
            buildCcsMs: buildCcs,
            sessionInitMs: sessionInit,
            sessionCreateMs: sessionCreate,
            addStepsMs: addSteps,
            foldAndProveMs: foldAndProve,
            verifyMs: verify,
            totalMs: total,
            steps: steps,
            verifyOk: verifyOk,
            spartanProveMs: spartanProveMs,
            spartanVerifyMs: spartanVerifyMs,
            spartanVerifyOk: spartanVerifyOk,
            spartanSnarkBytes: spartanSnarkBytes,
            spartanVkBytes: spartanVkBytes,
            spartanVkAndSnarkBytes: spartanVkAndSnarkBytes
        )
    }

    private static func int(_ any: Any?) -> Int? {
        if let n = any as? NSNumber { return n.intValue }
        if let n = any as? Int { return n }
        if let s = any as? String, let n = Int(s) { return n }
        return nil
    }

    private static func double(_ any: Any?) -> Double? {
        if let n = any as? NSNumber { return n.doubleValue }
        if let n = any as? Double { return n }
        if let s = any as? String, let n = Double(s) { return n }
        return nil
    }
}

extension NeoFoldWasmWebViewService: WKNavigationDelegate {
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        isReady = false
        readyContinuation?.resume(throwing: error)
        readyContinuation = nil
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        isReady = false
        readyContinuation?.resume(throwing: error)
        readyContinuation = nil
    }
}

extension NeoFoldWasmWebViewService: WKScriptMessageHandler {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        handleBridgeMessage(message.body)
    }
}

@MainActor
private func callJS(
    _ webView: WKWebView,
    body: String,
    arguments: [String: Any] = [:]
) async throws -> Any {
    try await withCheckedThrowingContinuation { continuation in
        webView.callAsyncJavaScript(body, arguments: arguments, in: nil, in: .page) { result in
            switch result {
            case .success(let any):
                continuation.resume(returning: any as Any)
            case .failure(let error):
                continuation.resume(throwing: error)
            }
        }
    }
}
