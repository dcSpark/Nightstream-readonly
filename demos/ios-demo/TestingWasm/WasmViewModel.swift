import Foundation
import Combine

@MainActor
final class WasmViewModel: ObservableObject {
    private enum WasmRuntimeKind {
        case webView
        case javaScriptCore
    }

    enum ProverBackend: String, CaseIterable, Identifiable {
        case wasm
        case native
        case both

        var id: String { rawValue }

        var title: String {
            switch self {
            case .wasm:
                return "WASM"
            case .native:
                return "Native"
            case .both:
                return "Both"
            }
        }

        var includesWasm: Bool {
            self == .wasm || self == .both
        }

        var includesNative: Bool {
            self == .native || self == .both
        }
    }

    enum Example: String, CaseIterable, Identifiable {
        case toySquare = "toy_square"
        case toySquareFolding8Steps = "toy_square_folding_8_steps"
        case poseidon2Batch1 = "poseidon2_ic_batch_1"

        var id: String { rawValue }

        var title: String {
            switch self {
            case .toySquare:
                return "Load toy circuit"
            case .toySquareFolding8Steps:
                return "Load toy folding circuit (8 steps)"
            case .poseidon2Batch1:
                return "Load Poseidon2 IC (batch 1)"
            }
        }

        var logLine: String {
            switch self {
            case .toySquare:
                return "Loaded toy circuit."
            case .toySquareFolding8Steps:
                return "Loaded toy folding circuit (8 steps)."
            case .poseidon2Batch1:
                return "Loaded Poseidon2 IC circuit batch 1."
            }
        }
    }

    @Published private(set) var selectedExample: Example?
    @Published var wasmStatus: String = "WASM: loading…"
    @Published var threadsStatus: String = "Threads: checking…"
    @Published var circuitJson: String = ""
    @Published var circuitSource: String = ""
    @Published var circuitSizeBytes: Int = 0
    @Published var circuitPreview: String = ""
    @Published var logText: String = ""
    @Published var isRunning: Bool = false
    @Published var proverBackend: ProverBackend = NeoFoldNativeService.isAvailable ? .both : .wasm
    @Published var compressSpartan: Bool = false
    @Published var spartanShareURL: URL?

    @Published var lastRunInputBytes: Int = 0
    @Published var lastWasmTimings: NeoFoldWasmTimings?
    @Published var lastNativeTimings: NeoFoldNativeTimings?
    @Published var lastNativeVerifyOk: Bool?
    @Published var lastNativeSteps: Int?

    let inlineEditorMaxBytes: Int = 60_000
    private let previewMaxChars: Int = 6_000

    private let logFlushIntervalNs: UInt64 = 75_000_000
    private let maxLogCharacters: Int = 30_000
    private let logTruncationNotice = "… (output truncated) …\n"
    private var pendingLogLines: [String] = []
    private var logFlushTask: Task<Void, Never>?
    private var wasmRuntime: WasmRuntimeKind = .webView

    func onAppear() async {
        await loadWasmRuntime()

        if circuitJson.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            loadExample(.poseidon2Batch1)
        }
    }

    func clearLog() {
        logFlushTask?.cancel()
        logFlushTask = nil
        pendingLogLines.removeAll(keepingCapacity: true)
        logText = ""
    }

    func appendLog(_ line: String) {
        pendingLogLines.append(line)
        scheduleLogFlush()
    }

    func loadExample(_ example: Example) {
        let previousSelection = selectedExample
        selectedExample = example

        Task {
            do {
                let txt = try await Task.detached(priority: .userInitiated) {
                    try Self.loadTextResource(name: example.rawValue, ext: "json", subdirectories: [
                        "examples",
                        "Resources/examples",
                        "Resources",
                    ])
                }.value

                setCircuitJson(txt, source: example.title)
                appendLog("\(example.logLine) (\(txt.utf8.count) bytes).")
            } catch {
                selectedExample = previousSelection
                appendLog("ERROR: Failed to load example: \(error.localizedDescription)")
            }
        }
    }

    func importJsonFile(from url: URL) {
        Task {
            do {
                let txt = try await Task.detached(priority: .userInitiated) {
                    try String(contentsOf: url, encoding: .utf8)
                }.value

                setCircuitJson(txt, source: "File: \(url.lastPathComponent)")
                selectedExample = nil
                appendLog("Loaded file \"\(url.lastPathComponent)\" (\(txt.utf8.count) bytes).")
            } catch {
                appendLog("ERROR: Failed to read file: \(error.localizedDescription)")
            }
        }
    }

    func runProveVerify() {
        let json = circuitJson
        guard !json.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            appendLog("No JSON provided.")
            return
        }
        guard !isRunning else {
            appendLog("Run already in progress.")
            return
        }

        isRunning = true
        clearLog()
        spartanShareURL = nil
        lastRunInputBytes = json.utf8.count
        lastWasmTimings = nil
        lastNativeTimings = nil
        lastNativeVerifyOk = nil
        lastNativeSteps = nil

        Task {
            do {
                if proverBackend.includesWasm {
                    appendLog("=== WASM ===")
                    let result: NeoFoldRunResult
                    switch wasmRuntime {
                    case .webView:
                        result = try await NeoFoldWasmWebViewService.shared.runProveVerify(
                            json: json,
                            doSpartan: compressSpartan
                        ) { [weak self] line in
                            self?.handleWasmLogLine(line)
                        }
                    case .javaScriptCore:
                        result = try await NeoFoldWasmService.shared.runProveVerify(
                            json: json,
                            doSpartan: compressSpartan
                        ) { [weak self] line in
                            self?.appendLog(line)
                        }
                    }

                    lastWasmTimings = result.timings

                    if let snark = result.spartanSnark, let filename = result.spartanFilename {
                        spartanShareURL = try writeTempFile(named: filename, data: snark)
                    }
                }

                if proverBackend.includesNative {
                    if proverBackend.includesWasm { appendLog("") }
                    appendLog("=== NATIVE (NeoFoldFFI) ===")
                    let result = try await NeoFoldNativeService.shared.runProveVerify(
                        json: json,
                        doSpartan: compressSpartan
                    ) { [weak self] line in
                        self?.appendLog(line)
                    }
                    lastNativeTimings = result.timings
                    lastNativeVerifyOk = result.verifyOk
                    lastNativeSteps = result.steps
                }
            } catch {
                appendLog("ERROR: \(error.localizedDescription)")
            }

            flushLogsNow()
            isRunning = false
        }
    }

    private func loadWasmRuntime() async {
        do {
            try await NeoFoldWasmWebViewService.shared.ensureLoaded()
            let caps = try await NeoFoldWasmWebViewService.shared.getCapabilities()
            guard caps.singleBundlePresent else {
                throw NeoFoldWasmWebViewError.webRootMissing
            }

            wasmRuntime = .webView
            wasmStatus = "WASM: loaded (WKWebView)"
            threadsStatus = formatThreadsStatus(caps)
            return
        } catch {
            // Fall back to JavaScriptCore.
        }

        do {
            try await NeoFoldWasmService.shared.ensureLoaded()
            wasmRuntime = .javaScriptCore
            wasmStatus = "WASM: loaded (JavaScriptCore)"
            threadsStatus = "Threads: disabled (iOS JavaScriptCore)"
        } catch {
            wasmStatus = "WASM: error (\(error.localizedDescription))"
            threadsStatus = "Threads: unavailable"
        }
    }

    private func formatThreadsStatus(_ caps: NeoFoldWasmWebViewCapabilities) -> String {
        if !caps.crossOriginIsolated {
            return "Threads: disabled (crossOriginIsolated=false)"
        }
        if !caps.supportsThreadsRuntime {
            return "Threads: disabled (no SharedArrayBuffer/Atomics)"
        }
        if !caps.threadsBundlePresent {
            return "Threads: supported, bundle missing"
        }
        if caps.defaultThreads > 0 {
            return "Threads: available (\(caps.defaultThreads) workers)"
        }
        return "Threads: available"
    }

    private func handleWasmLogLine(_ line: String) {
        if line.hasPrefix("Threads: initialized (") {
            threadsStatus = line.replacingOccurrences(of: "Threads: initialized", with: "Threads: enabled")
        } else if line.hasPrefix("Threads init failed;") {
            threadsStatus = "Threads: disabled (init failed)"
        }
        appendLog(line)
    }

    private func scheduleLogFlush() {
        if logFlushTask != nil { return }

        logFlushTask = Task { @MainActor in
            defer { logFlushTask = nil }
            try? await Task.sleep(nanoseconds: logFlushIntervalNs)
            flushLogsNow()
        }
    }

    private func flushLogsNow() {
        guard !pendingLogLines.isEmpty else { return }
        let chunk = pendingLogLines.joined(separator: "\n") + "\n"
        pendingLogLines.removeAll(keepingCapacity: true)
        logText.append(contentsOf: chunk)
        truncateLogIfNeeded()
    }

    private func truncateLogIfNeeded() {
        guard logText.count > maxLogCharacters else { return }
        let budget = max(0, maxLogCharacters - logTruncationNotice.count)
        logText = logTruncationNotice + logText.suffix(budget)
    }

    func setCircuitJsonFromEditor(_ json: String) {
        setCircuitJson(json, source: "Custom JSON")
        selectedExample = nil
        appendLog("Updated circuit JSON (\(json.utf8.count) bytes).")
    }

    var isCircuitLarge: Bool {
        circuitSizeBytes > inlineEditorMaxBytes
    }

    private func setCircuitJson(_ json: String, source: String) {
        circuitJson = json
        circuitSource = source
        circuitSizeBytes = json.utf8.count
        circuitPreview = String(json.prefix(previewMaxChars))
    }

    nonisolated private static func loadTextResource(name: String, ext: String, subdirectories: [String]) throws -> String {
        if let url = Bundle.main.url(forResource: name, withExtension: ext) {
            return try String(contentsOf: url, encoding: .utf8)
        }
        for subdir in subdirectories {
            if let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdir) {
                return try String(contentsOf: url, encoding: .utf8)
            }
        }
        throw WasmRuntimeError.resourceMissing("\(name).\(ext)")
    }

    private func writeTempFile(named filename: String, data: Data) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent(filename, isDirectory: false)
        try data.write(to: url, options: .atomic)
        return url
    }
}
