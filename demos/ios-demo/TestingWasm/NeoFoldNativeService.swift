import Foundation
import os
import os.signpost

#if canImport(NeoFoldFFI)
import NeoFoldFFI
#endif

struct NeoFoldNativeRunResult {
    let rawResultJson: String
    let timings: NeoFoldNativeTimings?
    let verifyOk: Bool?
    let steps: Int?
}

struct NeoFoldNativeTimings {
    let ajtaiSetupMs: Double?
    let buildCcsMs: Double?
    let sessionInitMs: Double?
    let sessionCreateMs: Double?
    let addStepsMs: Double?
    let foldAndProveMs: Double?
    let foldStepsMs: [Double]?
    let verifyMs: Double?
    let totalMs: Double?
    let spartanProveMs: Double?
    let spartanVerifyMs: Double?
    let spartanVerifyOk: Bool?
    let spartanSnarkBytes: Int?
    let spartanVkBytes: Int?
    let spartanVkAndSnarkBytes: Int?
}

enum NeoFoldNativeError: Error, LocalizedError {
    case frameworkUnavailable(String)
    case invalidUtf8Result
    case ffiError(code: Int32, message: String)

    var errorDescription: String? {
        switch self {
        case .frameworkUnavailable(let message):
            return message
        case .invalidUtf8Result:
            return "Native neo-fold returned invalid UTF-8."
        case .ffiError(let code, let message):
            return "NeoFoldFFI error (code \(code)): \(message)"
        }
    }
}

final class NeoFoldNativeService {
    static let shared = NeoFoldNativeService()

    static var isAvailable: Bool {
        #if canImport(NeoFoldFFI)
        return true
        #else
        return false
        #endif
    }

    private static let subsystem = "com.midnight.TestingWasm"
    private static let logger = Logger(subsystem: subsystem, category: "neo-fold-native")
    private static let signpostLog = OSLog(subsystem: subsystem, category: "neo-fold-native")

    private let queue = DispatchQueue(label: "TestingWasm.neoFold.native", qos: .userInitiated)
}

#if canImport(NeoFoldFFI)
extension NeoFoldNativeService {
    func runProveVerify(
        json: String,
        doSpartan: Bool,
        log: @escaping @MainActor (String) -> Void
    ) async throws -> NeoFoldNativeRunResult {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let result = try Self.runProveVerifySync(json: json, doSpartan: doSpartan, log: log)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func runProveVerifySync(
        json: String,
        doSpartan: Bool,
        log: @escaping @MainActor (String) -> Void
    ) throws -> NeoFoldNativeRunResult {
        func emit(_ line: String) {
            Task { @MainActor in
                log(line)
            }
        }

        let inputBytes = json.utf8.count
        logger.info("Native run start (bytes=\(inputBytes, privacy: .public), spartan=\(doSpartan, privacy: .public))")

        let signpostID = OSSignpostID(log: signpostLog)
        os_signpost(
            .begin,
            log: signpostLog,
            name: "NeoFold Native Run",
            signpostID: signpostID,
            "bytes=%{public}d spartan=%{public}d",
            inputBytes,
            doSpartan ? 1 : 0
        )

        emit("Running native NeoFoldFFI workflow…")
        emit("Input JSON size: \(fmtBytes(Double(inputBytes)))")
        if doSpartan {
            emit("Spartan2: enabled")
        }
        if let parsed = tryParseTestExport(json: json) {
            emit("Input export: constraints=\(parsed.constraints) variables=\(parsed.variables) steps=\(parsed.steps)")
        }

        let jsonData = Data(json.utf8)
        let bytes = [UInt8](jsonData)

        var outPtr: UnsafeMutablePointer<UInt8>?
        var outLen: Int = 0
        var errPtr: UnsafeMutablePointer<UInt8>?
        var errLen: Int = 0

        let doSpartanFlag: Int32 = doSpartan ? 1 : 0

        let rc: Int32 = bytes.withUnsafeBytes { rawBuf in
            let base = rawBuf.bindMemory(to: UInt8.self).baseAddress
            return neo_fold_run_wasm_demo_workflow_json(
                base,
                bytes.count,
                doSpartanFlag,
                &outPtr,
                &outLen,
                &errPtr,
                &errLen
            )
        }

        defer {
            if let outPtr {
                neo_fold_free_bytes(outPtr, outLen)
            }
            if let errPtr {
                neo_fold_free_bytes(errPtr, errLen)
            }
        }

        guard rc == 0 else {
            let msg: String
            if let errPtr, errLen > 0 {
                let data = Data(bytes: errPtr, count: errLen)
                msg = String(data: data, encoding: .utf8) ?? "<non-utf8 error>"
            } else {
                msg = "Unknown error"
            }
            os_signpost(.end, log: signpostLog, name: "NeoFold Native Run", signpostID: signpostID, "rc=%{public}d", rc)
            logger.error("Native run failed (rc=\(rc, privacy: .public))")
            throw NeoFoldNativeError.ffiError(code: rc, message: msg)
        }

        guard let outPtr, outLen > 0 else {
            os_signpost(.end, log: signpostLog, name: "NeoFold Native Run", signpostID: signpostID, "rc=%{public}d empty=1", rc)
            throw NeoFoldNativeError.ffiError(code: rc, message: "No output returned.")
        }

        let outData = Data(bytes: outPtr, count: outLen)
        guard let outJson = String(data: outData, encoding: .utf8) else {
            os_signpost(.end, log: signpostLog, name: "NeoFold Native Run", signpostID: signpostID, "rc=%{public}d utf8=0", rc)
            throw NeoFoldNativeError.invalidUtf8Result
        }

        os_signpost(.end, log: signpostLog, name: "NeoFold Native Run", signpostID: signpostID, "rc=%{public}d", rc)

        let parsed = Self.parseSummary(json: outData)
        let timings = parsed.timings
        if let timings, let create = timings.sessionCreateMs {
            emit("Session ready (approx \(fmtMs(create)))")
        }
        if let params = parsed.root?["params"] as? [String: Any] {
            let parts: [String] = [
                "b=\(jsString(params["b"]))",
                "d=\(jsString(params["d"]))",
                "kappa=\(jsString(params["kappa"]))",
                "k_rho=\(jsString(params["k_rho"]))",
                "T=\(jsString(params["T"]))",
                "s=\(jsString(params["s"]))",
                "lambda=\(jsString(params["lambda"]))",
            ]
            emit("Params: \(parts.joined(separator: " "))")
        }

        if let circuit = parsed.root?["circuit"] as? [String: Any] {
            emit(
                "Circuit (R1CS): constraints=\(jsString(circuit["r1cs_constraints"])) variables=\(jsString(circuit["r1cs_variables"])) padded_n=\(jsString(circuit["r1cs_padded_n"])) A_nnz=\(jsString(circuit["r1cs_a_nnz"])) B_nnz=\(jsString(circuit["r1cs_b_nnz"])) C_nnz=\(jsString(circuit["r1cs_c_nnz"]))"
            )

            let nonzeroRatio = (circuit["witness_nonzero_ratio"] as? NSNumber)?.doubleValue
            let nonzeroPercent: String
            if let nonzeroRatio {
                nonzeroPercent = String(format: "%.2f%%", nonzeroRatio * 100.0)
            } else {
                nonzeroPercent = "?"
            }

            emit(
                "Witness: steps=\(jsString(circuit["witness_steps"])) fields_total=\(jsString(circuit["witness_fields_total"])) fields_min=\(jsString(circuit["witness_fields_min"])) fields_max=\(jsString(circuit["witness_fields_max"])) nonzero=\(jsString(circuit["witness_nonzero_fields_total"])) (\(nonzeroPercent))"
            )

            emit(
                "Circuit (CCS): n=\(jsString(circuit["ccs_n"])) m=\(jsString(circuit["ccs_m"])) t=\(jsString(circuit["ccs_t"])) max_degree=\(jsString(circuit["ccs_max_degree"])) poly_terms=\(jsString(circuit["ccs_poly_terms"])) nnz_total=\(jsString(circuit["ccs_matrix_nnz_total"]))"
            )

            if let nnz = circuit["ccs_matrix_nnz"] {
                if let json = jsonString(nnz, pretty: true), json != "[]" {
                    emit("CCS matrices nnz: \(json)")
                }
            }
        }

        if let timings, let ajtai = timings.ajtaiSetupMs, let build = timings.buildCcsMs, let sessionInit = timings.sessionInitMs {
            emit("Timings: ajtai=\(fmtMs(ajtai)) build_ccs=\(fmtMs(build)) session_init=\(fmtMs(sessionInit))")
        }

        if timings != nil {
            emit("Adding witness steps…")
        }
        if let timings, let add = timings.addStepsMs {
            emit("Timings: add_steps_total=\(fmtMs(add))")
        }

        if timings != nil {
            emit("Folding + proving…")
        }
        if let timings, let prove = timings.foldAndProveMs {
            emit("Timings: prove=\(fmtMs(prove))")
        }
        if let timings, let foldSteps = timings.foldStepsMs, !foldSteps.isEmpty {
            emit("Folding prove per-step: \(fmtMsList(foldSteps))")
            let sum = foldSteps.reduce(0, +)
            let avg = sum / Double(foldSteps.count)
            let min = foldSteps.min() ?? 0
            let max = foldSteps.max() ?? 0
            emit("Folding prove per-step stats: avg=\(fmtMs(avg)) min=\(fmtMs(min)) max=\(fmtMs(max))")
        }

        if timings != nil {
            emit("Verifying folding proof…")
        }
        if let timings, let verify = timings.verifyMs {
            emit("Timings: verify=\(fmtMs(verify))")
        }
        if let verifyOk = parsed.verifyOk, let steps = parsed.steps, let timings, let total = timings.totalMs {
            emit("OK: verify_ok=\(verifyOk) steps=\(steps) (total \(fmtMs(total)))")
        }

        if let proofEstimate = parsed.root?["proof_estimate"] as? [String: Any] {
            emit(
                "Proof estimate: proof_steps=\(jsString(proofEstimate["proof_steps"])) final_acc_len=\(jsString(proofEstimate["final_accumulator_len"]))"
            )
            emit(
                "Proof estimate: commitments fold_lane=\(jsString(proofEstimate["fold_lane_commitments"])) mem_cpu_val=\(jsString(proofEstimate["mem_cpu_val_claim_commitments"])) val_lane=\(jsString(proofEstimate["val_lane_commitments"])) total=\(jsString(proofEstimate["total_commitments"]))"
            )
            let estimatedBytes = (proofEstimate["estimated_commitment_bytes"] as? NSNumber)?.doubleValue ?? 0
            emit(
                "Proof estimate: commitment_bytes=\(jsString(proofEstimate["commitment_bytes"])) (d=\(jsString(proofEstimate["commitment_d"])) kappa=\(jsString(proofEstimate["commitment_kappa"]))) estimated_commitment_bytes=\(fmtBytes(estimatedBytes))"
            )
        }

        if let folding = parsed.root?["folding"] as? [String: Any] {
            if let kIn = folding["k_in"], let json = jsonString(kIn, pretty: true), json != "[]" {
                emit("Folding k_in per step: \(json)")
            }
            if let accLenAfter = folding["acc_len_after"], let json = jsonString(accLenAfter, pretty: true), json != "[]" {
                emit("Folding accumulator len after step: \(json)")
            }
        }

        if doSpartan, let timings, let spProve = timings.spartanProveMs {
            emit("Compressing with Spartan2…")

            let sizeParts: [String?] = [
                timings.spartanSnarkBytes.map { "snark=\(fmtBytes(Double($0)))" },
                timings.spartanVkBytes.map { "vk=\(fmtBytes(Double($0)))" },
                timings.spartanVkAndSnarkBytes.map { "total(vk+snark)=\(fmtBytes(Double($0)))" },
            ]

            emit("Spartan2: prove=\(fmtMs(spProve)) \(sizeParts.compactMap { $0 }.joined(separator: " "))")

            if let spVerify = timings.spartanVerifyMs {
                emit("Spartan2: verify=\(fmtMs(spVerify)) ok=\(String(timings.spartanVerifyOk ?? false))")
            }
        }

        emit("\n\nRaw result:")
        if let root = parsed.root, let pretty = jsonString(root, pretty: true) {
            emit(pretty)
        } else {
            emit(outJson)
        }

        return NeoFoldNativeRunResult(
            rawResultJson: outJson,
            timings: timings,
            verifyOk: parsed.verifyOk,
            steps: parsed.steps
        )
    }

    private static func parseSummary(json data: Data) -> (root: [String: Any]?, timings: NeoFoldNativeTimings?, verifyOk: Bool?, steps: Int?) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (nil, nil, nil, nil)
        }

        let timings: NeoFoldNativeTimings? = {
            guard let timings = obj["timings_ms"] as? [String: Any] else { return nil }

            func number(_ key: String) -> Double? {
                if let n = timings[key] as? NSNumber { return n.doubleValue }
                return nil
            }

            let spartan: (prove: Double?, verify: Double?, ok: Bool?) = {
                guard let sp = obj["spartan"] as? [String: Any] else { return (nil, nil, nil) }
                let prove = (sp["prove_ms"] as? NSNumber)?.doubleValue
                let verify = (sp["verify_ms"] as? NSNumber)?.doubleValue
                let ok: Bool? = {
                    if let b = sp["verify_ok"] as? Bool { return b }
                    if let n = sp["verify_ok"] as? NSNumber { return n.boolValue }
                    return nil
                }()
                return (prove, verify, ok)
            }()

            let foldSteps: [Double]? = {
                guard let raw = timings["fold_steps"] as? [Any] else { return nil }
                let values = raw.compactMap { ($0 as? NSNumber)?.doubleValue }
                return values.isEmpty ? nil : values
            }()

            let spartanBytes: (snark: Int?, vk: Int?, packed: Int?) = {
                guard let sp = obj["spartan"] as? [String: Any] else { return (nil, nil, nil) }
                let snark = (sp["snark_bytes"] as? NSNumber)?.intValue
                let vk = (sp["vk_bytes"] as? NSNumber)?.intValue
                let packed = (sp["vk_and_snark_bytes"] as? NSNumber)?.intValue
                return (snark, vk, packed)
            }()

            return NeoFoldNativeTimings(
                ajtaiSetupMs: number("ajtai_setup"),
                buildCcsMs: number("build_ccs"),
                sessionInitMs: number("session_init"),
                sessionCreateMs: {
                    guard
                        let total = number("total"),
                        let add = number("add_steps_total"),
                        let prove = number("fold_and_prove"),
                        let verify = number("verify")
                    else { return nil }
                    return max(0.0, total - add - prove - verify)
                }(),
                addStepsMs: number("add_steps_total"),
                foldAndProveMs: number("fold_and_prove"),
                foldStepsMs: foldSteps,
                verifyMs: number("verify"),
                totalMs: number("total"),
                spartanProveMs: spartan.prove,
                spartanVerifyMs: spartan.verify,
                spartanVerifyOk: spartan.ok,
                spartanSnarkBytes: spartanBytes.snark,
                spartanVkBytes: spartanBytes.vk,
                spartanVkAndSnarkBytes: spartanBytes.packed
            )
        }()

        let verifyOk: Bool? = {
            if let b = obj["verify_ok"] as? Bool { return b }
            if let n = obj["verify_ok"] as? NSNumber { return n.boolValue }
            return nil
        }()

        let steps: Int? = {
            if let n = obj["steps"] as? NSNumber { return n.intValue }
            return nil
        }()

        return (obj, timings, verifyOk, steps)
    }

    private struct TestExportSummary {
        var constraints: String
        var variables: String
        var steps: String
    }

    private static func tryParseTestExport(json: String) -> TestExportSummary? {
        guard let data = json.data(using: .utf8) else { return nil }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }

        let constraints = (obj["num_constraints"] as? NSNumber)?.stringValue ?? "?"
        let variables = (obj["num_variables"] as? NSNumber)?.stringValue ?? "?"
        let steps: String
        if let witness = obj["witness"] as? [Any] {
            steps = String(witness.count)
        } else {
            steps = "?"
        }

        return TestExportSummary(constraints: constraints, variables: variables, steps: steps)
    }
}
#else
extension NeoFoldNativeService {
    func runProveVerify(
        json _: String,
        doSpartan _: Bool,
        log _: @escaping @MainActor (String) -> Void
    ) async throws -> NeoFoldNativeRunResult {
        throw NeoFoldNativeError.frameworkUnavailable(
            "NeoFoldFFI.xcframework is missing. Build it with `./scripts/build_native.sh` (or `./scripts/build.sh --native`)."
        )
    }
}
#endif

private func fmtMs(_ ms: Double) -> String {
    guard ms.isFinite else { return String(ms) }
    return String(format: "%.1f ms", ms)
}

private func fmtBytes(_ bytes: Double) -> String {
    guard bytes.isFinite else { return String(bytes) }
    if bytes < 1024 { return String(format: "%.0f B", bytes) }
    let kb = bytes / 1024
    if kb < 1024 { return String(format: "%.2f KB", kb) }
    let mb = kb / 1024
    return String(format: "%.2f MB", mb)
}

private func fmtMsList(_ values: [Double], maxItems: Int = 32) -> String {
    let shown = values.prefix(maxItems).map(fmtMs)
    let more = values.count > maxItems ? " … (+\(values.count - maxItems) more)" : ""
    return "[\(shown.joined(separator: ", "))]\(more)"
}

private func jsString(_ value: Any?) -> String {
    guard let value else { return "?" }
    if value is NSNull { return "?" }
    if let s = value as? String { return s }
    if let n = value as? NSNumber { return n.stringValue }
    return String(describing: value)
}

private func jsonString(_ value: Any, pretty: Bool) -> String? {
    if !JSONSerialization.isValidJSONObject(value) { return nil }
    let opts: JSONSerialization.WritingOptions = pretty ? [.prettyPrinted, .sortedKeys] : []
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: opts) else { return nil }
    return String(data: data, encoding: .utf8)
}
