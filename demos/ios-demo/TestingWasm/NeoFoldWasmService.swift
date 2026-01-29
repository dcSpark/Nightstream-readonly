import Foundation
import JavaScriptCore
import os
import os.signpost

struct NeoFoldWasmTimings {
    let ajtaiSetupMs: Double
    let buildCcsMs: Double
    let sessionInitMs: Double
    let sessionCreateMs: Double
    let addStepsMs: Double
    let foldAndProveMs: Double
    let verifyMs: Double
    let totalMs: Double
    let steps: Int
    let verifyOk: Bool
    let spartanProveMs: Double?
    let spartanVerifyMs: Double?
    let spartanVerifyOk: Bool?
    let spartanSnarkBytes: Int?
    let spartanVkBytes: Int?
    let spartanVkAndSnarkBytes: Int?
}

struct NeoFoldRunResult {
    var timings: NeoFoldWasmTimings?
    var spartanSnark: Data?
    var spartanFilename: String?
}

final class NeoFoldWasmService {
    static let shared = NeoFoldWasmService()

    private static let subsystem = "com.midnight.TestingWasm"
    private static let logger = Logger(subsystem: subsystem, category: "neo-fold")
    private static let signpostLog = OSLog(subsystem: subsystem, category: "neo-fold")

    private let queue = DispatchQueue(label: "TestingWasm.neoFold.wasm", qos: .userInitiated)
    private var runtime: NeoFoldWasmRuntime?

    func ensureLoaded() async throws {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    try self.ensureLoadedSync()
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func runProveVerify(
        json: String,
        doSpartan: Bool,
        log: @escaping @MainActor (String) -> Void
    ) async throws -> NeoFoldRunResult {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    try self.ensureLoadedSync()
                    guard let runtime = self.runtime else {
                        throw WasmRuntimeError.jsException("WASM runtime not initialized")
                    }

                    let result = try self.runProveVerifySync(runtime: runtime, json: json, doSpartan: doSpartan, log: log)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func ensureLoadedSync() throws {
        if runtime != nil { return }

        guard let wasmURL = Self.resourceURL(name: "neo_fold_demo_bg", ext: "wasm", subdirectories: [
            "wasm",
            "Resources/wasm",
            "Resources",
        ]) else {
            throw WasmRuntimeError.resourceMissing("neo_fold_demo_bg.wasm")
        }

        guard let jsURL = Self.resourceURL(name: "neo_fold_demo", ext: "js", subdirectories: [
            "wasm",
            "Resources/wasm",
            "Resources",
        ]) else {
            throw WasmRuntimeError.resourceMissing("neo_fold_demo.js")
        }

        let wasmData = try Data(contentsOf: wasmURL)
        let glueScript = try String(contentsOf: jsURL, encoding: .utf8)
        runtime = try NeoFoldWasmRuntime(wasmData: wasmData, glueScript: glueScript)
    }

    private func runProveVerifySync(
        runtime: NeoFoldWasmRuntime,
        json: String,
        doSpartan: Bool,
        log: @escaping @MainActor (String) -> Void
    ) throws -> NeoFoldRunResult {
        func emit(_ line: String) {
            Task { @MainActor in
                log(line)
            }
        }

        func msSince(_ start: CFAbsoluteTime) -> Double {
            (CFAbsoluteTimeGetCurrent() - start) * 1000
        }

        let inputBytes = json.utf8.count
        Self.logger.info("Run start (bytes=\(inputBytes, privacy: .public), spartan=\(doSpartan, privacy: .public))")
        let runSignpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(
            .begin,
            log: Self.signpostLog,
            name: "NeoFold Run",
            signpostID: runSignpostID,
            "bytes=%{public}d spartan=%{public}d",
            inputBytes,
            doSpartan ? 1 : 0
        )

        emit("Running prove+verify…")
        emit("Input JSON size: \(fmtBytes(Double(json.utf8.count)))")

        if let parsed = tryParseTestExport(json: json) {
            emit("Input export: constraints=\(parsed.constraints) variables=\(parsed.variables) steps=\(parsed.steps)")
        }

        let totalStart = CFAbsoluteTimeGetCurrent()

        let createStart = CFAbsoluteTimeGetCurrent()
        let session = try runtime.makeSession(circuitJson: json)
        let createMs = msSince(createStart)

        defer {
            _ = try? runtime.call(session, "free")
        }

        let setup = try runtime.call(session, "setup_timings_ms")
        let params = try runtime.call(session, "params_summary")
        let circuit = try runtime.call(session, "circuit_summary")

        let ajtaiSetupMs = jsDouble(setup.forProperty("ajtai_setup"))
        let buildCcsMs = jsDouble(setup.forProperty("build_ccs"))
        let sessionInitMs = jsDouble(setup.forProperty("session_init"))

        emit("Session ready (\(fmtMs(createMs)))")
        Self.logger.info("Session created (ms=\(String(format: "%.1f", createMs), privacy: .public))")

        if !params.isUndefined {
            let parts = [
                "b=\(jsString(params.forProperty("b")))",
                "d=\(jsString(params.forProperty("d")))",
                "kappa=\(jsString(params.forProperty("kappa")))",
                "k_rho=\(jsString(params.forProperty("k_rho")))",
                "T=\(jsString(params.forProperty("T")))",
                "s=\(jsString(params.forProperty("s")))",
                "lambda=\(jsString(params.forProperty("lambda")))",
            ]
            emit("Params: \(parts.joined(separator: " "))")
        }

        if !circuit.isUndefined {
            emit(
                "Circuit (R1CS): constraints=\(jsString(circuit.forProperty("r1cs_constraints"))) variables=\(jsString(circuit.forProperty("r1cs_variables"))) padded_n=\(jsString(circuit.forProperty("r1cs_padded_n"))) A_nnz=\(jsString(circuit.forProperty("r1cs_a_nnz"))) B_nnz=\(jsString(circuit.forProperty("r1cs_b_nnz"))) C_nnz=\(jsString(circuit.forProperty("r1cs_c_nnz")))"
            )

            let nonzeroRatio = circuit.forProperty("witness_nonzero_ratio")?.toDouble()
            let nonzeroPercent: String
            if let nonzeroRatio {
                nonzeroPercent = String(format: "%.2f%%", nonzeroRatio * 100.0)
            } else {
                nonzeroPercent = "?"
            }

            emit(
                "Witness: steps=\(jsString(circuit.forProperty("witness_steps"))) fields_total=\(jsString(circuit.forProperty("witness_fields_total"))) fields_min=\(jsString(circuit.forProperty("witness_fields_min"))) fields_max=\(jsString(circuit.forProperty("witness_fields_max"))) nonzero=\(jsString(circuit.forProperty("witness_nonzero_fields_total"))) (\(nonzeroPercent))"
            )

            emit(
                "Circuit (CCS): n=\(jsString(circuit.forProperty("ccs_n"))) m=\(jsString(circuit.forProperty("ccs_m"))) t=\(jsString(circuit.forProperty("ccs_t"))) max_degree=\(jsString(circuit.forProperty("ccs_max_degree"))) poly_terms=\(jsString(circuit.forProperty("ccs_poly_terms"))) nnz_total=\(jsString(circuit.forProperty("ccs_matrix_nnz_total")))"
            )

            if let ccsMatrixNnz = circuit.forProperty("ccs_matrix_nnz"), !ccsMatrixNnz.isUndefined {
                if let json = try? runtime.safeStringify(ccsMatrixNnz), json != "[]" {
                    emit("CCS matrices nnz: \(json)")
                }
            }
        }

        if !setup.isUndefined {
            emit(
                "Timings: ajtai=\(fmtMs(ajtaiSetupMs)) build_ccs=\(fmtMs(buildCcsMs)) session_init=\(fmtMs(sessionInitMs))"
            )
        }

        emit("Adding witness steps…")
        let addStart = CFAbsoluteTimeGetCurrent()
        let addSignpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(.begin, log: Self.signpostLog, name: "Add Witness Steps", signpostID: addSignpostID)
        _ = try runtime.call(session, "add_steps_from_test_export_json", with: [json])
        let addMs = msSince(addStart)
        os_signpost(
            .end,
            log: Self.signpostLog,
            name: "Add Witness Steps",
            signpostID: addSignpostID,
            "ms=%{public}.1f",
            addMs
        )
        emit("Timings: add_steps_total=\(fmtMs(addMs))")
        Self.logger.info("Add steps (ms=\(String(format: "%.1f", addMs), privacy: .public))")

        emit("Folding + proving…")
        let proveStart = CFAbsoluteTimeGetCurrent()
        let proveSignpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(.begin, log: Self.signpostLog, name: "Fold + Prove", signpostID: proveSignpostID)
        let foldProof = try runtime.call(session, "fold_and_prove")
        let proveMs = msSince(proveStart)
        os_signpost(
            .end,
            log: Self.signpostLog,
            name: "Fold + Prove",
            signpostID: proveSignpostID,
            "ms=%{public}.1f",
            proveMs
        )

        defer {
            _ = try? runtime.call(foldProof, "free")
        }

        let foldStepsValue = try runtime.call(foldProof, "fold_step_ms")
        emit("Timings: prove=\(fmtMs(proveMs))")
        Self.logger.info("Fold+Prove (ms=\(String(format: "%.1f", proveMs), privacy: .public))")

        if let foldSteps = foldStepsValue.toArray()?.compactMap({ ($0 as? NSNumber)?.doubleValue }), !foldSteps.isEmpty {
            emit("Folding prove per-step: \(fmtMsList(foldSteps))")
            let sum = foldSteps.reduce(0, +)
            let avg = sum / Double(foldSteps.count)
            let min = foldSteps.min() ?? 0
            let max = foldSteps.max() ?? 0
            emit("Folding prove per-step stats: avg=\(fmtMs(avg)) min=\(fmtMs(min)) max=\(fmtMs(max))")
        }

        emit("Verifying folding proof…")
        let verifyStart = CFAbsoluteTimeGetCurrent()
        let verifySignpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(.begin, log: Self.signpostLog, name: "Verify", signpostID: verifySignpostID)
        let verifyOkValue = try runtime.call(session, "verify", with: [foldProof])
        let verifyOk = verifyOkValue.toBool()
        let verifyMs = msSince(verifyStart)
        os_signpost(
            .end,
            log: Self.signpostLog,
            name: "Verify",
            signpostID: verifySignpostID,
            "ms=%{public}.1f ok=%{public}d",
            verifyMs,
            verifyOk ? 1 : 0
        )
        emit("Timings: verify=\(fmtMs(verifyMs))")
        Self.logger.info("Verify (ms=\(String(format: "%.1f", verifyMs), privacy: .public), ok=\(verifyOk, privacy: .public))")

        let steps = (try? runtime.call(foldProof, "step_count").toInt32()).map(Int.init) ?? 0
        let totalMs = msSince(totalStart)
        emit("OK: verify_ok=\(verifyOk) steps=\(steps) (total \(fmtMs(totalMs)))")
        os_signpost(
            .end,
            log: Self.signpostLog,
            name: "NeoFold Run",
            signpostID: runSignpostID,
            "total_ms=%{public}.1f steps=%{public}d ok=%{public}d",
            totalMs,
            steps,
            verifyOk ? 1 : 0
        )
        Self.logger.info(
            "Run done (steps=\(steps, privacy: .public), ok=\(verifyOk, privacy: .public), total_ms=\(String(format: "%.1f", totalMs), privacy: .public))"
        )

        let proofEstimate = try runtime.call(foldProof, "proof_estimate")
        if !proofEstimate.isUndefined {
            emit(
                "Proof estimate: proof_steps=\(jsString(proofEstimate.forProperty("proof_steps"))) final_acc_len=\(jsString(proofEstimate.forProperty("final_accumulator_len")))"
            )
            emit(
                "Proof estimate: commitments fold_lane=\(jsString(proofEstimate.forProperty("fold_lane_commitments"))) mem_cpu_val=\(jsString(proofEstimate.forProperty("mem_cpu_val_claim_commitments"))) val_lane=\(jsString(proofEstimate.forProperty("val_lane_commitments"))) total=\(jsString(proofEstimate.forProperty("total_commitments")))"
            )
            let estimatedBytes = jsDouble(proofEstimate.forProperty("estimated_commitment_bytes"))
            emit(
                "Proof estimate: commitment_bytes=\(jsString(proofEstimate.forProperty("commitment_bytes"))) (d=\(jsString(proofEstimate.forProperty("commitment_d"))) kappa=\(jsString(proofEstimate.forProperty("commitment_kappa")))) estimated_commitment_bytes=\(fmtBytes(estimatedBytes))"
            )
        }

        let folding = try runtime.call(foldProof, "folding_summary")
        if !folding.isUndefined {
            let kInStr: String
            if let kIn = folding.forProperty("k_in") {
                kInStr = (try? runtime.safeStringify(kIn)) ?? ""
            } else {
                kInStr = ""
            }
            if !kInStr.isEmpty { emit("Folding k_in per step: \(kInStr)") }

            let accLenAfterStr: String
            if let accLenAfter = folding.forProperty("acc_len_after") {
                accLenAfterStr = (try? runtime.safeStringify(accLenAfter)) ?? ""
            } else {
                accLenAfterStr = ""
            }
            if !accLenAfterStr.isEmpty { emit("Folding accumulator len after step: \(accLenAfterStr)") }
        }

        var runResult = NeoFoldRunResult()
        var spartanProveMsValue: Double?
        var spartanVerifyMsValue: Double?
        var spartanVerifyOkValue: Bool?
        var spartanSnarkBytesValue: Int?
        var spartanVkBytesValue: Int?
        var spartanVkAndSnarkBytesValue: Int?

        if doSpartan {
            emit("Compressing with Spartan2…")
            let spStart = CFAbsoluteTimeGetCurrent()
            let spartanProveSignpostID = OSSignpostID(log: Self.signpostLog)
            os_signpost(.begin, log: Self.signpostLog, name: "Spartan Prove", signpostID: spartanProveSignpostID)
            let spartan = try runtime.call(session, "spartan_prove", with: [foldProof])
            let spartanProveMs = msSince(spStart)
            os_signpost(
                .end,
                log: Self.signpostLog,
                name: "Spartan Prove",
                signpostID: spartanProveSignpostID,
                "ms=%{public}.1f",
                spartanProveMs
            )

            defer {
                _ = try? runtime.call(spartan, "free")
            }

            let snarkBytesValue = try runtime.call(spartan, "bytes")
            runtime.setGlobal("__swift_tmp_u8", value: snarkBytesValue)
            let bytesArrayValue = try runtime.evaluate("Array.from(__swift_tmp_u8)")
            let bytesArray = bytesArrayValue.toArray() ?? []
            let snarkBytes = bytesArray.compactMap { (value: Any) -> UInt8? in
                guard let n = value as? NSNumber else { return nil }
                return UInt8(truncating: n)
            }
            runResult.spartanSnark = Data(snarkBytes)
            runResult.spartanFilename = "neo_fold_spartan_snark_\(Int(Date().timeIntervalSince1970)).bin"

            let vkLen = jsDouble(try? runtime.call(spartan, "vk_bytes_len"))
            let packedLen = jsDouble(try? runtime.call(spartan, "vk_and_snark_bytes_len"))

            let sizeParts: [String] = [
                "snark=\(fmtBytes(Double(snarkBytes.count)))",
                vkLen > 0 ? "vk=\(fmtBytes(vkLen))" : nil,
                packedLen > 0 ? "total(vk+snark)=\(fmtBytes(packedLen))" : nil,
            ].compactMap { $0 }

            emit("Spartan2: prove=\(fmtMs(spartanProveMs)) \(sizeParts.joined(separator: " "))")
            Self.logger.info("Spartan prove (ms=\(String(format: "%.1f", spartanProveMs), privacy: .public))")

            let spVerifyStart = CFAbsoluteTimeGetCurrent()
            let spartanVerifySignpostID = OSSignpostID(log: Self.signpostLog)
            os_signpost(.begin, log: Self.signpostLog, name: "Spartan Verify", signpostID: spartanVerifySignpostID)
            let spartanVerifyOk = (try runtime.call(session, "spartan_verify", with: [spartan])).toBool()
            let spartanVerifyMs = msSince(spVerifyStart)
            os_signpost(
                .end,
                log: Self.signpostLog,
                name: "Spartan Verify",
                signpostID: spartanVerifySignpostID,
                "ms=%{public}.1f ok=%{public}d",
                spartanVerifyMs,
                spartanVerifyOk ? 1 : 0
            )
            emit("Spartan2: verify=\(fmtMs(spartanVerifyMs)) ok=\(String(spartanVerifyOk))")
            Self.logger.info(
                "Spartan verify (ms=\(String(format: "%.1f", spartanVerifyMs), privacy: .public), ok=\(spartanVerifyOk, privacy: .public))"
            )

            spartanProveMsValue = spartanProveMs
            spartanVerifyMsValue = spartanVerifyMs
            spartanVerifyOkValue = spartanVerifyOk
            spartanSnarkBytesValue = snarkBytes.count
            spartanVkBytesValue = vkLen > 0 ? Int(vkLen) : nil
            spartanVkAndSnarkBytesValue = packedLen > 0 ? Int(packedLen) : nil

            let meta: [String: Any] = [
                "prove_ms": spartanProveMs,
                "verify_ms": spartanVerifyMs,
                "verify_ok": spartanVerifyOk,
                "snark_bytes": snarkBytes.count,
                "vk_bytes": vkLen > 0 ? Int(vkLen) : NSNull(),
                "vk_and_snark_bytes": packedLen > 0 ? Int(packedLen) : NSNull(),
            ]
            runtime.setGlobal("__swift_raw_spartan", value: meta)
        } else {
            runtime.setGlobal("__swift_raw_spartan", value: NSNull())
        }

        runResult.timings = NeoFoldWasmTimings(
            ajtaiSetupMs: ajtaiSetupMs,
            buildCcsMs: buildCcsMs,
            sessionInitMs: sessionInitMs,
            sessionCreateMs: createMs,
            addStepsMs: addMs,
            foldAndProveMs: proveMs,
            verifyMs: verifyMs,
            totalMs: totalMs,
            steps: steps,
            verifyOk: verifyOk,
            spartanProveMs: spartanProveMsValue,
            spartanVerifyMs: spartanVerifyMsValue,
            spartanVerifyOk: spartanVerifyOkValue,
            spartanSnarkBytes: spartanSnarkBytesValue,
            spartanVkBytes: spartanVkBytesValue,
            spartanVkAndSnarkBytes: spartanVkAndSnarkBytesValue
        )

        let timings: [String: Any] = [
            "ajtai_setup": jsDouble(setup.forProperty("ajtai_setup")),
            "build_ccs": jsDouble(setup.forProperty("build_ccs")),
            "session_init": jsDouble(setup.forProperty("session_init")),
            "add_steps_total": addMs,
            "fold_and_prove": proveMs,
            "fold_steps": foldStepsValue.toArray()?.compactMap { ($0 as? NSNumber)?.doubleValue } ?? [],
            "verify": verifyMs,
            "total": totalMs,
        ]

        runtime.setGlobal("__swift_raw_steps", value: steps)
        runtime.setGlobal("__swift_raw_verify_ok", value: verifyOk)
        runtime.setGlobal("__swift_raw_circuit", value: circuit)
        runtime.setGlobal("__swift_raw_params", value: params)
        runtime.setGlobal("__swift_raw_timings", value: timings)
        runtime.setGlobal("__swift_raw_proof_estimate", value: proofEstimate)
        runtime.setGlobal("__swift_raw_folding", value: folding)

        let rawValue = try runtime.evaluate("""
        ({
          steps: __swift_raw_steps,
          verify_ok: __swift_raw_verify_ok,
          circuit: __swift_raw_circuit,
          params: __swift_raw_params,
          timings_ms: __swift_raw_timings,
          proof_estimate: __swift_raw_proof_estimate,
          folding: __swift_raw_folding,
          spartan: __swift_raw_spartan,
        })
        """)

        emit("\n\nRaw result:")
        emit(try runtime.safeStringify(rawValue))

        return runResult
    }

    private static func resourceURL(name: String, ext: String, subdirectories: [String]) -> URL? {
        if let url = Bundle.main.url(forResource: name, withExtension: ext) { return url }
        for subdir in subdirectories {
            if let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdir) {
                return url
            }
        }
        return nil
    }

    private struct TestExportSummary {
        var constraints: String
        var variables: String
        var steps: String
    }

    private func tryParseTestExport(json: String) -> TestExportSummary? {
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

private func jsString(_ value: JSValue?) -> String {
    guard let value, !value.isUndefined, !value.isNull else { return "?" }
    return value.toString()
}

private func jsDouble(_ value: JSValue?) -> Double {
    guard let value, !value.isUndefined, !value.isNull else { return 0 }
    return value.toDouble()
}

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
