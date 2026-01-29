import Foundation
import JavaScriptCore

enum WasmRuntimeError: Error, LocalizedError {
    case resourceMissing(String)
    case webAssemblyUnavailable
    case jsException(String)
    case missingGlobal(String)
    case missingExport(String)

    var errorDescription: String? {
        switch self {
        case .resourceMissing(let name):
            return "Missing resource: \(name)"
        case .webAssemblyUnavailable:
            return "WebAssembly is not available in JavaScriptCore on this device."
        case .jsException(let message):
            return "JavaScript error: \(message)"
        case .missingGlobal(let name):
            return "Missing JavaScript global: \(name)"
        case .missingExport(let name):
            return "Missing WASM export: \(name)"
        }
    }
}

final class NeoFoldWasmRuntime {
    private let context: JSContext
    private let wasmBindgen: JSValue
    private let safeStringifyFn: JSValue

    init(wasmData: Data, glueScript: String) throws {
        guard let context = JSContext() else {
            throw WasmRuntimeError.jsException("Unable to create JSContext")
        }

        var lastException: JSValue?
        context.exceptionHandler = { _, exception in
            lastException = exception
        }

        let supportsWasm = context.evaluateScript("typeof WebAssembly !== 'undefined'")
        if supportsWasm?.toBool() != true {
            throw WasmRuntimeError.webAssemblyUnavailable
        }

        context.evaluateScript(Self.polyfillsAndHelpers)
        if let exception = lastException ?? context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }

        context.setObject([UInt8](wasmData), forKeyedSubscript: "wasmBytes" as NSString)
        context.evaluateScript(glueScript)
        if let exception = lastException ?? context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }

        // wasm-pack `--target no-modules` emits `let wasm_bindgen = ...`, which is a global lexical
        // binding (not a property on the global object). Bridge it to `globalThis.wasm_bindgen` so
        // Swift can reliably access it via `objectForKeyedSubscript`.
        context.evaluateScript("""
        (function () {
          if (typeof globalThis === 'undefined') return;
          if (typeof wasm_bindgen !== 'undefined') {
            globalThis.wasm_bindgen = wasm_bindgen;
          }
        })();
        """)
        if let exception = lastException ?? context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }

        let wasmBindgen: JSValue? = {
            if let v = context.objectForKeyedSubscript("wasm_bindgen"), !v.isUndefined, !v.isNull {
                return v
            }
            let v = context.evaluateScript("typeof wasm_bindgen !== 'undefined' ? wasm_bindgen : undefined")
            if let v, !v.isUndefined, !v.isNull { return v }
            return nil
        }()

        guard let wasmBindgen else {
            throw WasmRuntimeError.missingGlobal("wasm_bindgen")
        }

        self.context = context
        self.wasmBindgen = wasmBindgen

        guard
            let initSync = wasmBindgen.forProperty("initSync"),
            !initSync.isUndefined,
            !initSync.isNull
        else {
            throw WasmRuntimeError.missingExport("initSync")
        }

        context.exception = nil
        let wasmBytesValue = context.evaluateScript("new Uint8Array(wasmBytes)")
        _ = initSync.call(withArguments: [wasmBytesValue as Any])
        if let exception = lastException ?? context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }

        if let initPanicHook = wasmBindgen.forProperty("init_panic_hook"), !initPanicHook.isUndefined {
            context.exception = nil
            _ = initPanicHook.call(withArguments: [])
            if let exception = lastException ?? context.exception {
                throw WasmRuntimeError.jsException(exception.toString())
            }
        }

        guard
            let safeStringifyFn = context.objectForKeyedSubscript("__swift_safeStringify"),
            !safeStringifyFn.isUndefined,
            !safeStringifyFn.isNull
        else {
            throw WasmRuntimeError.missingGlobal("__swift_safeStringify")
        }
        self.safeStringifyFn = safeStringifyFn
    }

    func makeSession(circuitJson: String) throws -> JSValue {
        guard
            let ctor = wasmBindgen.forProperty("NeoFoldSession"),
            !ctor.isUndefined,
            !ctor.isNull
        else {
            throw WasmRuntimeError.missingExport("NeoFoldSession")
        }

        context.exception = nil
        let session = ctor.construct(withArguments: [circuitJson])
        if let exception = context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }

        guard let session, !session.isUndefined, !session.isNull else {
            throw WasmRuntimeError.jsException("Failed to construct NeoFoldSession")
        }

        return session
    }

    func call(_ object: JSValue, _ method: String, with arguments: [Any] = []) throws -> JSValue {
        context.exception = nil
        let result = object.invokeMethod(method, withArguments: arguments)
        if let exception = context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }
        guard let result else {
            throw WasmRuntimeError.jsException("JavaScript returned nil for \(method)")
        }
        return result
    }

    func safeStringify(_ value: JSValue) throws -> String {
        context.exception = nil
        let str = safeStringifyFn.call(withArguments: [value])
        if let exception = context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }
        return str?.toString() ?? ""
    }

    func evaluate(_ script: String) throws -> JSValue {
        context.exception = nil
        let value = context.evaluateScript(script)
        if let exception = context.exception {
            throw WasmRuntimeError.jsException(exception.toString())
        }
        guard let value else {
            throw WasmRuntimeError.jsException("JavaScript returned nil for script evaluation")
        }
        return value
    }

    func setGlobal(_ name: String, value: Any) {
        context.setObject(value, forKeyedSubscript: name as NSString)
    }

    private static let polyfillsAndHelpers = """
    (function () {
      if (typeof globalThis === 'undefined') {
        // eslint-disable-next-line no-undef
        this.globalThis = this;
      }

      if (typeof console === 'undefined') {
        globalThis.console = { log: function () {}, warn: function () {}, error: function () {} };
      }

      function __swift_utf8Encode(str) {
        str = String(str);
        var len = str.length;
        var out = new Uint8Array(len * 4);
        var p = 0;
        for (var i = 0; i < len; i++) {
          var cp = str.codePointAt(i);
          if (cp > 0xffff) i++;

          if (cp <= 0x7f) {
            out[p++] = cp;
          } else if (cp <= 0x7ff) {
            out[p++] = 0xc0 | (cp >> 6);
            out[p++] = 0x80 | (cp & 0x3f);
          } else if (cp <= 0xffff) {
            out[p++] = 0xe0 | (cp >> 12);
            out[p++] = 0x80 | ((cp >> 6) & 0x3f);
            out[p++] = 0x80 | (cp & 0x3f);
          } else {
            out[p++] = 0xf0 | (cp >> 18);
            out[p++] = 0x80 | ((cp >> 12) & 0x3f);
            out[p++] = 0x80 | ((cp >> 6) & 0x3f);
            out[p++] = 0x80 | (cp & 0x3f);
          }
        }
        return out.subarray(0, p);
      }

      function __swift_utf8Decode(bytes) {
        if (bytes == null) return '';
        if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
        if (ArrayBuffer.isView(bytes)) bytes = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        var out = '';
        var i = 0;
        while (i < bytes.length) {
          var b0 = bytes[i++] & 0xff;
          if (b0 < 0x80) {
            out += String.fromCharCode(b0);
            continue;
          }

          var cp = 0;
          if ((b0 & 0xe0) === 0xc0) {
            var b1 = bytes[i++] & 0x3f;
            cp = ((b0 & 0x1f) << 6) | b1;
          } else if ((b0 & 0xf0) === 0xe0) {
            var b1 = bytes[i++] & 0x3f;
            var b2 = bytes[i++] & 0x3f;
            cp = ((b0 & 0x0f) << 12) | (b1 << 6) | b2;
          } else {
            var b1 = bytes[i++] & 0x3f;
            var b2 = bytes[i++] & 0x3f;
            var b3 = bytes[i++] & 0x3f;
            cp = ((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3;
          }

          if (cp <= 0xffff) {
            out += String.fromCharCode(cp);
          } else {
            cp -= 0x10000;
            out += String.fromCharCode(0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
          }
        }
        return out;
      }

      if (typeof TextEncoder === 'undefined') {
        globalThis.TextEncoder = function TextEncoder() {};
        TextEncoder.prototype.encode = function (arg) { return __swift_utf8Encode(arg); };
        TextEncoder.prototype.encodeInto = function (arg, view) {
          var buf = __swift_utf8Encode(arg);
          view.set(buf);
          return { read: String(arg).length, written: buf.length };
        };
      }

      if (typeof TextDecoder === 'undefined') {
        globalThis.TextDecoder = function TextDecoder(_label, _opts) {};
        TextDecoder.prototype.decode = function (input) { return __swift_utf8Decode(input); };
      }

      globalThis.__swift_safeStringify = function (value) {
        return JSON.stringify(value, function (_k, v) { return typeof v === 'bigint' ? v.toString() : v; }, 2);
      };
    })();
    """
}
