import Foundation
import Network

enum LocalhostStaticServerError: Error, LocalizedError {
    case invalidBaseDirectory(URL)
    case serverNotReady
    case invalidRequest
    case unsupportedMethod(String)
    case pathTraversalBlocked
    case fileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .invalidBaseDirectory(let url):
            return "Invalid web root directory: \(url.path)"
        case .serverNotReady:
            return "Local web server not ready."
        case .invalidRequest:
            return "Invalid HTTP request."
        case .unsupportedMethod(let method):
            return "Unsupported HTTP method: \(method)"
        case .pathTraversalBlocked:
            return "Blocked path traversal attempt."
        case .fileNotFound(let path):
            return "File not found: \(path)"
        }
    }
}

final class LocalhostStaticServer {
    private let webRoot: URL
    private let queue = DispatchQueue(label: "TestingWasm.localhostStaticServer", qos: .userInitiated)

    private var listener: NWListener?
    private var startContinuation: CheckedContinuation<URL, Error>?

    init(webRoot: URL) throws {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: webRoot.path, isDirectory: &isDir), isDir.boolValue else {
            throw LocalhostStaticServerError.invalidBaseDirectory(webRoot)
        }
        self.webRoot = webRoot
    }

    func start() async throws -> URL {
        if let listener, let port = listener.port {
            return Self.baseURL(port: port)
        }

        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                if let listener = self.listener, let port = listener.port {
                    continuation.resume(returning: Self.baseURL(port: port))
                    return
                }

                self.startContinuation = continuation

                do {
                    let params = NWParameters.tcp
                    let listener = try NWListener(using: params, on: .any)
                    self.listener = listener

                    listener.newConnectionHandler = { [weak self] connection in
                        self?.handleConnection(connection)
                    }

                    listener.stateUpdateHandler = { [weak self] state in
                        guard let self else { return }
                        switch state {
                        case .ready:
                            guard let port = listener.port else {
                                self.resumeStart(throwing: LocalhostStaticServerError.serverNotReady)
                                return
                            }
                            self.resumeStart(returning: Self.baseURL(port: port))
                        case .failed(let error):
                            self.resumeStart(throwing: error)
                        default:
                            break
                        }
                    }

                    listener.start(queue: self.queue)
                } catch {
                    self.resumeStart(throwing: error)
                }
            }
        }
    }

    func stop() {
        queue.async {
            self.listener?.cancel()
            self.listener = nil
            self.startContinuation = nil
        }
    }

    private func resumeStart(returning url: URL) {
        guard let cont = startContinuation else { return }
        startContinuation = nil
        cont.resume(returning: url)
    }

    private func resumeStart(throwing error: Error) {
        guard let cont = startContinuation else { return }
        startContinuation = nil
        cont.resume(throwing: error)
    }

    private static func baseURL(port: NWEndpoint.Port) -> URL {
        // Use localhost so WebKit treats it as a secure context (needed for SharedArrayBuffer).
        URL(string: "http://localhost:\(port.rawValue)")!
    }

    private func handleConnection(_ connection: NWConnection) {
        connection.stateUpdateHandler = { state in
            switch state {
            case .ready:
                self.receiveRequest(on: connection, buffer: Data())
            case .failed, .cancelled:
                connection.cancel()
            default:
                break
            }
        }
        connection.start(queue: queue)
    }

    private func receiveRequest(on connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, isComplete, error in
            if let error {
                self.sendErrorResponse(on: connection, status: 500, message: "Receive error: \(error)")
                return
            }

            var buffer = buffer
            if let data { buffer.append(data) }

            if let headerRange = buffer.range(of: Data([13, 10, 13, 10])) {
                let headerData = buffer.prefix(upTo: headerRange.lowerBound)
                self.handleRequestHeader(headerData, on: connection)
                return
            }

            if isComplete {
                self.sendErrorResponse(on: connection, status: 400, message: "Incomplete request")
                return
            }

            self.receiveRequest(on: connection, buffer: buffer)
        }
    }

    private struct HTTPRequest {
        let method: String
        let path: String
        let headers: [String: String]
    }

    private func handleRequestHeader(_ headerData: Data, on connection: NWConnection) {
        guard let headerString = String(data: headerData, encoding: .utf8) else {
            sendErrorResponse(on: connection, status: 400, message: "Invalid header encoding")
            return
        }

        do {
            let request = try parseRequest(headerString)
            let response = try buildResponse(for: request)
            connection.send(content: response, completion: .contentProcessed { _ in
                connection.cancel()
            })
        } catch {
            sendErrorResponse(on: connection, status: 500, message: error.localizedDescription)
        }
    }

    private func parseRequest(_ headerString: String) throws -> HTTPRequest {
        let lines = headerString.components(separatedBy: "\r\n").filter { !$0.isEmpty }
        guard let first = lines.first else { throw LocalhostStaticServerError.invalidRequest }

        let parts = first.split(separator: " ")
        guard parts.count >= 2 else { throw LocalhostStaticServerError.invalidRequest }

        let method = String(parts[0])
        let path = String(parts[1])

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let idx = line.firstIndex(of: ":") else { continue }
            let name = line[..<idx].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: idx)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }

        return HTTPRequest(method: method, path: path, headers: headers)
    }

    private func buildResponse(for request: HTTPRequest) throws -> Data {
        let method = request.method.uppercased()
        let wantsBody: Bool
        switch method {
        case "GET":
            wantsBody = true
        case "HEAD":
            wantsBody = false
        default:
            throw LocalhostStaticServerError.unsupportedMethod(method)
        }

        var path = request.path
        if let q = path.firstIndex(of: "?") { path = String(path[..<q]) }
        if let hash = path.firstIndex(of: "#") { path = String(path[..<hash]) }
        if path.isEmpty { path = "/" }
        if path == "/" { path = "/index.html" }

        let decoded = path.removingPercentEncoding ?? path
        let relative = decoded.hasPrefix("/") ? String(decoded.dropFirst()) : decoded
        if relative.contains("..") {
            throw LocalhostStaticServerError.pathTraversalBlocked
        }

        let baseResolved = webRoot.standardizedFileURL.resolvingSymlinksInPath()
        let basePath = baseResolved.path.hasSuffix("/") ? baseResolved.path : baseResolved.path + "/"

        var fileURL = baseResolved.appendingPathComponent(relative, isDirectory: false)
        fileURL = fileURL.standardizedFileURL.resolvingSymlinksInPath()

        if !fileURL.path.hasPrefix(basePath) {
            throw LocalhostStaticServerError.pathTraversalBlocked
        }

        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: fileURL.path, isDirectory: &isDir), isDir.boolValue {
            fileURL = fileURL.appendingPathComponent("index.html", isDirectory: false)
        }

        guard FileManager.default.fileExists(atPath: fileURL.path, isDirectory: &isDir), !isDir.boolValue else {
            throw LocalhostStaticServerError.fileNotFound(decoded)
        }

        let body = wantsBody ? (try Data(contentsOf: fileURL)) : Data()
        let contentType = contentTypeForPath(fileURL.path)
        let headers: [String: String] = [
            "Content-Type": contentType,
            "Content-Length": "\(wantsBody ? body.count : (try fileSize(fileURL)))",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Cache-Control": "no-store",
            "Connection": "close",
        ]

        var headerLines = "HTTP/1.1 200 OK\r\n"
        for (k, v) in headers {
            headerLines += "\(k): \(v)\r\n"
        }
        headerLines += "\r\n"

        var out = Data(headerLines.utf8)
        if wantsBody { out.append(body) }
        return out
    }

    private func sendErrorResponse(on connection: NWConnection, status: Int, message: String) {
        let body = Data("\(message)\n".utf8)
        let headers: [String: String] = [
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": "\(body.count)",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Cache-Control": "no-store",
            "Connection": "close",
        ]

        var headerLines = "HTTP/1.1 \(status) Error\r\n"
        for (k, v) in headers {
            headerLines += "\(k): \(v)\r\n"
        }
        headerLines += "\r\n"

        var out = Data(headerLines.utf8)
        out.append(body)

        connection.send(content: out, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private func fileSize(_ url: URL) throws -> Int {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        return values.fileSize ?? 0
    }

    private func contentTypeForPath(_ path: String) -> String {
        switch URL(fileURLWithPath: path).pathExtension.lowercased() {
        case "html":
            return "text/html; charset=utf-8"
        case "js", "mjs":
            return "text/javascript; charset=utf-8"
        case "css":
            return "text/css; charset=utf-8"
        case "json":
            return "application/json; charset=utf-8"
        case "wasm":
            return "application/wasm"
        case "svg":
            return "image/svg+xml"
        case "png":
            return "image/png"
        case "jpg", "jpeg":
            return "image/jpeg"
        default:
            return "application/octet-stream"
        }
    }
}
