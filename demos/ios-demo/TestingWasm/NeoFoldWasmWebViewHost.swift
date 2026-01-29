import SwiftUI
import WebKit

struct NeoFoldWasmWebViewHost: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView {
        (try? NeoFoldWasmWebViewService.shared.makeOrGetWebView()) ?? WKWebView(frame: .zero)
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

