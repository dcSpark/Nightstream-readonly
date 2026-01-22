#!/usr/bin/env python3

import argparse
import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # Required for SharedArrayBuffer / wasm threads.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Keep everything same-origin by default.
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # Reduce cache-related confusion when rebuilding wasm.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    mimetypes.add_type("application/wasm", ".wasm")

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--dir", default=os.getcwd())
    args = parser.parse_args()

    os.chdir(args.dir)
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"Serving http://{args.bind}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

