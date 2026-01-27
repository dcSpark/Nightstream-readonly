const COI_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

function withCoiHeaders(response) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(COI_HEADERS)) headers.set(key, value);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export default {
  async fetch(request, env) {
    let response = await env.ASSETS.fetch(request);

    if (response.status === 404) {
      const url = new URL(request.url);
      const isAssetPath = url.pathname.includes(".");
      if (!isAssetPath) {
        response = await env.ASSETS.fetch(new Request(new URL("/index.html", url), request));
      }
    }

    return withCoiHeaders(response);
  },
};
