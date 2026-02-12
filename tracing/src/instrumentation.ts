/**
 * Next.js instrumentation hook — runs once when the server starts.
 *
 * If we're running inside a Kubernetes cluster, strip HTTP proxy env vars
 * so that fetch() calls to cluster-internal services (*.svc.cluster.local)
 * and the infini-gram API don't get routed through a corporate proxy.
 */
export function register() {
  if (
    typeof process !== "undefined" &&
    process.env.KUBERNETES_SERVICE_HOST &&
    process.env.KUBERNETES_SERVICE_HOST !== ""
  ) {
    // Remove proxy env vars that would intercept cluster-internal traffic
    const proxyVars = [
      "https_proxy",
      "http_proxy",
      "HTTPS_PROXY",
      "HTTP_PROXY",
    ];
    for (const key of proxyVars) {
      if (process.env[key]) {
        delete process.env[key];
      }
    }

    // Ensure cluster-internal DNS is excluded from any remaining proxy
    const noProxyAddition = ",.svc.cluster.local";
    if (process.env.NO_PROXY) {
      process.env.NO_PROXY += noProxyAddition;
    }
    if (process.env.no_proxy) {
      process.env.no_proxy += noProxyAddition;
    }

    console.log(
      "[vllm-tracing] Running inside Kubernetes — proxy env vars cleared"
    );
  }
}
