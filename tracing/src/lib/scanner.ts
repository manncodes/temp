import { VllmDeployment } from "./types";

/**
 * Attempt to discover vLLM deployments by:
 * 1. Running kubectl to list services (if available)
 * 2. Probing provided manual endpoints
 * 3. Querying each endpoint's /v1/models to verify it's a vLLM OpenAI-compatible server
 */
export async function scanCluster(
  namespaces: string,
  manualEndpoints: string,
  defaultPort: number
): Promise<VllmDeployment[]> {
  const deployments: VllmDeployment[] = [];
  const seen = new Set<string>();

  // 1. Try kubectl discovery
  const k8sEndpoints = await discoverFromKubernetes(namespaces, defaultPort);
  for (const ep of k8sEndpoints) {
    if (!seen.has(ep.endpoint)) {
      seen.add(ep.endpoint);
      deployments.push(ep);
    }
  }

  // 2. Probe manual endpoints
  const manualLines = manualEndpoints
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  for (const line of manualLines) {
    const endpoint = normalizeEndpoint(line, defaultPort);
    if (seen.has(endpoint)) continue;
    seen.add(endpoint);

    const result = await probeEndpoint(endpoint, line);
    deployments.push(result);
  }

  return deployments;
}

async function discoverFromKubernetes(
  namespaces: string,
  defaultPort: number
): Promise<VllmDeployment[]> {
  const results: VllmDeployment[] = [];

  try {
    const nsList = namespaces
      ? namespaces.split(",").map((n) => n.trim())
      : ["default"];

    for (const ns of nsList) {
      // We use the Kubernetes API from within the cluster if available
      // This works when the app is deployed inside the cluster
      const k8sHost = process.env.KUBERNETES_SERVICE_HOST;
      if (!k8sHost) continue;

      const token = await readK8sToken();
      if (!token) continue;

      const svcUrl = `https://${k8sHost}:${process.env.KUBERNETES_SERVICE_PORT || 443}/api/v1/namespaces/${ns}/services`;

      const res = await fetch(svcUrl, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) continue;

      const data = await res.json();
      const services = data.items || [];

      for (const svc of services) {
        const name: string = svc.metadata?.name || "";
        const labels: Record<string, string> = svc.metadata?.labels || {};

        // Heuristic: look for services likely to be vLLM
        const isLikelyVllm =
          name.includes("vllm") ||
          name.includes("llm") ||
          name.includes("model") ||
          name.includes("inference") ||
          labels["app.kubernetes.io/name"]?.includes("vllm") ||
          labels["app"]?.includes("vllm");

        if (!isLikelyVllm) continue;

        const ports = svc.spec?.ports || [];
        const httpPort =
          ports.find(
            (p: { name?: string; port: number }) =>
              p.name === "http" || p.port === 8000 || p.port === defaultPort
          )?.port || defaultPort;

        const endpoint = `http://${name}.${ns}.svc.cluster.local:${httpPort}`;
        const result = await probeEndpoint(endpoint, name);
        results.push(result);
      }
    }
  } catch {
    // kubectl / k8s API not available — that's fine, user can add manual endpoints
  }

  return results;
}

async function readK8sToken(): Promise<string | null> {
  try {
    const fs = await import("fs/promises");
    return (
      await fs.readFile(
        "/var/run/secrets/kubernetes.io/serviceaccount/token",
        "utf-8"
      )
    ).trim();
  } catch {
    return null;
  }
}

async function probeEndpoint(
  endpoint: string,
  label: string
): Promise<VllmDeployment> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    const res = await fetch(`${endpoint}/v1/models`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      return {
        name: label,
        endpoint,
        model: "—",
        status: "offline",
      };
    }

    const data = await res.json();
    const models = data.data || [];
    const modelId = models[0]?.id || "unknown";

    return {
      name: label,
      endpoint,
      model: modelId,
      status: "online",
      metadata: {
        modelCount: String(models.length),
        allModels: models.map((m: { id: string }) => m.id).join(", "),
      },
    };
  } catch {
    return {
      name: label,
      endpoint,
      model: "—",
      status: "offline",
    };
  }
}

function normalizeEndpoint(raw: string, defaultPort: number): string {
  let ep = raw.trim();
  if (!ep.startsWith("http://") && !ep.startsWith("https://")) {
    ep = `http://${ep}`;
  }
  // Add port if missing
  try {
    const url = new URL(ep);
    if (!raw.includes(":") || url.port === "") {
      url.port = String(defaultPort);
    }
    // Remove trailing slash
    return url.toString().replace(/\/$/, "");
  } catch {
    return ep;
  }
}
