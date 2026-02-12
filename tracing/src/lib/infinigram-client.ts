/**
 * Client-side infini-gram tracing.
 * Calls our /api/trace Next.js route which proxies to api.infini-gram.io
 * server-side, avoiding CORS issues.
 */
import { TraceResult } from "./types";

export async function traceResponseClient(
  responseText: string,
  index?: string,
  onProgress?: (done: number, total: number) => void
): Promise<TraceResult> {
  onProgress?.(0, 1);

  const res = await fetch("/api/trace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: responseText,
      index: index || undefined,
      mode: "full",
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Trace failed: ${res.status} — ${body}`);
  }

  const result: TraceResult = await res.json();
  if ((result as unknown as { error?: string }).error) {
    throw new Error((result as unknown as { error: string }).error);
  }

  onProgress?.(1, 1);
  return result;
}

export { AVAILABLE_INDEXES } from "./infinigram";
