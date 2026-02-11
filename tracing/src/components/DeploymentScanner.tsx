"use client";

import { useState } from "react";
import { Search, Server, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { VllmDeployment, ScanConfig } from "@/lib/types";

interface Props {
  onSelect: (deployment: VllmDeployment) => void;
  selected: VllmDeployment | null;
}

export default function DeploymentScanner({ onSelect, selected }: Props) {
  const [config, setConfig] = useState<ScanConfig>({
    namespaces: "",
    manualEndpoints: "http://localhost:8000",
    defaultPort: 8000,
  });
  const [deployments, setDeployments] = useState<VllmDeployment[]>([]);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState("");

  async function scan() {
    setScanning(true);
    setError("");
    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setDeployments(data.deployments);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Scan failed");
    } finally {
      setScanning(false);
    }
  }

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--border)] flex items-center gap-2">
        <Search className="w-4 h-4 text-[var(--accent)]" />
        <h2 className="font-semibold text-sm">Cluster Scanner</h2>
      </div>

      <div className="p-4 space-y-3">
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">
            Kubernetes Namespaces (comma-separated)
          </label>
          <input
            type="text"
            value={config.namespaces}
            onChange={(e) =>
              setConfig({ ...config, namespaces: e.target.value })
            }
            placeholder="default, ml-serving, inference"
            className="w-full px-3 py-1.5 text-sm rounded bg-[var(--bg)] border border-[var(--border)] focus:border-[var(--accent)] outline-none"
          />
        </div>

        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">
            Manual Endpoints (one per line)
          </label>
          <textarea
            value={config.manualEndpoints}
            onChange={(e) =>
              setConfig({ ...config, manualEndpoints: e.target.value })
            }
            placeholder={"http://vllm-server:8000\nhttp://10.0.1.50:8000"}
            rows={3}
            className="w-full px-3 py-1.5 text-sm rounded bg-[var(--bg)] border border-[var(--border)] focus:border-[var(--accent)] outline-none resize-none font-mono"
          />
        </div>

        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="block text-xs text-[var(--text-muted)] mb-1">
              Default Port
            </label>
            <input
              type="number"
              value={config.defaultPort}
              onChange={(e) =>
                setConfig({ ...config, defaultPort: Number(e.target.value) })
              }
              className="w-full px-3 py-1.5 text-sm rounded bg-[var(--bg)] border border-[var(--border)] focus:border-[var(--accent)] outline-none"
            />
          </div>
          <button
            onClick={scan}
            disabled={scanning}
            className="px-4 py-1.5 text-sm font-medium rounded bg-[var(--accent)] hover:bg-[var(--accent-light)] disabled:opacity-50 transition-colors flex items-center gap-2"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${scanning ? "animate-spin" : ""}`}
            />
            {scanning ? "Scanning…" : "Scan"}
          </button>
        </div>

        {error && (
          <p className="text-xs text-[var(--red)]">{error}</p>
        )}
      </div>

      {deployments.length > 0 && (
        <div className="border-t border-[var(--border)]">
          <div className="px-4 py-2 text-xs text-[var(--text-muted)]">
            Found {deployments.length} deployment
            {deployments.length !== 1 ? "s" : ""}
          </div>
          <ul>
            {deployments.map((d) => (
              <li key={d.endpoint}>
                <button
                  onClick={() => onSelect(d)}
                  className={`w-full text-left px-4 py-2.5 flex items-center gap-3 hover:bg-[var(--bg-card-hover)] transition-colors ${
                    selected?.endpoint === d.endpoint
                      ? "bg-[var(--bg-card-hover)] border-l-2 border-[var(--accent)]"
                      : ""
                  }`}
                >
                  {d.status === "online" ? (
                    <Wifi className="w-4 h-4 text-[var(--green)] shrink-0" />
                  ) : (
                    <WifiOff className="w-4 h-4 text-[var(--red)] shrink-0" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <Server className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                      <span className="text-sm font-medium truncate">
                        {d.name}
                      </span>
                    </div>
                    <div className="text-xs text-[var(--text-muted)] truncate mt-0.5">
                      {d.model} — {d.endpoint}
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
