"use client";

import { useState } from "react";
import { Layers } from "lucide-react";
import { VllmDeployment } from "@/lib/types";
import DeploymentScanner from "@/components/DeploymentScanner";
import ChatPanel from "@/components/ChatPanel";
import TracePanel from "@/components/TracePanel";

export default function Home() {
  const [selectedDeployment, setSelectedDeployment] =
    useState<VllmDeployment | null>(null);
  const [lastResponse, setLastResponse] = useState("");

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top bar */}
      <header className="px-6 py-3 border-b border-[var(--border)] flex items-center gap-3 shrink-0">
        <Layers className="w-5 h-5 text-[var(--accent)]" />
        <h1 className="font-bold text-lg tracking-tight">vLLM Tracing</h1>
        <span className="text-xs text-[var(--text-muted)] ml-1">
          Scan deployments &middot; Chat &middot; Trace with infini-gram
        </span>
      </header>

      {/* Main layout */}
      <div className="flex-1 flex min-h-0">
        {/* Left sidebar: scanner */}
        <aside className="w-80 shrink-0 border-r border-[var(--border)] overflow-y-auto p-4">
          <DeploymentScanner
            onSelect={setSelectedDeployment}
            selected={selectedDeployment}
          />
        </aside>

        {/* Center: chat */}
        <main className="flex-1 flex flex-col min-w-0 p-4">
          <ChatPanel
            deployment={selectedDeployment}
            onResponse={setLastResponse}
          />
        </main>

        {/* Right sidebar: trace */}
        <aside className="w-[480px] shrink-0 border-l border-[var(--border)] overflow-y-auto p-4">
          <TracePanel responseText={lastResponse} />
        </aside>
      </div>
    </div>
  );
}
