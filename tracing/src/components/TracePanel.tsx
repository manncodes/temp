"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Fingerprint,
  Loader2,
  ChevronDown,
  ChevronRight,
  BookOpen,
  BarChart3,
  Database,
} from "lucide-react";
import { TraceResult, TraceSegment, TraceDocument } from "@/lib/types";
import { AVAILABLE_INDEXES } from "@/lib/infinigram";

interface Props {
  responseText: string;
}

export default function TracePanel({ responseText }: Props) {
  const [traceResult, setTraceResult] = useState<TraceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(AVAILABLE_INDEXES[0].id);
  const [expandedSegment, setExpandedSegment] = useState<number | null>(null);

  const trace = useCallback(async () => {
    if (!responseText.trim()) return;
    setLoading(true);
    setError("");
    setTraceResult(null);

    try {
      const res = await fetch("/api/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: responseText,
          index: selectedIndex,
          mode: "full",
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setTraceResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Trace failed");
    } finally {
      setLoading(false);
    }
  }, [responseText, selectedIndex]);

  useEffect(() => {
    if (responseText) {
      trace();
    }
  }, [responseText, trace]);

  function heatClass(segment: TraceSegment): string {
    if (segment.count <= 0) return "";
    if (segment.count < 5) return "heat-0";
    if (segment.count < 50) return "heat-1";
    if (segment.count < 500) return "heat-2";
    return "heat-3";
  }

  function heatLabel(segment: TraceSegment): string {
    if (segment.count <= 0) return "not found";
    if (segment.count < 5) return "rare";
    if (segment.count < 50) return "uncommon";
    if (segment.count < 500) return "common";
    return "very common";
  }

  const stats = traceResult
    ? {
        total: traceResult.segments.length,
        found: traceResult.segments.filter((s) => s.count > 0).length,
        withDocs: traceResult.segments.filter(
          (s) => s.documents.length > 0
        ).length,
        avgCount:
          traceResult.segments.reduce(
            (sum, s) => sum + Math.max(0, s.count),
            0
          ) / traceResult.segments.length,
      }
    : null;

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Fingerprint className="w-4 h-4 text-[var(--accent)]" />
          <h2 className="font-semibold text-sm">infini-gram Trace</h2>
        </div>
        <select
          value={selectedIndex}
          onChange={(e) => setSelectedIndex(e.target.value)}
          className="text-xs px-2 py-1 rounded bg-[var(--bg)] border border-[var(--border)] outline-none"
        >
          {AVAILABLE_INDEXES.map((idx) => (
            <option key={idx.id} value={idx.id}>
              {idx.label}
            </option>
          ))}
        </select>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 min-h-0">
        {!responseText && (
          <div className="text-center text-[var(--text-muted)] text-sm py-12">
            Send a message and receive a response to trace it against training
            corpora.
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center gap-3 py-12">
            <Loader2 className="w-6 h-6 text-[var(--accent)] animate-spin" />
            <span className="text-sm text-[var(--text-muted)]">
              Tracing response through infini-gram…
            </span>
          </div>
        )}

        {error && (
          <div className="text-sm text-[var(--red)] bg-[var(--red)]/10 rounded p-3">
            {error}
          </div>
        )}

        {traceResult && !loading && (
          <div className="space-y-4">
            {/* Stats bar */}
            {stats && (
              <div className="grid grid-cols-4 gap-2">
                <StatCard
                  icon={<BarChart3 className="w-3.5 h-3.5" />}
                  label="Segments"
                  value={String(stats.total)}
                />
                <StatCard
                  icon={<Database className="w-3.5 h-3.5" />}
                  label="Found"
                  value={`${stats.found}/${stats.total}`}
                />
                <StatCard
                  icon={<BookOpen className="w-3.5 h-3.5" />}
                  label="With Sources"
                  value={String(stats.withDocs)}
                />
                <StatCard
                  icon={<BarChart3 className="w-3.5 h-3.5" />}
                  label="Avg Count"
                  value={stats.avgCount.toFixed(0)}
                />
              </div>
            )}

            {/* Heatmap text */}
            <div>
              <h3 className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wider">
                Response Heatmap
              </h3>
              <div className="text-sm leading-relaxed bg-[var(--bg)] rounded-lg p-3 border border-[var(--border)]">
                {traceResult.segments.map((seg, i) => (
                  <span
                    key={i}
                    className={`cursor-pointer rounded-sm px-0.5 ${heatClass(seg)} ${
                      expandedSegment === i
                        ? "ring-2 ring-[var(--accent)] ring-offset-1 ring-offset-[var(--bg)]"
                        : ""
                    }`}
                    onClick={() =>
                      setExpandedSegment(expandedSegment === i ? null : i)
                    }
                    title={`count: ${seg.count} | ${heatLabel(seg)}`}
                  >
                    {seg.text}{" "}
                  </span>
                ))}
              </div>

              {/* Legend */}
              <div className="flex items-center gap-4 mt-2 text-xs text-[var(--text-muted)]">
                <span>Click a segment for details.</span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-2 rounded-sm heat-0 inline-block" />{" "}
                  rare
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-2 rounded-sm heat-1 inline-block" />{" "}
                  uncommon
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-2 rounded-sm heat-2 inline-block" />{" "}
                  common
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-2 rounded-sm heat-3 inline-block" />{" "}
                  very common
                </span>
              </div>
            </div>

            {/* Segment details */}
            <div>
              <h3 className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wider">
                Segment Details
              </h3>
              <div className="space-y-1">
                {traceResult.segments.map((seg, i) => (
                  <SegmentRow
                    key={i}
                    segment={seg}
                    index={i}
                    expanded={expandedSegment === i}
                    onToggle={() =>
                      setExpandedSegment(expandedSegment === i ? null : i)
                    }
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-[var(--bg)] rounded-lg border border-[var(--border)] p-2">
      <div className="flex items-center gap-1 text-[var(--text-muted)] mb-1">
        {icon}
        <span className="text-[10px] uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}

function SegmentRow({
  segment,
  index,
  expanded,
  onToggle,
}: {
  segment: TraceSegment;
  index: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded border border-[var(--border)] overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-[var(--bg-card-hover)] transition-colors"
      >
        {expanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
        )}
        <span className="text-xs text-[var(--text-muted)] w-6 text-right shrink-0">
          #{index + 1}
        </span>
        <span className="text-sm truncate flex-1">{segment.text}</span>
        <span className="text-xs text-[var(--text-muted)] shrink-0 tabular-nums">
          {segment.count >= 0 ? segment.count.toLocaleString() : "—"} hits
        </span>
      </button>

      {expanded && (
        <div className="px-3 py-3 border-t border-[var(--border)] bg-[var(--bg)] space-y-3">
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-[var(--text-muted)]">Count:</span>{" "}
              <span className="font-medium">
                {segment.count >= 0 ? segment.count.toLocaleString() : "N/A"}
              </span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Probability:</span>{" "}
              <span className="font-medium">
                {segment.prob >= 0 ? (segment.prob * 100).toFixed(2) + "%" : "N/A"}
              </span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Effective N:</span>{" "}
              <span className="font-medium">{segment.effectiveN || "—"}</span>
            </div>
          </div>

          {segment.documents.length > 0 ? (
            <div>
              <h4 className="text-xs text-[var(--text-muted)] mb-2 flex items-center gap-1">
                <BookOpen className="w-3 h-3" />
                Source Documents ({segment.documents.length})
              </h4>
              <div className="space-y-2">
                {segment.documents.map((doc, di) => (
                  <DocumentCard key={di} doc={doc} />
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-[var(--text-muted)] italic">
              No source documents found for this segment.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function DocumentCard({ doc }: { doc: TraceDocument }) {
  return (
    <div className="rounded bg-[var(--bg-card)] border border-[var(--border)] p-2">
      <div className="flex items-center gap-2 text-[10px] text-[var(--text-muted)] mb-1">
        <span>Doc #{doc.doc_ix}</span>
        <span>|</span>
        <span>{doc.doc_len?.toLocaleString()} tokens</span>
        {doc.metadata && (
          <>
            <span>|</span>
            <span className="truncate">{doc.metadata}</span>
          </>
        )}
      </div>
      <p className="text-xs leading-relaxed font-mono whitespace-pre-wrap break-words max-h-32 overflow-y-auto">
        {doc.passage}
      </p>
    </div>
  );
}
