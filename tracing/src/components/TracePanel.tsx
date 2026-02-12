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
  Zap,
} from "lucide-react";
import { TraceResult, TraceSegment, TraceDocument } from "@/lib/types";
import { AVAILABLE_INDEXES, traceResponseClient } from "@/lib/infinigram-client";

interface Props {
  responseText: string;
  onTraceComplete: (result: TraceResult | null) => void;
}

export default function TracePanel({ responseText, onTraceComplete }: Props) {
  const [traceResult, setTraceResult] = useState<TraceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [error, setError] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(AVAILABLE_INDEXES[0].id);
  const [expandedSegment, setExpandedSegment] = useState<number | null>(null);

  const trace = useCallback(async () => {
    if (!responseText.trim()) return;
    setLoading(true);
    setError("");
    setTraceResult(null);
    onTraceComplete(null);
    setProgress({ done: 0, total: 0 });

    try {
      const result = await traceResponseClient(
        responseText,
        selectedIndex,
        (done, total) => setProgress({ done, total })
      );
      setTraceResult(result);
      onTraceComplete(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Trace failed");
    } finally {
      setLoading(false);
    }
  }, [responseText, selectedIndex, onTraceComplete]);

  useEffect(() => {
    if (responseText) {
      trace();
    }
  }, [responseText, trace]);

  /** Map normalized score → heat CSS class */
  function heatClass(segment: TraceSegment): string {
    const s = segment.normalizedScore;
    if (s <= 0) return "";
    if (s < 0.25) return "heat-0";
    if (s < 0.5) return "heat-1";
    if (s < 0.75) return "heat-2";
    return "heat-3";
  }

  const stats = traceResult
    ? (() => {
        const segs = traceResult.segments;
        const matched = segs.filter((s) => s.count > 0);
        const avgScore =
          segs.length > 0
            ? segs.reduce((sum, s) => sum + s.normalizedScore, 0) / segs.length
            : 0;
        const avgMatchRatio =
          matched.length > 0
            ? matched.reduce((sum, s) => sum + s.matchRatio, 0) /
              matched.length
            : 0;
        return {
          total: segs.length,
          found: matched.length,
          withDocs: segs.filter((s) => s.documents.length > 0).length,
          avgScore,
          avgMatchRatio,
        };
      })()
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
          className="text-xs px-2 py-1 rounded bg-[var(--bg)] border border-[var(--border)] outline-none max-w-[260px]"
        >
          <optgroup label="infini-gram mini — Common Crawl">
            {AVAILABLE_INDEXES.filter(
              (idx) => idx.engine === "mini" && idx.id.startsWith("v2_cc")
            ).map((idx) => (
              <option key={idx.id} value={idx.id}>
                {idx.label} ({idx.size})
              </option>
            ))}
          </optgroup>
          <optgroup label="infini-gram mini — Curated">
            {AVAILABLE_INDEXES.filter(
              (idx) =>
                idx.engine === "mini" && !idx.id.startsWith("v2_cc")
            ).map((idx) => (
              <option key={idx.id} value={idx.id}>
                {idx.label} ({idx.size})
              </option>
            ))}
          </optgroup>
          <optgroup label="infini-gram (original)">
            {AVAILABLE_INDEXES.filter(
              (idx) => idx.engine === "original"
            ).map((idx) => (
              <option key={idx.id} value={idx.id}>
                {idx.label} ({idx.size})
              </option>
            ))}
          </optgroup>
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
              Tracing segments… {progress.done}/{progress.total}
            </span>
            {progress.total > 0 && (
              <div className="w-48 h-1.5 bg-[var(--border)] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--accent)] rounded-full transition-all duration-300"
                  style={{
                    width: `${(progress.done / progress.total) * 100}%`,
                  }}
                />
              </div>
            )}
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
                  label="Matched"
                  value={`${stats.found}/${stats.total}`}
                />
                <StatCard
                  icon={<Zap className="w-3.5 h-3.5" />}
                  label="Avg Score"
                  value={`${(stats.avgScore * 100).toFixed(0)}%`}
                />
                <StatCard
                  icon={<BookOpen className="w-3.5 h-3.5" />}
                  label="Avg Match"
                  value={`${(stats.avgMatchRatio * 100).toFixed(0)}%`}
                />
              </div>
            )}

            {/* Legend */}
            <div className="flex items-center gap-4 text-xs text-[var(--text-muted)]">
              <span>Memorization:</span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-2 rounded-sm heat-0 inline-block" />
                low
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-2 rounded-sm heat-1 inline-block" />
                moderate
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-2 rounded-sm heat-2 inline-block" />
                high
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-2 rounded-sm heat-3 inline-block" />
                verbatim
              </span>
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
                    heatClass={heatClass(seg)}
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

/** Format a score as a colored bar + percentage */
function ScoreBar({ score, label }: { score: number; label: string }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 75
      ? "bg-red-500"
      : pct >= 50
        ? "bg-orange-400"
        : pct >= 25
          ? "bg-yellow-400"
          : pct > 0
            ? "bg-green-400"
            : "bg-gray-300";
  return (
    <div className="flex items-center gap-2">
      <span className="text-[var(--text-muted)] text-xs w-20 shrink-0">
        {label}
      </span>
      <div className="flex-1 h-1.5 bg-[var(--border)] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.max(pct, 2)}%` }}
        />
      </div>
      <span className="text-xs tabular-nums w-10 text-right font-medium">
        {pct}%
      </span>
    </div>
  );
}

function SegmentRow({
  segment,
  index,
  heatClass,
  expanded,
  onToggle,
}: {
  segment: TraceSegment;
  index: number;
  heatClass: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  const scorePct = Math.round(segment.normalizedScore * 100);
  return (
    <div className="rounded border border-[var(--border)] overflow-hidden">
      <button
        onClick={onToggle}
        className={`w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-[var(--bg-card-hover)] transition-colors ${heatClass}`}
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
          {scorePct > 0 ? `${scorePct}%` : "—"}
        </span>
      </button>

      {expanded && (
        <div className="px-3 py-3 border-t border-[var(--border)] bg-[var(--bg)] space-y-3">
          {/* Score bars */}
          <div className="space-y-1.5">
            <ScoreBar
              score={segment.normalizedScore}
              label="Score"
            />
            <ScoreBar
              score={segment.matchRatio}
              label="Match"
            />
          </div>

          {/* Details grid */}
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-[var(--text-muted)]">Count:</span>{" "}
              <span className="font-medium">
                {segment.count >= 0 ? segment.count.toLocaleString() : "N/A"}
              </span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Matched:</span>{" "}
              <span className="font-medium">
                {segment.matchedWords > 0
                  ? `${segment.matchedWords} words`
                  : "none"}
              </span>
            </div>
            <div>
              <span className="text-[var(--text-muted)]">Probability:</span>{" "}
              <span className="font-medium">
                {segment.prob >= 0
                  ? (segment.prob * 100).toFixed(2) + "%"
                  : "N/A"}
              </span>
            </div>
          </div>

          {/* Show the matched substring */}
          {segment.matchedText && (
            <div className="text-xs">
              <span className="text-[var(--text-muted)]">
                Verbatim match:
              </span>
              <p className="mt-1 font-mono bg-[var(--bg-card)] border border-[var(--border)] rounded px-2 py-1.5 text-[var(--accent)] break-words">
                &quot;{segment.matchedText}&quot;
              </p>
            </div>
          )}

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
        <span>{doc.doc_len?.toLocaleString()} chars</span>
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
