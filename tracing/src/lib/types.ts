export interface VllmDeployment {
  name: string;
  endpoint: string;
  model: string;
  status: "online" | "offline" | "unknown";
  metadata?: Record<string, string>;
}

export interface ScanConfig {
  /** Kubernetes namespace(s) to scan, comma-separated. Empty = all namespaces */
  namespaces: string;
  /** Additional manual endpoints to probe, one per line */
  manualEndpoints: string;
  /** Port to probe on discovered services */
  defaultPort: number;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface TraceSegment {
  text: string;
  count: number;
  prob: number;
  /** effective n-gram length used by infini-gram */
  effectiveN: number;
  documents: TraceDocument[];
}

export interface TraceDocument {
  doc_ix: number;
  doc_len: number;
  disp_len: number;
  passage: string;
  metadata?: string;
}

export interface TraceResult {
  segments: TraceSegment[];
  fullText: string;
  index: string;
  totalTokens: number;
}

export interface InfinigramCountResult {
  count: number;
  approx?: boolean;
  latency?: number;
}

export interface InfinigramProbResult {
  prob: number;
  prompt_cnt: number;
  cont_cnt: number;
  suffix_len?: number;
}

export interface InfinigramSearchResult {
  cnt: number;
  approx?: boolean;
  documents: TraceDocument[];
}

export interface InfinigramNtdResult {
  result_by_token_id: Record<string, number>;
  approx?: boolean;
}

/** infini-gram mini find result */
export interface InfinigramMiniFindResult {
  cnt: number;
  segment_by_shard: [number, number][];
  latency: number;
}

/** infini-gram mini get_doc_by_rank result */
export interface InfinigramMiniDocResult {
  doc_ix: number;
  doc_len: number;
  disp_len: number;
  needle_offset: number;
  text: string;
  latency: number;
}

/** Which engine an index belongs to */
export type InfinigramEngine = "original" | "mini";

export interface IndexEntry {
  id: string;
  label: string;
  engine: InfinigramEngine;
  size?: string;
}
