/**
 * Server-side infini-gram client.
 * Supports BOTH the original infini-gram API and infini-gram mini API.
 * Used by /api/trace route — fetches APIs directly (no CORS).
 */
import {
  InfinigramCountResult,
  InfinigramProbResult,
  InfinigramMiniFindResult,
  InfinigramMiniDocResult,
  TraceDocument,
  TraceSegment,
  TraceResult,
  IndexEntry,
} from "./types";

const INFINIGRAM_API = "https://api.infini-gram.io/";
const INFINIGRAM_MINI_API = "https://api.infini-gram-mini.io/";
export const DEFAULT_INDEX = "v4_rpj_llama_s4";

// ──────────────────────────────────────────────
// All available indexes across BOTH engines
// ──────────────────────────────────────────────

export const AVAILABLE_INDEXES: IndexEntry[] = [
  // ── infini-gram mini: Common Crawl (massive, 8–10 TB each) ──
  { id: "v2_cc-2025-30", label: "Common Crawl Jul 2025", engine: "mini", size: "9.0 TB" },
  { id: "v2_cc-2025-26", label: "Common Crawl Jun 2025", engine: "mini", size: "8.7 TB" },
  { id: "v2_cc-2025-21", label: "Common Crawl May 2025", engine: "mini", size: "9.2 TB" },
  { id: "v2_cc-2025-18", label: "Common Crawl Apr 2025", engine: "mini", size: "10.5 TB" },
  { id: "v2_cc-2025-13", label: "Common Crawl Mar 2025", engine: "mini", size: "10.4 TB" },
  { id: "v2_cc-2025-08", label: "Common Crawl Feb 2025", engine: "mini", size: "8.2 TB" },
  { id: "v2_cc-2025-05", label: "Common Crawl Jan 2025", engine: "mini", size: "9.1 TB" },

  // ── infini-gram mini: Curated datasets ──
  { id: "v2_dclm_all", label: "DCLM-baseline", engine: "mini", size: "16.7 TB" },
  { id: "v2_piletrain", label: "Pile Train", engine: "mini", size: "1.3 TB" },
  { id: "v2_pileval", label: "Pile Val", engine: "mini", size: "1.3 GB" },

  // ── infini-gram original: Token-level indexes ──
  { id: "v4_rpj_llama_s4", label: "RPJ+Dolma+Pile+C4 (Llama)", engine: "original", size: "~5 TB" },
  { id: "v4_dolma-v1_7_llama", label: "Dolma v1.7 (Llama)", engine: "original", size: "~3 TB" },
  { id: "v4_rpj_llama", label: "RedPajama (Llama)", engine: "original", size: "~1.4 TB" },
  { id: "v4_piletrain_llama", label: "Pile Train (Llama)", engine: "original", size: "~800 GB" },
  { id: "v4_c4train_llama", label: "C4 Train (Llama)", engine: "original", size: "~350 GB" },
  { id: "v4_rpj_gpt2", label: "RedPajama (GPT-2)", engine: "original", size: "~1.4 TB" },
  { id: "v4_dolma-v1_7_gpt2", label: "Dolma v1.7 (GPT-2)", engine: "original", size: "~3 TB" },
];

/** Look up which engine an index belongs to */
export function getEngine(indexId: string): "original" | "mini" {
  const entry = AVAILABLE_INDEXES.find((e) => e.id === indexId);
  return entry?.engine ?? "original";
}

function apiUrl(indexId: string): string {
  return getEngine(indexId) === "mini" ? INFINIGRAM_MINI_API : INFINIGRAM_API;
}

// ──────────────────────────────────────────────
// Low-level query helpers
// ──────────────────────────────────────────────

async function queryRaw(
  endpoint: string,
  payload: Record<string, unknown>
) {
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`infini-gram API error: ${res.status} ${res.statusText} — ${body}`);
  }
  const json = await res.json();
  if (json.error) {
    throw new Error(`infini-gram error: ${json.error}`);
  }
  return json;
}

// ──────────────────────────────────────────────
// Count (works on BOTH engines)
// ──────────────────────────────────────────────

export async function countNgram(
  text: string,
  index: string = DEFAULT_INDEX
): Promise<InfinigramCountResult> {
  return queryRaw(apiUrl(index), {
    index,
    query_type: "count",
    query: text,
  });
}

// ──────────────────────────────────────────────
// Probability (original engine only)
// ──────────────────────────────────────────────

export async function probNgram(
  text: string,
  index: string = DEFAULT_INDEX
): Promise<InfinigramProbResult> {
  if (getEngine(index) === "mini") {
    // Mini doesn't support prob queries — return a sentinel
    return { prob: -1, prompt_cnt: 0, cont_cnt: 0 };
  }
  return queryRaw(apiUrl(index), {
    index,
    query_type: "infgram_prob",
    query: text,
  });
}

// ──────────────────────────────────────────────
// Document search
// ──────────────────────────────────────────────

/** Original engine: single-step search_docs */
async function searchDocsOriginal(
  text: string,
  index: string
): Promise<TraceDocument[]> {
  const result = await queryRaw(INFINIGRAM_API, {
    index,
    query_type: "search_docs",
    query: text,
    max_disp_len: 500,
    max_clause_freq: 50000,
    max_diff_tokens: 100,
  });
  return result.documents || [];
}

/**
 * Mini engine: two-step find → get_doc_by_rank.
 * Follows the same shard/rank mapping as the HuggingFace Spaces reference.
 */
async function searchDocsMini(
  text: string,
  index: string,
  maxDocs: number = 3
): Promise<TraceDocument[]> {
  // Step 1: find
  const findResult: InfinigramMiniFindResult = await queryRaw(
    INFINIGRAM_MINI_API,
    {
      index,
      query_type: "find",
      query: text,
    }
  );

  if (findResult.cnt === 0 || !findResult.segment_by_shard) {
    return [];
  }

  const shards = findResult.segment_by_shard;
  const cnt = findResult.cnt;

  // Map a flat global index → (shard, rank), same as HF reference:
  //   cnt_by_shard = [end - start for (start, end) in segment_by_shard]
  //   walk through shards until idx falls within the shard's range
  function globalToShardRank(globalIdx: number): { s: number; rank: number } {
    let remaining = globalIdx;
    for (let s = 0; s < shards.length; s++) {
      const shardCnt = shards[s][1] - shards[s][0];
      if (remaining < shardCnt) {
        return { s, rank: shards[s][0] + remaining };
      }
      remaining -= shardCnt;
    }
    return { s: 0, rank: shards[0][0] };
  }

  // Pick spread-out indices (first, random from middle, last)
  const indices: number[] = [0];
  if (cnt > 1 && maxDocs > 1) {
    indices.push(Math.floor(cnt / 2));
  }
  if (cnt > 2 && maxDocs > 2) {
    indices.push(cnt - 1);
  }

  const fetched = await Promise.all(
    indices.slice(0, maxDocs).map(async (globalIdx) => {
      const { s, rank } = globalToShardRank(globalIdx);
      try {
        // HF reference sends `query` along with get_doc_by_rank
        const doc = await queryRaw(INFINIGRAM_MINI_API, {
          index,
          query_type: "get_doc_by_rank",
          query: text,
          s,
          rank,
          max_ctx_len: 500,
        });
        // Response may have `text` (per docs) or `spans` (per HF reference)
        const passage =
          typeof doc.text === "string"
            ? doc.text
            : Array.isArray(doc.spans)
              ? doc.spans.map((sp: [string, string | null]) => sp[0]).join("")
              : "";
        return {
          doc_ix: doc.doc_ix,
          doc_len: doc.doc_len,
          disp_len: doc.disp_len,
          passage,
          metadata: doc.metadata,
        } satisfies TraceDocument;
      } catch {
        return null;
      }
    })
  );

  const docs: TraceDocument[] = [];
  for (const d of fetched) {
    if (d) docs.push(d);
  }
  return docs;
}

/** Unified doc search — dispatches to the right engine */
export async function searchDocs(
  text: string,
  maxDocs: number = 3,
  index: string = DEFAULT_INDEX
): Promise<TraceDocument[]> {
  if (getEngine(index) === "mini") {
    return searchDocsMini(text, index, maxDocs);
  }
  return searchDocsOriginal(text, index);
}

// ──────────────────────────────────────────────
// Text utilities
// ──────────────────────────────────────────────

/** Strip markdown formatting so infini-gram gets clean plain text */
export function stripMarkdown(text: string): string {
  return (
    text
      .replace(/\*{1,3}([^*]+)\*{1,3}/g, "$1")
      .replace(/_{1,3}([^_]+)_{1,3}/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
      .replace(/^>\s+/gm, "")
      .replace(/^[-*+]\s+/gm, "")
      .replace(/^\d+\.\s+/gm, "")
      .replace(/^---+$/gm, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim()
  );
}

export function splitIntoChunks(text: string): string[] {
  const raw = text.match(/[^.!?\n]+[.!?\n]?/g) || [text];
  const chunks: string[] = [];

  for (const r of raw) {
    const trimmed = r.trim();
    if (!trimmed) continue;
    const wordCount = trimmed.split(/\s+/).length;

    if (wordCount <= 30) {
      chunks.push(trimmed);
    } else {
      const subParts = trimmed.split(/[,;:]/);
      let buffer = "";
      for (const part of subParts) {
        if (buffer && (buffer + part).split(/\s+/).length > 25) {
          chunks.push(buffer.trim());
          buffer = part;
        } else {
          buffer += (buffer ? "," : "") + part;
        }
      }
      if (buffer.trim()) chunks.push(buffer.trim());
    }
  }

  return chunks;
}

// ──────────────────────────────────────────────
// Full trace (server-side, called from /api/trace)
// ──────────────────────────────────────────────

export async function traceResponse(
  responseText: string,
  index: string = DEFAULT_INDEX
): Promise<TraceResult> {
  const cleanText = stripMarkdown(responseText);
  const chunks = splitIntoChunks(cleanText);
  const segments: TraceSegment[] = [];

  for (const chunk of chunks) {
    if (chunk.trim().length < 4) {
      segments.push({
        text: chunk,
        count: -1,
        prob: -1,
        effectiveN: 0,
        documents: [],
      });
      continue;
    }

    try {
      const [countResult, probResult] = await Promise.all([
        countNgram(chunk, index),
        probNgram(chunk, index),
      ]);

      let documents: TraceDocument[] = [];
      if (countResult.count > 0) {
        try {
          documents = await searchDocs(chunk, 3, index);
        } catch {
          // search_docs can fail for very common n-grams
        }
      }

      segments.push({
        text: chunk,
        count: countResult.count,
        prob: probResult.prob,
        effectiveN: probResult.suffix_len ?? chunk.split(/\s+/).length,
        documents,
      });
    } catch (err) {
      console.warn(
        `infini-gram trace failed for chunk "${chunk.slice(0, 40)}…":`,
        err
      );
      segments.push({
        text: chunk,
        count: -1,
        prob: -1,
        effectiveN: 0,
        documents: [],
      });
    }
  }

  return {
    segments,
    fullText: responseText,
    index,
    totalTokens: chunks.length,
  };
}
