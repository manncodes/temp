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
// Maximal matching (OLMoTrace-style)
// ──────────────────────────────────────────────

/**
 * Binary search for the longest prefix of `words` (by word count) that
 * has count > 0 in the corpus. Returns the matched word count and count.
 *
 * For the mini engine this is character-level exact matching, so full
 * sentences rarely match — but shorter sub-phrases often do.
 */
async function findLongestMatchingPrefix(
  words: string[],
  index: string
): Promise<{ matchedWords: number; matchedText: string; count: number }> {
  if (words.length === 0) {
    return { matchedWords: 0, matchedText: "", count: 0 };
  }

  // First try the full text
  const fullText = words.join(" ");
  const fullResult = await countNgram(fullText, index);
  if (fullResult.count > 0) {
    return {
      matchedWords: words.length,
      matchedText: fullText,
      count: fullResult.count,
    };
  }

  // Binary search on prefix length (by word count)
  let lo = 1;
  let hi = words.length - 1;
  let bestLen = 0;
  let bestCount = 0;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const prefix = words.slice(0, mid).join(" ");
    const result = await countNgram(prefix, index);
    if (result.count > 0) {
      bestLen = mid;
      bestCount = result.count;
      lo = mid + 1; // try longer
    } else {
      hi = mid - 1; // try shorter
    }
  }

  return {
    matchedWords: bestLen,
    matchedText: bestLen > 0 ? words.slice(0, bestLen).join(" ") : "",
    count: bestCount,
  };
}

/**
 * For each chunk, also try suffixes (starting from later words) to find
 * additional matches beyond just the prefix. This catches cases where
 * the end of a sentence is memorized but the beginning isn't.
 */
async function findBestMatch(
  words: string[],
  index: string
): Promise<{ matchedWords: number; matchedText: string; count: number; startWord: number }> {
  // Try prefix first (most common case, fastest)
  const prefix = await findLongestMatchingPrefix(words, index);

  // If we matched most of the chunk from the start, that's good enough
  if (prefix.matchedWords >= words.length * 0.5) {
    return { ...prefix, startWord: 0 };
  }

  // Also try from the middle and from the end
  let best = { ...prefix, startWord: 0 };

  const midStart = Math.floor(words.length / 2);
  if (midStart > 0 && midStart < words.length) {
    const midMatch = await findLongestMatchingPrefix(
      words.slice(midStart),
      index
    );
    if (midMatch.matchedWords > best.matchedWords) {
      best = { ...midMatch, startWord: midStart };
    }
  }

  // Try last third
  const lateStart = Math.floor(words.length * 0.67);
  if (lateStart > midStart && lateStart < words.length) {
    const lateMatch = await findLongestMatchingPrefix(
      words.slice(lateStart),
      index
    );
    if (lateMatch.matchedWords > best.matchedWords) {
      best = { ...lateMatch, startWord: lateStart };
    }
  }

  return best;
}

// ──────────────────────────────────────────────
// Segment normalization
// ──────────────────────────────────────────────

/**
 * Compute a normalized memorization score for a segment.
 *
 * Score combines:
 *  - matchRatio: what fraction of the chunk matched verbatim (0–1)
 *  - countSignal: log-scaled count relative to expected (higher = more copies)
 *  - lengthBonus: longer verbatim matches are more significant
 *
 * Result is in [0, 1] where 1 = highly memorized.
 */
function computeNormalizedScore(
  matchedWords: number,
  totalWords: number,
  count: number
): number {
  if (matchedWords === 0 || count === 0) return 0;

  const matchRatio = matchedWords / totalWords;

  // Log-scale the count: log2(count+1) / log2(threshold)
  // A count of ~1000 is "saturated" at 1.0
  const countSignal = Math.min(1, Math.log2(count + 1) / Math.log2(1000));

  // Bonus for longer matches: 2-word match = 0.3, 5-word = 0.65, 10+ = ~1.0
  const lengthBonus = Math.min(1, Math.log2(matchedWords + 1) / Math.log2(12));

  // Weighted combination
  return Math.min(1, matchRatio * 0.3 + countSignal * 0.3 + lengthBonus * 0.4);
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
  const isMini = getEngine(index) === "mini";

  // Optional: get corpus size for context (empty string count = total bytes)
  let corpusSize: number | undefined;
  try {
    const csResult = await countNgram("", index);
    corpusSize = csResult.count;
  } catch {
    // not critical
  }

  for (const chunk of chunks) {
    const trimmed = chunk.trim();
    if (trimmed.length < 4) {
      segments.push({
        text: chunk,
        matchedText: "",
        count: 0,
        matchRatio: 0,
        normalizedScore: 0,
        matchedWords: 0,
        prob: -1,
        effectiveN: 0,
        documents: [],
      });
      continue;
    }

    try {
      const words = trimmed.split(/\s+/);

      // Step 1: Find longest verbatim match (OLMoTrace-style)
      const match = isMini
        ? await findBestMatch(words, index)
        : await findLongestMatchingPrefix(words, index);

      const matchRatio =
        match.matchedWords > 0 ? match.matchedWords / words.length : 0;

      // Step 2: Get probability (original engine only)
      const probResult = await probNgram(trimmed, index);

      // Step 3: Compute normalized score
      const normalizedScore = computeNormalizedScore(
        match.matchedWords,
        words.length,
        match.count
      );

      // Step 4: Retrieve source documents for the matched substring
      let documents: TraceDocument[] = [];
      if (match.count > 0 && match.matchedText) {
        try {
          documents = await searchDocs(match.matchedText, 3, index);
        } catch {
          // doc retrieval can fail for very common strings
        }
      }

      segments.push({
        text: chunk,
        matchedText: match.matchedText,
        count: match.count,
        matchRatio,
        normalizedScore,
        matchedWords: match.matchedWords,
        prob: probResult.prob,
        effectiveN:
          (probResult.suffix_len ?? match.matchedWords) || words.length,
        documents,
      });
    } catch (err) {
      console.warn(
        `infini-gram trace failed for chunk "${chunk.slice(0, 40)}…":`,
        err
      );
      segments.push({
        text: chunk,
        matchedText: "",
        count: -1,
        matchRatio: 0,
        normalizedScore: 0,
        matchedWords: 0,
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
    corpusSize,
  };
}
