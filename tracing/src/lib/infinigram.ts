/**
 * Server-side infini-gram client.
 * Used by /api/trace route — fetches api.infini-gram.io directly (no CORS).
 */
import {
  InfinigramCountResult,
  InfinigramProbResult,
  InfinigramSearchResult,
  TraceDocument,
  TraceSegment,
  TraceResult,
} from "./types";

const INFINIGRAM_API = "https://api.infini-gram.io/";
export const DEFAULT_INDEX = "v4_rpj_llama_s4";

export async function queryInfinigram(payload: Record<string, unknown>) {
  const res = await fetch(INFINIGRAM_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index: DEFAULT_INDEX, ...payload }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `infini-gram API error: ${res.status} ${res.statusText} — ${body}`
    );
  }
  return res.json();
}

export async function countNgram(
  text: string,
  index?: string
): Promise<InfinigramCountResult> {
  return queryInfinigram({
    query_type: "count",
    query: text,
    ...(index && { index }),
  });
}

export async function probNgram(
  text: string,
  index?: string
): Promise<InfinigramProbResult> {
  return queryInfinigram({
    query_type: "infgram_prob",
    query: text,
    ...(index && { index }),
  });
}

export async function searchDocs(
  text: string,
  maxDocs: number = 3,
  index?: string
): Promise<InfinigramSearchResult> {
  return queryInfinigram({
    query_type: "search_docs",
    query: text,
    max_disp_len: 500,
    max_clause_freq: 50000,
    max_diff_tokens: 100,
    ...(index && { index }),
  });
}

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

/**
 * Split text into sentence-level chunks for tracing.
 * Targets 5–30 words per chunk for good n-gram matches.
 */
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

/**
 * Full trace: split into chunks, query infini-gram for each.
 * Called server-side from /api/trace.
 */
export async function traceResponse(
  responseText: string,
  index?: string
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
          const searchResult = await searchDocs(chunk, 3, index);
          documents = searchResult.documents || [];
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
    index: index || DEFAULT_INDEX,
    totalTokens: chunks.length,
  };
}

export const AVAILABLE_INDEXES = [
  { id: "v4_rpj_llama_s4", label: "RedPajama + Dolma + Pile + C4 (Llama)" },
  { id: "v4_dolma-v1_7_llama", label: "Dolma v1.7 (Llama)" },
  { id: "v4_rpj_llama", label: "RedPajama (Llama)" },
  { id: "v4_piletrain_llama", label: "Pile Train (Llama)" },
  { id: "v4_c4train_llama", label: "C4 Train (Llama)" },
  { id: "v4_rpj_gpt2", label: "RedPajama (GPT-2)" },
  { id: "v4_dolma-v1_7_gpt2", label: "Dolma v1.7 (GPT-2)" },
];
