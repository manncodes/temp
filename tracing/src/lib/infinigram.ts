import {
  InfinigramCountResult,
  InfinigramProbResult,
  InfinigramSearchResult,
  TraceDocument,
  TraceSegment,
  TraceResult,
} from "./types";

const INFINIGRAM_API = "https://api.infini-gram.io/";
const DEFAULT_INDEX = "v4_rpj_llama_s4";

async function query(payload: Record<string, unknown>) {
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
  return query({
    query_type: "count",
    query: text,
    ...(index && { index }),
  });
}

export async function probNgram(
  text: string,
  index?: string
): Promise<InfinigramProbResult> {
  return query({
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
  return query({
    query_type: "search_docs",
    query: text,
    max_disp_len: 500,
    max_clause_freq: 50000,
    max_diff_tokens: 100,
    ...(index && { index }),
  });
}

/** Strip markdown formatting so infini-gram gets clean plain text */
function stripMarkdown(text: string): string {
  return (
    text
      // bold/italic
      .replace(/\*{1,3}([^*]+)\*{1,3}/g, "$1")
      .replace(/_{1,3}([^_]+)_{1,3}/g, "$1")
      // inline code
      .replace(/`([^`]+)`/g, "$1")
      // headers
      .replace(/^#{1,6}\s+/gm, "")
      // links [text](url)
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      // images
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
      // blockquotes
      .replace(/^>\s+/gm, "")
      // bullet points
      .replace(/^[-*+]\s+/gm, "")
      // numbered lists
      .replace(/^\d+\.\s+/gm, "")
      // horizontal rules
      .replace(/^---+$/gm, "")
      // extra whitespace
      .replace(/\n{3,}/g, "\n\n")
      .trim()
  );
}

/**
 * Trace a full response text by splitting it into n-gram chunks,
 * querying infini-gram for each, and finding source documents.
 * This runs CLIENT-SIDE in the browser, calling the infini-gram API directly.
 */
export async function traceResponse(
  responseText: string,
  index?: string,
  onProgress?: (done: number, total: number) => void
): Promise<TraceResult> {
  const cleanText = stripMarkdown(responseText);
  const chunks = splitIntoChunks(cleanText);
  const segments: TraceSegment[] = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    onProgress?.(i, chunks.length);

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
      console.warn(`infini-gram trace failed for chunk "${chunk.slice(0, 40)}…":`, err);
      segments.push({
        text: chunk,
        count: -1,
        prob: -1,
        effectiveN: 0,
        documents: [],
      });
    }
  }

  onProgress?.(chunks.length, chunks.length);

  return {
    segments,
    fullText: responseText,
    index: index || DEFAULT_INDEX,
    totalTokens: chunks.length,
  };
}

/**
 * Split text into sentence-level chunks for tracing.
 * Targets 5–30 words per chunk for good n-gram matches.
 */
function splitIntoChunks(text: string): string[] {
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

export const AVAILABLE_INDEXES = [
  { id: "v4_rpj_llama_s4", label: "RedPajama + Dolma + Pile + C4 (Llama)" },
  { id: "v4_dolma-v1_7_llama", label: "Dolma v1.7 (Llama)" },
  { id: "v4_rpj_llama", label: "RedPajama (Llama)" },
  { id: "v4_piletrain_llama", label: "Pile Train (Llama)" },
  { id: "v4_c4train_llama", label: "C4 Train (Llama)" },
  { id: "v4_rpj_gpt2", label: "RedPajama (GPT-2)" },
  { id: "v4_dolma-v1_7_gpt2", label: "Dolma v1.7 (GPT-2)" },
];
