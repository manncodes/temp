import {
  InfinigramCountResult,
  InfinigramProbResult,
  InfinigramSearchResult,
  TraceSegment,
  TraceDocument,
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
    throw new Error(`infini-gram API error: ${res.status} ${res.statusText}`);
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

/**
 * Trace a full response text by splitting it into overlapping n-gram windows,
 * computing infini-gram probabilities for each, and retrieving source documents
 * for low-probability (memorized) segments.
 */
export async function traceResponse(
  responseText: string,
  index?: string
): Promise<TraceResult> {
  // Split into sentences / meaningful chunks for tracing
  const chunks = splitIntoChunks(responseText);
  const segments: TraceSegment[] = [];

  for (const chunk of chunks) {
    if (chunk.trim().length < 3) {
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
      // Run count and probability queries in parallel
      const [countResult, probResult] = await Promise.all([
        countNgram(chunk, index),
        probNgram(chunk, index),
      ]);

      let documents: TraceDocument[] = [];
      // If the n-gram appears in the corpus, find source documents
      if (countResult.count > 0) {
        try {
          const searchResult = await searchDocs(chunk, 3, index);
          documents = searchResult.documents || [];
        } catch {
          // search_docs can fail for very common n-grams; that's okay
        }
      }

      segments.push({
        text: chunk,
        count: countResult.count,
        prob: probResult.prob,
        effectiveN: probResult.suffix_len ?? chunk.split(/\s+/).length,
        documents,
      });
    } catch {
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

/**
 * Split text into sentence-level chunks for tracing.
 * We try to keep chunks between 5–40 words to get meaningful n-gram lookups.
 */
function splitIntoChunks(text: string): string[] {
  // Split on sentence boundaries
  const raw = text.match(/[^.!?\n]+[.!?\n]?/g) || [text];
  const chunks: string[] = [];

  for (const r of raw) {
    const trimmed = r.trim();
    if (!trimmed) continue;
    const wordCount = trimmed.split(/\s+/).length;

    if (wordCount <= 40) {
      chunks.push(trimmed);
    } else {
      // Break long sentences into smaller pieces at clause boundaries
      const subParts = trimmed.split(/[,;:]/);
      let buffer = "";
      for (const part of subParts) {
        if (buffer && (buffer + part).split(/\s+/).length > 30) {
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
  { id: "v4_rpj_llama_s4", label: "RedPajama + Dolma + Pile + C4 (Llama tokenizer)" },
  { id: "v4_dolma-v1_7_llama", label: "Dolma v1.7 (Llama tokenizer)" },
  { id: "v4_rpj_llama", label: "RedPajama (Llama tokenizer)" },
  { id: "v4_piletrain_llama", label: "Pile Train (Llama tokenizer)" },
  { id: "v4_c4train_llama", label: "C4 Train (Llama tokenizer)" },
  { id: "v4_rpj_gpt2", label: "RedPajama (GPT-2 tokenizer)" },
  { id: "v4_dolma-v1_7_gpt2", label: "Dolma v1.7 (GPT-2 tokenizer)" },
];
