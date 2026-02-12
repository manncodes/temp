/**
 * Diagnostic tests for infini-gram APIs.
 * Tests both the original and mini engines to verify connectivity,
 * index names, count queries, and document retrieval.
 *
 * Run with: npx tsx src/lib/__tests__/infinigram-api.test.ts
 */

const MINI_API = "https://api.infini-gram-mini.io/";
const ORIGINAL_API = "https://api.infini-gram.io/";

interface TestResult {
  name: string;
  pass: boolean;
  detail: string;
  data?: unknown;
}

const results: TestResult[] = [];

async function postJson(url: string, payload: Record<string, unknown>) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const json = await res.json();
  return { status: res.status, json };
}

function record(name: string, pass: boolean, detail: string, data?: unknown) {
  results.push({ name, pass, detail, data });
  const icon = pass ? "\x1b[32mPASS\x1b[0m" : "\x1b[31mFAIL\x1b[0m";
  console.log(`[${icon}] ${name}`);
  console.log(`       ${detail}`);
  if (data) console.log(`       data:`, JSON.stringify(data).slice(0, 300));
  console.log();
}

// ─────────────────────────────────────────────
// Test 1: Mini API – index name format
// The docs curl example uses "v2_cc-2025-05" (hyphen between cc and year).
// Verify this format works AND that the wrong format fails.
// ─────────────────────────────────────────────
async function testMiniIndexNameFormats() {
  const query = "University of Washington";

  // Correct format: hyphen (from docs curl example)
  const correct = await postJson(MINI_API, {
    index: "v2_cc-2025-05",
    query_type: "count",
    query,
  });

  record(
    "Mini: v2_cc-2025-05 (hyphen — correct per docs)",
    !correct.json.error && typeof correct.json.count === "number" && correct.json.count > 0,
    correct.json.error
      ? `ERROR: ${correct.json.error}`
      : `count=${correct.json.count}, latency=${correct.json.latency}ms`,
    correct.json
  );

  // Wrong format: underscore instead of hyphen
  const wrong = await postJson(MINI_API, {
    index: "v2_cc_2025-05",
    query_type: "count",
    query,
  });

  record(
    "Mini: v2_cc_2025-05 (underscore — expected to fail)",
    !!wrong.json.error,
    wrong.json.error
      ? `Correctly rejected: ${wrong.json.error}`
      : `WARNING: unexpectedly accepted! count=${wrong.json.count}`,
    wrong.json
  );
}

// ─────────────────────────────────────────────
// Test 2: Mini API – count on all CC indexes
// ─────────────────────────────────────────────
async function testMiniCountAllCCIndexes() {
  const ccIndexes = [
    "v2_cc-2025-30",
    "v2_cc-2025-26",
    "v2_cc-2025-21",
    "v2_cc-2025-18",
    "v2_cc-2025-13",
    "v2_cc-2025-08",
    "v2_cc-2025-05",
  ];

  for (const idx of ccIndexes) {
    const r = await postJson(MINI_API, {
      index: idx,
      query_type: "count",
      query: "why is sky blue",
    });
    record(
      `Mini count: ${idx}`,
      !r.json.error && typeof r.json.count === "number",
      r.json.error
        ? `ERROR: ${r.json.error}`
        : `count=${r.json.count}, latency=${r.json.latency}ms`,
      r.json
    );
  }
}

// ─────────────────────────────────────────────
// Test 3: Mini API – count on DCLM + Pile
// ─────────────────────────────────────────────
async function testMiniCountCurated() {
  const indexes = ["v2_dclm_all", "v2_piletrain", "v2_pileval"];
  for (const idx of indexes) {
    const r = await postJson(MINI_API, {
      index: idx,
      query_type: "count",
      query: "machine learning",
    });
    record(
      `Mini count: ${idx}`,
      !r.json.error && typeof r.json.count === "number",
      r.json.error
        ? `ERROR: ${r.json.error}`
        : `count=${r.json.count}, latency=${r.json.latency}ms`,
      r.json
    );
  }
}

// ─────────────────────────────────────────────
// Test 4: Mini API – find + get_doc_by_rank (two-step doc retrieval)
// ─────────────────────────────────────────────
async function testMiniFindAndGetDoc() {
  const idx = "v2_cc-2025-05";
  const query = "machine learning";

  // Step 1: find
  const findR = await postJson(MINI_API, {
    index: idx,
    query_type: "find",
    query,
  });

  record(
    `Mini find: ${idx}`,
    !findR.json.error && typeof findR.json.cnt === "number" && findR.json.cnt > 0,
    findR.json.error
      ? `ERROR: ${findR.json.error}`
      : `cnt=${findR.json.cnt}, shards=${findR.json.segment_by_shard?.length}`,
    { cnt: findR.json.cnt, shards_count: findR.json.segment_by_shard?.length }
  );

  if (findR.json.error || !findR.json.segment_by_shard) return;

  // Step 2: pick first non-empty shard
  const shards: [number, number][] = findR.json.segment_by_shard;
  let pickedShard = -1;
  let pickedRank = -1;
  for (let s = 0; s < shards.length; s++) {
    if (shards[s][1] > shards[s][0]) {
      pickedShard = s;
      pickedRank = shards[s][0];
      break;
    }
  }

  if (pickedShard < 0) {
    record("Mini get_doc_by_rank", false, "No non-empty shard found");
    return;
  }

  const docR = await postJson(MINI_API, {
    index: idx,
    query_type: "get_doc_by_rank",
    s: pickedShard,
    rank: pickedRank,
    max_ctx_len: 200,
  });

  record(
    `Mini get_doc_by_rank: shard=${pickedShard} rank=${pickedRank}`,
    !docR.json.error && typeof docR.json.text === "string" && docR.json.text.length > 0,
    docR.json.error
      ? `ERROR: ${docR.json.error}`
      : `doc_ix=${docR.json.doc_ix}, doc_len=${docR.json.doc_len}, preview="${(docR.json.text || "").slice(0, 120)}..."`,
    { doc_ix: docR.json.doc_ix, doc_len: docR.json.doc_len }
  );
}

// ─────────────────────────────────────────────
// Test 5: Original API – count
// ─────────────────────────────────────────────
async function testOriginalCount() {
  const r = await postJson(ORIGINAL_API, {
    index: "v4_rpj_llama_s4",
    query_type: "count",
    query: "natural language processing",
  });

  record(
    "Original count: v4_rpj_llama_s4",
    !r.json.error && typeof r.json.count === "number",
    r.json.error
      ? `ERROR: ${r.json.error}`
      : `count=${r.json.count}, latency=${r.json.latency}ms`,
    r.json
  );
}

// ─────────────────────────────────────────────
// Test 6: Mini API – short vs long query (character-level matching)
// infini-gram mini does EXACT string matching at the character level.
// Longer strings = fewer matches. This test shows the expected behavior.
// ─────────────────────────────────────────────
async function testMiniQueryLengths() {
  const idx = "v2_cc-2025-30";
  const queries = [
    "the",
    "machine learning",
    "why is sky blue",
    "The sky appears blue because of a phenomenon called Rayleigh scattering",
    "In the field of artificial intelligence, transformer-based language models have demonstrated remarkable capabilities across a wide range of natural language processing tasks",
  ];

  for (const q of queries) {
    const r = await postJson(MINI_API, {
      index: idx,
      query_type: "count",
      query: q,
    });
    record(
      `Mini count len=${q.length}: "${q.slice(0, 50)}${q.length > 50 ? "..." : ""}"`,
      !r.json.error && typeof r.json.count === "number",
      r.json.error
        ? `ERROR: ${r.json.error}`
        : `count=${r.json.count}`,
      r.json
    );
  }
}

// ─────────────────────────────────────────────
// Test 7: Original API – search_docs
// ─────────────────────────────────────────────
async function testOriginalSearchDocs() {
  const r = await postJson(ORIGINAL_API, {
    index: "v4_rpj_llama_s4",
    query_type: "search_docs",
    query: "natural language processing",
    max_disp_len: 200,
    max_clause_freq: 50000,
    max_diff_tokens: 100,
  });

  const docCount = r.json.documents?.length ?? 0;
  record(
    "Original search_docs: v4_rpj_llama_s4",
    !r.json.error && docCount > 0,
    r.json.error
      ? `ERROR: ${r.json.error}`
      : `Got ${docCount} documents`,
    { cnt: r.json.cnt, doc_count: docCount }
  );
}

// ─────────────────────────────────────────────
// Run all
// ─────────────────────────────────────────────
async function main() {
  console.log("=== infini-gram API Diagnostic Tests ===\n");

  try {
    await testMiniIndexNameFormats();
  } catch (e) {
    record("Mini index name test", false, `Network error: ${e}`);
  }

  try {
    await testMiniCountAllCCIndexes();
  } catch (e) {
    record("Mini CC indexes", false, `Network error: ${e}`);
  }

  try {
    await testMiniCountCurated();
  } catch (e) {
    record("Mini curated indexes", false, `Network error: ${e}`);
  }

  try {
    await testMiniFindAndGetDoc();
  } catch (e) {
    record("Mini find+doc", false, `Network error: ${e}`);
  }

  try {
    await testOriginalCount();
  } catch (e) {
    record("Original count", false, `Network error: ${e}`);
  }

  try {
    await testMiniQueryLengths();
  } catch (e) {
    record("Mini query lengths", false, `Network error: ${e}`);
  }

  try {
    await testOriginalSearchDocs();
  } catch (e) {
    record("Original search_docs", false, `Network error: ${e}`);
  }

  console.log("\n=== SUMMARY ===");
  const passed = results.filter((r) => r.pass).length;
  const failed = results.filter((r) => !r.pass).length;
  console.log(`\x1b[32m${passed} passed\x1b[0m, \x1b[31m${failed} failed\x1b[0m out of ${results.length} tests`);

  if (failed > 0) {
    console.log("\nFailed tests:");
    for (const r of results.filter((r) => !r.pass)) {
      console.log(`  \x1b[31m✗\x1b[0m ${r.name}: ${r.detail}`);
    }
  }
}

main().catch(console.error);
