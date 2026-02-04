# Structured Output Generation for LLMs: State of the Art (2024-2025)

## Overview

Structured output generation ensures LLM outputs conform to specific formats (JSON, SQL, XML, code) with guaranteed validity. The field has converged on two complementary approaches: **constrained decoding** (inference-time grammar enforcement) and **reinforcement learning** (training-time schema adherence). This summary covers the key papers, frameworks, and techniques as of early 2025.

---

## 1. Constrained Decoding (Inference-Time)

### Core Mechanism: Token Masking

At each decoding step, before sampling the next token, the logit distribution is modified to **mask out tokens that would violate the target grammar**. This guarantees 100% structural compliance with zero training data.

```
logits = model(input_ids)
mask = grammar_engine.get_valid_token_mask(current_state)
logits[~mask] = -inf
next_token = sample(logits)
grammar_engine.advance(next_token)
```

### Constraint Types (Weakest to Strongest)

| Type | Power | Example |
|---|---|---|
| **Choice list** | Enumerate valid outputs | `["yes", "no", "maybe"]` |
| **Regex** | Regular languages | `[A-Z][a-z]+ \d{4}` for "Name Year" |
| **JSON Schema** | Nested/typed structures | `{"type": "object", "properties": {...}}` |
| **Context-Free Grammar (CFG)** | Recursive/nested structures | Full JSON, SQL, Python, XML |
| **Type systems** | Semantic + syntactic | TypeScript type-constrained code generation |

### Automaton-Based Approaches

**Finite State Machines (FSMs)** — Used for regex constraints. Each state represents progress through the pattern; transitions are token-level. Outlines and lm-format-enforcer use this approach.

**Pushdown Automata (PDAs)** — Required for CFGs because they need a stack to handle recursion (nested brackets, nested JSON objects). XGrammar pioneered this at scale.

**Deterministic PDAs (DPDAs)** — Pre3 (ACL 2025) showed that LR(1) grammars can be converted to DPDAs, enabling faster mask computation because each (state, stack_top) pair has exactly one valid transition set.

### Key Optimization: Adaptive Token Mask Caching

The vocabulary can be 128K+ tokens. Computing valid masks at every step is expensive. XGrammar's breakthrough was splitting tokens into:
- **Context-independent tokens** — validity depends only on the grammar state, not the stack. These masks can be precomputed and cached.
- **Context-dependent tokens** — validity depends on the stack state (e.g., tokens that cross grammar rule boundaries). These are computed at runtime but are a small fraction (~3-5%) of the vocabulary.

This achieves **<40 µs per token** for JSON Schema, essentially zero overhead.

---

## 2. Key Frameworks (2024-2025)

### XGrammar (mlc-ai) — arxiv:2411.15100
- **Approach:** Byte-level pushdown automaton for CFGs
- **Performance:** Up to 100x faster than prior art; <40µs/token for JSON Schema
- **Integration:** Default backend in vLLM v0.8+, used by MLC-LLM, SGLang
- **Key insight:** Adaptive token mask cache splits vocab into context-independent (precomputed) and context-dependent (runtime) tokens
- **Supports:** JSON Schema, arbitrary EBNF grammars, regex

### Outlines (dottxt-ai) — Production framework
- **Approach:** FSM-based structured generation
- **Strengths:** Unicode support, HuggingFace ecosystem integration, vLLM backend
- **Supports:** Regex, JSON Schema, CFG (via interegular)
- **Note:** Being superseded by XGrammar for CFG tasks due to performance

### Guidance (Microsoft) — guidance-ai/guidance
- **Approach:** Token healing + CFG enforcement + template-based generation
- **Strengths:** Template syntax mixing free-form text with structured blocks, compatible with Transformers and llama.cpp
- **Unique feature:** "Token healing" fixes tokenization artifacts at constraint boundaries

### lm-format-enforcer — Regex/JSON Schema enforcement
- **Approach:** Character-level constraint tracking
- **Supports:** Regex, JSON Schema, beam search integration

### SynCode (UIUC) — arxiv:2403.01632
- **Approach:** Grammar-constrained generation using incremental parsing
- **Supports:** Built-in grammars for JSON, Python, Go, SQL
- **Unique:** Uses incremental Earley parser for runtime constraint checking

### genlm-control — Constrained decoding as posterior inference
- **Approach:** Sequential Monte Carlo (SMC) for arbitrary programmable constraints
- **Unique:** Supports both syntactic AND semantic constraints; treats constrained generation as probabilistic inference

---

## 3. Reinforcement Learning for Structured Output (Training-Time)

### Schema Reinforcement Learning (SRL) — arxiv:2502.18878, ACL 2025
**Key contributions:**
- **SchemaBench:** Benchmark with ~40K JSON schemas across two tasks:
  - Schema-only generation (generate valid JSON from schema alone)
  - Schema-constrained reasoning (answer questions with schema-valid output)
- **Thought of Structure (ToS):** Chain-of-thought adapted for structural reasoning. The model first reasons about the target structure before generating it, similar to how CoT helps with math.
- **Schema RL:** Uses a fine-grained schema validator as the reward function in RL training. The validator provides token-level or field-level rewards rather than binary valid/invalid.
- **Finding:** Even frontier models (GPT-4o, Claude) struggle with complex schemas. RL training with structural rewards significantly improves schema adherence.

### Think Inside the JSON — arxiv:2502.14905
**Key contributions:**
- Builds on DeepSeek R1's RL framework (GRPO) for a 1.5B parameter model
- Two-stage training:
  1. GRPO on 20K unstructured-to-structured samples (~20 hours on 8×H100)
  2. SFT on 10K reasoning samples (~3 hours on 1×A100)
- Custom reward functions for schema adherence under Group Relative Policy Optimization
- Shows even small models can achieve robust JSON compliance with targeted RL

### RL-Struct — arxiv:2512.00319
**Key contributions:**
- Lightweight framework using GRPO with hierarchical reward function
- **38% less peak VRAM** than PPO by eliminating the critic network
- Shows 4B models can match or beat larger models on JSON validity
- Key insight: SFT learns semantic content but fails to capture rigid syntactic constraints of formal languages. RL acts as a non-differentiable regularizer penalizing syntactic deviations.

### SLOT — arxiv:2505.04016
**Key contributions:**
- Model-agnostic framework that **decouples formatting from the NL task**
- Two metrics: Schema Accuracy (structural validity) and Content Similarity (semantic preservation)
- With SFT, lightweight open-weight models outperform larger proprietary models
- Task-agnostic: works across different structured output requirements

---

## 4. Hybrid Approaches (Reasoning + Constraints)

### CRANE — arxiv:2502.09061, ICML 2025
**The key insight:** Strict grammar constraints during generation HURT reasoning ability. The constraint grammar restricts the token space so much that the model can't express intermediate reasoning steps.

**Solution:** Augment the output grammar with delimiter structure: `S1 G S2`
- `S1`: Unconstrained reasoning section (free-form text)
- `G`: Grammar-constrained output section (strict format)
- `S2`: Optional continuation

**How it works:**
1. Model generates free-form reasoning (chain-of-thought)
2. At a delimiter token, switches to grammar-constrained decoding
3. Generates the structured answer with guaranteed validity
4. Can switch back to free-form if needed

**Results on GSM-Symbolic (math) and FOLIO (logic):**
- Consistently outperforms both unconstrained generation and pure constrained decoding
- Tested on Qwen2.5, Llama-3.1, DeepSeek variants
- Slightly more syntax errors than pure constrained generation, but much higher functional accuracy

**Why this matters:** It resolves the fundamental tension between "LLM needs freedom to reason" and "output must be structurally valid."

### Thinking Before Constraining — arxiv:2601.07525
- Unified decoding framework that schedules when to apply constraints
- Lets the model "think" unconstrained, then "write" constrained

---

## 5. Practical Implementation in vLLM

vLLM (v0.8+) supports structured outputs via three backends:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# JSON Schema
sampling = SamplingParams(
    temperature=0.7,
    guided_decoding={
        "json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age"]
        }
    }
)

# Regex
sampling = SamplingParams(
    guided_decoding={"regex": r"\d{3}-\d{2}-\d{4}"}
)

# CFG (EBNF)
sampling = SamplingParams(
    guided_decoding={"grammar": 'root ::= "yes" | "no"'}
)

# Choice
sampling = SamplingParams(
    guided_decoding={"choice": ["positive", "negative", "neutral"]}
)
```

Backend selection: XGrammar (default, fastest), Outlines, lm-format-enforcer.

---

## 6. Benchmarks

### SchemaBench (from SRL paper, 2025)
- ~40K JSON schemas of varying complexity
- Tests: nested objects, arrays, enums, conditional schemas, recursive refs
- Finding: Models fail most on `allOf`/`anyOf`/`oneOf` combinators and `$ref` recursion

### JSONSchemaBench (EPFL, 2025) — github.com/epfl-dlab/jsonschemabench
- Benchmarks constrained decoding engines (not models)
- Tests: Guidance, Outlines, XGrammar, OpenAI structured outputs
- Measures: compilation time, generation throughput, schema coverage

### GSM-Symbolic & FOLIO (used by CRANE)
- GSM-Symbolic: symbolic math requiring structured output
- FOLIO: first-order logic reasoning with formal output requirements

---

## 7. Key Takeaways for Implementation

1. **For inference-time guarantees:** Use XGrammar (via vLLM or SGLang). It's the fastest, supports JSON Schema and arbitrary CFGs, and has near-zero overhead.

2. **For training better structured models:** Use GRPO-based RL (SRL or RL-Struct approach) with a schema validator as the reward function. This is more effective than SFT alone for structural compliance.

3. **For reasoning + structure:** Use the CRANE approach — let the model reason freely, then switch to constrained decoding for the output. This preserves reasoning quality while guaranteeing format.

4. **For production APIs:** Combine constrained decoding with retry logic (Instructor pattern) for maximum reliability.

5. **The fundamental lesson:** SFT teaches models *what* to say; constrained decoding ensures *how* it's formatted; RL teaches models to *internalize* structural rules. The best systems combine all three.

---

## 8. Paper Reference List

| Paper | ID | Venue | Key Contribution |
|---|---|---|---|
| XGrammar | 2411.15100 | ICLR 2025 | PDA-based constrained decoding, 100x speedup |
| Schema RL (SRL) | 2502.18878 | ACL 2025 | SchemaBench + RL with schema validator rewards |
| CRANE | 2502.09061 | ICML 2025 | Grammar augmentation preserving reasoning ability |
| Think Inside the JSON | 2502.14905 | Preprint 2025 | GRPO for 1.5B model JSON adherence |
| RL-Struct | 2512.00319 | Preprint 2025 | Lightweight GRPO, 38% less VRAM than PPO |
| SLOT | 2505.04016 | Preprint 2025 | Model-agnostic format decoupling |
| Pre3 (DPDA) | 2506.03887 | ACL 2025 | Deterministic PDA for faster constrained decoding |
| Flexible Grammar-CD | 2502.05111 | Preprint 2025 | PDA optimizations for grammar-constrained decoding |
| Thinking Before Constraining | 2601.07525 | Preprint 2026 | Unified framework scheduling constraint application |
| SynCode | 2403.01632 | EMNLP 2024 | Incremental Earley parser for grammar constraints |
| GRAMMAR-LLM | - | ACL Findings 2025 | LLprefix + DPDA approach |
| Generating Structured Outputs (Benchmark) | 2501.10868 | Preprint 2025 | Comprehensive benchmark and studies |

## 9. Open-Source Tools

| Tool | GitHub | Approach |
|---|---|---|
| XGrammar | mlc-ai/xgrammar | PDA + adaptive mask cache |
| Outlines | dottxt-ai/outlines | FSM + HF ecosystem |
| Guidance | guidance-ai/guidance | Token healing + templates |
| lm-format-enforcer | noamgat/lm-format-enforcer | Regex/JSON Schema |
| SynCode | uiuc-focal-lab/syncode | Incremental Earley parsing |
| genlm-control | genlm/genlm-control | SMC for semantic+syntactic |
| Instructor | jxnl/instructor | Try-reject-repeat (API-level) |
| JSONSchemaBench | epfl-dlab/jsonschemabench | Engine benchmarking |
| Formatron | Dan-wanna-M/formatron | Regex/JSON Schema/CFG |
| Litelines | alonsosilvaallende/litelines | Regex/JSON Schema + FSM viz |
