"""
Question decomposer for RL curriculum generation.
Decomposes hard problems into skill trees, sub-questions, and contextual variations.
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator

from pydantic import BaseModel, Field, field_validator, model_validator
from openai import AsyncOpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if "KUBERNETES_SERVICE_HOST" in os.environ:
    for v in ("NO_PROXY", "no_proxy"):
        if os.getenv(v):
            os.environ[v] += ",.svc.cluster.local"


# --- Constants ---
class Error(str, Enum):
    DECOMPOSITION_FAILED = "decomposition_failed"
    SOLVE_FAILED = "solve_failed"
    VERIFY_FAILED = "verification_failed"
    BREADTH_FAILED = "breadth_failed"


class Warning(str, Enum):
    LENIENT_SCHEMA = "used_lenient_schema"
    NOT_SELF_CONTAINED = "not_self_contained"
    NOT_ORDERED = "sub_questions_not_ordered"
    TOO_MANY_SKILLS = "too_many_skills"
    NO_SKILLS = "no_skills"


SELF_REFERENCE_PATTERNS = ["previous", "above", "earlier", "last question", "prior", "result of"]


# --- Taxonomy ---
SKILLS = {
    "foundations": ["arithmetic", "fractions", "percentages", "ratios", "basic_algebra",
                    "linear_equations", "inequalities", "basic_geometry", "coordinate_geometry",
                    "trigonometry", "set_theory", "logic", "boolean_algebra"],
    "discrete": ["combinatorics", "permutations", "combinations", "pigeonhole", "graph_theory",
                 "trees", "graph_traversal", "shortest_path", "number_theory", "divisibility",
                 "primes", "modular_arithmetic", "gcd_lcm", "recurrences", "generating_functions"],
    "algorithms": ["sorting", "searching", "binary_search", "dynamic_programming", "memoization",
                   "greedy", "divide_and_conquer", "backtracking", "branch_and_bound",
                   "string_matching", "hashing"],
    "optimization": ["linear_programming", "convex_optimization", "gradient_descent",
                     "constraint_satisfaction", "game_theory", "minimax", "nash_equilibrium"],
    "probability": ["counting", "probability_basics", "conditional_probability", "bayes",
                    "expected_value", "variance", "distributions", "markov_chains", "random_walks"],
    "reasoning": ["case_analysis", "proof_by_contradiction", "induction", "pattern_recognition",
                  "abstraction", "decomposition", "spatial_reasoning", "temporal_reasoning"],
}
CONTEXTS = ["finance", "physics", "biology", "chemistry", "computer_science",
            "game_theory", "logistics", "social_networks", "sports", "economics"]
VALID_SKILLS = frozenset(s for cat in SKILLS.values() for s in cat)
VALID_CONTEXTS = frozenset(CONTEXTS)


def _skill_list() -> str:
    return "\n".join(f"  {k}: {', '.join(v)}" for k, v in SKILLS.items())


# --- Schemas ---
class Skill(BaseModel):
    name: str
    level: int = Field(ge=1, le=10)
    prerequisites: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v not in VALID_SKILLS:
            raise ValueError(f"Unknown skill: {v}")
        return v


class SubQuestion(BaseModel):
    question: str
    difficulty: int = Field(ge=1, le=10)
    target_skill: str

    @field_validator("target_skill")
    @classmethod
    def validate_skill(cls, v: str) -> str:
        if v not in VALID_SKILLS:
            raise ValueError(f"Unknown skill: {v}")
        return v


class SubQuestionStrict(SubQuestion):
    @field_validator("question")
    @classmethod
    def validate_self_contained(cls, v: str) -> str:
        v_lower = v.lower()
        if any(p in v_lower for p in SELF_REFERENCE_PATTERNS):
            raise ValueError("Sub-question references other questions")
        return v


class ContextVariation(BaseModel):
    context: str
    question: str
    constraint_twist: Optional[str] = None
    mapping: Optional[str] = None

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: str) -> str:
        if v not in VALID_CONTEXTS:
            raise ValueError(f"Unknown context: {v}")
        return v


class DecompositionStrict(BaseModel):
    estimated_difficulty: int = Field(ge=1, le=10)
    core_skills: List[Skill] = Field(min_length=2, max_length=4)
    sub_questions: List[SubQuestionStrict] = Field(min_length=3, max_length=5)
    reasoning_steps: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_progression(self):
        diffs = [sq.difficulty for sq in self.sub_questions]
        if diffs != sorted(diffs):
            raise ValueError("Sub-questions must be ordered easy→hard")
        if diffs[-1] >= self.estimated_difficulty:
            raise ValueError("Final sub-question must be easier than original")
        return self


class DecompositionLenient(BaseModel):
    estimated_difficulty: int = Field(ge=1, le=10, default=5)
    core_skills: List[Skill] = Field(default_factory=list)
    sub_questions: List[SubQuestion] = Field(default_factory=list)
    reasoning_steps: int = Field(ge=1, default=3)


class Breadth(BaseModel):
    core_concept: str
    variations: List[ContextVariation] = Field(default_factory=list)


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class Verification(BaseModel):
    is_correct: bool
    critique: str
    final_answer: Optional[str] = None


# --- Prompts ---
DECOMPOSE_PROMPT = f"""Analyze this problem and create a learning curriculum.

TASK:
1. Estimate original difficulty (1-10 scale below)
2. Identify 2-4 CORE skills (not generic ones like "logic" or "basic_algebra")
3. Create 3-5 sub-questions as stepping stones (each MUST be self-contained)
4. Order sub-questions strictly easy→hard

DIFFICULTY SCALE:
- L1-2: Direct formula application, 1-2 steps
- L3-4: Two concepts combined, requires some insight
- L5-6: Multiple concepts, non-obvious approach needed
- L7-8: Complex integration, key insight required, 8+ steps
- L9-10: Competition/olympiad level, creative leaps needed

SKILLS (use ONLY from this list, pick 2-4 most specific):
{_skill_list()}

CRITICAL RULES:
- Each sub-question MUST be solvable WITHOUT seeing other sub-questions
- NO phrases like "using the previous result", "from above", "as shown earlier"
- Start sub-questions ~3 levels below original difficulty
- Put simple verification cases FIRST (e.g., "check n=1,2,3" before general proof)
- Avoid generic skills like "logic", "basic_algebra" unless truly core

BAD example: "Using the result from Q2, prove..." (references Q2)
GOOD example: "Prove that for any odd prime p, p² ≡ 1 (mod 8)" (self-contained)

JSON only."""

BREADTH_PROMPT = f"""Generate STRUCTURALLY DIFFERENT problems that test the same core skill.

CRITICAL: DO NOT just relabel variables or change the domain. The mathematical structure must CHANGE.

TYPES OF STRUCTURAL VARIATION (pick different ones for each):
1. TRANSFORM: Change the formula (p²+2 → p²-2, 2p+1, p³+4, p+p², etc.)
2. GOAL: Change what's asked (find all → count how many → find smallest → prove none exist)
3. DIRECTION: Reverse the problem (given output, find input)
4. CONSTRAINT: Add/remove conditions (consecutive, bounded, distinct, coprime)
5. GENERALIZE: Make it parametric (for which k does p²+k work?)
6. COMPOSITION: Combine with another operation (apply twice, alternate)

BAD VARIATION (just relabeling):
  Original: "Find primes p where p²+2 is prime"
  Bad: "Find atomic numbers Z where Z²+2 is prime" ← SAME FORMULA

GOOD VARIATIONS:
  - "Find primes p where p²-2 is prime" (different formula)
  - "Count primes p<1000 where p²+2 is prime" (count vs list)
  - "Find the smallest k such that 3²+k is prime" (reversed)
  - "Find consecutive primes p,q where p²+q² is prime" (composition)
  - "Prove: for any prime p>3, p²+2 is composite" (prove vs find)

CONTEXTS (use for realistic framing, NOT the core structure): {', '.join(CONTEXTS)}

For each variation:
- context: domain for realistic framing
- question: complete problem with DIFFERENT STRUCTURE
- constraint_twist: which structural change type you used

JSON only."""

ANSWER_PROMPT = """Solve step by step, showing all work.
JSON format: {"answer": "<final answer>", "confidence": <0-1>, "reasoning": "<detailed steps>"}
If answer is numeric, just give the number. If proof, summarize conclusion."""

VERIFY_PROMPT = """Verify this solution rigorously. Check:
1. Are all steps logically valid?
2. Are edge cases handled?
3. Is the final answer correct?

JSON: {"is_correct": true/false, "critique": "<specific issues or 'correct'>", "final_answer": "<corrected answer if wrong, else null>"}"""


# --- Config ---
@dataclass
class DecomposerConfig:
    base_url: str = "http://0.0.0.0:8000/v1"
    api_key: str = "dummy"
    model: Optional[str] = None
    max_concurrent: int = 64
    timeout: float = 120.0
    retries: int = 3
    temp_decompose: float = 0.7
    temp_breadth: float = 0.9
    temp_solve: float = 0.3
    temp_verify: float = 0.2


# --- Decomposer ---
class Decomposer:
    def __init__(self, config: Optional[DecomposerConfig] = None, **kwargs):
        if config:
            self.cfg = config
        else:
            self.cfg = DecomposerConfig(**kwargs)
        self.client = AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.timeout
        )
        self._model_resolved = False

    async def _ensure_model(self):
        """Resolve model name once."""
        if self._model_resolved:
            return
        if not self.cfg.model:
            try:
                models = await self.client.models.list()
                self.cfg.model = models.data[0].id
                log.info(f"Using model: {self.cfg.model}")
            except Exception as e:
                log.warning(f"Failed to list models: {e}, using 'default'")
                self.cfg.model = "default"
        self._model_resolved = True

    async def _call(self, system: str, user: str, schema: type, temp: float) -> Optional[Any]:
        """Make LLM call with retries."""
        for attempt in range(self.cfg.retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.cfg.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ],
                        response_format=schema,
                        temperature=temp
                    ),
                    timeout=self.cfg.timeout
                )
                return resp.choices[0].message.parsed
            except asyncio.TimeoutError:
                log.warning(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                log.warning(f"Call failed attempt {attempt + 1}: {type(e).__name__}: {e}")
            if attempt < self.cfg.retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
        return None

    async def decompose(self, question: str, solution: str = None,
                        difficulty: int = None, hints: List[str] = None) -> tuple:
        """Decompose with strict→lenient fallback."""
        ctx = f"## Problem\n{question}"
        if difficulty:
            ctx += f"\n\n## Known Difficulty: {difficulty}/10"
        if hints:
            ctx += "\n\n## Hints\n" + "\n".join(f"{i}. {h}" for i, h in enumerate(hints, 1))
        if solution:
            ctx += f"\n\n## Reference Solution\n{solution}"

        result = await self._call(DECOMPOSE_PROMPT, ctx, DecompositionStrict, self.cfg.temp_decompose)
        if result:
            return result, []

        log.info("Strict decomposition failed, trying lenient")
        result = await self._call(DECOMPOSE_PROMPT, ctx, DecompositionLenient, self.cfg.temp_decompose)
        return result, [Warning.LENIENT_SCHEMA] if result else []

    async def expand(self, question: str, solution: str = None) -> Optional[Breadth]:
        """Generate breadth variations."""
        ctx = f"## Problem\n{question}"
        if solution:
            ctx += f"\n\n## Solution Approach\n{solution}"
        return await self._call(BREADTH_PROMPT, ctx, Breadth, self.cfg.temp_breadth)

    async def solve(self, question: str) -> Optional[Answer]:
        """Solve a question."""
        return await self._call(ANSWER_PROMPT, question, Answer, self.cfg.temp_solve)

    async def verify(self, question: str, answer: str) -> Optional[Verification]:
        """Verify an answer."""
        ctx = f"## Question\n{question}\n\n## Proposed Answer\n{answer}"
        return await self._call(VERIFY_PROMPT, ctx, Verification, self.cfg.temp_verify)

    async def _solve_and_verify(self, sq: SubQuestion) -> Dict[str, Any]:
        """Solve and verify a single sub-question."""
        result = {"question": sq.question, "difficulty": sq.difficulty, "skill": sq.target_skill}

        ans = await self.solve(sq.question)
        if not ans:
            result.update(answer=None, verified=False, error=Error.SOLVE_FAILED.value)
            return result

        ver = await self.verify(sq.question, ans.answer)
        if ver and ver.is_correct:
            result.update(answer=ans.answer, verified=True, confidence=ans.confidence)
        elif ver and ver.final_answer:
            result.update(answer=ver.final_answer, verified=True, corrected=True)
        else:
            result.update(
                answer=ans.answer,
                verified=False,
                critique=ver.critique if ver else Error.VERIFY_FAILED.value
            )
        return result

    async def _solve_all_subquestions(self, sub_questions: List[SubQuestion]) -> List[Dict]:
        """Solve all sub-questions in parallel."""
        tasks = [self._solve_and_verify(sq) for sq in sub_questions]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _collect_warnings(self, result: Dict) -> List[str]:
        """Collect quality warnings."""
        warnings = []
        subs = result.get("sub_questions", [])

        for i, sq in enumerate(subs):
            q_lower = sq.get("question", "").lower()
            if any(p in q_lower for p in SELF_REFERENCE_PATTERNS):
                warnings.append(f"{Warning.NOT_SELF_CONTAINED.value}_{i}")

        diffs = [sq.get("difficulty", 0) for sq in subs]
        if diffs and diffs != sorted(diffs):
            warnings.append(Warning.NOT_ORDERED.value)

        n_skills = len(result.get("skills", []))
        if n_skills > 5:
            warnings.append(Warning.TOO_MANY_SKILLS.value)
        elif n_skills == 0:
            warnings.append(Warning.NO_SKILLS.value)

        return warnings

    async def process(self, question: str, solution: str = None, difficulty: int = None,
                      hints: List[str] = None, breadth: bool = True) -> Dict[str, Any]:
        """Process a single question end-to-end."""
        await self._ensure_model()
        result = {"question": question, "success": False, "warnings": []}

        # Decompose
        decomp, warnings = await self.decompose(question, solution, difficulty, hints)
        if not decomp:
            result["error"] = Error.DECOMPOSITION_FAILED.value
            return result

        result["warnings"].extend(w.value if isinstance(w, Warning) else w for w in warnings)
        result["difficulty"] = decomp.estimated_difficulty
        result["skills"] = [s.model_dump() for s in decomp.core_skills] if decomp.core_skills else []
        result["steps"] = decomp.reasoning_steps

        # Solve sub-questions in parallel
        if decomp.sub_questions:
            solved = await self._solve_all_subquestions(decomp.sub_questions)
            result["sub_questions"] = [
                r if isinstance(r, dict) else {"error": str(r)[:100]}
                for r in solved
            ]
        else:
            result["sub_questions"] = []

        # Breadth expansion
        if breadth:
            exp = await self.expand(question, solution)
            if exp and exp.variations:
                result["breadth"] = {
                    "core": exp.core_concept,
                    "variations": [v.model_dump() for v in exp.variations]
                }
            else:
                result["warnings"].append(Warning.BREADTH_FAILED.value)

        result["warnings"].extend(self._collect_warnings(result))
        result["success"] = True
        return result

    async def process_batch(self, items: List[Dict], pbar=None,
                            breadth: bool = True) -> AsyncIterator[Dict]:
        """Process batch with continuous batching."""
        await self._ensure_model()
        n, idx, pending = len(items), 0, {}

        def submit(i: int):
            it = items[i]
            task = asyncio.create_task(self.process(
                it["question"], it.get("solution"),
                it.get("difficulty"), it.get("hints"), breadth
            ))
            pending[task] = i

        # Initial fill
        while idx < n and len(pending) < self.cfg.max_concurrent:
            submit(idx)
            idx += 1

        # Process as completed
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                i = pending.pop(task)
                try:
                    r = task.result()
                except Exception as e:
                    log.error(f"Task {i} failed: {e}")
                    r = {"index": i, "success": False, "error": str(e)[:100]}
                r["index"] = i
                if pbar:
                    pbar.update(1)
                yield r
                if idx < n:
                    submit(idx)
                    idx += 1


# --- Simple API ---
async def decompose_question(question: str, solution: str = None,
                             base_url: str = "http://0.0.0.0:8000/v1",
                             breadth: bool = True) -> Dict[str, Any]:
    """
    Decompose a single question into skills, sub-questions, and variations.

    Usage:
        In Jupyter: result = await decompose_question("...")
        In scripts: result = asyncio.run(decompose_question("..."))

    Returns:
        dict with: question, difficulty, skills, steps, sub_questions, breadth, warnings, success
    """
    d = Decomposer(base_url=base_url)
    return await d.process(question, solution, breadth=breadth)


# --- Visualization ---
def render(result: Dict) -> str:
    """Render decomposition as ASCII tree."""
    if not result.get("success"):
        return f"[ERROR] {result.get('error')}"

    q_short = result["question"][:60]
    lines = [
        f"[L{result.get('difficulty', '?')}] {q_short}...",
        f"│  steps={result.get('steps', '?')}"
    ]

    skills = result.get("skills", [])
    subs = result.get("sub_questions", [])
    brd = result.get("breadth", {})
    has_more = bool(subs or brd)

    if skills:
        lines.append("├── SKILLS")
        for i, s in enumerate(skills):
            pre = f" ← {','.join(s['prerequisites'])}" if s.get("prerequisites") else ""
            connector = "└" if i == len(skills) - 1 else "├"
            prefix = "│" if has_more else " "
            lines.append(f"{prefix}   {connector}── [{s['level']:2d}] {s['name']}{pre}")

    if subs:
        has_breadth = bool(brd.get("variations"))
        lines.append(f"{'├' if has_breadth else '└'}── CURRICULUM")
        for i, sq in enumerate(subs):
            v = "✓" if sq.get("verified") else "✗"
            connector = "└" if i == len(subs) - 1 else "├"
            prefix = "│" if has_breadth else " "
            lines.append(f"{prefix}   {connector}── {v} [L{sq.get('difficulty', '?')}] {sq.get('skill', '?')}")
            sub_prefix = "  " if i == len(subs) - 1 else "│ "
            lines.append(f"{prefix}   {sub_prefix}   {sq.get('question', '')[:45]}...")

    if brd.get("variations"):
        lines.append(f"└── VARIATIONS ({brd.get('core', '?')[:30]})")
        for i, v in enumerate(brd["variations"]):
            connector = "└" if i == len(brd["variations"]) - 1 else "├"
            twist = (v.get("constraint_twist") or "")[:25]
            lines.append(f"    {connector}── [{v['context']}] {twist}")
            sub_prefix = "  " if i == len(brd["variations"]) - 1 else "│ "
            lines.append(f"    {sub_prefix}   {v['question'][:40]}...")

    return "\n".join(lines)


def aggregate_skills(results: List[Dict]) -> Dict:
    """Aggregate skills across multiple results."""
    if not results:
        return {"skills": {}, "total": 0}

    counts: Dict[str, int] = {}
    levels: Dict[str, List[int]] = {}
    prereqs: Dict[str, set] = {}

    for r in results:
        if not r.get("success"):
            continue
        for s in r.get("skills", []):
            name = s["name"]
            counts[name] = counts.get(name, 0) + 1
            levels.setdefault(name, []).append(s["level"])
            prereqs.setdefault(name, set()).update(s.get("prerequisites", []))

    tree = {
        name: {
            "count": counts[name],
            "avg": sum(levels[name]) / len(levels[name]),
            "max": max(levels[name]),
            "prereqs": list(prereqs[name])
        }
        for name in counts
    }

    return {
        "skills": dict(sorted(tree.items(), key=lambda x: -x[1]["count"])),
        "total": len(results)
    }


# --- CLI Helpers ---
def extract_from_row(row: Dict) -> Dict:
    """Extract fields from a parquet/JSONL row."""
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        question = prompt[0].get("content", "")
    else:
        question = str(prompt or "")

    rm = row.get("reward_model", {})
    solution = rm.get("ground_truth") if isinstance(rm, dict) else row.get("solution")

    hints = row.get("hints")
    if isinstance(hints, str):
        try:
            hints = json.loads(hints)
        except json.JSONDecodeError:
            hints = None

    diff = row.get("estimated_difficulty")
    difficulty = int(diff) if diff else None

    return {"question": question, "solution": solution, "hints": hints, "difficulty": difficulty}


async def run_cli(args):
    """Main CLI entry point."""
    import pandas as pd

    # Visualize mode
    if args.visualize:
        if args.input.endswith(".jsonl"):
            with open(args.input) as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            df = pd.read_parquet(args.input)
            data = [df.iloc[i].to_dict() for i in range(len(df))]

        successful = [d for d in data if d.get("success")]
        for r in successful[:args.visualize]:
            print(render(r), "\n")
        return

    # Process mode
    df = pd.read_parquet(args.input)
    items = [extract_from_row(df.iloc[i].to_dict()) for i in range(len(df))]

    if args.min_diff:
        items = [it for it in items if (it.get("difficulty") or 0) >= args.min_diff]
        log.info(f"Filtered to {len(items)} items with difficulty >= {args.min_diff}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    config = DecomposerConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.concurrent,
        timeout=args.timeout
    )
    decomposer = Decomposer(config)

    results = {}
    chunk, chunk_idx = [], 0

    def save_chunk():
        nonlocal chunk, chunk_idx
        if not chunk:
            return
        path = out / f"chunk_{chunk_idx:04d}.jsonl"
        with open(path, "w") as f:
            for c in chunk:
                f.write(json.dumps(c, default=str) + "\n")
        log.info(f"Saved {path}")
        chunk = []
        chunk_idx += 1

    pbar = tqdm(total=len(items), desc="Processing")
    async for r in decomposer.process_batch(items, pbar, not args.no_breadth):
        results[r["index"]] = r
        chunk.append(r)
        if len(chunk) >= args.chunk:
            save_chunk()
    pbar.close()
    save_chunk()

    # Save skill tree
    tree = aggregate_skills(list(results.values()))
    tree_path = out / "skills.json"
    with open(tree_path, "w") as f:
        json.dump(tree, f, indent=2)

    ok = sum(1 for r in results.values() if r.get("success"))
    log.info(f"Done: {ok}/{len(items)} successful")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Question decomposer for RL curriculum generation")
    p.add_argument("input", help="Input parquet or JSONL file")
    p.add_argument("output", nargs="?", default="./out", help="Output directory")
    p.add_argument("--base-url", default="http://0.0.0.0:8000/v1", help="vLLM server URL")
    p.add_argument("--api-key", default="dummy", help="API key")
    p.add_argument("--model", default=None, help="Model name (auto-detected if not set)")
    p.add_argument("--concurrent", type=int, default=32, help="Max concurrent requests")
    p.add_argument("--chunk", type=int, default=500, help="Chunk size for saving")
    p.add_argument("--timeout", type=float, default=180, help="Request timeout in seconds")
    p.add_argument("--min-diff", type=int, default=None, help="Filter by minimum difficulty")
    p.add_argument("--no-breadth", action="store_true", help="Skip breadth expansion")
    p.add_argument("--visualize", type=int, default=None, metavar="N", help="Visualize N samples")
    asyncio.run(run_cli(p.parse_args()))


if __name__ == "__main__":
    main()
