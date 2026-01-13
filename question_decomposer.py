"""
Question decomposer for RL curriculum generation.
Decomposes hard problems into skill trees, sub-questions, and contextual variations.
"""
import asyncio, json, os, re
from typing import List, Optional, Dict, Any, AsyncIterator
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from openai import AsyncOpenAI
from tqdm import tqdm

if "KUBERNETES_SERVICE_HOST" in os.environ:
    for v in ("NO_PROXY", "no_proxy"):
        if os.getenv(v): os.environ[v] += ",.svc.cluster.local"

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
VALID_SKILLS = {s for cat in SKILLS.values() for s in cat}
VALID_CONTEXTS = set(CONTEXTS)

def _skill_list() -> str:
    return "\n".join(f"  {k}: {', '.join(v)}" for k, v in SKILLS.items())

# --- Schemas (lenient versions for retry) ---
class Skill(BaseModel):
    name: str
    level: int = Field(ge=1, le=10)
    prerequisites: List[str] = Field(default_factory=list)
    @field_validator("name")
    @classmethod
    def _v(cls, v):
        if v not in VALID_SKILLS: raise ValueError(f"Unknown skill: {v}")
        return v

class SubQuestion(BaseModel):
    question: str
    difficulty: int = Field(ge=1, le=10)
    target_skill: str
    @field_validator("target_skill")
    @classmethod
    def _v(cls, v):
        if v not in VALID_SKILLS: raise ValueError(f"Unknown skill: {v}")
        return v

class SubQuestionStrict(SubQuestion):
    """Strict version that validates self-containment."""
    @field_validator("question")
    @classmethod
    def _check_self_contained(cls, v):
        bad = ["previous", "above", "earlier", "last question", "prior", "result of"]
        if any(b in v.lower() for b in bad):
            raise ValueError("Sub-question references other questions")
        return v

class ContextVariation(BaseModel):
    context: str
    question: str
    constraint_twist: Optional[str] = None
    mapping: Optional[str] = None  # fallback for old format
    @field_validator("context")
    @classmethod
    def _v(cls, v):
        if v not in VALID_CONTEXTS: raise ValueError(f"Unknown context: {v}")
        return v

class DecompositionStrict(BaseModel):
    """Strict schema - used first attempt."""
    estimated_difficulty: int = Field(ge=1, le=10)
    core_skills: List[Skill] = Field(min_length=2, max_length=4)
    sub_questions: List[SubQuestionStrict] = Field(min_length=3, max_length=5)
    reasoning_steps: int = Field(ge=1)
    @model_validator(mode="after")
    def _check_progression(self):
        diffs = [sq.difficulty for sq in self.sub_questions]
        if diffs != sorted(diffs):
            raise ValueError("Sub-questions must be ordered easy→hard")
        if diffs[-1] >= self.estimated_difficulty:
            raise ValueError("Final sub-question must be easier than original")
        return self

class DecompositionLenient(BaseModel):
    """Lenient schema - used on retry."""
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

BREADTH_PROMPT = f"""Generate variations of this problem in different real-world contexts.

CRITICAL: Variations must differ STRUCTURALLY, not just relabel variables.

Good variation: Changes constraints, adds/removes conditions, different goal
Bad variation: Same problem with "stocks" instead of "numbers"

CONTEXTS (use ONLY these): {', '.join(CONTEXTS)}

For each variation specify:
- context: the domain
- question: complete self-contained problem
- constraint_twist: what structural element differs (NOT just domain mapping)

Example constraint twists:
- "discrete vs continuous"
- "minimize vs maximize"
- "existence vs counting"
- "bounded vs unbounded domain"
- "additional constraint: values must be distinct"

JSON only."""

ANSWER_PROMPT = """Solve step by step, showing all work.
JSON format: {"answer": "<final answer>", "confidence": <0-1>, "reasoning": "<detailed steps>"}
If answer is numeric, just give the number. If proof, summarize conclusion."""

VERIFY_PROMPT = """Verify this solution rigorously. Check:
1. Are all steps logically valid?
2. Are edge cases handled?
3. Is the final answer correct?

JSON: {"is_correct": true/false, "critique": "<specific issues or 'correct'>", "final_answer": "<corrected answer if wrong, else null>"}"""

# --- Decomposer ---
class Decomposer:
    def __init__(self, base_url="http://0.0.0.0:8000/v1", api_key="dummy",
                 model=None, max_concurrent=64, timeout=120.0, retries=3):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model, self.timeout, self.retries = model, timeout, retries
        self.max_concurrent = max_concurrent

    async def _init(self):
        if not self.model:
            try: self.model = (await self.client.models.list()).data[0].id
            except: self.model = "default"

    async def _call(self, system: str, user: str, schema: type, temp=0.7):
        for i in range(self.retries + 1):
            try:
                r = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        response_format=schema, temperature=temp),
                    timeout=self.timeout)
                return r.choices[0].message.parsed
            except Exception as e:
                if i == self.retries: return None
                await asyncio.sleep(0.5 * 2**i)

    async def decompose(self, q: str, sol: str = None, diff: int = None, hints: List[str] = None):
        ctx = f"## Problem\n{q}"
        if diff: ctx += f"\n\n## Known Difficulty: {diff}/10"
        if hints: ctx += "\n\n## Hints\n" + "\n".join(f"{i}. {h}" for i, h in enumerate(hints, 1))
        if sol: ctx += f"\n\n## Reference Solution\n{sol}"

        # Try strict first, fall back to lenient
        result = await self._call(DECOMPOSE_PROMPT, ctx, DecompositionStrict)
        if result:
            return result, []

        result = await self._call(DECOMPOSE_PROMPT, ctx, DecompositionLenient)
        warnings = ["used_lenient_schema"] if result else []
        return result, warnings

    async def expand(self, q: str, sol: str = None):
        ctx = f"## Problem\n{q}" + (f"\n\n## Solution Approach\n{sol}" if sol else "")
        return await self._call(BREADTH_PROMPT, ctx, Breadth, temp=0.9)

    async def solve(self, q: str):
        return await self._call(ANSWER_PROMPT, q, Answer, temp=0.3)

    async def verify(self, q: str, ans: str):
        return await self._call(VERIFY_PROMPT, f"## Question\n{q}\n\n## Proposed Answer\n{ans}", Verification, temp=0.2)

    def _validate_quality(self, result: Dict) -> List[str]:
        """Check quality issues and return warnings (don't fail)."""
        warnings = []
        subs = result.get("sub_questions", [])

        # Check self-contained
        for i, sq in enumerate(subs):
            q = sq.get("question", "").lower()
            if any(b in q for b in ["previous", "above", "earlier", "prior"]):
                warnings.append(f"sub_q_{i}_not_self_contained")

        # Check ordering
        diffs = [sq.get("difficulty", 0) for sq in subs]
        if diffs and diffs != sorted(diffs):
            warnings.append("sub_questions_not_ordered")

        # Check skills count
        skills = result.get("skills", [])
        if len(skills) > 5:
            warnings.append("too_many_skills")
        if len(skills) == 0:
            warnings.append("no_skills")

        return warnings

    async def process(self, q: str, sol: str = None, diff: int = None,
                      hints: List[str] = None, breadth: bool = True) -> Dict[str, Any]:
        await self._init()
        result = {"question": q, "success": False, "warnings": []}

        # Decompose with fallback
        decomp_result = await self.decompose(q, sol, diff, hints)
        if decomp_result is None or decomp_result[0] is None:
            result["error"] = "decomposition_failed"
            return result

        decomp, warnings = decomp_result
        result["warnings"].extend(warnings)
        result["difficulty"] = decomp.estimated_difficulty
        result["skills"] = [s.model_dump() for s in decomp.core_skills] if decomp.core_skills else []
        result["steps"] = decomp.reasoning_steps

        # Process sub-questions with solve+verify (keep partial results)
        subs = []
        for sq in decomp.sub_questions:
            sub = {"question": sq.question, "difficulty": sq.difficulty, "skill": sq.target_skill}
            try:
                ans = await self.solve(sq.question)
                if ans:
                    ver = await self.verify(sq.question, ans.answer)
                    if ver and ver.is_correct:
                        sub.update(answer=ans.answer, verified=True, confidence=ans.confidence)
                    elif ver and ver.final_answer:
                        sub.update(answer=ver.final_answer, verified=True, corrected=True)
                    else:
                        sub.update(answer=ans.answer, verified=False,
                                  critique=ver.critique if ver else "verification_failed")
                else:
                    sub.update(answer=None, verified=False, error="solve_failed")
            except Exception as e:
                sub.update(answer=None, verified=False, error=str(e)[:100])
            subs.append(sub)
        result["sub_questions"] = subs

        # Breadth expansion (don't fail if this fails)
        if breadth:
            try:
                exp = await self.expand(q, sol)
                if exp and exp.variations:
                    result["breadth"] = {
                        "core": exp.core_concept,
                        "variations": [v.model_dump() for v in exp.variations]
                    }
                else:
                    result["warnings"].append("breadth_failed")
            except Exception:
                result["warnings"].append("breadth_failed")

        # Quality validation (warnings only)
        result["warnings"].extend(self._validate_quality(result))

        result["success"] = True
        return result

    async def process_batch(self, items: List[Dict], pbar=None, breadth=True) -> AsyncIterator[Dict]:
        await self._init()
        n, idx, pending = len(items), 0, {}

        def submit(i):
            it = items[i]
            task = asyncio.create_task(self.process(
                it["question"], it.get("solution"), it.get("difficulty"),
                it.get("hints"), breadth))
            pending[task] = i

        while idx < n and len(pending) < self.max_concurrent:
            submit(idx); idx += 1

        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                i = pending.pop(t)
                r = t.result()
                r["index"] = i
                if pbar: pbar.update(1)
                yield r
                if idx < n: submit(idx); idx += 1


# --- Simple API ---
async def decompose_question(question: str, solution: str = None,
                             base_url: str = "http://0.0.0.0:8000/v1",
                             breadth: bool = True) -> Dict[str, Any]:
    """
    Decompose a single question into skills, sub-questions, and variations.

    In Jupyter: result = await decompose_question("...")
    In scripts: result = asyncio.run(decompose_question("..."))
    """
    d = Decomposer(base_url=base_url)
    return await d.process(question, solution, breadth=breadth)


# --- Visualization ---
def render(r: Dict) -> str:
    if not r.get("success"): return f"[ERROR] {r.get('error')}"

    lines = [f"[L{r.get('difficulty', '?')}] {r['question'][:60]}...", f"│  steps={r.get('steps', '?')}"]

    skills = r.get("skills", [])
    subs = r.get("sub_questions", [])
    brd = r.get("breadth", {})

    if skills:
        lines.append("├── SKILLS")
        for i, s in enumerate(skills):
            pre = " ← " + ",".join(s["prerequisites"]) if s.get("prerequisites") else ""
            c = "└" if i == len(skills)-1 else "├"
            lines.append(f"{'│' if subs or brd else ' '}   {c}── [{s['level']:2d}] {s['name']}{pre}")

    if subs:
        lines.append(f"{'├' if brd else '└'}── CURRICULUM")
        for i, sq in enumerate(subs):
            v = "✓" if sq.get("verified") else "✗"
            c = "└" if i == len(subs)-1 else "├"
            lines.append(f"{'│' if brd else ' '}   {c}── {v} [L{sq['difficulty']}] {sq['skill']}")
            lines.append(f"{'│' if brd else ' '}   {'  ' if i==len(subs)-1 else '│ '}   {sq['question'][:45]}...")

    if brd.get("variations"):
        lines.append(f"└── VARIATIONS ({brd.get('core', '?')})")
        for i, v in enumerate(brd["variations"]):
            c = "└" if i == len(brd["variations"])-1 else "├"
            twist = v.get("constraint_twist", "")[:25]
            lines.append(f"    {c}── [{v['context']}] {twist}")
            lines.append(f"    {'  ' if i==len(brd['variations'])-1 else '│ '}   {v['question'][:40]}...")

    return "\n".join(lines)


def aggregate_skills(results: List[Dict]) -> Dict:
    counts, levels, prereqs = {}, {}, {}
    for r in results:
        if not r.get("success"): continue
        for s in r.get("skills", []):
            n = s["name"]
            counts[n] = counts.get(n, 0) + 1
            levels.setdefault(n, []).append(s["level"])
            prereqs.setdefault(n, set()).update(s.get("prerequisites", []))

    tree = {n: {"count": counts[n], "avg": sum(levels[n])/len(levels[n]),
                "max": max(levels[n]), "prereqs": list(prereqs[n])}
            for n in counts}
    return {"skills": dict(sorted(tree.items(), key=lambda x: -x[1]["count"])),
            "total": len(results)}


# --- CLI ---
def _extract(row):
    p = row.get("prompt")
    q = p[0].get("content", "") if isinstance(p, list) and p and isinstance(p[0], dict) else str(p or "")
    rm = row.get("reward_model", {})
    sol = rm.get("ground_truth") if isinstance(rm, dict) else row.get("solution")
    hints = row.get("hints")
    if isinstance(hints, str):
        try: hints = json.loads(hints)
        except: hints = None
    diff = row.get("estimated_difficulty")
    return {"question": q, "solution": sol, "hints": hints, "difficulty": int(diff) if diff else None}


async def _run(args):
    import pandas as pd

    if args.visualize:
        data = ([json.loads(l) for l in open(args.input) if l.strip()]
                if args.input.endswith(".jsonl")
                else [pd.read_parquet(args.input).iloc[i].to_dict()
                      for i in range(len(pd.read_parquet(args.input)))])
        for r in [d for d in data if d.get("success")][:args.visualize]:
            print(render(r), "\n")
        return

    df = pd.read_parquet(args.input)
    items = [_extract(df.iloc[i].to_dict()) for i in range(len(df))]
    if args.min_diff:
        items = [it for it in items if (it.get("difficulty") or 0) >= args.min_diff]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    d = Decomposer(args.base_url, args.api_key, args.model, args.concurrent, args.timeout)
    results, chunk, cidx = {}, [], 0

    pbar = tqdm(total=len(items), desc="Processing")
    async for r in d.process_batch(items, pbar, not args.no_breadth):
        results[r["index"]] = r
        chunk.append(r)
        if len(chunk) >= args.chunk:
            with open(out / f"chunk_{cidx:04d}.jsonl", "w") as f:
                for c in chunk: f.write(json.dumps(c, default=str) + "\n")
            chunk, cidx = [], cidx + 1
    pbar.close()

    if chunk:
        with open(out / f"chunk_{cidx:04d}.jsonl", "w") as f:
            for c in chunk: f.write(json.dumps(c, default=str) + "\n")

    tree = aggregate_skills(list(results.values()))
    with open(out / "skills.json", "w") as f: json.dump(tree, f, indent=2)

    ok = sum(1 for r in results.values() if r.get("success"))
    print(f"\nDone: {ok}/{len(items)} successful")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default="./out")
    p.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--model", default=None)
    p.add_argument("--concurrent", type=int, default=32)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--min-diff", type=int, default=None)
    p.add_argument("--no-breadth", action="store_true")
    p.add_argument("--visualize", type=int, default=None)
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
