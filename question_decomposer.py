"""
Question decomposer for RL curriculum generation.
Decomposes hard problems into skill trees, sub-questions, and contextual variations.
"""
import asyncio, json, os
from typing import List, Optional, Dict, Any, AsyncIterator
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
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

DIFFICULTY_GUIDE = """L1-2: Single concept, 1-2 steps | L3-4: Two concepts, 3-4 steps
L5-6: Multiple concepts, 5-7 steps | L7-8: Complex, 8-12 steps | L9-10: Competition level"""

def _skill_list() -> str:
    return "\n".join(f"  {k}: {', '.join(v)}" for k, v in SKILLS.items())

# --- Schemas ---
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
    answer: Optional[str] = None
    @field_validator("target_skill")
    @classmethod
    def _v(cls, v):
        if v not in VALID_SKILLS: raise ValueError(f"Unknown skill: {v}")
        return v

class ContextVariation(BaseModel):
    context: str
    question: str
    mapping: str
    @field_validator("context")
    @classmethod
    def _v(cls, v):
        if v not in VALID_CONTEXTS: raise ValueError(f"Unknown context: {v}")
        return v

class Decomposition(BaseModel):
    required_skills: List[Skill]
    sub_questions: List[SubQuestion] = Field(min_length=3, max_length=6)
    reasoning_steps: int = Field(ge=1)

class Breadth(BaseModel):
    core_concept: str
    abstract_structure: str
    variations: List[ContextVariation] = Field(min_length=3, max_length=5)

class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class Verification(BaseModel):
    is_correct: bool
    critique: str
    final_answer: Optional[str] = None

# --- Prompts ---
DECOMPOSE_PROMPT = f"""Create a skill-tree curriculum for this problem.
1. Identify required skills (with prerequisites as DAG)
2. Generate 3-6 sub-questions building up to original (start ~3 levels below)
3. Count minimum reasoning steps

Skills (use ONLY these):
{_skill_list()}

Difficulty: {DIFFICULTY_GUIDE}

Sub-questions must be SELF-CONTAINED. JSON only."""

BREADTH_PROMPT = f"""Extract the core concept and generate variations in different contexts.
Variations must test the SAME skill in different domains.

Contexts (use ONLY these): {', '.join(CONTEXTS)}

Each variation: same algorithm, self-contained, realistic. JSON only."""

ANSWER_PROMPT = "Solve step by step. JSON: {answer, confidence: 0-1, reasoning}"
VERIFY_PROMPT = "Verify this answer rigorously. JSON: {is_correct, critique, final_answer or null}"

# --- Decomposer ---
class Decomposer:
    def __init__(self, base_url="http://0.0.0.0:8000/v1", api_key="dummy",
                 model=None, max_concurrent=64, timeout=120.0, retries=2):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model, self.timeout, self.retries = model, timeout, retries
        self.max_concurrent = max_concurrent

    async def _init(self):
        if not self.model:
            try: self.model = (await self.client.models.list()).data[0].id
            except: self.model = "default"

    async def _call(self, system: str, user: str, schema: type):
        for i in range(self.retries + 1):
            try:
                r = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        response_format=schema, temperature=0.7),
                    timeout=self.timeout)
                return r.choices[0].message.parsed
            except Exception:
                if i == self.retries: return None
                await asyncio.sleep(0.5 * 2**i)

    async def decompose(self, q: str, sol: str = None, diff: int = None, hints: List[str] = None):
        ctx = f"## Problem\n{q}"
        if diff: ctx += f"\n\n## Difficulty: {diff}/10"
        if hints: ctx += "\n\n## Hints\n" + "\n".join(f"{i}. {h}" for i, h in enumerate(hints, 1))
        if sol: ctx += f"\n\n## Solution\n{sol}"
        return await self._call(DECOMPOSE_PROMPT, ctx, Decomposition)

    async def expand(self, q: str, sol: str = None):
        ctx = f"## Problem\n{q}" + (f"\n\n## Solution\n{sol}" if sol else "")
        return await self._call(BREADTH_PROMPT, ctx, Breadth)

    async def solve(self, q: str):
        return await self._call(ANSWER_PROMPT, q, Answer)

    async def verify(self, q: str, ans: str):
        return await self._call(VERIFY_PROMPT, f"## Question\n{q}\n\n## Answer\n{ans}", Verification)

    async def process(self, q: str, sol: str = None, diff: int = None,
                      hints: List[str] = None, breadth: bool = True) -> Dict[str, Any]:
        """Process a single question. Returns full decomposition result."""
        await self._init()
        result = {"question": q, "difficulty": diff, "hints": hints, "success": False}

        decomp = await self.decompose(q, sol, diff, hints)
        if not decomp:
            result["error"] = "decomposition_failed"
            return result

        result["skills"] = [s.model_dump() for s in decomp.required_skills]
        result["steps"] = decomp.reasoning_steps

        subs = []
        for sq in decomp.sub_questions:
            sub = {"question": sq.question, "difficulty": sq.difficulty, "skill": sq.target_skill}
            ans = await self.solve(sq.question)
            if ans:
                ver = await self.verify(sq.question, ans.answer)
                if ver and ver.is_correct:
                    sub.update(answer=ans.answer, verified=True, confidence=ans.confidence)
                elif ver and ver.final_answer:
                    sub.update(answer=ver.final_answer, verified=True, corrected=True)
                else:
                    sub.update(answer=ans.answer, verified=False,
                              critique=ver.critique if ver else "failed")
            else:
                sub.update(answer=None, verified=False, error="solve_failed")
            subs.append(sub)
        result["sub_questions"] = subs

        if breadth:
            exp = await self.expand(q, sol)
            if exp:
                result["breadth"] = {
                    "core": exp.core_concept,
                    "abstract": exp.abstract_structure,
                    "variations": [v.model_dump() for v in exp.variations]
                }

        result["success"] = True
        return result

    async def process_batch(self, items: List[Dict], pbar=None, breadth=True) -> AsyncIterator[Dict]:
        """Process batch with continuous batching."""
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
            lines.append(f"{'│' if subs or brd else ' '}   {'└' if i==len(skills)-1 else '├'}── [{s['level']:2d}] {s['name']}{pre}")

    if subs:
        lines.append(f"{'├' if brd else '└'}── DEPTH")
        for i, sq in enumerate(subs):
            v = "✓" if sq.get("verified") else "✗"
            lines.append(f"{'│' if brd else ' '}   {'└' if i==len(subs)-1 else '├'}── {v} [L{sq['difficulty']}] {sq['skill']}")
            lines.append(f"{'│' if brd else ' '}   {'  ' if i==len(subs)-1 else '│ '}   {sq['question'][:40]}...")

    if brd.get("variations"):
        lines.append(f"└── BREADTH ({brd.get('core', '?')})")
        for i, v in enumerate(brd["variations"]):
            lines.append(f"    {'└' if i==len(brd['variations'])-1 else '├'}── [{v['context']}] {v['question'][:40]}...")

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
