"""
Skill-Tree Question Decomposer for RL Curriculum Generation.

Takes hard questions (zero/low pass rate) and decomposes them into:
1. Required skills (skill tree nodes)
2. Prerequisite sub-questions at lower difficulty
3. Verified Q&A pairs via debate

Reuses hints + difficulty from hint_generator.py if available.

Usage: python question_decomposer.py input.parquet output_dir/ --base-url http://0.0.0.0:8000/v1
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, AsyncIterator
from pathlib import Path

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm import tqdm

if "KUBERNETES_SERVICE_HOST" in os.environ:
    for var in ("NO_PROXY", "no_proxy"):
        if os.getenv(var):
            os.environ[var] += ",.svc.cluster.local"


# =============================================================================
# Schemas
# =============================================================================

class Skill(BaseModel):
    name: str = Field(description="Skill name, e.g., 'modular_arithmetic', 'dynamic_programming'")
    level: int = Field(ge=1, le=10, description="Mastery level required (1=basic, 10=expert)")
    prerequisites: List[str] = Field(default_factory=list, description="Skill names that must be unlocked first")


class SubQuestion(BaseModel):
    question: str
    difficulty: int = Field(ge=1, le=10)
    target_skill: str = Field(description="Which skill this sub-question trains")
    answer: Optional[str] = None


class QuestionDecomposition(BaseModel):
    """Output of Pass 1: Decompose hard question into skill tree + curriculum."""
    required_skills: List[Skill] = Field(description="Skills needed, with prerequisites")
    sub_questions: List[SubQuestion] = Field(
        min_length=3, max_length=6,
        description="Curriculum from easiest to hardest, building up to original"
    )
    reasoning_steps: int = Field(ge=1, description="Min reasoning steps to solve original")


class VerifiedAnswer(BaseModel):
    """Output of Pass 2: Answer with verification."""
    answer: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class DebateVerification(BaseModel):
    """Output of debate verification."""
    is_correct: bool
    critique: str
    final_answer: Optional[str] = Field(None, description="Corrected answer if original was wrong")


# =============================================================================
# Prompts
# =============================================================================

DECOMPOSE_SYSTEM = '''You are an expert at creating learning curricula from hard problems.

Given a problem with its difficulty and progressive hints, create a skill-tree curriculum:
1. Identify required skills (with prerequisites forming a DAG)
2. Generate 3-6 sub-questions building up to the original
   - Start ~3 difficulty levels below
   - Each targets one skill
   - Use the hints to inform what skills/concepts are needed
3. Count minimum reasoning steps

Skill naming: snake_case like "modular_arithmetic", "graph_traversal", "dynamic_programming"
Sub-questions must be SELF-CONTAINED.

JSON only.'''

DECOMPOSE_SYSTEM_NO_HINTS = '''You are an expert at creating learning curricula from hard problems.

Given a problem, create a skill-tree curriculum:
1. Estimate difficulty (1-10) and identify required skills (with prerequisites)
2. Generate 3-6 sub-questions building up to the original
3. Count minimum reasoning steps

Skill naming: snake_case. Sub-questions must be SELF-CONTAINED. JSON only.'''

ANSWER_SYSTEM = '''Solve step by step. Be precise.
JSON: {answer: <final>, confidence: 0-1, reasoning: <steps>}'''

DEBATE_SYSTEM = '''Verify this answer. Be rigorous.
JSON: {is_correct: bool, critique: <explanation>, final_answer: <corrected or null>}'''


# =============================================================================
# Decomposer
# =============================================================================

class QuestionDecomposer:
    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000/v1",
        api_key: str = "dummy",
        model: Optional[str] = None,
        max_concurrent: int = 64,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    async def _init_model(self):
        if not self.model:
            try:
                models = await self.client.models.list()
                self.model = models.data[0].id
            except Exception:
                self.model = "unknown"

    async def _call(self, system: str, user: str, response_format: type) -> Optional[Any]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        response_format=response_format,
                        temperature=0.7,
                    ),
                    timeout=self.timeout,
                )
                return resp.choices[0].message.parsed
            except Exception as e:
                if attempt == self.max_retries:
                    return None
                await asyncio.sleep(0.5 * (2 ** attempt))
        return None

    # -------------------------------------------------------------------------
    # Pass 1: Decompose
    # -------------------------------------------------------------------------
    async def decompose(
        self,
        question: str,
        solution: Optional[str] = None,
        difficulty: Optional[int] = None,
        hints: Optional[List[str]] = None,
    ) -> Optional[QuestionDecomposition]:
        # Build context with available info
        context = f"## Problem\n{question}"
        if difficulty:
            context += f"\n\n## Difficulty: {difficulty}/10"
        if hints:
            context += "\n\n## Progressive Hints (easy→hard)"
            for i, h in enumerate(hints, 1):
                context += f"\n{i}. {h}"
        if solution:
            context += f"\n\n## Reference Solution\n{solution}"

        system = DECOMPOSE_SYSTEM if hints else DECOMPOSE_SYSTEM_NO_HINTS
        return await self._call(system, context, QuestionDecomposition)

    # -------------------------------------------------------------------------
    # Pass 2: Generate answers for sub-questions
    # -------------------------------------------------------------------------
    async def answer(self, question: str) -> Optional[VerifiedAnswer]:
        return await self._call(ANSWER_SYSTEM, question, VerifiedAnswer)

    # -------------------------------------------------------------------------
    # Pass 3: Debate verification
    # -------------------------------------------------------------------------
    async def verify(self, question: str, proposed_answer: str) -> Optional[DebateVerification]:
        user = f"## Question\n{question}\n\n## Proposed Answer\n{proposed_answer}"
        return await self._call(DEBATE_SYSTEM, user, DebateVerification)

    # -------------------------------------------------------------------------
    # Full pipeline for one question
    # -------------------------------------------------------------------------
    async def process_one(
        self,
        question: str,
        solution: Optional[str],
        idx: int,
        difficulty: Optional[int] = None,
        hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        result = {
            "index": idx,
            "original_question": question,
            "original_difficulty": difficulty,
            "original_hints": hints,
            "success": False,
        }

        # Pass 1: Decompose (reusing hints + difficulty if available)
        decomp = await self.decompose(question, solution, difficulty, hints)
        if not decomp:
            result["error"] = "decomposition_failed"
            return result

        result["required_skills"] = [s.model_dump() for s in decomp.required_skills]
        result["reasoning_steps"] = decomp.reasoning_steps

        # Pass 2 + 3: Answer and verify each sub-question
        verified_subs = []
        for sq in decomp.sub_questions:
            sub_result = {
                "question": sq.question,
                "difficulty": sq.difficulty,
                "target_skill": sq.target_skill,
            }

            # Generate answer
            ans = await self.answer(sq.question)
            if not ans:
                sub_result["answer"] = None
                sub_result["verified"] = False
                sub_result["error"] = "answer_generation_failed"
            else:
                # Verify via debate
                verification = await self.verify(sq.question, ans.answer)
                if verification and verification.is_correct:
                    sub_result["answer"] = ans.answer
                    sub_result["verified"] = True
                    sub_result["confidence"] = ans.confidence
                elif verification and verification.final_answer:
                    # Use corrected answer
                    sub_result["answer"] = verification.final_answer
                    sub_result["verified"] = True
                    sub_result["was_corrected"] = True
                else:
                    sub_result["answer"] = ans.answer
                    sub_result["verified"] = False
                    sub_result["critique"] = verification.critique if verification else "verification_failed"

            verified_subs.append(sub_result)

        result["sub_questions"] = verified_subs
        result["success"] = True
        return result

    # -------------------------------------------------------------------------
    # Continuous processing
    # -------------------------------------------------------------------------
    async def process_continuous(
        self,
        questions: List[str],
        solutions: List[Optional[str]],
        difficulties: List[Optional[int]],
        hints_list: List[Optional[List[str]]],
        pbar: Optional[tqdm] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        await self._init_model()
        n = len(questions)
        next_idx = 0
        pending: Dict[asyncio.Task, int] = {}

        def submit(idx: int) -> asyncio.Task:
            task = asyncio.create_task(
                self.process_one(
                    questions[idx], solutions[idx], idx,
                    difficulties[idx], hints_list[idx]
                )
            )
            pending[task] = idx
            return task

        while next_idx < n and len(pending) < self.max_concurrent:
            submit(next_idx)
            next_idx += 1

        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                del pending[task]
                result = task.result()
                if pbar:
                    pbar.update(1)
                yield result
                if next_idx < n:
                    submit(next_idx)
                    next_idx += 1


# =============================================================================
# Skill Tree Aggregation
# =============================================================================

def aggregate_skill_tree(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate skill requirements across all questions to build global skill tree."""
    skill_counts: Dict[str, int] = {}
    skill_levels: Dict[str, List[int]] = {}
    skill_prereqs: Dict[str, set] = {}

    for r in results:
        if not r.get("success"):
            continue
        for skill in r.get("required_skills", []):
            name = skill["name"]
            skill_counts[name] = skill_counts.get(name, 0) + 1
            skill_levels.setdefault(name, []).append(skill["level"])
            skill_prereqs.setdefault(name, set()).update(skill.get("prerequisites", []))

    tree = {}
    for name in skill_counts:
        tree[name] = {
            "count": skill_counts[name],
            "avg_level": sum(skill_levels[name]) / len(skill_levels[name]),
            "max_level": max(skill_levels[name]),
            "prerequisites": list(skill_prereqs[name]),
        }

    sorted_skills = sorted(tree.items(), key=lambda x: -x[1]["count"])
    return {"skills": dict(sorted_skills), "total_questions": len(results)}


# =============================================================================
# Visualization
# =============================================================================

def visualize_decomposition(result: Dict[str, Any], max_width: int = 80) -> str:
    """Generate ASCII visualization of a decomposed question."""
    if not result.get("success"):
        return f"[FAILED] {result.get('error', 'unknown error')}"

    lines = []
    diff = result.get("original_difficulty", "?")
    steps = result.get("reasoning_steps", "?")

    # Header
    lines.append("=" * max_width)
    lines.append(f"QUESTION DECOMPOSITION  [Difficulty: {diff}/10, Steps: {steps}]")
    lines.append("=" * max_width)

    # Original question (truncated)
    q = result.get("original_question", "")[:200]
    lines.append(f"\n📋 ORIGINAL: {q}{'...' if len(result.get('original_question', '')) > 200 else ''}")

    # Original hints if available
    hints = result.get("original_hints")
    if hints:
        lines.append("\n💡 HINTS (from hint_generator):")
        for i, h in enumerate(hints, 1):
            lines.append(f"   {i}. {h[:70]}{'...' if len(h) > 70 else ''}")

    # Skill tree
    lines.append("\n🌳 SKILL TREE:")
    skills = result.get("required_skills", [])
    for skill in skills:
        prereqs = skill.get("prerequisites", [])
        prereq_str = f" ← requires: {', '.join(prereqs)}" if prereqs else ""
        lines.append(f"   [{skill['level']:2d}] {skill['name']}{prereq_str}")

    # Sub-questions curriculum
    lines.append("\n📚 CURRICULUM (easy → hard):")
    lines.append("   ┌" + "─" * 60)
    subs = result.get("sub_questions", [])
    for i, sq in enumerate(subs):
        verified = "✓" if sq.get("verified") else "✗"
        corrected = " (corrected)" if sq.get("was_corrected") else ""
        skill = sq.get("target_skill", "?")
        diff = sq.get("difficulty", "?")

        lines.append(f"   │ Q{i+1} [L{diff}] ({skill})")
        q_text = sq.get("question", "")[:55]
        lines.append(f"   │    {q_text}{'...' if len(sq.get('question', '')) > 55 else ''}")

        ans = sq.get("answer")
        if ans:
            ans_text = str(ans)[:50]
            lines.append(f"   │    → {verified} {ans_text}{corrected}")
        else:
            lines.append(f"   │    → {verified} [no answer]")

        if i < len(subs) - 1:
            lines.append("   │")
            lines.append("   ▼")

    lines.append("   └" + "─" * 60)
    lines.append("")

    return "\n".join(lines)


def visualize_skill_tree(tree: Dict[str, Any], top_n: int = 15) -> str:
    """Visualize global skill tree aggregation."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"GLOBAL SKILL TREE  [{tree.get('total_questions', 0)} questions analyzed]")
    lines.append("=" * 70)

    skills = tree.get("skills", {})
    for i, (name, data) in enumerate(list(skills.items())[:top_n]):
        count = data["count"]
        avg = data["avg_level"]
        max_lvl = data["max_level"]
        prereqs = data.get("prerequisites", [])

        bar_len = int(count / max(s["count"] for s in skills.values()) * 30)
        bar = "█" * bar_len

        lines.append(f"\n{name}")
        lines.append(f"   Count: {count:4d} {bar}")
        lines.append(f"   Level: avg={avg:.1f}, max={max_lvl}")
        if prereqs:
            lines.append(f"   Requires: {', '.join(prereqs[:5])}")

    if len(skills) > top_n:
        lines.append(f"\n... and {len(skills) - top_n} more skills")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def extract_problem(row: Dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        msg = prompt[0]
        if isinstance(msg, dict):
            return msg.get("content", "")
    return str(prompt) if prompt else ""


def extract_solution(row: Dict[str, Any]) -> Optional[str]:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if not gt:
            return None
        return gt if isinstance(gt, str) else json.dumps(gt)
    sol = row.get("solution") or row.get("answer") or row.get("ground_truth")
    if sol:
        return sol if isinstance(sol, str) else json.dumps(sol)
    return None


def extract_hints(row: Dict[str, Any]) -> Optional[List[str]]:
    """Extract hints from enriched parquet row."""
    hints = row.get("hints")
    if not hints:
        return None
    if isinstance(hints, str):
        try:
            return json.loads(hints)
        except Exception:
            return None
    if isinstance(hints, list):
        return hints
    return None


def extract_difficulty(row: Dict[str, Any]) -> Optional[int]:
    """Extract difficulty from enriched parquet row."""
    diff = row.get("estimated_difficulty")
    if diff is None:
        return None
    try:
        return int(diff)
    except Exception:
        return None


async def process_parquet(
    input_path: str,
    output_dir: str,
    decomposer: QuestionDecomposer,
    chunk_size: int = 500,
    filter_difficulty: Optional[int] = None,
):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    n = len(df)
    print(f"Loaded {n} rows from {input_path}")

    rows = [df.iloc[i].to_dict() for i in range(n)]
    questions = [extract_problem(r) for r in rows]
    solutions = [extract_solution(r) for r in rows]

    # Extract pre-computed hints and difficulty (from hint_generator.py)
    difficulties = [extract_difficulty(r) for r in rows]
    hints_list = [extract_hints(r) for r in rows]

    has_hints = sum(1 for h in hints_list if h)
    has_diff = sum(1 for d in difficulties if d)
    print(f"  Found {has_diff} with difficulty, {has_hints} with hints (from hint_generator)")

    # Filter by difficulty if requested
    if filter_difficulty and has_diff > 0:
        indices = [i for i in range(n) if (difficulties[i] or 0) >= filter_difficulty]
        print(f"Filtering to {len(indices)} questions with difficulty >= {filter_difficulty}")
        questions = [questions[i] for i in indices]
        solutions = [solutions[i] for i in indices]
        difficulties = [difficulties[i] for i in indices]
        hints_list = [hints_list[i] for i in indices]
        rows = [rows[i] for i in indices]
        n = len(questions)

    results: Dict[int, Dict] = {}
    chunk_idx = 0
    last_saved = 0

    def save_chunk(start: int, end: int):
        nonlocal chunk_idx
        chunk_path = output_dir / f"decomposed_{chunk_idx:04d}.jsonl"
        with open(chunk_path, "w") as f:
            for i in range(start, end):
                if i in results:
                    f.write(json.dumps(results[i], default=str) + "\n")
        print(f"\nSaved {chunk_path}")
        chunk_idx += 1

    pbar = tqdm(total=n, desc="Decomposing questions", unit="q")

    async for result in decomposer.process_continuous(questions, solutions, difficulties, hints_list, pbar):
        results[result["index"]] = result

        while last_saved in results:
            last_saved += 1
        if last_saved > 0 and last_saved % chunk_size == 0:
            save_chunk(last_saved - chunk_size, last_saved)

    pbar.close()

    if last_saved % chunk_size != 0:
        save_chunk((last_saved // chunk_size) * chunk_size, last_saved)

    # Aggregate skill tree
    all_results = [results[i] for i in range(n) if i in results]
    skill_tree = aggregate_skill_tree(all_results)

    tree_path = output_dir / "skill_tree.json"
    with open(tree_path, "w") as f:
        json.dump(skill_tree, f, indent=2)
    print(f"\nSkill tree: {tree_path}")
    print(visualize_skill_tree(skill_tree))

    # Save a sample visualization
    success_results = [r for r in all_results if r.get("success")]
    if success_results:
        sample_path = output_dir / "sample_visualization.txt"
        with open(sample_path, "w") as f:
            for r in success_results[:3]:
                f.write(visualize_decomposition(r) + "\n\n")
        print(f"\nSample visualizations: {sample_path}")

    success = len(success_results)
    print(f"\nDone: {success}/{n} successfully decomposed")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_file", help="Input parquet or JSONL (enriched or decomposed)")
    p.add_argument("output_dir", nargs="?", default="./decomposed", help="Output directory")
    p.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--model", default=None)
    p.add_argument("--max-concurrent", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--min-difficulty", type=int, default=None, help="Only process questions >= this difficulty")
    p.add_argument("--visualize", type=int, default=None, metavar="N", help="Just visualize N samples from decomposed JSONL, don't process")
    args = p.parse_args()

    # Visualize mode: just show samples from already-decomposed data
    if args.visualize is not None:
        if args.input_file.endswith(".jsonl"):
            with open(args.input_file) as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            import pandas as pd
            df = pd.read_parquet(args.input_file)
            data = [df.iloc[i].to_dict() for i in range(len(df))]

        success = [d for d in data if d.get("success")]
        print(f"Loaded {len(data)} results, {len(success)} successful\n")

        for r in success[:args.visualize]:
            print(visualize_decomposition(r))
            print()
        return

    # Full processing mode
    decomposer = QuestionDecomposer(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )

    asyncio.run(process_parquet(
        args.input_file,
        args.output_dir,
        decomposer,
        args.chunk_size,
        args.min_difficulty,
    ))


if __name__ == "__main__":
    main()
