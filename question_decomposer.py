"""
Skill-Tree Question Decomposer for RL Curriculum Generation.

Takes hard questions (zero/low pass rate) and decomposes them into:
1. Required skills (skill tree nodes)
2. Prerequisite sub-questions at lower difficulty
3. Verified Q&A pairs via debate

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
    original_difficulty: int = Field(ge=1, le=10)
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

DECOMPOSE_SYSTEM = '''You are an expert at analyzing problem difficulty and creating learning curricula.

Given a hard problem, you must:
1. Estimate difficulty (1-10)
2. Identify required skills as a skill tree (with prerequisites)
3. Generate 3-6 sub-questions that build up to solving the original
   - Start at difficulty ~3 below the original
   - Each sub-question targets one skill
   - Progress from foundational to advanced
4. Count minimum reasoning steps needed

Skill naming: use snake_case like "modular_arithmetic", "graph_traversal", "integration_by_parts"

Sub-questions must be SELF-CONTAINED (solvable without seeing the original).

JSON only.'''

ANSWER_SYSTEM = '''Solve this problem step by step. Be precise and show your work.
Output JSON with: answer (final answer), confidence (0-1), reasoning (your steps).'''

DEBATE_SYSTEM = '''You are a critical reviewer. Given a question and proposed answer:
1. Check if the answer is correct
2. Identify any errors in reasoning
3. If wrong, provide the correct answer

Be rigorous. Output JSON with: is_correct (bool), critique (explanation), final_answer (corrected answer if needed, else null).'''


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
    async def decompose(self, question: str, solution: Optional[str] = None) -> Optional[QuestionDecomposition]:
        context = f"## Problem\n{question}"
        if solution:
            context += f"\n\n## Reference Solution\n{solution}"
        return await self._call(DECOMPOSE_SYSTEM, context, QuestionDecomposition)

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
    async def process_one(self, question: str, solution: Optional[str], idx: int) -> Dict[str, Any]:
        result = {
            "index": idx,
            "original_question": question,
            "success": False,
        }

        # Pass 1: Decompose
        decomp = await self.decompose(question, solution)
        if not decomp:
            result["error"] = "decomposition_failed"
            return result

        result["original_difficulty"] = decomp.original_difficulty
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
        pbar: Optional[tqdm] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        await self._init_model()
        n = len(questions)
        next_idx = 0
        pending: Dict[asyncio.Task, int] = {}

        def submit(idx: int) -> asyncio.Task:
            task = asyncio.create_task(self.process_one(questions[idx], solutions[idx], idx))
            pending[task] = idx
            return task

        # Initial fill
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

    # Build tree
    tree = {}
    for name in skill_counts:
        tree[name] = {
            "count": skill_counts[name],
            "avg_level": sum(skill_levels[name]) / len(skill_levels[name]),
            "max_level": max(skill_levels[name]),
            "prerequisites": list(skill_prereqs[name]),
        }

    # Sort by frequency (most needed skills first)
    sorted_skills = sorted(tree.items(), key=lambda x: -x[1]["count"])
    return {"skills": dict(sorted_skills), "total_questions": len(results)}


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

    # Optionally filter by difficulty if we have it
    if filter_difficulty and "estimated_difficulty" in df.columns:
        indices = [i for i in range(n) if df.iloc[i].get("estimated_difficulty", 0) >= filter_difficulty]
        print(f"Filtering to {len(indices)} questions with difficulty >= {filter_difficulty}")
        questions = [questions[i] for i in indices]
        solutions = [solutions[i] for i in indices]
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

    async for result in decomposer.process_continuous(questions, solutions, pbar):
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
    print(f"\nSkill tree saved to {tree_path}")

    # Summary
    success = sum(1 for r in all_results if r.get("success"))
    print(f"Done: {success}/{n} successfully decomposed")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_file", help="Input parquet or JSONL")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--model", default=None)
    p.add_argument("--max-concurrent", type=int, default=32, help="Lower than hint gen due to multi-pass")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--min-difficulty", type=int, default=None, help="Only process questions >= this difficulty")
    args = p.parse_args()

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
