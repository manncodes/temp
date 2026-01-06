"""
Generate difficulty + progressive hints for POPE-style RL training.
Uses continuous batching to keep vLLM saturated. Saves checkpoints as chunk files.

Usage: python hint_generator.py train.parquet output_dir/ --base-url http://0.0.0.0:8000/v1
"""

import asyncio
import json
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, AsyncIterator
from pathlib import Path

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm import tqdm

if "KUBERNETES_SERVICE_HOST" in os.environ:
    for var in ("NO_PROXY", "no_proxy"):
        if os.getenv(var):
            os.environ[var] += ",.svc.cluster.local"


class ProgressiveHints(BaseModel):
    estimated_difficulty: int = Field(ge=1, le=10)
    hints: List[str] = Field(min_length=5, max_length=5)


SYSTEM_PROMPT = '''Generate difficulty (1-10) and 5 progressive hints for this problem.

Difficulty: 1-2 trivial, 3-4 standard, 5-6 needs insight, 7-8 multiple insights, 9-10 competition-level.

Hints must be strictly progressive:
1. Orientation: Problem type only ("This is a DP problem")
2. Approach: General technique ("Use memoization")
3. Key Insight: The "aha" moment ("dp[i] = max(dp[i-1], dp[i-2] + val[i])")
4. Structure: Solution outline with steps
5. Near-Complete: Everything except final computation

Rules: Hint 1 must not enable solving. Hint 5 makes it trivial. 1-3 sentences each. JSON only.'''


@dataclass
class Result:
    index: int
    hints: Optional[ProgressiveHints]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.hints is not None


class HintGenerator:
    """Continuous batching hint generator - keeps vLLM saturated."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000/v1",
        api_key: str = "dummy",
        model: Optional[str] = None,
        max_concurrent: int = 128,
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

    def _make_prompt(self, problem: str, solution: str) -> str:
        return f"## Problem\n{problem}\n\n## Reference Solution\n{solution}"

    async def _generate_one(self, problem: str, solution: str, idx: int) -> Result:
        for attempt in range(self.max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": self._make_prompt(problem, solution)},
                        ],
                        response_format=ProgressiveHints,
                        temperature=0.7,
                    ),
                    timeout=self.timeout,
                )
                parsed = resp.choices[0].message.parsed
                if parsed:
                    return Result(idx, parsed)
                raise ValueError("Parse failed")
            except asyncio.CancelledError:
                return Result(idx, None, "cancelled")
            except Exception as e:
                if attempt == self.max_retries:
                    return Result(idx, None, str(e))
                await asyncio.sleep(0.5 * (2 ** attempt))
        return Result(idx, None, "exhausted retries")

    async def generate_continuous(
        self,
        problems: List[str],
        solutions: List[str],
        pbar: Optional[tqdm] = None,
    ) -> AsyncIterator[Result]:
        """
        Continuous batching: maintains max_concurrent in-flight requests.
        Yields results as they complete, immediately backfilling with new requests.
        """
        await self._init_model()
        n = len(problems)
        next_idx = 0
        pending: Dict[asyncio.Task, int] = {}

        def submit(idx: int) -> asyncio.Task:
            task = asyncio.create_task(self._generate_one(problems[idx], solutions[idx], idx))
            pending[task] = idx
            return task

        # Initial fill
        while next_idx < n and len(pending) < self.max_concurrent:
            submit(next_idx)
            next_idx += 1

        # Process as they complete, backfill immediately
        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                del pending[task]
                result = task.result()
                if pbar:
                    pbar.update(1)
                yield result

                # Backfill: submit next request immediately
                if next_idx < n:
                    submit(next_idx)
                    next_idx += 1


def extract_problem(row: Dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        msg = prompt[0]
        if isinstance(msg, dict):
            return msg.get("content", "")
    return str(prompt) if prompt else ""


def extract_solution(row: Dict[str, Any]) -> str:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth", "")
        return gt if isinstance(gt, str) else json.dumps(gt)
    return str(rm) if rm else ""


async def enrich_parquet(
    input_path: str,
    output_dir: str,
    generator: HintGenerator,
    chunk_size: int = 1000,
):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    n = len(df)
    print(f"Loaded {n} rows from {input_path}")

    # Pre-extract all problems/solutions
    rows = [df.iloc[i].to_dict() for i in range(n)]
    problems = [extract_problem(r) for r in rows]
    solutions = [extract_solution(r) for r in rows]

    # Track results by index
    results: Dict[int, Result] = {}
    success_count = 0
    chunk_idx = 0
    last_saved = 0

    def save_chunk(start: int, end: int):
        """Save a chunk of results to JSONL."""
        nonlocal chunk_idx
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_path, "w") as f:
            for i in range(start, end):
                if i not in results:
                    continue
                r = results[i]
                row = rows[i].copy()
                row["estimated_difficulty"] = r.hints.estimated_difficulty if r.success else None
                row["hints"] = r.hints.hints if r.success else None
                row["hint_generation_success"] = r.success
                if not r.success:
                    row["hint_generation_error"] = r.error
                f.write(json.dumps(row, default=str) + "\n")
        print(f"\nSaved {chunk_path} ({end - start} rows)")
        chunk_idx += 1

    # Global progress bar across entire dataset
    pbar = tqdm(total=n, desc="Generating hints", unit="row")

    async for result in generator.generate_continuous(problems, solutions, pbar):
        results[result.index] = result
        if result.success:
            success_count += 1

        # Save chunk when we have chunk_size completed results in order
        # Find how many consecutive results we have from last_saved
        while last_saved in results:
            last_saved += 1

        # Save when we have a full chunk of consecutive results
        completed_consecutive = last_saved
        if completed_consecutive > 0 and completed_consecutive % chunk_size == 0:
            chunk_start = completed_consecutive - chunk_size
            save_chunk(chunk_start, completed_consecutive)

    pbar.close()

    # Save remaining results
    if last_saved % chunk_size != 0:
        chunk_start = (last_saved // chunk_size) * chunk_size
        save_chunk(chunk_start, last_saved)

    # Also save complete parquet
    all_diff, all_hints, all_ok, all_err = [], [], [], []
    for i in range(n):
        r = results.get(i)
        if r and r.success:
            all_diff.append(r.hints.estimated_difficulty)
            all_hints.append(json.dumps(r.hints.hints))
            all_ok.append(True)
            all_err.append(None)
        else:
            all_diff.append(None)
            all_hints.append(None)
            all_ok.append(False)
            all_err.append(r.error if r else "missing")

    df["estimated_difficulty"] = all_diff
    df["hints"] = all_hints
    df["hint_generation_success"] = all_ok
    df["hint_generation_error"] = all_err

    final_path = output_dir / "enriched.parquet"
    df.to_parquet(final_path, index=False)
    print(f"\nDone: {success_count}/{n} ({100*success_count/n:.1f}%)")
    print(f"Saved: {final_path}")


async def enrich_jsonl(
    input_path: str,
    output_dir: str,
    generator: HintGenerator,
    chunk_size: int = 1000,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    n = len(data)
    print(f"Loaded {n} rows from {input_path}")

    problems = [extract_problem(d) for d in data]
    solutions = [extract_solution(d) for d in data]

    results: Dict[int, Result] = {}
    success_count = 0
    chunk_idx = 0
    last_saved = 0

    def save_chunk(start: int, end: int):
        nonlocal chunk_idx
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_path, "w") as f:
            for i in range(start, end):
                if i not in results:
                    continue
                r = results[i]
                row = data[i].copy()
                row["estimated_difficulty"] = r.hints.estimated_difficulty if r.success else None
                row["hints"] = r.hints.hints if r.success else None
                row["hint_generation_success"] = r.success
                if not r.success:
                    row["hint_generation_error"] = r.error
                f.write(json.dumps(row, default=str) + "\n")
        print(f"\nSaved {chunk_path}")
        chunk_idx += 1

    pbar = tqdm(total=n, desc="Generating hints", unit="row")

    async for result in generator.generate_continuous(problems, solutions, pbar):
        results[result.index] = result
        if result.success:
            success_count += 1

        while last_saved in results:
            last_saved += 1

        if last_saved > 0 and last_saved % chunk_size == 0:
            save_chunk(last_saved - chunk_size, last_saved)

    pbar.close()

    if last_saved % chunk_size != 0:
        save_chunk((last_saved // chunk_size) * chunk_size, last_saved)

    print(f"\nDone: {success_count}/{n} ({100*success_count/n:.1f}%)")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_file", help="Input parquet or JSONL")
    p.add_argument("output_dir", help="Output directory for chunks and final parquet")
    p.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--model", default=None)
    p.add_argument("--max-concurrent", type=int, default=128, help="In-flight requests (keep high for vLLM)")
    p.add_argument("--chunk-size", type=int, default=1000, help="Save checkpoint every N rows")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    gen = HintGenerator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )

    if args.input_file.endswith(".parquet"):
        asyncio.run(enrich_parquet(args.input_file, args.output_dir, gen, args.chunk_size))
    else:
        asyncio.run(enrich_jsonl(args.input_file, args.output_dir, gen, args.chunk_size))


if __name__ == "__main__":
    main()
